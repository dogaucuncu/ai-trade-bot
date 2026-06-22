"""
VWAP reversion strategy — fade stretched moves away from the rolling VWAP.

Computes a rolling Volume-Weighted Average Price (VWAP) and trades the snap-back
when price deviates too far from it. Because VWAP weights by volume it is a
sturdier "fair value" anchor than a plain moving average, which makes it a good
fit for intraday mean-reversion — especially on equities.

Entry / exit logic
------------------
* **BUY**  when price trades ``band_pct`` *below* VWAP (stretched down).
* **SELL** when price trades ``band_pct`` *above* VWAP (stretched up).
* Take-profit / stop-loss are percentage-based; the position can also exit early
  once price reverts back through the VWAP.
* Designed for **5 m** / **15 m** intraday timeframes.

Usage
-----
>>> strategy = VWAPReversionStrategy()
>>> signal = strategy.analyze(ohlcv_df)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from src.strategy.base import BaseStrategy, Position, Signal, TradeAction


class VWAPReversionStrategy(BaseStrategy):
    """Rolling-VWAP reversion strategy (5 m / 15 m charts)."""

    DEFAULT_CONFIG: dict[str, Any] = {
        "vwap_period": 20,       # rolling VWAP window
        "band_pct": 0.015,       # deviation from VWAP to trigger (1.5%)
        "target_pct": 0.015,     # 1.5% take-profit
        "stop_loss_pct": 0.015,  # 1.5% stop-loss
        "volume_period": 20,
        "volume_threshold": 1.0,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name="vwap_reversion", config=merged)

    # ------------------------------------------------------------------ vwap

    def _rolling_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Volume-weighted average price over a rolling window."""
        period = self.config["vwap_period"]
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = (typical * df["volume"]).rolling(window=period).sum()
        vol = df["volume"].rolling(window=period).sum().replace(0, pd.NA)
        return pv / vol

    # ------------------------------------------------------------------ core

    def analyze(self, df: pd.DataFrame) -> Signal:
        """Produce a VWAP-reversion signal from OHLCV data."""
        symbol: str = df.attrs.get("symbol", "UNKNOWN")

        if not self.validate_dataframe(df):
            return self._hold_signal(symbol)

        close = df["close"]
        current_price = float(close.iloc[-1])

        vwap = self._rolling_vwap(df)
        current_vwap = vwap.iloc[-1]
        if pd.isna(current_vwap) or current_vwap == 0:
            return self._hold_signal(symbol)
        current_vwap = float(current_vwap)

        deviation = (current_price - current_vwap) / current_vwap

        vol_ratio = self.calc_volume_ratio(df["volume"], self.config["volume_period"])
        current_vol = float(vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else 1.0

        band = self.config["band_pct"]
        meta: dict[str, Any] = {
            "vwap": round(current_vwap, 8),
            "deviation_pct": round(deviation * 100, 4),
            "volume_ratio": round(current_vol, 4),
            "current_price": current_price,
            "timeframe": self.config.get("timeframe", "15m"),
        }

        # Confidence scales with how far past the band the price is stretched.
        excess = (abs(deviation) - band) / band if band else 0.0
        confidence = min(1.0, 0.5 + max(0.0, excess) * 0.5)

        if deviation < -band:
            stop_loss = current_price * (1 - self.config["stop_loss_pct"])
            take_profit = current_price * (1 + self.config["target_pct"])
            action = TradeAction.BUY
            logger.info(
                "[vwap_reversion] BUY — {} dev={:.2%} price={:.6f} conf={:.2f}",
                symbol, deviation, current_price, confidence,
            )
        elif deviation > band:
            stop_loss = current_price * (1 + self.config["stop_loss_pct"])
            take_profit = current_price * (1 - self.config["target_pct"])
            action = TradeAction.SELL
            logger.info(
                "[vwap_reversion] SELL — {} dev={:.2%} price={:.6f} conf={:.2f}",
                symbol, deviation, current_price, confidence,
            )
        else:
            return self._hold_signal(symbol)

        return Signal(
            symbol=symbol,
            action=action,
            confidence=round(confidence, 4),
            stop_loss_price=round(stop_loss, 8),
            take_profit_price=round(take_profit, 8),
            strategy_name=self.name,
            metadata=meta,
        )

    # ---------------------------------------------------------------- gates

    def should_enter(self, df: pd.DataFrame, signal: Signal) -> bool:
        """Confirm with a confidence floor and minimum volume."""
        if signal.action == TradeAction.HOLD:
            return False
        if signal.confidence < 0.45:
            logger.debug("[vwap_reversion] Confidence {:.2f} below threshold", signal.confidence)
            return False
        vol_ratio = self.calc_volume_ratio(df["volume"], self.config["volume_period"])
        if float(vol_ratio.iloc[-1]) < self.config["volume_threshold"]:
            logger.debug("[vwap_reversion] Volume too low — skip entry")
            return False
        return True

    def should_exit(self, df: pd.DataFrame, position: Position) -> bool:
        """Exit on SL / TP, or once price reverts through the VWAP."""
        current_price = float(df["close"].iloc[-1])
        if position.is_stop_hit(current_price) or position.is_tp_hit(current_price):
            logger.info(
                "[vwap_reversion] SL/TP hit for {} @ {:.6f}", position.symbol, current_price
            )
            return True

        vwap = self._rolling_vwap(df).iloc[-1]
        if pd.isna(vwap):
            return False
        vwap = float(vwap)
        # Reverted back to fair value — take the mean-reversion profit/cut.
        if position.side == TradeAction.BUY and current_price >= vwap:
            logger.info("[vwap_reversion] Reverted to VWAP (long) — exit {}", position.symbol)
            return True
        if position.side == TradeAction.SELL and current_price <= vwap:
            logger.info("[vwap_reversion] Reverted to VWAP (short) — exit {}", position.symbol)
            return True
        return False
