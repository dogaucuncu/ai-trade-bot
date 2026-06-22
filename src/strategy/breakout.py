"""
Breakout strategy — trade Donchian channel breakouts with volume confirmation.

Enters in the direction of a range break: when price closes above the highest
high of the last *N* bars (Donchian upper), or below the lowest low (Donchian
lower). A volume spike confirms the break to filter out false moves.

Entry / exit logic
------------------
* **BUY**  when close breaks above the prior ``channel_period`` high.
* **SELL** when close breaks below the prior ``channel_period`` low.
* Stop-loss / take-profit are percentage-based; an ATR floor widens the stop in
  volatile regimes so normal noise doesn't knock the position out.
* Designed for **1 h** / **4 h** trend timeframes.

Usage
-----
>>> strategy = BreakoutStrategy()
>>> signal = strategy.analyze(ohlcv_df)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from src.strategy.base import BaseStrategy, Position, Signal, TradeAction


class BreakoutStrategy(BaseStrategy):
    """Donchian-channel breakout strategy (1 h / 4 h charts)."""

    DEFAULT_CONFIG: dict[str, Any] = {
        "channel_period": 20,    # Donchian look-back
        "target_pct": 0.03,      # 3% take-profit
        "stop_loss_pct": 0.02,   # 2% stop-loss
        "atr_period": 14,
        "atr_mult": 1.5,         # ATR floor for the stop distance
        "volume_period": 20,
        "volume_threshold": 1.3, # require a volume spike on the break
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name="breakout", config=merged)

    # ------------------------------------------------------------------ core

    def analyze(self, df: pd.DataFrame) -> Signal:
        """Produce a breakout signal from OHLCV data."""
        symbol: str = df.attrs.get("symbol", "UNKNOWN")

        if not self.validate_dataframe(df):
            return self._hold_signal(symbol)

        period = self.config["channel_period"]
        if len(df) < period + 2:
            return self._hold_signal(symbol)

        close = df["close"]
        current_price = float(close.iloc[-1])

        # Donchian channel from *prior* bars (exclude the current bar so the
        # break is genuine, not the bar setting its own high).
        upper = float(df["high"].iloc[-(period + 1):-1].max())
        lower = float(df["low"].iloc[-(period + 1):-1].min())

        vol_ratio = self.calc_volume_ratio(df["volume"], self.config["volume_period"])
        current_vol = float(vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else 1.0

        atr = self.calc_atr(df, self.config["atr_period"])
        current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

        meta: dict[str, Any] = {
            "donchian_upper": round(upper, 8),
            "donchian_lower": round(lower, 8),
            "volume_ratio": round(current_vol, 4),
            "atr": round(current_atr, 8),
            "current_price": current_price,
            "timeframe": self.config.get("timeframe", "1h"),
        }

        # Confidence: how decisively the break clears the channel, plus volume.
        channel_width = (upper - lower) or current_price
        vol_bonus = min(0.20, max(0.0, (current_vol - 1.0) * 0.15))

        if current_price > upper:
            break_excess = (current_price - upper) / channel_width
            confidence = min(1.0, 0.5 + break_excess * 5 + vol_bonus)
            stop_dist = max(
                current_price * self.config["stop_loss_pct"],
                current_atr * self.config["atr_mult"],
            )
            stop_loss = current_price - stop_dist
            take_profit = current_price * (1 + self.config["target_pct"])
            action = TradeAction.BUY
            logger.info(
                "[breakout] BUY — {} price={:.6f} > upper={:.6f} conf={:.2f}",
                symbol, current_price, upper, confidence,
            )

        elif current_price < lower:
            break_excess = (lower - current_price) / channel_width
            confidence = min(1.0, 0.5 + break_excess * 5 + vol_bonus)
            stop_dist = max(
                current_price * self.config["stop_loss_pct"],
                current_atr * self.config["atr_mult"],
            )
            stop_loss = current_price + stop_dist
            take_profit = current_price * (1 - self.config["target_pct"])
            action = TradeAction.SELL
            logger.info(
                "[breakout] SELL — {} price={:.6f} < lower={:.6f} conf={:.2f}",
                symbol, current_price, lower, confidence,
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
        """Confirm the break with confidence and a volume spike."""
        if signal.action == TradeAction.HOLD:
            return False
        if signal.confidence < 0.5:
            logger.debug("[breakout] Confidence {:.2f} below threshold", signal.confidence)
            return False

        vol_ratio = self.calc_volume_ratio(df["volume"], self.config["volume_period"])
        if float(vol_ratio.iloc[-1]) < self.config["volume_threshold"]:
            logger.debug("[breakout] Volume too low — skip break")
            return False

        return True

    def should_exit(self, df: pd.DataFrame, position: Position) -> bool:
        """Exit on SL / TP only (trend is given room to run)."""
        current_price = float(df["close"].iloc[-1])
        if position.is_stop_hit(current_price) or position.is_tp_hit(current_price):
            logger.info(
                "[breakout] SL/TP hit for {} @ {:.6f}", position.symbol, current_price
            )
            return True
        return False
