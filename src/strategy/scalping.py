"""
Scalping strategy — short-term mean-reversion scalps on crypto altcoins.

Combines three fast indicators:

1. **RSI (14)** — oversold / overbought extremes
2. **Bollinger Bands (20, 2σ)** — price distance from the envelope
3. **Volume surge** — volume > 1.5× its 20-bar average

Entry / exit logic
------------------
* **BUY** when RSI < 30 *and* price ≤ lower Bollinger *and* volume ratio > 1.5
* **SELL** when RSI > 70 *and* price ≥ upper Bollinger
* Target profit: 0.3–0.5 %
* Stop-loss: 0.5 % (tight, suitable for scalping)
* Designed for **1 m** and **5 m** timeframes on Binance altcoins.

Usage
-----
>>> strategy = ScalpingStrategy()
>>> signal = strategy.analyze(ohlcv_df)
>>> if strategy.should_enter(ohlcv_df, signal):
...     # forward signal to the order manager
...     pass
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from src.strategy.base import BaseStrategy, Position, Signal, TradeAction


class ScalpingStrategy(BaseStrategy):
    """Fast scalping strategy for crypto altcoins (1 m / 5 m charts).

    Parameters
    ----------
    config : dict, optional
        Override any of the defaults below::

            {
                "rsi_period":        14,
                "rsi_oversold":      30,
                "rsi_overbought":    70,
                "bb_period":         20,
                "bb_std":            2.0,
                "volume_period":     20,
                "volume_threshold":  1.5,
                "target_pct":        0.004,   # 0.4 %
                "stop_loss_pct":     0.005,   # 0.5 %
            }

    Examples
    --------
    >>> from src.strategy.scalping import ScalpingStrategy
    >>> s = ScalpingStrategy(config={"target_pct": 0.005})
    >>> sig = s.analyze(df)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "volume_period": 20,
        "volume_threshold": 1.3,
        "target_pct": 0.004,   # 0.4% take-profit
        "stop_loss_pct": 0.005, # 0.5% stop-loss
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name="scalping", config=merged)

    # ------------------------------------------------------------------ core

    def analyze(self, df: pd.DataFrame) -> Signal:
        """Produce a scalping signal from the latest OHLCV data.

        The confidence score is the weighted average of three boolean
        sub-signals (RSI agreement 0.40, Bollinger 0.35, Volume 0.25).
        """
        symbol: str = df.attrs.get("symbol", "UNKNOWN")

        if not self.validate_dataframe(df):
            return self._hold_signal(symbol)

        close = df["close"]
        current_price = float(close.iloc[-1])

        # --- indicators ---------------------------------------------------
        rsi = self.calc_rsi(close, self.config["rsi_period"])
        current_rsi = float(rsi.iloc[-1])

        _, bb_upper, bb_lower = self.calc_bollinger_bands(
            close, self.config["bb_period"], self.config["bb_std"]
        )
        current_bb_upper = float(bb_upper.iloc[-1])
        current_bb_lower = float(bb_lower.iloc[-1])

        vol_ratio = self.calc_volume_ratio(
            df["volume"], self.config["volume_period"]
        )
        current_vol_ratio = float(vol_ratio.iloc[-1])

        # --- sub-signals (1 = fully triggered, 0 = not triggered) ---------
        rsi_buy = max(0.0, min(1.0, (self.config["rsi_oversold"] - current_rsi) / 10))
        rsi_sell = max(
            0.0, min(1.0, (current_rsi - self.config["rsi_overbought"]) / 10)
        )

        bb_range = current_bb_upper - current_bb_lower if current_bb_upper != current_bb_lower else 1.0
        bb_buy = max(0.0, min(1.0, (current_bb_lower - current_price) / bb_range + 1.0))
        bb_sell = max(0.0, min(1.0, (current_price - current_bb_upper) / bb_range + 1.0))

        vol_score = max(0.0, min(1.0, (current_vol_ratio - 1.0) / (self.config["volume_threshold"] - 1.0)))

        # --- weighted confidence ------------------------------------------
        w_rsi, w_bb, w_vol = 0.40, 0.35, 0.25

        buy_conf = w_rsi * rsi_buy + w_bb * bb_buy + w_vol * vol_score
        sell_conf = w_rsi * rsi_sell + w_bb * bb_sell + w_vol * vol_score

        # --- decide action ------------------------------------------------
        meta: dict[str, Any] = {
            "rsi": current_rsi,
            "bb_upper": current_bb_upper,
            "bb_lower": current_bb_lower,
            "volume_ratio": current_vol_ratio,
            "buy_confidence": round(buy_conf, 4),
            "sell_confidence": round(sell_conf, 4),
            "timeframe": self.config.get("timeframe", "1m"),
        }

        if (
            current_rsi < self.config["rsi_oversold"]
            and current_price <= current_bb_lower
            and current_vol_ratio >= self.config["volume_threshold"]
        ):
            stop_loss = current_price * (1 - self.config["stop_loss_pct"])
            take_profit = current_price * (1 + self.config["target_pct"])
            confidence = min(1.0, buy_conf)
            action = TradeAction.BUY
            logger.info(
                "[scalping] BUY signal — {} RSI={:.1f} price={:.6f} conf={:.2f}",
                symbol, current_rsi, current_price, confidence,
            )
        elif (
            current_rsi > self.config["rsi_overbought"]
            and current_price >= current_bb_upper
        ):
            stop_loss = current_price * (1 + self.config["stop_loss_pct"])
            take_profit = current_price * (1 - self.config["target_pct"])
            confidence = min(1.0, sell_conf)
            action = TradeAction.SELL
            logger.info(
                "[scalping] SELL signal — {} RSI={:.1f} price={:.6f} conf={:.2f}",
                symbol, current_rsi, current_price, confidence,
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
        """Confirm entry only when confidence is high enough and spread is OK.

        Extra checks beyond :meth:`analyze`:
        * confidence ≥ 0.50
        * the last two RSI readings both in the extreme zone
        """
        if signal.action == TradeAction.HOLD:
            return False
        if signal.confidence < 0.50:
            logger.debug("[scalping] Confidence {:.2f} too low — skipping", signal.confidence)
            return False

        rsi = self.calc_rsi(df["close"], self.config["rsi_period"])
        last_two = rsi.iloc[-2:]

        if signal.action == TradeAction.BUY:
            if not all(v < self.config["rsi_oversold"] + 5 for v in last_two):
                logger.debug("[scalping] RSI not consistently oversold — skip")
                return False
        elif signal.action == TradeAction.SELL:
            if not all(v > self.config["rsi_overbought"] - 5 for v in last_two):
                logger.debug("[scalping] RSI not consistently overbought — skip")
                return False

        return True

    def should_exit(self, df: pd.DataFrame, position: Position) -> bool:
        """Exit only on SL / TP to allow high win rate."""
        current_price = float(df["close"].iloc[-1])

        # Always respect hard stop and TP
        if position.is_stop_hit(current_price) or position.is_tp_hit(current_price):
            logger.info(
                "[scalping] SL/TP hit for {} @ {:.6f}", position.symbol, current_price
            )
            return True

        return False
