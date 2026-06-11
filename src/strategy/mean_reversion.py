"""
Mean-reversion strategy — trade the return to the moving-average mean.

Uses the **Z-score** of the close price relative to its EMA(21).

Entry / exit logic
------------------
* **BUY** when Z-score < −2  (price significantly below the mean)
* **SELL** when Z-score > +2  (price significantly above the mean)
* Target profit: 1–2 %
* Stop-loss: 1.5 %
* Best on **15 m** and **1 h** timeframes.

Usage
-----
>>> strategy = MeanReversionStrategy()
>>> signal = strategy.analyze(ohlcv_df)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from src.strategy.base import BaseStrategy, Position, Signal, TradeAction


class MeanReversionStrategy(BaseStrategy):
    """Z-score mean-reversion strategy (15 m / 1 h charts).

    Parameters
    ----------
    config : dict, optional
        Override any of the defaults::

            {
                "ema_period":      21,
                "z_score_period":  21,
                "z_entry":         2.0,
                "z_exit":          0.5,
                "target_pct":      0.015,   # 1.5 %
                "stop_loss_pct":   0.015,   # 1.5 %
                "volume_period":   20,
                "volume_threshold": 1.2,
            }

    Examples
    --------
    >>> from src.strategy.mean_reversion import MeanReversionStrategy
    >>> mr = MeanReversionStrategy()
    >>> sig = mr.analyze(df)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "ema_period": 21,
        "z_score_period": 21,
        "z_entry": 2.0,
        "z_exit": 0.5,
        "target_pct": 0.015,    # 1.5% take-profit
        "stop_loss_pct": 0.015, # 1.5% stop-loss
        "volume_period": 20,
        "volume_threshold": 1.2,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name="mean_reversion", config=merged)

    # ------------------------------------------------------------------ core

    def analyze(self, df: pd.DataFrame) -> Signal:
        """Produce a mean-reversion signal from OHLCV data."""
        symbol: str = df.attrs.get("symbol", "UNKNOWN")

        if not self.validate_dataframe(df):
            return self._hold_signal(symbol)

        close = df["close"]
        current_price = float(close.iloc[-1])

        # --- indicators ---------------------------------------------------
        ema = self.calc_ema(close, self.config["ema_period"])
        current_ema = float(ema.iloc[-1])

        z_score = self.calc_z_score(
            close, ema, self.config["z_score_period"]
        )
        current_z = float(z_score.iloc[-1]) if not pd.isna(z_score.iloc[-1]) else 0.0

        vol_ratio = self.calc_volume_ratio(
            df["volume"], self.config["volume_period"]
        )
        current_vol = float(vol_ratio.iloc[-1])

        # --- confidence calculation ---------------------------------------
        # Confidence scales linearly from z_entry to z_entry + 1
        z_entry = self.config["z_entry"]
        z_excess = abs(current_z) - z_entry
        z_confidence = max(0.0, min(1.0, 0.5 + z_excess * 0.5))

        # Volume adds a small bonus
        vol_bonus = min(0.15, max(0.0, (current_vol - 1.0) * 0.10))

        # Distance from EMA as fraction — larger distance = higher confidence
        ema_dist_pct = abs(current_price - current_ema) / current_ema if current_ema else 0
        dist_bonus = min(0.15, ema_dist_pct * 5)

        meta: dict[str, Any] = {
            "z_score": round(current_z, 4),
            "ema": round(current_ema, 8),
            "volume_ratio": round(current_vol, 4),
            "ema_distance_pct": round(ema_dist_pct * 100, 4),
            "timeframe": self.config.get("timeframe", "15m"),
        }

        # --- decision -----------------------------------------------------
        if current_z < -z_entry:
            confidence = min(1.0, z_confidence + vol_bonus + dist_bonus)
            stop_loss = current_price * (1 - self.config["stop_loss_pct"])
            take_profit = current_price * (1 + self.config["target_pct"])
            action = TradeAction.BUY
            logger.info(
                "[mean_reversion] BUY — {} z={:.2f} price={:.6f} conf={:.2f}",
                symbol, current_z, current_price, confidence,
            )

        elif current_z > z_entry:
            confidence = min(1.0, z_confidence + vol_bonus + dist_bonus)
            stop_loss = current_price * (1 + self.config["stop_loss_pct"])
            take_profit = current_price * (1 - self.config["target_pct"])
            action = TradeAction.SELL
            logger.info(
                "[mean_reversion] SELL — {} z={:.2f} price={:.6f} conf={:.2f}",
                symbol, current_z, current_price, confidence,
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
        """Confirm entry with volume and a consistency check.

        * Confidence ≥ 0.45
        * Volume ratio above threshold
        * Z-score direction consistent across last 2 bars
        """
        if signal.action == TradeAction.HOLD:
            return False
        if signal.confidence < 0.45:
            logger.debug(
                "[mean_reversion] Confidence {:.2f} below threshold", signal.confidence
            )
            return False

        vol_ratio = self.calc_volume_ratio(
            df["volume"], self.config["volume_period"]
        )
        if float(vol_ratio.iloc[-1]) < self.config["volume_threshold"]:
            logger.debug("[mean_reversion] Volume too low — skip entry")
            return False

        ema = self.calc_ema(df["close"], self.config["ema_period"])
        z = self.calc_z_score(df["close"], ema, self.config["z_score_period"])
        last_two = z.iloc[-2:]

        if signal.action == TradeAction.BUY and not all(v < -1.5 for v in last_two):
            logger.debug("[mean_reversion] Z-score not consistently low — skip")
            return False
        if signal.action == TradeAction.SELL and not all(v > 1.5 for v in last_two):
            logger.debug("[mean_reversion] Z-score not consistently high — skip")
            return False

        return True

    def should_exit(self, df: pd.DataFrame, position: Position) -> bool:
        """Exit only on SL / TP to allow high win rate."""
        current_price = float(df["close"].iloc[-1])

        if position.is_stop_hit(current_price) or position.is_tp_hit(current_price):
            logger.info(
                "[mean_reversion] SL/TP hit for {} @ {:.6f}",
                position.symbol, current_price,
            )
            return True

        return False
