"""
Momentum strategy — trend-following on 1 h / 4 h charts.

Uses a combination of:

1. **MACD crossover** (12, 26, 9) — primary trend signal
2. **EMA(9) / EMA(21) crossover** — trend confirmation
3. **Volume confirmation** — above-average volume validates the move
4. **ATR-based trailing stop** — adaptive exit mechanism

Entry / exit logic
------------------
* **BUY** on bullish MACD crossover + EMA(9) > EMA(21) + volume rising
* **SELL** on bearish MACD crossover + EMA(9) < EMA(21)
* Uses ATR-based trailing stop-loss (2× ATR from recent high)
* Target: 2–5 % profit
* Works on **1 h** and **4 h** timeframes.

Usage
-----
>>> strategy = MomentumStrategy()
>>> signal = strategy.analyze(ohlcv_df)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from src.strategy.base import BaseStrategy, Position, Signal, TradeAction


class MomentumStrategy(BaseStrategy):
    """MACD + EMA crossover momentum strategy (1 h / 4 h charts).

    Parameters
    ----------
    config : dict, optional
        Override any of the defaults::

            {
                "macd_fast":         12,
                "macd_slow":         26,
                "macd_signal":       9,
                "ema_fast":          9,
                "ema_slow":          21,
                "atr_period":        14,
                "atr_multiplier":    2.0,
                "volume_period":     20,
                "volume_threshold":  1.2,
                "target_pct":        0.035,  # 3.5 %
                "stop_loss_pct":     0.02,   # 2.0 %
            }

    Examples
    --------
    >>> from src.strategy.momentum import MomentumStrategy
    >>> m = MomentumStrategy(config={"target_pct": 0.05})
    >>> sig = m.analyze(df)
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "ema_fast": 9,
        "ema_slow": 21,
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "volume_period": 20,
        "volume_threshold": 1.2,
        "target_pct": 0.035,    # 3.5% take-profit
        "stop_loss_pct": 0.02,  # 2.0% stop-loss
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name="momentum", config=merged)

    # ------------------------------------------------------------------ core

    def analyze(self, df: pd.DataFrame) -> Signal:
        """Produce a momentum signal from OHLCV data."""
        symbol: str = df.attrs.get("symbol", "UNKNOWN")

        if not self.validate_dataframe(df):
            return self._hold_signal(symbol)

        close = df["close"]
        current_price = float(close.iloc[-1])

        # --- indicators ---------------------------------------------------
        macd_line, macd_signal, macd_hist = self.calc_macd(
            close,
            self.config["macd_fast"],
            self.config["macd_slow"],
            self.config["macd_signal"],
        )
        cur_macd = float(macd_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        cur_macd_sig = float(macd_signal.iloc[-1])
        prev_macd_sig = float(macd_signal.iloc[-2])
        cur_hist = float(macd_hist.iloc[-1])

        ema_fast = self.calc_ema(close, self.config["ema_fast"])
        ema_slow = self.calc_ema(close, self.config["ema_slow"])
        cur_ema_fast = float(ema_fast.iloc[-1])
        cur_ema_slow = float(ema_slow.iloc[-1])
        prev_ema_fast = float(ema_fast.iloc[-2])
        prev_ema_slow = float(ema_slow.iloc[-2])

        atr = self.calc_atr(df, self.config["atr_period"])
        cur_atr = float(atr.iloc[-1])

        vol_ratio = self.calc_volume_ratio(
            df["volume"], self.config["volume_period"]
        )
        cur_vol = float(vol_ratio.iloc[-1])

        # --- crossover detection ------------------------------------------
        macd_bull_cross = prev_macd <= prev_macd_sig and cur_macd > cur_macd_sig
        macd_bear_cross = prev_macd >= prev_macd_sig and cur_macd < cur_macd_sig
        ema_bull_cross = cur_ema_fast > cur_ema_slow
        ema_bear_cross = cur_ema_fast < cur_ema_slow

        # Also check for "near crossover" (within one bar)
        macd_bull_near = (cur_macd - cur_macd_sig) > 0 and abs(prev_macd - prev_macd_sig) < cur_atr * 0.1
        macd_bear_near = (cur_macd - cur_macd_sig) < 0 and abs(prev_macd - prev_macd_sig) < cur_atr * 0.1

        # --- sub-signal scoring -------------------------------------------
        # MACD crossover weight: 0.40
        macd_buy_score = 1.0 if macd_bull_cross else (0.6 if macd_bull_near else 0.0)
        macd_sell_score = 1.0 if macd_bear_cross else (0.6 if macd_bear_near else 0.0)

        # EMA alignment weight: 0.30
        ema_buy_score = 1.0 if ema_bull_cross else 0.0
        ema_sell_score = 1.0 if ema_bear_cross else 0.0

        # Volume confirmation weight: 0.20
        vol_score = min(1.0, max(0.0, (cur_vol - 1.0) / (self.config["volume_threshold"] - 1.0)))

        # Histogram momentum weight: 0.10
        hist_score_buy = min(1.0, max(0.0, cur_hist / cur_atr)) if cur_atr > 0 else 0.0
        hist_score_sell = min(1.0, max(0.0, -cur_hist / cur_atr)) if cur_atr > 0 else 0.0

        w_macd, w_ema, w_vol, w_hist = 0.40, 0.30, 0.20, 0.10
        buy_conf = (
            w_macd * macd_buy_score
            + w_ema * ema_buy_score
            + w_vol * vol_score
            + w_hist * hist_score_buy
        )
        sell_conf = (
            w_macd * macd_sell_score
            + w_ema * ema_sell_score
            + w_vol * vol_score
            + w_hist * hist_score_sell
        )

        meta: dict[str, Any] = {
            "macd": round(cur_macd, 8),
            "macd_signal": round(cur_macd_sig, 8),
            "macd_histogram": round(cur_hist, 8),
            "ema_fast": round(cur_ema_fast, 8),
            "ema_slow": round(cur_ema_slow, 8),
            "atr": round(cur_atr, 8),
            "volume_ratio": round(cur_vol, 4),
            "macd_bull_cross": macd_bull_cross,
            "macd_bear_cross": macd_bear_cross,
            "timeframe": self.config.get("timeframe", "1h"),
        }

        # --- decision -----------------------------------------------------
        if macd_bull_cross and ema_bull_cross and cur_vol >= self.config["volume_threshold"]:
            atr_stop = current_price - self.config["atr_multiplier"] * cur_atr
            pct_stop = current_price * (1 - self.config["stop_loss_pct"])
            stop_loss = max(atr_stop, pct_stop)  # use the tighter of the two
            take_profit = current_price * (1 + self.config["target_pct"])
            confidence = min(1.0, buy_conf)
            action = TradeAction.BUY
            logger.info(
                "[momentum] BUY — {} MACD cross + EMA alignment, conf={:.2f}",
                symbol, confidence,
            )

        elif macd_bear_cross and ema_bear_cross:
            atr_stop = current_price + self.config["atr_multiplier"] * cur_atr
            pct_stop = current_price * (1 + self.config["stop_loss_pct"])
            stop_loss = min(atr_stop, pct_stop)
            take_profit = current_price * (1 - self.config["target_pct"])
            confidence = min(1.0, sell_conf)
            action = TradeAction.SELL
            logger.info(
                "[momentum] SELL — {} MACD cross + EMA alignment, conf={:.2f}",
                symbol, confidence,
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
        """Confirm momentum entry.

        * Confidence ≥ 0.55
        * MACD histogram increasing (for buys) / decreasing (for sells)
        """
        if signal.action == TradeAction.HOLD:
            return False
        if signal.confidence < 0.55:
            logger.debug("[momentum] Confidence {:.2f} too low", signal.confidence)
            return False

        _, _, hist = self.calc_macd(
            df["close"],
            self.config["macd_fast"],
            self.config["macd_slow"],
            self.config["macd_signal"],
        )

        if signal.action == TradeAction.BUY:
            if float(hist.iloc[-1]) <= float(hist.iloc[-2]):
                logger.debug("[momentum] Histogram not expanding — skip buy")
                return False
        elif signal.action == TradeAction.SELL:
            if float(hist.iloc[-1]) >= float(hist.iloc[-2]):
                logger.debug("[momentum] Histogram not contracting — skip sell")
                return False

        return True

    def should_exit(self, df: pd.DataFrame, position: Position) -> bool:
        """Exit only on SL / TP to allow high win rate."""
        current_price = float(df["close"].iloc[-1])

        if position.is_stop_hit(current_price) or position.is_tp_hit(current_price):
            logger.info(
                "[momentum] SL/TP hit for {} @ {:.6f}",
                position.symbol, current_price,
            )
            return True

        return False

    def calc_trailing_stop(
        self, df: pd.DataFrame, position: Position
    ) -> float:
        """Compute an ATR-based trailing stop for an open position.

        For longs the stop trails below the highest close since entry.
        For shorts the stop trails above the lowest close since entry.

        Returns
        -------
        float
            New stop-loss price.
        """
        atr = self.calc_atr(df, self.config["atr_period"])
        cur_atr = float(atr.iloc[-1])

        # Filter bars after entry
        mask = df["timestamp"] >= position.entry_time
        relevant = df.loc[mask, "close"] if mask.any() else df["close"]

        if position.side == TradeAction.BUY:
            highest = float(relevant.max())
            new_stop = highest - self.config["atr_multiplier"] * cur_atr
            return max(new_stop, position.stop_loss)  # only move up
        else:
            lowest = float(relevant.min())
            new_stop = lowest + self.config["atr_multiplier"] * cur_atr
            return min(new_stop, position.stop_loss)  # only move down
