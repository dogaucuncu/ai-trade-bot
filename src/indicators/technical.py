"""
AI Trade Bot — Technical indicator calculations.

Wraps the ``ta`` (Technical Analysis) library behind a clean, static
interface.  Every method accepts and returns a pandas DataFrame so
indicators can be chained easily.

Usage::

    from src.indicators.technical import TechnicalIndicators as TI

    df = await collector.fetch_crypto_ohlcv("DOGE/USDT", "5m")
    df = TI.add_all_indicators(df)
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


class TechnicalIndicators:
    """Collection of static methods that enrich an OHLCV DataFrame with
    standard technical indicators.

    All methods mutate the DataFrame **in-place** and also return it for
    convenient chaining.  Columns added by each method are named with a
    descriptive prefix (e.g. ``rsi_14``, ``macd_line``).
    """

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    @staticmethod
    def add_ema(
        df: pd.DataFrame,
        periods: tuple[int, ...] = (9, 21, 50, 200),
    ) -> pd.DataFrame:
        """Add Exponential Moving Averages for multiple look-back windows.

        New columns: ``ema_9``, ``ema_21``, ``ema_50``, ``ema_200``.
        """
        for p in periods:
            col = f"ema_{p}"
            indicator = EMAIndicator(close=df["close"], window=p)
            df[col] = indicator.ema_indicator()
        logger.debug("Added EMA columns: {}", [f"ema_{p}" for p in periods])
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Add MACD line, signal line, and histogram.

        New columns: ``macd_line``, ``macd_signal``, ``macd_histogram``.
        """
        macd = MACD(
            close=df["close"],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal,
        )
        df["macd_line"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()
        logger.debug("Added MACD ({},{},{})", fast, slow, signal)
        return df

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index.

        New column: ``rsi_14`` (or ``rsi_{period}``).
        """
        col = f"rsi_{period}"
        rsi = RSIIndicator(close=df["close"], window=period)
        df[col] = rsi.rsi()
        logger.debug("Added {}", col)
        return df

    @staticmethod
    def add_stochastic_rsi(
        df: pd.DataFrame,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> pd.DataFrame:
        """Add Stochastic RSI (%K and %D).

        New columns: ``stoch_rsi_k``, ``stoch_rsi_d``.
        """
        stoch = StochRSIIndicator(
            close=df["close"],
            window=period,
            smooth1=smooth_k,
            smooth2=smooth_d,
        )
        df["stoch_rsi_k"] = stoch.stochrsi_k()
        df["stoch_rsi_d"] = stoch.stochrsi_d()
        logger.debug("Added Stochastic RSI ({})", period)
        return df

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.DataFrame:
        """Add Bollinger Bands (upper, middle, lower, bandwidth, %b).

        New columns: ``bb_upper``, ``bb_middle``, ``bb_lower``,
        ``bb_bandwidth``, ``bb_pband``.
        """
        bb = BollingerBands(
            close=df["close"],
            window=period,
            window_dev=int(std_dev),
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_bandwidth"] = bb.bollinger_wband()
        df["bb_pband"] = bb.bollinger_pband()
        logger.debug("Added Bollinger Bands ({}, {}σ)", period, std_dev)
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range.

        New column: ``atr_14`` (or ``atr_{period}``).
        """
        col = f"atr_{period}"
        atr = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=period,
        )
        df[col] = atr.average_true_range()
        logger.debug("Added {}", col)
        return df

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume-Weighted Average Price (intraday approximation).

        VWAP is computed from the current DataFrame (assumes a single
        session).  New column: ``vwap``.
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        df["vwap"] = cumulative_tp_vol / cumulative_vol.replace(0, float("nan"))
        logger.debug("Added VWAP")
        return df

    @staticmethod
    def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Volume Simple Moving Average.

        New column: ``volume_sma_20`` (or ``volume_sma_{period}``).
        """
        col = f"volume_sma_{period}"
        df[col] = df["volume"].rolling(window=period).mean()
        logger.debug("Added {}", col)
        return df

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Apply **every** indicator with default parameters.

        Returns the enriched DataFrame (also mutated in-place).  Drops any
        rows that are entirely NaN (warm-up period) at the end.
        """
        if df.empty:
            logger.warning("Empty DataFrame — skipping indicator calculation.")
            return df

        logger.info(
            "Computing all indicators on {} rows …", len(df)
        )

        cls.add_ema(df)
        cls.add_macd(df)
        cls.add_rsi(df)
        cls.add_stochastic_rsi(df)
        cls.add_bollinger_bands(df)
        cls.add_atr(df)
        cls.add_vwap(df)
        cls.add_volume_sma(df)

        # Drop warm-up NaN rows (keep at least 50 % of data)
        min_keep = max(len(df) // 2, 1)
        indicator_cols = [
            c for c in df.columns
            if c not in {"timestamp", "open", "high", "low", "close", "volume"}
        ]
        before = len(df)
        df.dropna(subset=indicator_cols, how="all", inplace=True)
        if len(df) < min_keep:
            logger.warning(
                "Aggressive NaN drop: kept only {}/{} rows — consider"
                " fetching more candles.",
                len(df),
                before,
            )

        logger.info(
            "All indicators added — {} columns, {} rows.",
            len(df.columns),
            len(df),
        )
        return df
