"""
Base strategy module — abstract foundation for all trading strategies.

Defines the core abstractions every strategy must implement:
- ``TradeAction`` enum for BUY / SELL / HOLD decisions
- ``Signal`` dataclass carrying the full trade recommendation
- ``Position`` dataclass tracking an open position
- ``BaseStrategy`` ABC that concrete strategies subclass

Usage
-----
>>> class MyStrategy(BaseStrategy):
...     def analyze(self, df):   ...
...     def should_enter(self, df, signal): ...
...     def should_exit(self, df, position): ...
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TradeAction(str, Enum):
    """Possible trade actions a strategy can recommend."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signal:
    """Immutable trade signal emitted by a strategy.

    Attributes
    ----------
    symbol : str
        Ticker / trading pair, e.g. ``"BTC/USDT"`` or ``"AAPL"``.
    action : TradeAction
        Recommended action.
    confidence : float
        Confidence score in ``[0.0, 1.0]``.
    stop_loss_price : float
        Absolute price level for the stop-loss order.
    take_profit_price : float
        Absolute price level for the take-profit order.
    strategy_name : str
        Human-readable name of the strategy that produced the signal.
    timestamp : datetime
        UTC timestamp of signal generation.
    metadata : dict[str, Any]
        Free-form dict for extra strategy-specific data (indicator values,
        timeframe, etc.).

    Examples
    --------
    >>> sig = Signal(
    ...     symbol="SOL/USDT",
    ...     action=TradeAction.BUY,
    ...     confidence=0.82,
    ...     stop_loss_price=142.50,
    ...     take_profit_price=148.00,
    ...     strategy_name="scalping",
    ... )
    """

    symbol: str
    action: TradeAction
    confidence: float
    stop_loss_price: float
    take_profit_price: float
    strategy_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass(slots=True)
class Position:
    """Represents an open position being tracked by the bot.

    Attributes
    ----------
    symbol : str
        Trading pair / ticker.
    side : TradeAction
        ``BUY`` (long) or ``SELL`` (short).
    entry_price : float
        Fill price of the entry order.
    quantity : float
        Position size in base units.
    stop_loss : float
        Current stop-loss price (may be updated for trailing stops).
    take_profit : float
        Target take-profit price.
    entry_time : datetime
        UTC time the position was opened.
    strategy_name : str
        Strategy that initiated the trade.
    unrealized_pnl : float
        Current unrealised profit/loss (updated by the engine).
    position_id : str
        Unique identifier.

    Examples
    --------
    >>> pos = Position(
    ...     symbol="ETH/USDT",
    ...     side=TradeAction.BUY,
    ...     entry_price=3_050.00,
    ...     quantity=0.005,
    ...     stop_loss=3_020.00,
    ...     take_profit=3_100.00,
    ...     entry_time=datetime.now(timezone.utc),
    ...     strategy_name="momentum",
    ... )
    """

    symbol: str
    side: TradeAction
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    strategy_name: str
    unrealized_pnl: float = 0.0
    position_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Exchange order id of the protective stop-loss resting on the exchange
    # (live mode only). Cancelled before the engine market-closes the position.
    stop_order_id: str | None = None

    # -- helpers -------------------------------------------------------------

    def update_pnl(self, current_price: float) -> None:
        """Re-calculate unrealised P&L given the latest market price."""
        if self.side == TradeAction.BUY:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

    def pnl_pct(self, current_price: float) -> float:
        """Return P&L as a percentage of the entry value."""
        if self.entry_price == 0:
            return 0.0
        if self.side == TradeAction.BUY:
            return ((current_price - self.entry_price) / self.entry_price) * 100.0
        return ((self.entry_price - current_price) / self.entry_price) * 100.0

    def is_stop_hit(self, current_price: float) -> bool:
        """Check whether the current price has breached the stop-loss."""
        if self.side == TradeAction.BUY:
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def is_tp_hit(self, current_price: float) -> bool:
        """Check whether the current price has reached the take-profit."""
        if self.side == TradeAction.BUY:
            return current_price >= self.take_profit
        return current_price <= self.take_profit


# ---------------------------------------------------------------------------
# Abstract base strategy
# ---------------------------------------------------------------------------


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses **must** implement:

    * :meth:`analyze`       – produce a :class:`Signal` from a DataFrame
    * :meth:`should_enter`  – gate entry based on extra conditions
    * :meth:`should_exit`   – decide if an open position should be closed

    Parameters
    ----------
    name : str
        Human-readable strategy name (used in logging / signals).
    config : dict[str, Any]
        Strategy-specific configuration knobs.

    Examples
    --------
    >>> class DummyStrategy(BaseStrategy):
    ...     def analyze(self, df):
    ...         return Signal(
    ...             symbol="TEST", action=TradeAction.HOLD, confidence=0.0,
    ...             stop_loss_price=0, take_profit_price=0,
    ...             strategy_name=self.name,
    ...         )
    ...     def should_enter(self, df, signal): return False
    ...     def should_exit(self, df, position): return False
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config = config or {}
        self._trade_history: list[dict[str, Any]] = []
        logger.info("Strategy initialised: {}", self.name)

    # -- abstract interface --------------------------------------------------

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Signal:
        """Analyse the latest OHLCV data and produce a trade signal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least ``open, high, low, close, volume`` columns,
            indexed by timestamp in ascending order.

        Returns
        -------
        Signal
            Trade recommendation.
        """

    @abstractmethod
    def should_enter(self, df: pd.DataFrame, signal: Signal) -> bool:
        """Final confirmation gate before executing a new entry.

        May add extra filters (e.g. volume spike, spread check).
        """

    @abstractmethod
    def should_exit(self, df: pd.DataFrame, position: Position) -> bool:
        """Decide whether an existing position should be closed early.

        Called each tick / bar to allow strategies to manage exits
        beyond the static stop-loss / take-profit.
        """

    # -- shared utilities ----------------------------------------------------

    @staticmethod
    def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute the Relative Strength Index (RSI).

        Parameters
        ----------
        series : pd.Series
            Close prices.
        period : int
            Look-back window (default 14).

        Returns
        -------
        pd.Series
            RSI values in ``[0, 100]``.
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def calc_ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calc_sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()

    @staticmethod
    def calc_bollinger_bands(
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands (middle, upper, lower).

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series]
            ``(middle, upper, lower)``
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        return middle, upper, lower

    @staticmethod
    def calc_macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, histogram.

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series]
            ``(macd_line, signal_line, histogram)``
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calc_atr(
        df: pd.DataFrame,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``high``, ``low``, ``close`` columns.
        period : int
            Look-back window (default 14).

        Returns
        -------
        pd.Series
            ATR values.
        """
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calc_z_score(
        series: pd.Series,
        mean_series: pd.Series,
        period: int = 21,
    ) -> pd.Series:
        """Z-score of *series* relative to *mean_series*.

        Parameters
        ----------
        series : pd.Series
            Observed values (e.g. close prices).
        mean_series : pd.Series
            Reference mean (e.g. EMA).
        period : int
            Look-back window for standard deviation.

        Returns
        -------
        pd.Series
        """
        diff = series - mean_series
        std = series.rolling(window=period).std().replace(0, np.nan)
        return diff / std

    @staticmethod
    def calc_volume_ratio(
        volume: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """Ratio of current volume to its rolling average."""
        avg = volume.rolling(window=period).mean()
        return volume / avg.replace(0, np.nan)

    # -- bookkeeping ---------------------------------------------------------

    def record_trade(self, signal: Signal, result: dict[str, Any]) -> None:
        """Append a completed trade to the internal history."""
        entry = {
            "signal": signal,
            "result": result,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._trade_history.append(entry)
        logger.debug(
            "[{}] Trade recorded – {} {} @ confidence {:.2f}",
            self.name,
            signal.action.value,
            signal.symbol,
            signal.confidence,
        )

    @property
    def win_rate(self) -> float:
        """Fraction of profitable trades in the recorded history."""
        if not self._trade_history:
            return 0.0
        wins = sum(
            1
            for t in self._trade_history
            if t.get("result", {}).get("pnl", 0) > 0
        )
        return wins / len(self._trade_history)

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Ensure *df* has the minimum required columns and rows.

        Returns
        -------
        bool
            ``True`` if valid, ``False`` otherwise (logs a warning).
        """
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            logger.warning("[{}] DataFrame missing columns: {}", self.name, missing)
            return False
        if len(df) < 30:
            logger.warning(
                "[{}] DataFrame has only {} rows (need ≥ 30)", self.name, len(df)
            )
            return False
        return True

    def _hold_signal(self, symbol: str) -> Signal:
        """Convenience: create a HOLD signal with zero confidence."""
        return Signal(
            symbol=symbol,
            action=TradeAction.HOLD,
            confidence=0.0,
            stop_loss_price=0.0,
            take_profit_price=0.0,
            strategy_name=self.name,
        )
