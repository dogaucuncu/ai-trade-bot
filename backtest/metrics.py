"""
Performance metrics — honest, timeframe-aware risk/return statistics.

Centralises every risk-adjusted metric used across the project so that the
strategy :class:`~backtest.backtester.Backtester` and the walk-forward ML
evaluator report the *same* numbers, computed the *same* (correct) way.

Why this module exists
----------------------
The original backtester annualised the Sharpe ratio with a hard-coded
``periods_per_year = 365`` regardless of the candle interval.  On 1-minute
bars that inflates Sharpe by ``sqrt(525600 / 365) ≈ 38×`` — a number that
looks spectacular and means nothing.  Annualisation must use the *real*
number of bars per year for the timeframe being traded.

All functions are pure and operate on plain Python lists / numpy arrays so
they can be reused without pulling in the rest of the stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ── Bars per year, per timeframe (crypto trades 24/7) ────────────────────
BARS_PER_YEAR: dict[str, float] = {
    "1m": 525_600,
    "3m": 175_200,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "6h": 1_460,
    "12h": 730,
    "1d": 365,
}


def bars_per_year(timeframe: str) -> float:
    """Return the number of bars in a year for *timeframe* (default 1d)."""
    return BARS_PER_YEAR.get(timeframe, 365)


# =========================================================================
# Core return helpers
# =========================================================================

def returns_from_equity(equity: Sequence[float]) -> np.ndarray:
    """Convert an equity curve into a series of simple per-bar returns.

    Zero or non-positive previous balances are skipped to avoid div-by-zero.
    """
    arr = np.asarray(equity, dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    prev = arr[:-1]
    cur = arr[1:]
    mask = prev > 0
    return (cur[mask] - prev[mask]) / prev[mask]


# =========================================================================
# Risk-adjusted ratios
# =========================================================================

def sharpe_ratio(
    returns: Sequence[float],
    timeframe: str = "1d",
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio computed from *per-bar* returns.

    Parameters
    ----------
    returns:
        Per-bar simple returns.
    timeframe:
        Candle interval — controls the annualisation factor.
    risk_free_rate:
        Annual risk-free rate (decimal). Converted to a per-bar rate.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0

    ppy = bars_per_year(timeframe)
    rf_per_bar = risk_free_rate / ppy
    excess = r - rf_per_bar

    std = float(np.std(excess, ddof=1))
    if std == 0 or math.isnan(std):
        return 0.0

    return float(np.mean(excess) / std * math.sqrt(ppy))


def sortino_ratio(
    returns: Sequence[float],
    timeframe: str = "1d",
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sortino ratio — penalises only downside deviation."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0

    ppy = bars_per_year(timeframe)
    rf_per_bar = risk_free_rate / ppy
    excess = r - rf_per_bar

    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std == 0 or math.isnan(downside_std):
        return 0.0

    return float(np.mean(excess) / downside_std * math.sqrt(ppy))


def max_drawdown(equity: Sequence[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0.20 = 20%)."""
    arr = np.asarray(equity, dtype=float)
    if arr.size < 2:
        return 0.0
    running_peak = np.maximum.accumulate(arr)
    # avoid div-by-zero on a zero peak
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_peak > 0, (running_peak - arr) / running_peak, 0.0)
    return float(np.max(dd))


def cagr(equity: Sequence[float], timeframe: str) -> float:
    """Compound annual growth rate implied by the equity curve."""
    arr = np.asarray(equity, dtype=float)
    if arr.size < 2 or arr[0] <= 0 or arr[-1] <= 0:
        return 0.0
    n_bars = arr.size - 1
    years = n_bars / bars_per_year(timeframe)
    if years <= 0:
        return 0.0
    return float((arr[-1] / arr[0]) ** (1 / years) - 1)


# =========================================================================
# Trade-based metrics
# =========================================================================

def profit_factor(pnls: Sequence[float]) -> float:
    """Gross profit divided by gross loss (>1 is profitable)."""
    p = np.asarray(pnls, dtype=float)
    gross_profit = float(p[p > 0].sum())
    gross_loss = float(-p[p < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def win_rate(pnls: Sequence[float]) -> float:
    """Fraction of trades that were profitable (0..1)."""
    p = np.asarray(pnls, dtype=float)
    if p.size == 0:
        return 0.0
    return float((p > 0).sum() / p.size)


def expectancy(pnls: Sequence[float]) -> float:
    """Average P&L per trade (in account currency)."""
    p = np.asarray(pnls, dtype=float)
    return float(p.mean()) if p.size else 0.0


# =========================================================================
# Benchmark
# =========================================================================

def buy_and_hold_return(close: Sequence[float]) -> float:
    """Total return of simply holding the asset over the window (decimal)."""
    arr = np.asarray(close, dtype=float)
    if arr.size < 2 or arr[0] <= 0:
        return 0.0
    return float(arr[-1] / arr[0] - 1)


# =========================================================================
# Aggregate container
# =========================================================================

@dataclass(slots=True)
class PerformanceReport:
    """Full set of honest performance metrics for one test window."""

    timeframe: str = "1d"
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_trades: int = 0
    total_fees: float = 0.0
    buy_and_hold_pct: float = 0.0
    excess_vs_hold_pct: float = 0.0

    @classmethod
    def from_run(
        cls,
        *,
        timeframe: str,
        initial_capital: float,
        equity_curve: Sequence[float],
        trade_pnls: Sequence[float],
        total_fees: float,
        close_prices: Sequence[float],
    ) -> "PerformanceReport":
        """Compute every metric from a single backtest run's raw output."""
        equity = list(equity_curve)
        final_capital = equity[-1] if equity else initial_capital
        rets = returns_from_equity(equity)

        total_ret = (
            (final_capital - initial_capital) / initial_capital
            if initial_capital > 0 else 0.0
        )
        bh = buy_and_hold_return(close_prices)

        return cls(
            timeframe=timeframe,
            initial_capital=round(initial_capital, 4),
            final_capital=round(final_capital, 4),
            total_return_pct=round(total_ret * 100, 4),
            cagr_pct=round(cagr(equity, timeframe) * 100, 4),
            sharpe=round(sharpe_ratio(rets, timeframe), 4),
            sortino=round(sortino_ratio(rets, timeframe), 4),
            max_drawdown_pct=round(max_drawdown(equity) * 100, 4),
            win_rate_pct=round(win_rate(trade_pnls) * 100, 2),
            profit_factor=round(profit_factor(trade_pnls), 4),
            expectancy=round(expectancy(trade_pnls), 6),
            total_trades=len(trade_pnls),
            total_fees=round(total_fees, 6),
            buy_and_hold_pct=round(bh * 100, 4),
            excess_vs_hold_pct=round((total_ret - bh) * 100, 4),
        )

    def summary(self) -> str:
        """Human-readable one-block summary."""
        pf = "inf" if self.profit_factor == float("inf") else f"{self.profit_factor:.2f}"
        # ASCII-only so it prints on any console (incl. Windows cp1254).
        return (
            f"{'-' * 56}\n"
            f"  Initial -> Final: ${self.initial_capital:.2f} -> ${self.final_capital:.2f}\n"
            f"  Total Return    : {self.total_return_pct:+.2f}%   "
            f"(CAGR {self.cagr_pct:+.1f}%)\n"
            f"  Buy & Hold      : {self.buy_and_hold_pct:+.2f}%\n"
            f"  EXCESS vs Hold  : {self.excess_vs_hold_pct:+.2f}%   "
            f"<-- the only number that matters\n"
            f"  Sharpe / Sortino: {self.sharpe:.2f} / {self.sortino:.2f}  ({self.timeframe})\n"
            f"  Max Drawdown    : {self.max_drawdown_pct:.2f}%\n"
            f"  Win Rate        : {self.win_rate_pct:.1f}%\n"
            f"  Profit Factor   : {pf}\n"
            f"  Expectancy/trade: ${self.expectancy:.4f}\n"
            f"  Trades / Fees   : {self.total_trades}  /  ${self.total_fees:.2f}\n"
            f"{'-' * 56}"
        )
