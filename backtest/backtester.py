"""
Backtesting engine — simulate strategies on historical data.

Replays OHLCV data through a strategy, applying realistic fees, slippage,
position sizing, and risk management to produce performance statistics.

Supports:
* Single-strategy backtests with detailed equity curves
* Multi-strategy comparison tables
* Walk-forward testing (train/test window splits)
* HTML report generation with Plotly charts
* Result persistence to ``backtest/results/``

Usage
-----
>>> from backtest.backtester import Backtester
>>> bt = Backtester()
>>> result = await bt.run(
...     strategy=my_strategy,
...     symbol="SOL/USDT",
...     timeframe="1h",
...     start_date="2024-01-01",
...     end_date="2024-06-01",
... )
>>> print(result.summary())
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.metrics import sharpe_ratio as _sharpe_ratio  # noqa: E402
from src.risk.manager import RiskConfig, RiskManager  # noqa: E402
from src.risk.position_sizer import PositionSizer  # noqa: E402
from src.strategy.base import (  # noqa: E402
    BaseStrategy,
    Position,
    Signal,
    TradeAction,
)


# =========================================================================
# Result data class
# =========================================================================

@dataclass(slots=True)
class BacktestResult:
    """Container for backtest output metrics.

    Attributes
    ----------
    strategy_name : str
        Name of the strategy tested.
    symbol : str
        Asset that was backtested.
    timeframe : str
        Candle timeframe used.
    start_date : str
        ISO start date.
    end_date : str
        ISO end date.
    initial_capital : float
        Starting capital in USD.
    final_capital : float
        Ending capital in USD.
    total_return : float
        Percentage return over the test period.
    sharpe_ratio : float
        Annualised Sharpe ratio (risk-free rate = 0).
    max_drawdown : float
        Maximum peak-to-trough drawdown (percentage).
    win_rate : float
        Fraction of trades that were profitable.
    total_trades : int
        Number of round-trip trades.
    avg_win : float
        Average profit on winning trades (USD).
    avg_loss : float
        Average loss on losing trades (USD, positive).
    profit_factor : float
        Gross profits / gross losses.
    equity_curve : list[dict[str, Any]]
        List of ``{timestamp, balance}`` dicts.
    trades : list[dict[str, Any]]
        Detailed per-trade records.
    """

    strategy_name: str = ""
    symbol: str = ""
    timeframe: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 50.0
    final_capital: float = 50.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)

    # -- helpers ----------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"{'─' * 50}\n"
            f"  Backtest: {self.strategy_name} on {self.symbol} ({self.timeframe})\n"
            f"  Period  : {self.start_date} → {self.end_date}\n"
            f"{'─' * 50}\n"
            f"  Initial Capital : ${self.initial_capital:.2f}\n"
            f"  Final Capital   : ${self.final_capital:.2f}\n"
            f"  Total Return    : {self.total_return:+.2f}%\n"
            f"  Sharpe Ratio    : {self.sharpe_ratio:.3f}\n"
            f"  Max Drawdown    : {self.max_drawdown:.2f}%\n"
            f"  Win Rate        : {self.win_rate:.1f}%\n"
            f"  Total Trades    : {self.total_trades}\n"
            f"  Avg Win         : ${self.avg_win:.4f}\n"
            f"  Avg Loss        : ${self.avg_loss:.4f}\n"
            f"  Profit Factor   : {self.profit_factor:.2f}\n"
            f"{'─' * 50}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (equity_curve and trades included)."""
        return asdict(self)


# =========================================================================
# Simulated trade tracker
# =========================================================================

@dataclass(slots=True)
class _SimPosition:
    """Internal position representation for the simulation."""

    symbol: str
    side: TradeAction
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: str
    strategy_name: str
    fees_paid: float = 0.0


# =========================================================================
# Backtester
# =========================================================================

class Backtester:
    """Event-driven backtesting engine with realistic simulation.

    Parameters
    ----------
    fee_rate : float
        Trading fee as a fraction (default 0.001 = 0.1 % per side for Binance).
    slippage_rate : float
        Simulated slippage as a fraction (default 0.0005 = 0.05 %).
    initial_capital : float
        Starting capital in USD.
    results_dir : Path | str | None
        Directory for saving result files.

    Examples
    --------
    >>> bt = Backtester()
    >>> result = await bt.run(strategy, "SOL/USDT", "1h",
    ...                       "2024-01-01", "2024-06-01")
    """

    def __init__(
        self,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        initial_capital: float = 50.0,
        results_dir: Path | str | None = None,
    ) -> None:
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.initial_capital = initial_capital

        self.results_dir = Path(
            results_dir or _PROJECT_ROOT / "backtest" / "results"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.position_sizer = PositionSizer(
            max_risk_pct=0.02,
            min_order_usd=5.0,
        )
        self.risk_manager = RiskManager(
            capital=initial_capital,
            config=RiskConfig(
                max_risk_per_trade=0.02,
                daily_loss_limit=0.03,
                max_drawdown=0.15,
                max_open_positions=3,
                max_portfolio_exposure=0.30,
                min_order_value_usd=5.0,
                min_confidence=0.45,
            ),
        )

        logger.info(
            "Backtester initialised — capital=${:.2f} fee={:.2%} slippage={:.2%}",
            initial_capital, fee_rate, slippage_rate,
        )

    # ================================================================= run

    async def run(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        initial_capital: float | None = None,
        data: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Execute a full backtest.

        Parameters
        ----------
        strategy : BaseStrategy
            Strategy instance to test.
        symbol : str
            Trading pair / ticker.
        timeframe : str
            Candle timeframe (e.g. ``"1h"``).
        start_date : str
            ISO start date (``"YYYY-MM-DD"``).
        end_date : str
            ISO end date.
        initial_capital : float | None
            Override starting capital.
        data : pd.DataFrame | None
            Pre-loaded OHLCV data.  If ``None``, synthetic data is generated
            for demonstration purposes.

        Returns
        -------
        BacktestResult
        """
        capital = initial_capital or self.initial_capital

        # Reset risk manager for this run
        self.risk_manager = RiskManager(
            capital=capital,
            config=self.risk_manager.config,
        )

        logger.info(
            "Starting backtest: {} on {} {} [{} → {}] capital=${:.2f}",
            strategy.name, symbol, timeframe, start_date, end_date, capital,
        )

        # Load or generate data
        df = data if data is not None else self._generate_synthetic_data(
            symbol, timeframe, start_date, end_date,
        )

        if df.empty:
            logger.error("No data available for backtest")
            return BacktestResult(
                strategy_name=strategy.name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=capital,
                final_capital=capital,
            )

        # ── Simulation loop ─────────────────────────────────────────────
        balance = capital
        peak_balance = capital
        max_drawdown = 0.0
        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        open_position: _SimPosition | None = None

        min_lookback = 30
        for i in range(min_lookback, len(df)):
            window = df.iloc[:i + 1].copy()
            window.attrs["symbol"] = symbol
            current_price = float(window["close"].iloc[-1])
            ts = str(window.index[-1])

            # ── Monitor open position ───────────────────────────────────
            if open_position is not None:
                hit_stop = False
                hit_tp = False

                if open_position.side == TradeAction.BUY:
                    hit_stop = current_price <= open_position.stop_loss
                    hit_tp = current_price >= open_position.take_profit
                else:
                    hit_stop = current_price >= open_position.stop_loss
                    hit_tp = current_price <= open_position.take_profit

                should_exit = hit_stop or hit_tp
                if not should_exit:
                    pos_obj = Position(
                        symbol=symbol,
                        side=open_position.side,
                        entry_price=open_position.entry_price,
                        quantity=open_position.quantity,
                        stop_loss=open_position.stop_loss,
                        take_profit=open_position.take_profit,
                        entry_time=datetime.now(timezone.utc),
                        strategy_name=open_position.strategy_name,
                    )
                    try:
                        should_exit = strategy.should_exit(window, pos_obj)
                    except Exception:
                        should_exit = False

                if should_exit:
                    exit_price = self._apply_slippage(
                        current_price,
                        side="sell" if open_position.side == TradeAction.BUY else "buy",
                    )
                    exit_fee = exit_price * open_position.quantity * self.fee_rate

                    if open_position.side == TradeAction.BUY:
                        pnl = (exit_price - open_position.entry_price) * open_position.quantity
                    else:
                        pnl = (open_position.entry_price - exit_price) * open_position.quantity

                    pnl -= open_position.fees_paid + exit_fee
                    balance += pnl

                    exit_reason = (
                        "stop_loss" if hit_stop
                        else "take_profit" if hit_tp
                        else "strategy_exit"
                    )
                    trades.append({
                        "entry_time": open_position.entry_time,
                        "exit_time": ts,
                        "symbol": symbol,
                        "side": open_position.side.value,
                        "entry_price": open_position.entry_price,
                        "exit_price": exit_price,
                        "quantity": open_position.quantity,
                        "pnl": round(pnl, 6),
                        "fees": round(open_position.fees_paid + exit_fee, 6),
                        "exit_reason": exit_reason,
                        "strategy": open_position.strategy_name,
                    })

                    self.risk_manager.close_position(
                        f"bt_{len(trades)}", pnl,
                    )
                    open_position = None

            # ── Generate signal ─────────────────────────────────────────
            if open_position is None:
                try:
                    signal = strategy.analyze(window)
                except Exception:
                    logger.debug("Strategy error at index {}", i)
                    equity_curve.append({"timestamp": ts, "balance": round(balance, 4)})
                    continue

                if signal.action in (TradeAction.BUY, TradeAction.SELL):
                    # Risk check
                    approved, _ = self.risk_manager.validate_trade(signal, balance)
                    if approved:
                        entry_price = self._apply_slippage(
                            current_price,
                            side="buy" if signal.action == TradeAction.BUY else "sell",
                        )
                        entry_fee = entry_price * self.fee_rate

                        # Position sizing via the sizer
                        try:
                            atr_val = float(
                                strategy.calc_atr(window).iloc[-1]
                            ) if len(window) >= 14 else entry_price * 0.01
                        except Exception:
                            atr_val = entry_price * 0.01

                        sizing = self.position_sizer.calculate(
                            balance=balance,
                            win_rate=max(0.40, strategy.win_rate),
                            avg_win=0.004,
                            avg_loss=0.003,
                            atr=atr_val,
                            price=entry_price,
                        )

                        if sizing.value_usd > 0 and sizing.value_usd <= balance:
                            quantity = sizing.units
                            total_fee = sizing.value_usd * self.fee_rate

                            open_position = _SimPosition(
                                symbol=symbol,
                                side=signal.action,
                                entry_price=entry_price,
                                quantity=quantity,
                                stop_loss=signal.stop_loss_price,
                                take_profit=signal.take_profit_price,
                                entry_time=ts,
                                strategy_name=signal.strategy_name,
                                fees_paid=total_fee,
                            )

                            self.risk_manager.register_open_position(
                                symbol=symbol,
                                value=sizing.value_usd,
                                side=signal.action.value,
                                position_id=f"bt_{len(trades) + 1}",
                            )

            # ── Equity tracking ─────────────────────────────────────────
            unrealised = 0.0
            if open_position is not None:
                if open_position.side == TradeAction.BUY:
                    unrealised = (current_price - open_position.entry_price) * open_position.quantity
                else:
                    unrealised = (open_position.entry_price - current_price) * open_position.quantity

            equity = balance + unrealised
            equity_curve.append({"timestamp": ts, "balance": round(equity, 4)})

            if equity > peak_balance:
                peak_balance = equity
            dd = ((peak_balance - equity) / peak_balance * 100) if peak_balance > 0 else 0.0
            if dd > max_drawdown:
                max_drawdown = dd

        # ── Force-close remaining position ──────────────────────────────
        if open_position is not None:
            final_price = float(df["close"].iloc[-1])
            exit_price = self._apply_slippage(
                final_price,
                side="sell" if open_position.side == TradeAction.BUY else "buy",
            )
            exit_fee = exit_price * open_position.quantity * self.fee_rate

            if open_position.side == TradeAction.BUY:
                pnl = (exit_price - open_position.entry_price) * open_position.quantity
            else:
                pnl = (open_position.entry_price - exit_price) * open_position.quantity
            pnl -= open_position.fees_paid + exit_fee
            balance += pnl

            trades.append({
                "entry_time": open_position.entry_time,
                "exit_time": str(df.index[-1]),
                "symbol": symbol,
                "side": open_position.side.value,
                "entry_price": open_position.entry_price,
                "exit_price": exit_price,
                "quantity": open_position.quantity,
                "pnl": round(pnl, 6),
                "fees": round(open_position.fees_paid + exit_fee, 6),
                "exit_reason": "end_of_backtest",
                "strategy": open_position.strategy_name,
            })

        # ── Compute metrics ─────────────────────────────────────────────
        result = self._compute_result(
            strategy_name=strategy.name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=capital,
            final_balance=balance,
            max_drawdown=max_drawdown,
            equity_curve=equity_curve,
            trades=trades,
        )

        logger.info("\n{}", result.summary())
        self._save_result(result)
        return result

    # =========================================================== comparison

    async def compare_strategies(
        self,
        strategies: list[BaseStrategy],
        symbol: str,
        timeframe: str,
        start_date: str = "2024-01-01",
        end_date: str = "2024-06-01",
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run multiple strategies on the same data and return a comparison.

        Parameters
        ----------
        strategies : list[BaseStrategy]
            Strategies to compare.
        symbol, timeframe, start_date, end_date : str
            Backtest parameters.
        data : pd.DataFrame | None
            Shared OHLCV data.

        Returns
        -------
        pd.DataFrame
            Comparison table with one row per strategy.
        """
        df_data = data if data is not None else self._generate_synthetic_data(
            symbol, timeframe, start_date, end_date,
        )

        rows: list[dict[str, Any]] = []
        for strat in strategies:
            result = await self.run(
                strategy=strat,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                data=df_data.copy(),
            )
            rows.append({
                "Strategy": result.strategy_name,
                "Return (%)": round(result.total_return, 2),
                "Sharpe": round(result.sharpe_ratio, 3),
                "Max DD (%)": round(result.max_drawdown, 2),
                "Win Rate (%)": round(result.win_rate, 1),
                "Trades": result.total_trades,
                "Profit Factor": round(result.profit_factor, 2),
                "Final Capital": round(result.final_capital, 2),
            })

        comparison = pd.DataFrame(rows)
        logger.info("Strategy comparison:\n{}", comparison.to_string(index=False))
        return comparison

    # ========================================================== walk-forward

    async def walk_forward(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        train_pct: float = 0.70,
        n_splits: int = 3,
        data: pd.DataFrame | None = None,
    ) -> list[BacktestResult]:
        """Walk-forward test — split data into train/test windows.

        Parameters
        ----------
        strategy : BaseStrategy
            Strategy to test.
        symbol, timeframe : str
            Asset and candle period.
        start_date, end_date : str
            Overall date range.
        train_pct : float
            Fraction of each split used for in-sample training.
        n_splits : int
            Number of sequential splits.
        data : pd.DataFrame | None
            Pre-loaded data.

        Returns
        -------
        list[BacktestResult]
            One result per out-of-sample test window.
        """
        df_data = data if data is not None else self._generate_synthetic_data(
            symbol, timeframe, start_date, end_date,
        )

        total_rows = len(df_data)
        split_size = total_rows // n_splits
        results: list[BacktestResult] = []

        for split_idx in range(n_splits):
            split_start = split_idx * split_size
            split_end = min(split_start + split_size, total_rows)
            split_data = df_data.iloc[split_start:split_end]

            train_end = int(len(split_data) * train_pct)
            test_data = split_data.iloc[train_end:]

            if len(test_data) < 31:
                logger.warning(
                    "Walk-forward split {} has only {} test rows — skipping",
                    split_idx, len(test_data),
                )
                continue

            test_start_str = str(test_data.index[0])[:10]
            test_end_str = str(test_data.index[-1])[:10]

            logger.info(
                "Walk-forward split {}/{}: test {} → {} ({} rows)",
                split_idx + 1, n_splits,
                test_start_str, test_end_str, len(test_data),
            )

            result = await self.run(
                strategy=strategy,
                symbol=symbol,
                timeframe=timeframe,
                start_date=test_start_str,
                end_date=test_end_str,
                data=test_data.copy(),
            )
            results.append(result)

        logger.info("Walk-forward completed: {} splits", len(results))
        return results

    # =========================================================== HTML report

    def plot_results(self, result: BacktestResult) -> Path:
        """Generate an interactive HTML report with Plotly charts.

        Parameters
        ----------
        result : BacktestResult
            Backtest result to visualise.

        Returns
        -------
        Path
            Path to the generated HTML file.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly not installed — pip install plotly")
            return self.results_dir / "error.html"

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Equity Curve",
                "Trade P&L Distribution",
                "Cumulative P&L",
            ),
            vertical_spacing=0.08,
            row_heights=[0.45, 0.25, 0.30],
        )

        # ── 1. Equity curve ─────────────────────────────────────────
        if result.equity_curve:
            timestamps = [p["timestamp"] for p in result.equity_curve]
            balances = [p["balance"] for p in result.equity_curve]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=balances,
                    mode="lines",
                    name="Equity",
                    line=dict(color="#7c3aed", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(124,58,237,0.1)",
                ),
                row=1, col=1,
            )
            fig.add_hline(
                y=result.initial_capital,
                line_dash="dash",
                line_color="rgba(255,255,255,0.3)",
                annotation_text="Initial Capital",
                row=1, col=1,
            )

        # ── 2. Trade P&L histogram ──────────────────────────────────
        if result.trades:
            pnls = [t["pnl"] for t in result.trades]
            colors = ["#10b981" if p >= 0 else "#ef4444" for p in pnls]
            fig.add_trace(
                go.Bar(
                    x=list(range(1, len(pnls) + 1)),
                    y=pnls,
                    marker_color=colors,
                    name="Trade P&L",
                ),
                row=2, col=1,
            )

        # ── 3. Cumulative P&L ───────────────────────────────────────
        if result.trades:
            cum_pnl = np.cumsum([t["pnl"] for t in result.trades]).tolist()
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cum_pnl) + 1)),
                    y=cum_pnl,
                    mode="lines+markers",
                    name="Cumulative P&L",
                    line=dict(color="#10b981", width=2),
                    marker=dict(size=4),
                ),
                row=3, col=1,
            )

        # ── Layout ──────────────────────────────────────────────────
        fig.update_layout(
            title=dict(
                text=(
                    f"Backtest: {result.strategy_name} on {result.symbol} "
                    f"({result.timeframe}) | "
                    f"Return: {result.total_return:+.2f}% | "
                    f"Sharpe: {result.sharpe_ratio:.3f} | "
                    f"Max DD: {result.max_drawdown:.2f}%"
                ),
                font=dict(size=14),
            ),
            template="plotly_dark",
            height=1000,
            showlegend=True,
            paper_bgcolor="#0a0e27",
            plot_bgcolor="#0f1332",
        )

        report_name = (
            f"{result.strategy_name}_{result.symbol.replace('/', '_')}"
            f"_{result.start_date}_{result.end_date}.html"
        )
        report_path = self.results_dir / report_name
        fig.write_html(str(report_path))
        logger.info("Report saved to {}", report_path)
        return report_path

    # =========================================================== internals

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply simulated slippage to a fill price.

        Buys fill slightly higher; sells fill slightly lower.
        """
        if side == "buy":
            return price * (1 + self.slippage_rate)
        return price * (1 - self.slippage_rate)

    def _compute_result(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        initial_capital: float,
        final_balance: float,
        max_drawdown: float,
        equity_curve: list[dict[str, Any]],
        trades: list[dict[str, Any]],
    ) -> BacktestResult:
        """Build a :class:`BacktestResult` from raw simulation output."""
        total_return = (
            ((final_balance - initial_capital) / initial_capital) * 100
            if initial_capital > 0 else 0.0
        )

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (abs(sum(losses)) / len(losses)) if losses else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        # Sharpe ratio — annualised with the CORRECT bars-per-year for this
        # timeframe (the old code hard-coded 365, inflating intraday Sharpe
        # by up to ~38x on 1m bars).
        sharpe = self._calc_sharpe(equity_curve, timeframe)

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=round(final_balance, 4),
            total_return=round(total_return, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_drawdown, 4),
            win_rate=round(win_rate, 2),
            total_trades=len(trades),
            avg_win=round(avg_win, 6),
            avg_loss=round(avg_loss, 6),
            profit_factor=round(profit_factor, 4),
            equity_curve=equity_curve,
            trades=trades,
        )

    @staticmethod
    def _calc_sharpe(
        equity_curve: list[dict[str, Any]],
        timeframe: str = "1d",
    ) -> float:
        """Annualised Sharpe ratio from an equity curve (timeframe-aware).

        Delegates to :func:`backtest.metrics.sharpe_ratio` so the
        annualisation factor matches the candle interval actually traded.
        """
        if len(equity_curve) < 2:
            return 0.0

        balances = [p["balance"] for p in equity_curve]
        returns: list[float] = []
        for i in range(1, len(balances)):
            prev = balances[i - 1]
            if prev > 0:
                returns.append((balances[i] - prev) / prev)

        return _sharpe_ratio(returns, timeframe)

    def _save_result(self, result: BacktestResult) -> None:
        """Persist a result to JSON in the results directory."""
        filename = (
            f"{result.strategy_name}_{result.symbol.replace('/', '_')}"
            f"_{result.start_date}_{result.end_date}.json"
        )
        path = self.results_dir / filename

        # Exclude the full equity curve from JSON to keep files small
        data = result.to_dict()
        data["equity_curve_length"] = len(data.pop("equity_curve", []))
        data["trades_count"] = len(data.get("trades", []))

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Result saved to {}", path)
        except Exception:
            logger.exception("Failed to save result to {}", path)

    @staticmethod
    def _generate_synthetic_data(
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Create realistic synthetic OHLCV data for testing.

        Uses geometric Brownian motion with mean-reverting volatility.
        """
        freq_map = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D",
        }
        freq = freq_map.get(timeframe, "1h")

        try:
            dates = pd.date_range(
                start=start_date, end=end_date, freq=freq, tz=timezone.utc,
            )
        except Exception:
            logger.error("Invalid date range: {} → {}", start_date, end_date)
            return pd.DataFrame()

        if len(dates) < 31:
            logger.warning(
                "Date range produces only {} candles — need ≥ 31",
                len(dates),
            )

        np.random.seed(42)
        n = len(dates)

        # Base price depends on symbol type
        if "/" in symbol:
            base_price = 150.0  # crypto-ish
        else:
            base_price = 180.0  # stock-ish

        # Geometric Brownian motion
        drift = 0.0001
        volatility = 0.015
        returns = np.random.normal(drift, volatility, n)
        prices = base_price * np.exp(np.cumsum(returns))

        # Build OHLCV
        high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_prices = prices * (1 + np.random.normal(0, 0.002, n))
        volume = np.random.lognormal(10, 1, n)

        df = pd.DataFrame(
            {
                "open": open_prices,
                "high": high,
                "low": low,
                "close": prices,
                "volume": volume,
            },
            index=dates[:n],
        )
        df.index.name = "timestamp"

        logger.info(
            "Generated {} synthetic candles for {} ({} → {})",
            len(df), symbol, start_date, end_date,
        )
        return df
