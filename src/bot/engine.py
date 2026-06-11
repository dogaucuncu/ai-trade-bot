"""
Bot engine — the main orchestrator that ties every component together.

The engine runs the core trading loop:

1. **Fetch data** for each tracked symbol
2. **Analyze** via the appropriate strategy (by timeframe)
3. **Risk-check** every signal through the RiskManager + CircuitBreaker
4. **Execute** approved orders through the correct exchange executor
5. **Monitor** open positions for exit signals and trailing-stop updates

Strategies run at different intervals:

* Scalping   → every 1 minute
* Mean Rev   → every 15 minutes
* Momentum   → every 1 hour

Usage
-----
>>> engine = BotEngine(capital=50.0)
>>> await engine.initialize()
>>> await engine.run()           # starts the main loop
>>> await engine.shutdown()      # graceful stop
"""

from __future__ import annotations

import asyncio
import signal as os_signal
import sys
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.execution.order_manager import OrderManager, OrderType
from src.risk.circuit_breaker import BreakerState, CircuitBreaker
from src.risk.manager import RiskManager
from src.risk.position_sizer import PositionSizer
from src.strategy.base import BaseStrategy, Position, Signal, TradeAction
from src.strategy.ensemble import EnsembleStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.ml_strategy import MLStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.scalping import ScalpingStrategy


class BotEngine:
    """Main trading-bot orchestrator.

    Parameters
    ----------
    capital : float
        Starting capital in USD (default $50).
    crypto_symbols : list[str] | None
        Crypto pairs to trade (default ``["SOL/USDT", "AVAX/USDT"]``).
    stock_symbols : list[str] | None
        US stock tickers (default ``["AAPL", "MSFT"]``).
    dry_run : bool
        If ``True``, log decisions but skip actual order placement.

    Examples
    --------
    >>> engine = BotEngine(capital=50.0, dry_run=True)
    >>> await engine.initialize()
    >>> await engine.run()
    """

    def __init__(
        self,
        capital: float = 50.0,
        crypto_symbols: list[str] | None = None,
        stock_symbols: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        self.capital = capital
        self.crypto_symbols = crypto_symbols or ["SOL/USDT", "AVAX/USDT"]
        self.stock_symbols = stock_symbols or ["AAPL", "MSFT"]
        self.dry_run = dry_run
        self._running = False
        self._shutdown_event = asyncio.Event()

        # ---- components (initialised in initialize()) --------------------
        self.risk_manager = RiskManager(capital=capital)
        self.circuit_breaker = CircuitBreaker(capital=capital)
        self.position_sizer = PositionSizer(
            max_risk_pct=0.02, min_order_usd=5.0
        )
        self.order_manager = OrderManager(risk_manager=self.risk_manager)

        # ---- strategies --------------------------------------------------
        self.scalping = ScalpingStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.momentum = MomentumStrategy()
        self.ml_strategy = MLStrategy(name="ml_1h")
        self.ensemble = EnsembleStrategy(
            strategies=[self.scalping, self.mean_reversion, self.momentum],
        )

        # ---- open positions ----------------------------------------------
        self._positions: dict[str, Position] = {}  # position_id -> Position

        # ---- executors (set up in initialize) ----------------------------
        self._binance_exec: Any = None
        self._alpaca_exec: Any = None

        # ---- tick counters -----------------------------------------------
        self._tick_count = 59

        logger.info(
            "BotEngine created — capital=${:.2f} crypto={} stocks={} dry_run={}",
            capital, self.crypto_symbols, self.stock_symbols, dry_run,
        )

    # ---------------------------------------------------------------- init

    async def initialize(self) -> None:
        """Initialise exchange connections and load markets.

        Separated from ``__init__`` so that async setup can happen
        after construction.
        """
        try:
            from config.settings import settings as _settings
            from src.execution.binance_exec import BinanceExecutor

            self._binance_exec = BinanceExecutor(
                api_key=_settings.binance.api_key,
                api_secret=_settings.binance.secret_key,
                testnet=_settings.binance.testnet,
            )
            await self._binance_exec.initialize()
        except Exception as exc:
            logger.warning("Binance executor init failed — {}", exc)

        try:
            from src.execution.alpaca_exec import AlpacaExecutor

            self._alpaca_exec = AlpacaExecutor(
                api_key=_settings.alpaca.api_key,
                api_secret=_settings.alpaca.secret_key,
                paper=_settings.alpaca.paper,
            )
        except Exception as exc:
            logger.warning("Alpaca executor init failed — {}", exc)

        logger.info("BotEngine initialised")

    # ------------------------------------------------------------ main loop

    async def run(self) -> None:
        """Start the main trading loop.

        The loop runs every 60 seconds (1-minute tick).  Scalping
        strategies fire every tick, mean reversion every 15 ticks,
        momentum every 60 ticks.
        """
        self._running = True
        self._install_signal_handlers()

        logger.info("=== BotEngine STARTED ===")

        try:
            while self._running and not self._shutdown_event.is_set():
                self._tick_count += 1
                tick = self._tick_count

                try:
                    await self._tick(tick)
                except Exception:
                    logger.exception("Unhandled exception in tick {}", tick)

                # Sleep 60 seconds (one-minute candle period)
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    pass  # normal: timeout means next tick

        finally:
            await self.shutdown()

    async def _tick(self, tick: int) -> None:
        """Execute one iteration of the trading loop."""
        logger.debug("--- Tick {} ---", tick)

        # 1. Circuit-breaker check
        balance = await self._get_balance()
        daily_pnl = self.risk_manager.get_daily_pnl()
        safe, alerts = self.circuit_breaker.check_all(
            balance=balance,
            daily_pnl=daily_pnl,
            api_healthy=self._binance_exec is not None,
        )

        if self.circuit_breaker.requires_close_all:
            logger.critical("EMERGENCY — closing all positions")
            await self._close_all_positions()
            return

        if not safe:
            logger.warning("Circuit breaker not safe — skipping tick: {}", alerts)
            return

        # 2. Update peak equity
        self.risk_manager.update_peak_equity(balance)

        # 3. Monitor open positions first
        await self._monitor_positions()

        # 4. Run strategies at their respective intervals
        # Scalping: every tick (1 min)
        if tick % 1 == 0:
            await self._run_strategy_on_symbols(
                self.scalping, self.crypto_symbols, "1m"
            )

        # Mean reversion: every 15 ticks (15 min)
        if tick % 15 == 0:
            await self._run_strategy_on_symbols(
                self.mean_reversion, self.crypto_symbols, "15m"
            )

        # Momentum: every 60 ticks (1 hour)
        if tick % 60 == 0:
            await self._run_strategy_on_symbols(
                self.momentum, self.crypto_symbols, "1h"
            )

        # ML Strategy: every 60 ticks (1 hour)
        if tick % 60 == 0:
            await self._run_strategy_on_symbols(
                self.ml_strategy, self.crypto_symbols, "1h"
            )

    # -------------------------------------------------- strategy execution

    async def _run_strategy_on_symbols(
        self,
        strategy: BaseStrategy,
        symbols: list[str],
        timeframe: str,
    ) -> None:
        """Run a single strategy across a list of symbols."""
        for symbol in symbols:
            try:
                df = await self._fetch_data(symbol, timeframe, limit=300)
                if df is None or df.empty:
                    continue

                # Add technical indicators before analysis
                from src.indicators.technical import TechnicalIndicators
                df = TechnicalIndicators.add_all_indicators(df)
                if df.empty:
                    continue

                df.attrs["symbol"] = symbol
                signal = strategy.analyze(df)

                if signal.action == TradeAction.HOLD:
                    continue

                if not strategy.should_enter(df, signal):
                    logger.debug(
                        "[engine] {} entry gate rejected for {}",
                        strategy.name, symbol,
                    )
                    continue

                await self._process_signal(signal)

            except Exception:
                logger.exception(
                    "[engine] Error running {} on {}", strategy.name, symbol
                )

    async def _process_signal(self, signal: Signal) -> None:
        """Risk-check and potentially execute a signal."""
        balance = await self._get_balance()

        # Risk validation
        approved, reason = self.risk_manager.validate_trade(signal, balance)
        if not approved:
            logger.info(
                "[engine] Trade rejected — {} {} — {}",
                signal.action.value, signal.symbol, reason,
            )
            return

        # Position sizing
        sizing = self.position_sizer.calculate(
            balance=balance,
            win_rate=max(0.40, self._get_strategy_win_rate(signal.strategy_name)),
            avg_win=0.004,
            avg_loss=0.003,
            atr=signal.metadata.get("atr", 0.01),
            price=signal.metadata.get(
                "current_price",
                (signal.take_profit_price + signal.stop_loss_price) / 2,
            ),
        )

        if sizing.value_usd <= 0:
            logger.info("[engine] Position size = $0 for {} — skip", signal.symbol)
            return

        if self.dry_run:
            # Paper mode: simulate the position so the dashboard shows it
            current_price = signal.metadata.get(
                "current_price",
                (signal.take_profit_price + signal.stop_loss_price) / 2,
            )
            units = sizing.value_usd / current_price if current_price > 0 else 0
            pos = Position(
                symbol=signal.symbol,
                side=signal.action,
                entry_price=current_price,
                quantity=units,
                stop_loss=signal.stop_loss_price,
                take_profit=signal.take_profit_price,
                entry_time=datetime.now(timezone.utc),
                strategy_name=signal.strategy_name,
            )
            self._positions[pos.position_id] = pos
            self.risk_manager.register_open_position(
                symbol=pos.symbol,
                value=sizing.value_usd,
                side=signal.action.value,
                position_id=pos.position_id,
            )
            logger.info(
                "[PAPER] Simulated {} {} — ${:.2f} @ {:.6f} conf={:.2f} SL={:.6f} TP={:.6f}",
                signal.action.value, signal.symbol, sizing.value_usd,
                current_price, signal.confidence,
                signal.stop_loss_price, signal.take_profit_price,
            )
            return

        # Execute
        executor = self._get_executor(signal.symbol)
        if executor is None:
            logger.warning("[engine] No executor for {}", signal.symbol)
            return

        order = await self.order_manager.place_order(
            signal=signal,
            position_size_usd=sizing.value_usd,
            executor=executor,
            order_type=OrderType.MARKET,
            balance=balance,
        )

        if order and order.status.value in ("SUBMITTED", "FILLED"):
            pos = Position(
                symbol=signal.symbol,
                side=signal.action,
                entry_price=order.filled_price or order.price,
                quantity=order.quantity,
                stop_loss=signal.stop_loss_price,
                take_profit=signal.take_profit_price,
                entry_time=datetime.now(timezone.utc),
                strategy_name=signal.strategy_name,
            )
            self._positions[pos.position_id] = pos
            self.risk_manager.register_open_position(
                symbol=pos.symbol,
                value=sizing.value_usd,
                side=signal.action.value,
                position_id=pos.position_id,
            )
            logger.info(
                "[engine] Position opened — {} {} ${:.2f}",
                signal.action.value, signal.symbol, sizing.value_usd,
            )

    # ------------------------------------------------- position monitoring

    async def _monitor_positions(self) -> None:
        """Check each open position for exit conditions and trailing stops."""
        closed_ids: list[str] = []

        for pid, pos in self._positions.items():
            try:
                df = await self._fetch_data(pos.symbol, "1m")
                if df is None or df.empty:
                    continue

                current_price = float(df["close"].iloc[-1])
                pos.update_pnl(current_price)

                # Check hard SL / TP
                should_exit = pos.is_stop_hit(current_price) or pos.is_tp_hit(
                    current_price
                )

                # Check strategy-specific exit
                if not should_exit:
                    strat = self._get_strategy_by_name(pos.strategy_name)
                    if strat:
                        df.attrs["symbol"] = pos.symbol
                        should_exit = strat.should_exit(df, pos)

                if should_exit:
                    await self._close_position(pos, current_price)
                    closed_ids.append(pid)
                else:
                    # Update trailing stop for momentum positions
                    if pos.strategy_name == "momentum":
                        new_stop = self.momentum.calc_trailing_stop(df, pos)
                        if new_stop != pos.stop_loss:
                            pos.stop_loss = new_stop
                            logger.debug(
                                "[engine] Trailing stop updated for {} → {:.6f}",
                                pos.symbol, new_stop,
                            )

            except Exception:
                logger.exception("[engine] Error monitoring position {}", pid)

        for pid in closed_ids:
            del self._positions[pid]

    async def _close_position(
        self, position: Position, current_price: float
    ) -> None:
        """Close a position and record the result."""
        pnl = position.unrealized_pnl
        pnl_pct = position.pnl_pct(current_price)

        if not self.dry_run:
            executor = self._get_executor(position.symbol)
            if executor:
                close_side = "sell" if position.side == TradeAction.BUY else "buy"
                try:
                    await executor.place_market_order(
                        position.symbol, close_side, position.quantity
                    )
                except Exception:
                    logger.exception(
                        "[engine] Failed to close position for {}", position.symbol
                    )

        self.risk_manager.close_position(position.position_id, pnl)
        logger.info(
            "[engine] Position closed — {} PnL=${:.4f} ({:.2f}%)",
            position.symbol, pnl, pnl_pct,
        )

    async def _close_all_positions(self) -> None:
        """Emergency: close every open position immediately."""
        for pid, pos in list(self._positions.items()):
            try:
                df = await self._fetch_data(pos.symbol, "1m")
                price = float(df["close"].iloc[-1]) if df is not None and not df.empty else pos.entry_price
                await self._close_position(pos, price)
            except Exception:
                logger.exception("[engine] Emergency close failed for {}", pos.symbol)
        self._positions.clear()

    # -------------------------------------------------------- data fetching

    async def _fetch_data(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> Any:
        """Fetch OHLCV data directly from Binance Public REST API.

        Uses aiohttp with ssl=False to bypass Windows SSL cert store issues.
        Returns a pandas DataFrame or ``None`` on failure.
        """
        import aiohttp
        import pandas as pd

        if "/" not in symbol:
            # Stock symbol — not yet implemented
            logger.debug("[engine] Stock data feed not implemented for {}", symbol)
            return None

        # Convert CCXT symbol format "BASE/QUOTE" → "BASEQUOTE" (e.g. DOGE/USDT → DOGEUSDT)
        binance_symbol = symbol.replace("/", "")

        # Map CCXT timeframes to Binance API intervals
        tf_map = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
            "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h",
            "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1d",
        }
        interval = tf_map.get(timeframe, "1m")

        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={binance_symbol}&interval={interval}&limit={limit}"
        )

        try:
            # ssl=False: Windows system cert store may block verification;
            # acceptable for dry-run / development. Use ssl=True in production
            # with proper OS cert store.
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=False)
            ) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "[engine] Binance API returned {} for {}", resp.status, symbol
                        )
                        return None
                    raw = await resp.json()

            # Binance kline format: [open_time, open, high, low, close, volume, ...]
            df = pd.DataFrame(
                raw,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ],
            )
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            return df

        except Exception:
            logger.exception("[engine] Failed to fetch data for {}", symbol)
            return None


    # ------------------------------------------------------------ helpers

    async def _get_balance(self) -> float:
        """Return the current total balance in USD.
        
        In dry-run mode, returns simulated capital to avoid
        circuit-breaker false alarms from $0 real balances.
        """
        if self.dry_run:
            return self.capital

        try:
            if self._binance_exec:
                bal = await self._binance_exec.get_balance("USDT")
                if bal > 0:
                    return bal
        except Exception:
            logger.exception("[engine] Balance fetch failed")
        return self.capital  # fallback

    def _get_executor(self, symbol: str) -> Any:
        """Pick the correct executor based on symbol format."""
        if "/" in symbol:
            return self._binance_exec
        return self._alpaca_exec

    def _get_strategy_by_name(self, name: str) -> BaseStrategy | None:
        """Look up a strategy instance by its name."""
        mapping: dict[str, BaseStrategy] = {
            "scalping": self.scalping,
            "mean_reversion": self.mean_reversion,
            "momentum": self.momentum,
            "ensemble": self.ensemble,
        }
        return mapping.get(name)

    def _get_strategy_win_rate(self, name: str) -> float:
        """Return the strategy's historical win rate (or default)."""
        strat = self._get_strategy_by_name(name)
        return strat.win_rate if strat else 0.5

    # ---------------------------------------------------------- shutdown

    async def shutdown(self) -> None:
        """Graceful shutdown: close executors and log summary."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        logger.info("=== BotEngine shutting down ===")

        # Close exchange connections
        if self._binance_exec:
            try:
                await self._binance_exec.close()
            except Exception:
                logger.exception("Error closing Binance connection")

        # Log final state
        logger.info(
            "Final state: {} open positions, {} total orders, daily PnL=${:.2f}",
            len(self._positions),
            self.order_manager.total_orders,
            self.risk_manager.get_daily_pnl(),
        )
        logger.info("=== BotEngine STOPPED ===")

    def _install_signal_handlers(self) -> None:
        """Register SIGINT / SIGTERM for graceful shutdown."""
        loop = asyncio.get_event_loop()

        def _handler() -> None:
            logger.info("Shutdown signal received")
            self._shutdown_event.set()

        if sys.platform != "win32":
            for sig in (os_signal.SIGINT, os_signal.SIGTERM):
                loop.add_signal_handler(sig, _handler)
        else:
            # Windows doesn't support add_signal_handler
            os_signal.signal(os_signal.SIGINT, lambda s, f: _handler())

    # ------------------------------------------------------------- status

    @property
    def status(self) -> dict[str, Any]:
        """Snapshot of the engine's current state."""
        return {
            "running": self._running,
            "tick": self._tick_count,
            "open_positions": len(self._positions),
            "circuit_breaker": self.circuit_breaker.state.value,
            "daily_pnl": self.risk_manager.get_daily_pnl(),
            "order_summary": self.order_manager.summary(),
        }
