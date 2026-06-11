"""
Trading scheduler — manages strategy execution timing with APScheduler.

Crypto strategies run 24 / 7.  Stock strategies only run during NYSE
market hours (9:30–16:00 Eastern Time, Mon–Fri).

Usage
-----
>>> scheduler = TradingScheduler(engine=bot_engine)
>>> scheduler.start()
>>> # ... runs in the background ...
>>> scheduler.stop()
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from loguru import logger

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore[import-untyped]
    from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
    from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]

    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not installed — TradingScheduler will not work")


# NYSE trading window (Eastern Time)
_ET = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


class TradingScheduler:
    """Schedule strategy jobs at the correct intervals and market windows.

    Parameters
    ----------
    engine : Any
        A :class:`BotEngine` instance (or compatible object) whose
        strategy-runner coroutines will be scheduled.
    timezone_str : str
        IANA timezone for cron expressions (default ``"America/New_York"``).

    Examples
    --------
    >>> from src.bot.engine import BotEngine
    >>> engine = BotEngine(capital=50.0)
    >>> scheduler = TradingScheduler(engine=engine)
    >>> scheduler.start()
    """

    def __init__(
        self,
        engine: Any,
        timezone_str: str = "America/New_York",
    ) -> None:
        if not APSCHEDULER_AVAILABLE:
            raise ImportError(
                "Install APScheduler: pip install apscheduler"
            )

        self.engine = engine
        self.tz = ZoneInfo(timezone_str)
        self._scheduler = AsyncIOScheduler(timezone=timezone_str)
        self._jobs: dict[str, str] = {}  # name -> job_id

        logger.info("TradingScheduler created — timezone={}", timezone_str)

    # ---------------------------------------------------------------- start

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        self._add_crypto_jobs()
        self._add_stock_jobs()
        self._add_housekeeping_jobs()
        self._scheduler.start()
        logger.info("Scheduler started with {} jobs", len(self._jobs))

    def stop(self) -> None:
        """Shut down the scheduler gracefully."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)
            logger.info("Scheduler stopped")

    # -------------------------------------------------------- crypto (24/7)

    def _add_crypto_jobs(self) -> None:
        """Schedule crypto strategies — run 24/7."""
        # Scalping — every 1 minute
        self.add_strategy_job(
            name="crypto_scalping",
            func=self._run_crypto_scalping,
            trigger=IntervalTrigger(minutes=1),
        )

        # Mean reversion — every 15 minutes
        self.add_strategy_job(
            name="crypto_mean_reversion",
            func=self._run_crypto_mean_reversion,
            trigger=IntervalTrigger(minutes=15),
        )

        # Momentum — every 1 hour
        self.add_strategy_job(
            name="crypto_momentum",
            func=self._run_crypto_momentum,
            trigger=IntervalTrigger(hours=1),
        )

    # ------------------------------------------------ stocks (market hours)

    def _add_stock_jobs(self) -> None:
        """Schedule stock strategies — NYSE hours only (9:30–16:00 ET, Mon–Fri)."""
        # Momentum on stocks — every hour during market hours
        self.add_strategy_job(
            name="stock_momentum",
            func=self._run_stock_momentum,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="10-15",  # full hours within market window
                minute=0,
                timezone=self.tz,
            ),
        )

        # Mean reversion on stocks — every 15 min during market hours
        self.add_strategy_job(
            name="stock_mean_reversion",
            func=self._run_stock_mean_reversion,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute="30,45,0,15",
                timezone=self.tz,
            ),
        )

    # --------------------------------------------------- housekeeping jobs

    def _add_housekeeping_jobs(self) -> None:
        """Add maintenance tasks."""
        # Position monitoring — every 30 seconds
        self.add_strategy_job(
            name="position_monitor",
            func=self._monitor_positions,
            trigger=IntervalTrigger(seconds=30),
        )

        # Ensemble weight update — every 4 hours
        self.add_strategy_job(
            name="weight_update",
            func=self._update_ensemble_weights,
            trigger=IntervalTrigger(hours=4),
        )

        # Daily summary — 00:05 UTC
        self.add_strategy_job(
            name="daily_summary",
            func=self._daily_summary,
            trigger=CronTrigger(hour=0, minute=5, timezone="UTC"),
        )

    # ----------------------------------------------------------- job mgmt

    def add_strategy_job(
        self,
        name: str,
        func: Callable[..., Any],
        trigger: Any,
        **kwargs: Any,
    ) -> str:
        """Add a new scheduled job.

        Parameters
        ----------
        name : str
            Human-readable job name.
        func : callable
            Async function to execute.
        trigger : IntervalTrigger | CronTrigger
            APScheduler trigger.

        Returns
        -------
        str
            Job ID.
        """
        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=name,
            name=name,
            replace_existing=True,
            misfire_grace_time=30,
            **kwargs,
        )
        self._jobs[name] = job.id
        logger.debug("Scheduled job: {}", name)
        return job.id

    def remove_job(self, name: str) -> None:
        """Remove a scheduled job by name."""
        job_id = self._jobs.pop(name, None)
        if job_id:
            self._scheduler.remove_job(job_id)
            logger.info("Removed job: {}", name)

    # -------------------------------------------------------- market hours

    @staticmethod
    def is_market_open() -> bool:
        """Check whether the NYSE is currently in session.

        Returns
        -------
        bool
        """
        now_et = datetime.now(_ET)
        if now_et.weekday() >= 5:
            return False
        return _MARKET_OPEN <= now_et.time() <= _MARKET_CLOSE

    # ------------------------------------------------- job implementations

    async def _run_crypto_scalping(self) -> None:
        """Execute the scalping strategy on all crypto symbols."""
        logger.debug("[scheduler] Running crypto scalping")
        try:
            await self.engine._run_strategy_on_symbols(
                self.engine.scalping,
                self.engine.crypto_symbols,
                "1m",
            )
        except Exception:
            logger.exception("[scheduler] Crypto scalping failed")

    async def _run_crypto_mean_reversion(self) -> None:
        """Execute mean reversion on crypto symbols."""
        logger.debug("[scheduler] Running crypto mean reversion")
        try:
            await self.engine._run_strategy_on_symbols(
                self.engine.mean_reversion,
                self.engine.crypto_symbols,
                "15m",
            )
        except Exception:
            logger.exception("[scheduler] Crypto mean reversion failed")

    async def _run_crypto_momentum(self) -> None:
        """Execute momentum on crypto symbols."""
        logger.debug("[scheduler] Running crypto momentum")
        try:
            await self.engine._run_strategy_on_symbols(
                self.engine.momentum,
                self.engine.crypto_symbols,
                "1h",
            )
        except Exception:
            logger.exception("[scheduler] Crypto momentum failed")

    async def _run_stock_momentum(self) -> None:
        """Execute momentum on stock symbols (market hours only)."""
        if not self.is_market_open():
            logger.debug("[scheduler] Market closed — skipping stock momentum")
            return
        logger.debug("[scheduler] Running stock momentum")
        try:
            await self.engine._run_strategy_on_symbols(
                self.engine.momentum,
                self.engine.stock_symbols,
                "1h",
            )
        except Exception:
            logger.exception("[scheduler] Stock momentum failed")

    async def _run_stock_mean_reversion(self) -> None:
        """Execute mean reversion on stock symbols (market hours only)."""
        if not self.is_market_open():
            return
        logger.debug("[scheduler] Running stock mean reversion")
        try:
            await self.engine._run_strategy_on_symbols(
                self.engine.mean_reversion,
                self.engine.stock_symbols,
                "15m",
            )
        except Exception:
            logger.exception("[scheduler] Stock mean reversion failed")

    async def _monitor_positions(self) -> None:
        """Periodic position monitoring."""
        try:
            await self.engine._monitor_positions()
        except Exception:
            logger.exception("[scheduler] Position monitoring failed")

    async def _update_ensemble_weights(self) -> None:
        """Periodically re-balance ensemble strategy weights."""
        try:
            self.engine.ensemble.update_weights()
        except Exception:
            logger.exception("[scheduler] Weight update failed")

    async def _daily_summary(self) -> None:
        """Log a daily trading summary."""
        status = self.engine.status
        logger.info(
            "=== DAILY SUMMARY === "
            "PnL=${:.2f} | Positions={} | Orders={} | Breaker={}",
            status["daily_pnl"],
            status["open_positions"],
            status.get("order_summary", {}),
            status["circuit_breaker"],
        )
