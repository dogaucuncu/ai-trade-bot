"""
Circuit breaker — emergency safeguard for dangerous market conditions.

Monitors the following conditions and transitions through states:

* ``NORMAL``   → all systems go
* ``CAUTIOUS`` → minor anomaly detected (e.g. stale data)
* ``HALTED``   → daily loss limit breached — no new entries
* ``EMERGENCY``→ max drawdown exceeded — close all positions immediately

Triggers
--------
1. Daily loss exceeds limit  → HALTED
2. Max drawdown exceeded     → EMERGENCY (close all)
3. API / connection lost     → HALTED
4. Flash crash (>5 % move in 1 min) → CAUTIOUS (pause new entries)
5. Stale data (no update for 30 s)  → CAUTIOUS

Usage
-----
>>> cb = CircuitBreaker(capital=50.0)
>>> safe, alerts = cb.check_all(
...     balance=47, daily_pnl=-1.2, last_prices=[100, 94], last_update_age_s=5
... )
>>> if not safe:
...     for alert in alerts:
...         print(alert)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger


class BreakerState(str, Enum):
    """Circuit-breaker states — severity increases downward."""

    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"
    HALTED = "HALTED"
    EMERGENCY = "EMERGENCY"


class CircuitBreaker:
    """Monitors for dangerous conditions and halts trading when necessary.

    Parameters
    ----------
    capital : float
        Starting capital (used for percentage thresholds).
    daily_loss_limit_pct : float
        Fraction of capital; daily loss beyond this triggers HALTED.
    max_drawdown_pct : float
        Fraction of peak equity; drawdown beyond this triggers EMERGENCY.
    flash_crash_pct : float
        Percentage move in 1 minute that qualifies as a flash crash.
    stale_data_seconds : float
        Seconds without a price update before data is considered stale.

    Examples
    --------
    >>> cb = CircuitBreaker(capital=50.0)
    >>> safe, alerts = cb.check_all(balance=45, daily_pnl=-1.0)
    """

    def __init__(
        self,
        capital: float = 50.0,
        daily_loss_limit_pct: float = 0.03,
        max_drawdown_pct: float = 0.15,
        flash_crash_pct: float = 5.0,
        stale_data_seconds: float = 30.0,
    ) -> None:
        self.capital = capital
        self.peak_equity: float = capital
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.flash_crash_pct = flash_crash_pct
        self.stale_data_seconds = stale_data_seconds

        self.state: BreakerState = BreakerState.NORMAL
        self._state_history: list[dict[str, Any]] = []
        self._api_healthy: bool = True

        logger.info(
            "CircuitBreaker initialised — capital=${:.2f} "
            "daily_limit={:.1%} max_dd={:.1%} flash={:.1f}%",
            capital,
            daily_loss_limit_pct,
            max_drawdown_pct,
            flash_crash_pct,
        )

    # ---------------------------------------------------------------- public

    def check_all(
        self,
        balance: float,
        daily_pnl: float = 0.0,
        last_prices: list[float] | None = None,
        last_update_age_s: float = 0.0,
        api_healthy: bool = True,
    ) -> tuple[bool, list[str]]:
        """Run every check and return overall safety status.

        Parameters
        ----------
        balance : float
            Current account balance (USD).
        daily_pnl : float
            Cumulative P&L today (negative means a loss).
        last_prices : list[float] | None
            Last two prices (most recent last) for flash-crash detection.
        last_update_age_s : float
            Seconds since the last price update was received.
        api_healthy : bool
            ``True`` if the exchange API connection is alive.

        Returns
        -------
        tuple[bool, list[str]]
            ``(safe, alerts)`` — *safe* is ``False`` if trading should
            be paused or stopped.
        """
        alerts: list[str] = []
        new_state = BreakerState.NORMAL

        # --- 1. Daily loss ------------------------------------------------
        daily_limit = self.capital * self.daily_loss_limit_pct
        if daily_pnl <= -daily_limit:
            msg = (
                f"Daily loss ${daily_pnl:.2f} exceeds limit "
                f"-${daily_limit:.2f} → HALTED"
            )
            alerts.append(msg)
            new_state = max(new_state, BreakerState.HALTED, key=_severity)

        # --- 2. Drawdown --------------------------------------------------
        if balance > self.peak_equity:
            self.peak_equity = balance

        drawdown = (
            (self.peak_equity - balance) / self.peak_equity
            if self.peak_equity > 0
            else 0
        )
        if drawdown >= self.max_drawdown_pct:
            msg = (
                f"Drawdown {drawdown:.1%} exceeds max "
                f"{self.max_drawdown_pct:.1%} → EMERGENCY"
            )
            alerts.append(msg)
            new_state = max(new_state, BreakerState.EMERGENCY, key=_severity)

        # --- 3. API health ------------------------------------------------
        self._api_healthy = api_healthy
        if not api_healthy:
            msg = "API connection lost → HALTED"
            alerts.append(msg)
            new_state = max(new_state, BreakerState.HALTED, key=_severity)

        # --- 4. Flash crash -----------------------------------------------
        if last_prices and len(last_prices) >= 2:
            prev, curr = last_prices[-2], last_prices[-1]
            if prev > 0:
                move_pct = abs(curr - prev) / prev * 100
                if move_pct >= self.flash_crash_pct:
                    msg = (
                        f"Flash crash detected: {move_pct:.1f}% move "
                        f"({prev:.4f} → {curr:.4f}) → CAUTIOUS"
                    )
                    alerts.append(msg)
                    new_state = max(new_state, BreakerState.CAUTIOUS, key=_severity)

        # --- 5. Stale data ------------------------------------------------
        if last_update_age_s >= self.stale_data_seconds:
            msg = (
                f"Stale data: no update for {last_update_age_s:.0f}s "
                f"(limit {self.stale_data_seconds:.0f}s) → CAUTIOUS"
            )
            alerts.append(msg)
            new_state = max(new_state, BreakerState.CAUTIOUS, key=_severity)

        # --- transition ---------------------------------------------------
        self._transition(new_state, alerts)
        safe = self.state in (BreakerState.NORMAL,)

        return safe, alerts

    def trigger_emergency_stop(self, reason: str = "manual") -> None:
        """Force an immediate EMERGENCY state.

        The bot engine should respond by closing all open positions.
        """
        self._transition(
            BreakerState.EMERGENCY,
            [f"EMERGENCY STOP triggered: {reason}"],
        )

    def reset(self) -> None:
        """Return to NORMAL state (e.g. after manual review)."""
        old = self.state
        self.state = BreakerState.NORMAL
        logger.info("[breaker] Reset from {} → NORMAL", old.value)
        self._log_transition(old, BreakerState.NORMAL, ["manual reset"])

    @property
    def is_safe(self) -> bool:
        """``True`` only when the breaker is in NORMAL state."""
        return self.state == BreakerState.NORMAL

    @property
    def allows_new_entries(self) -> bool:
        """``True`` when new positions can be opened (NORMAL only)."""
        return self.state == BreakerState.NORMAL

    @property
    def requires_close_all(self) -> bool:
        """``True`` when all positions must be closed immediately."""
        return self.state == BreakerState.EMERGENCY

    # ------------------------------------------------------------- internal

    def _transition(self, new_state: BreakerState, alerts: list[str]) -> None:
        """Transition to *new_state* if it differs from the current state."""
        if new_state == self.state:
            return

        old = self.state
        self.state = new_state
        logger.warning(
            "[breaker] State change: {} → {} | alerts={}",
            old.value, new_state.value, alerts,
        )
        self._log_transition(old, new_state, alerts)

    def _log_transition(
        self,
        old: BreakerState,
        new: BreakerState,
        alerts: list[str],
    ) -> None:
        self._state_history.append(
            {
                "from": old.value,
                "to": new.value,
                "alerts": alerts,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


# ----------------------------------------------------------------- helpers

_SEVERITY_ORDER = {
    BreakerState.NORMAL: 0,
    BreakerState.CAUTIOUS: 1,
    BreakerState.HALTED: 2,
    BreakerState.EMERGENCY: 3,
}


def _severity(state: BreakerState) -> int:
    return _SEVERITY_ORDER.get(state, 0)
