"""
Risk manager — central gate-keeper that validates every trade.

Every signal produced by a strategy must pass through the
:class:`RiskManager` before being forwarded to the execution layer.

Rules enforced
--------------
* **Max risk per trade**: 1–2 % of total capital ($0.50–$1.00 on $50)
* **Daily loss limit**: 3 % of capital
* **Max drawdown**: 15 % from equity peak
* **Max open positions**: 3 (small capital constraint)
* **Max portfolio exposure**: 30 % of balance in open positions

Usage
-----
>>> rm = RiskManager(capital=50.0)
>>> approved, reason = rm.validate_trade(signal, balance=48.5)
>>> if approved:
...     size = rm.calculate_position_size(signal, balance=48.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from loguru import logger

from src.strategy.base import Signal, TradeAction


@dataclass(slots=True)
class RiskConfig:
    """Configuration knobs for the risk manager.

    All percentage values are expressed as decimals (e.g. 0.02 = 2 %).
    """

    max_risk_per_trade: float = 0.02          # 2% of capital per trade
    daily_loss_limit: float = 0.03            # halt trading after 3% daily loss
    max_drawdown: float = 0.15                # emergency stop at 15% drawdown
    max_open_positions: int = 3               # small-capital constraint
    max_portfolio_exposure: float = 0.30      # max 30% of balance in open trades
    min_order_value_usd: float = 5.0          # Binance minimum ~$5–10
    min_confidence: float = 0.45             # reject low-confidence signals


class RiskManager:
    """Central risk gate-keeper.

    Parameters
    ----------
    capital : float
        Starting capital in USD (default $50).
    config : RiskConfig | None
        Override risk rules.

    Examples
    --------
    >>> rm = RiskManager(capital=50.0)
    >>> ok, reason = rm.validate_trade(signal, balance=49.0)
    >>> size = rm.calculate_position_size(signal, balance=49.0)
    """

    def __init__(
        self,
        capital: float = 50.0,
        config: RiskConfig | None = None,
    ) -> None:
        self.initial_capital = capital
        self.config = config or RiskConfig()
        self.peak_equity: float = capital

        # Daily P&L tracking
        self._daily_pnl: float = 0.0
        self._daily_date: date = datetime.now(timezone.utc).date()

        # Open position tracking
        self._open_positions: list[dict[str, Any]] = []
        self._trade_log: list[dict[str, Any]] = []

        logger.info(
            "RiskManager initialised — capital=${:.2f} max_risk={:.1%} "
            "daily_limit={:.1%} max_positions={}",
            capital,
            self.config.max_risk_per_trade,
            self.config.daily_loss_limit,
            self.config.max_open_positions,
        )

    # ---------------------------------------------------------------- public

    def validate_trade(
        self,
        signal: Signal,
        balance: float,
    ) -> tuple[bool, str]:
        """Run the signal through all risk rules.

        Parameters
        ----------
        signal : Signal
            The trade signal to validate.
        balance : float
            Current account balance in USD.

        Returns
        -------
        tuple[bool, str]
            ``(approved, reason)`` — *reason* explains rejection or
            says ``"approved"``.
        """
        self._roll_daily_pnl()

        # 1. HOLD signals need no validation
        if signal.action == TradeAction.HOLD:
            return False, "HOLD signal — no trade needed"

        # 2. Minimum confidence
        if signal.confidence < self.config.min_confidence:
            reason = (
                f"Confidence {signal.confidence:.2f} below minimum "
                f"{self.config.min_confidence:.2f}"
            )
            logger.warning("[risk] REJECTED — {}", reason)
            return False, reason

        # 3. Daily loss limit
        daily_limit = self.initial_capital * self.config.daily_loss_limit
        if self._daily_pnl <= -daily_limit:
            reason = (
                f"Daily loss ${self._daily_pnl:.2f} exceeds limit "
                f"-${daily_limit:.2f}"
            )
            logger.warning("[risk] REJECTED — {}", reason)
            return False, reason

        # 4. Max drawdown
        drawdown = (self.peak_equity - balance) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown >= self.config.max_drawdown:
            reason = (
                f"Drawdown {drawdown:.1%} exceeds max {self.config.max_drawdown:.1%}"
            )
            logger.warning("[risk] REJECTED — {}", reason)
            return False, reason

        # 5. Max open positions
        if len(self._open_positions) >= self.config.max_open_positions:
            reason = (
                f"Open positions ({len(self._open_positions)}) "
                f"at limit ({self.config.max_open_positions})"
            )
            logger.warning("[risk] REJECTED — {}", reason)
            return False, reason

        # 6. Portfolio exposure
        total_exposure = sum(p.get("value", 0) for p in self._open_positions)
        if balance > 0 and total_exposure / balance >= self.config.max_portfolio_exposure:
            reason = (
                f"Portfolio exposure {total_exposure / balance:.1%} "
                f"exceeds max {self.config.max_portfolio_exposure:.1%}"
            )
            logger.warning("[risk] REJECTED — {}", reason)
            return False, reason

        # 7. Verify stop-loss is set
        if signal.stop_loss_price <= 0:
            reason = "Signal has no stop-loss price"
            logger.warning("[risk] REJECTED — {}", reason)
            return False, reason

        logger.info(
            "[risk] APPROVED — {} {} conf={:.2f} balance=${:.2f}",
            signal.action.value, signal.symbol, signal.confidence, balance,
        )
        return True, "approved"

    def calculate_position_size(
        self,
        signal: Signal,
        balance: float,
        risk_per_trade: float | None = None,
    ) -> float:
        """Determine position size in USD respecting risk limits.

        The position is sized so that the maximum dollar loss (price →
        stop-loss) equals ``risk_per_trade`` fraction of *balance*.

        Parameters
        ----------
        signal : Signal
            Trade signal (needs ``stop_loss_price``).
        balance : float
            Current account balance in USD.
        risk_per_trade : float | None
            Override for ``config.max_risk_per_trade``.

        Returns
        -------
        float
            Position size **in USD**.  Returns 0.0 if any constraint
            is violated.
        """
        risk_frac = risk_per_trade or self.config.max_risk_per_trade
        max_risk_usd = balance * risk_frac  # e.g. $50 * 0.02 = $1.00

        current_price = signal.metadata.get(
            "current_price",
            (signal.take_profit_price + signal.stop_loss_price) / 2,
        )
        if current_price <= 0:
            logger.warning("[risk] Invalid current price — position size = 0")
            return 0.0

        # Risk per unit = distance from entry to stop-loss
        risk_per_unit = abs(current_price - signal.stop_loss_price)
        if risk_per_unit <= 0:
            logger.warning("[risk] Zero risk distance — position size = 0")
            return 0.0

        # Position size in base units, then convert to USD
        units = max_risk_usd / risk_per_unit
        position_usd = units * current_price

        # Cap by portfolio exposure limit
        max_exposure = balance * self.config.max_portfolio_exposure
        current_exposure = sum(p.get("value", 0) for p in self._open_positions)
        remaining_exposure = max(0, max_exposure - current_exposure)
        position_usd = min(position_usd, remaining_exposure)

        # Enforce minimum order value
        if position_usd < self.config.min_order_value_usd:
            logger.warning(
                "[risk] Position ${:.2f} below min order ${:.2f} — size = 0",
                position_usd, self.config.min_order_value_usd,
            )
            return 0.0

        logger.info(
            "[risk] Position sized: ${:.2f} (risk ${:.2f}, {:.1%} of balance)",
            position_usd, max_risk_usd, position_usd / balance if balance else 0,
        )
        return round(position_usd, 2)

    # -------------------------------------------------------------- tracking

    def register_open_position(
        self,
        symbol: str,
        value: float,
        side: str,
        position_id: str,
    ) -> None:
        """Register a new open position for exposure tracking."""
        self._open_positions.append(
            {
                "symbol": symbol,
                "value": value,
                "side": side,
                "position_id": position_id,
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.debug("[risk] Registered position {} value=${:.2f}", position_id, value)

    def close_position(self, position_id: str, realized_pnl: float) -> None:
        """Remove a closed position and update daily P&L."""
        self._open_positions = [
            p for p in self._open_positions if p.get("position_id") != position_id
        ]
        self._daily_pnl += realized_pnl
        if self._daily_pnl + self.initial_capital > self.peak_equity:
            self.peak_equity = self._daily_pnl + self.initial_capital

        self._trade_log.append(
            {
                "position_id": position_id,
                "pnl": realized_pnl,
                "closed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info(
            "[risk] Closed position {} PnL=${:.2f} daily_pnl=${:.2f}",
            position_id, realized_pnl, self._daily_pnl,
        )

    def get_daily_pnl(self) -> float:
        """Return the current-day cumulative P&L."""
        self._roll_daily_pnl()
        return self._daily_pnl

    def check_circuit_breaker(self, balance: float) -> tuple[bool, str]:
        """Quick circuit-breaker check (delegates to CircuitBreaker for full logic).

        Returns
        -------
        tuple[bool, str]
            ``(safe, message)``
        """
        self._roll_daily_pnl()

        daily_limit = self.initial_capital * self.config.daily_loss_limit
        if self._daily_pnl <= -daily_limit:
            return False, f"Daily loss limit breached: ${self._daily_pnl:.2f}"

        drawdown = (self.peak_equity - balance) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown >= self.config.max_drawdown:
            return False, f"Max drawdown breached: {drawdown:.1%}"

        return True, "OK"

    def update_peak_equity(self, current_equity: float) -> None:
        """Update the peak equity high-water mark."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    @property
    def open_position_count(self) -> int:
        """Number of currently tracked open positions."""
        return len(self._open_positions)

    # ------------------------------------------------------------- internal

    def _roll_daily_pnl(self) -> None:
        """Reset daily P&L if the date has changed (UTC)."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_date:
            logger.info(
                "[risk] New trading day — resetting daily PnL (was ${:.2f})",
                self._daily_pnl,
            )
            self._daily_pnl = 0.0
            self._daily_date = today
