"""
Order manager — manages the full lifecycle of trade orders.

Responsibilities:

* Create, monitor, cancel, and update orders
* Enforce risk checks via :class:`RiskManager` before every order
* Route to the correct executor (Binance / Alpaca)
* Track order state transitions

Usage
-----
>>> from src.risk.manager import RiskManager
>>> om = OrderManager(risk_manager=RiskManager(capital=50.0))
>>> order = om.place_order(signal, position_size_usd=10.0, executor=binance)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from loguru import logger

from src.risk.manager import RiskManager
from src.strategy.base import Signal, TradeAction


# ---------------------------------------------------------------------------
# Enums / data classes
# ---------------------------------------------------------------------------


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass(slots=True)
class Order:
    """Represents a single order throughout its lifecycle.

    Examples
    --------
    >>> o = Order(
    ...     symbol="SOL/USDT", side=TradeAction.BUY,
    ...     order_type=OrderType.MARKET, price=145.0, quantity=0.07,
    ... )
    """

    symbol: str
    side: TradeAction
    order_type: OrderType
    price: float
    quantity: float
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    exchange_order_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: datetime | None = None
    filled_price: float | None = None
    filled_quantity: float | None = None
    stop_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        )


# ---------------------------------------------------------------------------
# Executor protocol — Binance / Alpaca executors must implement this
# ---------------------------------------------------------------------------


class ExecutorProtocol(Protocol):
    """Minimal interface that exchange executors must satisfy."""

    async def place_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> dict[str, Any]: ...

    async def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict[str, Any]: ...

    async def place_stop_loss(
        self, symbol: str, side: str, quantity: float, stop_price: float
    ) -> dict[str, Any]: ...

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Order manager
# ---------------------------------------------------------------------------


class OrderManager:
    """Manages order creation, monitoring, cancellation, and updates.

    Every order goes through :class:`RiskManager` validation before
    submission to the exchange.

    Parameters
    ----------
    risk_manager : RiskManager
        Pre-trade risk gate.

    Examples
    --------
    >>> om = OrderManager(risk_manager=rm)
    >>> order = await om.place_order(signal, 10.0, executor)
    """

    def __init__(self, risk_manager: RiskManager) -> None:
        self.risk_manager = risk_manager
        self._orders: dict[str, Order] = {}
        logger.info("OrderManager initialised")

    # ---------------------------------------------------------------- place

    async def place_order(
        self,
        signal: Signal,
        position_size_usd: float,
        executor: ExecutorProtocol,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        balance: float | None = None,
    ) -> Order | None:
        """Create and submit an order after risk validation.

        Parameters
        ----------
        signal : Signal
            Strategy signal driving this order.
        position_size_usd : float
            Dollar value for the position.
        executor : ExecutorProtocol
            Exchange executor to submit the order through.
        order_type : OrderType
            MARKET (default), LIMIT, or STOP_LOSS.
        limit_price : float | None
            Required for LIMIT orders.
        balance : float | None
            Current balance — used for risk check.

        Returns
        -------
        Order | None
            The created order, or ``None`` if rejected by risk manager.
        """
        # --- risk check ---------------------------------------------------
        bal = balance or self.risk_manager.initial_capital
        approved, reason = self.risk_manager.validate_trade(signal, bal)
        if not approved:
            logger.warning(
                "[orders] Order REJECTED for {} — {}", signal.symbol, reason
            )
            return None

        # --- compute quantity in base units -------------------------------
        price = limit_price or signal.metadata.get(
            "current_price",
            (signal.take_profit_price + signal.stop_loss_price) / 2,
        )
        if price <= 0:
            logger.error("[orders] Invalid price for {} — cannot place order", signal.symbol)
            return None

        quantity = position_size_usd / price

        order = Order(
            symbol=signal.symbol,
            side=signal.action,
            order_type=order_type,
            price=price,
            quantity=round(quantity, 8),
            stop_price=signal.stop_loss_price if order_type == OrderType.STOP_LOSS else None,
            metadata={
                "signal_confidence": signal.confidence,
                "strategy": signal.strategy_name,
                "stop_loss": signal.stop_loss_price,
                "take_profit": signal.take_profit_price,
            },
        )

        # --- submit to exchange -------------------------------------------
        try:
            side_str = "buy" if signal.action == TradeAction.BUY else "sell"

            if order_type == OrderType.MARKET:
                result = await executor.place_market_order(
                    signal.symbol, side_str, order.quantity
                )
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    raise ValueError("limit_price required for LIMIT orders")
                result = await executor.place_limit_order(
                    signal.symbol, side_str, order.quantity, limit_price
                )
            elif order_type == OrderType.STOP_LOSS:
                result = await executor.place_stop_loss(
                    signal.symbol, side_str, order.quantity, signal.stop_loss_price
                )
            else:
                raise ValueError(f"Unknown order type: {order_type}")

            order.exchange_order_id = result.get("id") or result.get("order_id")
            order.status = OrderStatus.SUBMITTED
            order.filled_price = result.get("price")
            order.filled_quantity = result.get("filled")

            if result.get("status") in ("closed", "filled", "FILLED"):
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now(timezone.utc)

            logger.info(
                "[orders] Order {} submitted — {} {} {:.6f} @ {:.6f}",
                order.order_id, side_str.upper(), signal.symbol, order.quantity, price,
            )

        except Exception as exc:
            order.status = OrderStatus.FAILED
            order.metadata["error"] = str(exc)
            logger.error("[orders] Order FAILED — {} — {}", signal.symbol, exc)

        self._orders[order.order_id] = order
        return order

    # --------------------------------------------------------------- cancel

    async def cancel_order(
        self,
        order_id: str,
        executor: ExecutorProtocol,
    ) -> bool:
        """Cancel an active order on the exchange.

        Returns
        -------
        bool
            ``True`` if successfully cancelled.
        """
        order = self._orders.get(order_id)
        if order is None:
            logger.warning("[orders] Unknown order_id {}", order_id)
            return False
        if not order.is_active:
            logger.warning("[orders] Order {} is not active ({})", order_id, order.status.value)
            return False

        try:
            await executor.cancel_order(
                order.exchange_order_id or order.order_id,
                order.symbol,
            )
            order.status = OrderStatus.CANCELLED
            logger.info("[orders] Cancelled order {}", order_id)
            return True
        except Exception as exc:
            logger.error("[orders] Cancel failed for {} — {}", order_id, exc)
            return False

    # ---------------------------------------------------------- queries

    def get_open_orders(self) -> list[Order]:
        """Return all orders that are still active."""
        return [o for o in self._orders.values() if o.is_active]

    def get_order(self, order_id: str) -> Order | None:
        """Look up an order by its internal ID."""
        return self._orders.get(order_id)

    def get_orders_for_symbol(self, symbol: str) -> list[Order]:
        """Return all orders (any status) for a given symbol."""
        return [o for o in self._orders.values() if o.symbol == symbol]

    # --------------------------------------------------- trailing stop update

    async def update_trailing_stop(
        self,
        order_id: str,
        new_stop: float,
        executor: ExecutorProtocol,
    ) -> bool:
        """Cancel the old stop-loss order and place a new one at *new_stop*.

        Returns
        -------
        bool
            ``True`` if the new stop was successfully placed.
        """
        order = self._orders.get(order_id)
        if order is None:
            logger.warning("[orders] Cannot update stop — unknown order {}", order_id)
            return False

        # Cancel existing SL order
        cancelled = await self.cancel_order(order_id, executor)
        if not cancelled and order.is_active:
            return False

        # Place new stop-loss
        side_str = "sell" if order.side == TradeAction.BUY else "buy"
        try:
            result = await executor.place_stop_loss(
                order.symbol, side_str, order.quantity, new_stop
            )
            new_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.STOP_LOSS,
                price=new_stop,
                quantity=order.quantity,
                stop_price=new_stop,
                exchange_order_id=result.get("id"),
                status=OrderStatus.SUBMITTED,
                metadata={"parent_order": order_id, "trailing_stop": True},
            )
            self._orders[new_order.order_id] = new_order
            logger.info(
                "[orders] Updated trailing stop for {} → {:.6f}",
                order.symbol, new_stop,
            )
            return True
        except Exception as exc:
            logger.error("[orders] Trailing stop update failed — {}", exc)
            return False

    # -------------------------------------------------------------- summary

    @property
    def total_orders(self) -> int:
        return len(self._orders)

    def summary(self) -> dict[str, int]:
        """Count orders by status."""
        counts: dict[str, int] = {}
        for o in self._orders.values():
            counts[o.status.value] = counts.get(o.status.value, 0) + 1
        return counts
