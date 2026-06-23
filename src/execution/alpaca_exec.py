"""
Alpaca executor — execute US stock trades via the Alpaca Markets API.

Uses the ``alpaca-py`` SDK with **paper trading** enabled by default.
Handles market-hours checks and fractional share support for the
$12.50 stock allocation.

Usage
-----
>>> executor = AlpacaExecutor(paper=True)
>>> await executor.place_market_order("AAPL", "buy", 0.05)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger

try:
    from alpaca.trading.client import TradingClient  # type: ignore[import-untyped]
    from alpaca.trading.requests import (  # type: ignore[import-untyped]
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
        StopLossRequest,
        StopOrderRequest,
    )
    from alpaca.trading.enums import (  # type: ignore[import-untyped]
        OrderSide,
        OrderStatus as AlpacaOrderStatus,
        QueryOrderStatus,
        TimeInForce,
    )

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("alpaca-py not installed — AlpacaExecutor will not work")


# NYSE trading hours (Eastern Time)
_ET = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


class AlpacaExecutor:
    """Async-compatible executor for US stock trading via Alpaca.

    Parameters
    ----------
    api_key : str | None
        Alpaca API key (reads ``ALPACA_API_KEY`` env var if omitted).
    api_secret : str | None
        Alpaca secret (reads ``ALPACA_API_SECRET`` env var if omitted).
    paper : bool
        Use paper trading (default ``True``).
    max_retries : int
        Number of retries on transient failures.

    Examples
    --------
    >>> exec = AlpacaExecutor(paper=True)
    >>> await exec.place_market_order("AAPL", "buy", 0.05)
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        paper: bool = True,
        max_retries: int = 3,
    ) -> None:
        if not ALPACA_AVAILABLE:
            raise ImportError("Install alpaca-py: pip install alpaca-py")

        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET", "")
        self.paper = paper
        self.max_retries = max_retries

        self._client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=self.paper,
        )

        logger.info(
            "AlpacaExecutor created — paper={} retries={}",
            paper, max_retries,
        )

    # ----------------------------------------------------------- orders

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> dict[str, Any]:
        """Place a fractional-share market order.

        Parameters
        ----------
        symbol : str
            Stock ticker, e.g. ``"AAPL"``.
        side : str
            ``"buy"`` or ``"sell"``.
        quantity : float
            Number of shares (fractional OK).

        Returns
        -------
        dict
            Normalised order response.
        """
        if not self.is_market_open():
            logger.warning("[alpaca] Market is closed — order may be queued")

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=round(quantity, 6),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        result = await self._submit(request)
        logger.info(
            "[alpaca] Market {} {} {:.6f} — id={}",
            side.upper(), symbol, quantity, result.get("id"),
        )
        return result

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
    ) -> dict[str, Any]:
        """Place a limit order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = LimitOrderRequest(
            symbol=symbol,
            qty=round(quantity, 6),
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(price, 2),
        )
        result = await self._submit(request)
        logger.info(
            "[alpaca] Limit {} {} {:.6f} @ {:.2f} — id={}",
            side.upper(), symbol, quantity, price, result.get("id"),
        )
        return result

    async def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
    ) -> dict[str, Any]:
        """Place a stop order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        request = StopOrderRequest(
            symbol=symbol,
            qty=round(quantity, 6),
            side=order_side,
            time_in_force=TimeInForce.DAY,
            stop_price=round(stop_price, 2),
        )
        result = await self._submit(request)
        logger.info(
            "[alpaca] Stop {} {} {:.6f} trigger={:.2f} — id={}",
            side.upper(), symbol, quantity, stop_price, result.get("id"),
        )
        return result

    async def cancel_order(
        self, order_id: str, symbol: str = ""
    ) -> dict[str, Any]:
        """Cancel an open order."""

        def _cancel() -> dict[str, Any]:
            self._client.cancel_order_by_id(order_id)
            return {"id": order_id, "status": "cancelled"}

        result = await self._retry_sync(_cancel)
        logger.info("[alpaca] Cancelled order {}", order_id)
        return result

    # ---------------------------------------------------------- queries

    async def get_balance(self) -> dict[str, float]:
        """Return account cash and equity.

        Returns
        -------
        dict
            ``{"cash": float, "equity": float, "buying_power": float}``
        """

        def _fetch() -> dict[str, float]:
            acct = self._client.get_account()
            return {
                "cash": float(acct.cash),
                "equity": float(acct.equity),
                "buying_power": float(acct.buying_power),
            }

        result = await self._retry_sync(_fetch)
        logger.debug("[alpaca] Balance: {}", result)
        return result

    async def get_positions(self) -> list[dict[str, Any]]:
        """Return all open positions."""

        def _fetch() -> list[dict[str, Any]]:
            positions = self._client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "side": p.side.value,
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "market_value": float(p.market_value),
                }
                for p in positions
            ]

        result = await self._retry_sync(_fetch)
        logger.debug("[alpaca] {} open positions", len(result))
        return result

    async def get_open_orders(self) -> list[dict[str, Any]]:
        """Return all open orders."""

        def _fetch() -> list[dict[str, Any]]:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self._client.get_orders(filter=request)
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "qty": float(o.qty) if o.qty else 0,
                    "status": o.status.value,
                    "type": o.type.value,
                }
                for o in orders
            ]

        result = await self._retry_sync(_fetch)
        return result

    # -------------------------------------------------------- market hours

    @staticmethod
    def is_market_open() -> bool:
        """Check whether the NYSE is currently open (9:30–16:00 ET, weekdays).

        Returns
        -------
        bool
        """
        now_et = datetime.now(_ET)
        if now_et.weekday() >= 5:  # Saturday / Sunday
            return False
        current_time = now_et.time()
        return _MARKET_OPEN <= current_time <= _MARKET_CLOSE

    @staticmethod
    def next_market_open() -> datetime:
        """Return the next market open as a UTC datetime."""
        now_et = datetime.now(_ET)
        target = now_et.replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        if now_et.time() >= _MARKET_OPEN:
            # Already past open today, try next weekday
            target = target + timedelta(days=1)

        while target.weekday() >= 5:
            target = target + timedelta(days=1)

        return target.astimezone(timezone.utc)

    # -------------------------------------------------------------- retry

    async def _submit(self, request: Any) -> dict[str, Any]:
        """Submit an order request via the synchronous client, wrapped async."""

        def _do() -> dict[str, Any]:
            order = self._client.submit_order(request)
            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "status": order.status.value if order.status else "unknown",
                "filled": float(order.filled_qty) if order.filled_qty else 0,
                "price": float(order.filled_avg_price) if order.filled_avg_price else 0,
            }

        return await self._retry_sync(_do)

    async def _retry_sync(self, func: Any) -> Any:
        """Retry a synchronous function in an executor with backoff."""
        last_exc: Exception | None = None
        loop = asyncio.get_event_loop()

        for attempt in range(1, self.max_retries + 1):
            try:
                return await loop.run_in_executor(None, func)
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "[alpaca] Error attempt {}/{}: {} — retry in {}s",
                    attempt, self.max_retries, exc, wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"Alpaca API call failed after {self.max_retries} retries"
        ) from last_exc
