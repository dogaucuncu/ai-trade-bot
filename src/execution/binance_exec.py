"""
Binance executor — execute trades on Binance via CCXT.

Uses `ccxt.binance` (async) to interact with the Binance exchange.
Defaults to **testnet** for safe development and paper trading.

Features:

* Spot trading support
* Binance-specific lot-size and min-notional filters
* Retry logic for transient HTTP errors
* Balance and open-order queries

Usage
-----
>>> executor = BinanceExecutor(testnet=True)
>>> await executor.initialize()
>>> result = await executor.place_market_order("SOL/USDT", "buy", 0.07)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

try:
    import ccxt.async_support as ccxt  # type: ignore[import-untyped]
except ImportError:
    ccxt = None  # type: ignore[assignment]
    logger.warning("ccxt not installed — BinanceExecutor will not work")


class BinanceExecutor:
    """Async executor for Binance spot trading via CCXT.

    Parameters
    ----------
    api_key : str | None
        Binance API key (reads ``BINANCE_API_KEY`` env var if omitted).
    api_secret : str | None
        Binance secret (reads ``BINANCE_API_SECRET`` env var if omitted).
    testnet : bool
        Use Binance testnet (default ``True``).
    max_retries : int
        Number of retries on transient API failures.

    Examples
    --------
    >>> exec = BinanceExecutor(testnet=True)
    >>> await exec.initialize()
    >>> bal = await exec.get_balance()
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool = True,
        max_retries: int = 3,
    ) -> None:
        if ccxt is None:
            raise ImportError(
                "Install ccxt: pip install ccxt"
            )

        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_SECRET_KEY") or os.getenv("BINANCE_API_SECRET", "")
        self.testnet = testnet
        self.max_retries = max_retries
        self._exchange: ccxt.binance | None = None
        self._markets_loaded = False

        logger.info(
            "BinanceExecutor created — testnet={} retries={}",
            testnet, max_retries,
        )

    # ---------------------------------------------------------------- init

    async def initialize(self) -> None:
        """Create the CCXT exchange instance and load markets."""
        import ssl
        import certifi

        # Build a proper SSL context using certifi certs (fixes Windows Python 3.13 SSL errors)
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        config: dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "loadCurrencies": False,  # skip sapi/margin endpoints (restricted on most accounts)
            },
            "aiohttp_trust_env": False,
            "verify": ssl_context,  # use certifi-backed SSL context
        }
        if self.testnet:
            config["sandbox"] = True

        self._exchange = ccxt.binance(config)

        # Only enable sandbox if explicitly using testnet credentials
        if self.testnet:
            try:
                self._exchange.set_sandbox_mode(True)
            except Exception:
                pass  # Some CCXT versions handle this differently

        # load_markets with reload=False to avoid re-fetching currencies
        await self._exchange.load_markets()
        self._markets_loaded = True
        logger.info(
            "Binance exchange initialised — {} markets loaded",
            len(self._exchange.markets),
        )

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._exchange:
            await self._exchange.close()
            logger.info("Binance exchange connection closed")

    # ------------------------------------------------------------ orders

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> dict[str, Any]:
        """Place a market order.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"SOL/USDT"``.
        side : str
            ``"buy"`` or ``"sell"``.
        quantity : float
            Amount in base currency.

        Returns
        -------
        dict
            CCXT order response.
        """
        quantity = self._adjust_quantity(symbol, quantity)
        self._check_min_notional(symbol, quantity)

        result = await self._retry(
            self._exchange.create_order,  # type: ignore[union-attr]
            symbol, "market", side, quantity,
        )
        logger.info(
            "[binance] Market {} {} {:.8f} — id={}",
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
        quantity = self._adjust_quantity(symbol, quantity)
        price = self._adjust_price(symbol, price)
        self._check_min_notional(symbol, quantity, price)

        result = await self._retry(
            self._exchange.create_order,  # type: ignore[union-attr]
            symbol, "limit", side, quantity, price,
        )
        logger.info(
            "[binance] Limit {} {} {:.8f} @ {:.8f} — id={}",
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
        """Place a stop-loss market order (STOP_LOSS type on Binance)."""
        quantity = self._adjust_quantity(symbol, quantity)
        stop_price = self._adjust_price(symbol, stop_price)

        params = {"stopPrice": stop_price}
        result = await self._retry(
            self._exchange.create_order,  # type: ignore[union-attr]
            symbol, "STOP_LOSS", side, quantity, None, params,
        )
        logger.info(
            "[binance] StopLoss {} {} {:.8f} trigger={:.8f} — id={}",
            side.upper(), symbol, quantity, stop_price, result.get("id"),
        )
        return result

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Cancel an open order by exchange order ID."""
        result = await self._retry(
            self._exchange.cancel_order,  # type: ignore[union-attr]
            order_id, symbol,
        )
        logger.info("[binance] Cancelled order {} on {}", order_id, symbol)
        return result

    # ----------------------------------------------------------- queries

    async def get_balance(self, currency: str = "USDT") -> float:
        """Return free balance for *currency*."""
        balance = await self._retry(
            self._exchange.fetch_balance,  # type: ignore[union-attr]
        )
        free = float(balance.get("free", {}).get(currency, 0))
        logger.debug("[binance] Balance {} = {:.4f}", currency, free)
        return free

    async def get_open_orders(
        self, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """Return all open orders, optionally filtered by symbol."""
        orders = await self._retry(
            self._exchange.fetch_open_orders,  # type: ignore[union-attr]
            symbol,
        )
        logger.debug("[binance] {} open orders", len(orders))
        return orders

    # ------------------------------------------------ lot size / precision

    def _adjust_quantity(self, symbol: str, quantity: float) -> float:
        """Round *quantity* to the exchange's lot-size precision."""
        if not self._markets_loaded or not self._exchange:
            return quantity
        market = self._exchange.market(symbol)
        precision = market.get("precision", {}).get("amount", 8)
        step = market.get("limits", {}).get("amount", {}).get("min", 0)
        if step and step > 0:
            quantity = quantity - (quantity % step)
        return round(quantity, precision)

    def _adjust_price(self, symbol: str, price: float) -> float:
        """Round *price* to the exchange's price precision."""
        if not self._markets_loaded or not self._exchange:
            return price
        market = self._exchange.market(symbol)
        precision = market.get("precision", {}).get("price", 8)
        return round(price, precision)

    def _check_min_notional(
        self,
        symbol: str,
        quantity: float,
        price: float | None = None,
    ) -> None:
        """Warn if order value is below Binance's minimum notional ($5–10)."""
        if not self._markets_loaded or not self._exchange:
            return
        market = self._exchange.market(symbol)
        min_cost = market.get("limits", {}).get("cost", {}).get("min", 5)
        est_price = price or market.get("info", {}).get("lastPrice", 0)
        try:
            est_price = float(est_price)
        except (ValueError, TypeError):
            return
        notional = quantity * est_price
        if min_cost and notional < float(min_cost):
            logger.warning(
                "[binance] Order notional ${:.2f} below min ${:.2f} for {}",
                notional, float(min_cost), symbol,
            )

    # -------------------------------------------------------------- retry

    async def _retry(self, coro_func: Any, *args: Any, **kwargs: Any) -> Any:
        """Retry a CCXT call up to *max_retries* times on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await coro_func(*args, **kwargs)
            except (
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
                ccxt.RequestTimeout,
            ) as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "[binance] Transient error (attempt {}/{}): {} — retrying in {}s",
                    attempt, self.max_retries, exc, wait,
                )
                await asyncio.sleep(wait)
            except ccxt.ExchangeError as exc:
                logger.error("[binance] Exchange error (non-retryable): {}", exc)
                raise
        raise RuntimeError(
            f"Binance API call failed after {self.max_retries} retries"
        ) from last_exc
