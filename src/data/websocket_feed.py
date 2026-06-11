"""
AI Trade Bot — Binance WebSocket real-time feed.

Connects to the Binance WebSocket API for streaming ticker and kline
(candlestick) data with auto-reconnect and heartbeat monitoring.

Usage::

    from src.data.websocket_feed import WebSocketFeed

    feed = WebSocketFeed()

    @feed.on_kline
    async def handle(data: dict) -> None:
        print(data)

    await feed.subscribe_klines(["dogeusdt", "shibusdt"], interval="1m")
    await feed.start()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any, Optional

import websockets
import websockets.exceptions
from loguru import logger

# ── Constants ───────────────────────────────────────────────────────────
_BINANCE_WS_BASE = "wss://stream.binance.com:9443"
_BINANCE_WS_TESTNET = "wss://testnet.binance.vision/ws"

_INITIAL_RECONNECT_DELAY: float = 1.0    # seconds
_MAX_RECONNECT_DELAY: float = 60.0       # seconds
_HEARTBEAT_INTERVAL: float = 30.0        # seconds
_HEARTBEAT_TIMEOUT: float = 10.0         # pong timeout

# Type alias for user callbacks
_Callback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class WebSocketFeed:
    """Async Binance WebSocket feed with auto-reconnect.

    Parameters
    ----------
    testnet:
        If ``True``, connect to the Binance testnet stream.
    """

    def __init__(self, *, testnet: bool = True) -> None:
        self._base_url = _BINANCE_WS_TESTNET if testnet else _BINANCE_WS_BASE
        self._subscriptions: list[str] = []
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running: bool = False
        self._reconnect_delay: float = _INITIAL_RECONNECT_DELAY
        self._last_message_time: float = 0.0

        # User callbacks
        self._kline_callbacks: list[_Callback] = []
        self._ticker_callbacks: list[_Callback] = []
        self._raw_callbacks: list[_Callback] = []

        logger.info("WebSocketFeed created (testnet={})", testnet)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_kline(self, fn: _Callback) -> _Callback:
        """Decorator / registrar for kline (candlestick) events."""
        self._kline_callbacks.append(fn)
        return fn

    def on_ticker(self, fn: _Callback) -> _Callback:
        """Decorator / registrar for 24-hr mini-ticker events."""
        self._ticker_callbacks.append(fn)
        return fn

    def on_raw(self, fn: _Callback) -> _Callback:
        """Decorator / registrar for *all* raw messages."""
        self._raw_callbacks.append(fn)
        return fn

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    def subscribe_klines(
        self,
        symbols: list[str],
        interval: str = "1m",
    ) -> None:
        """Queue kline stream subscriptions (call before :meth:`start`).

        Parameters
        ----------
        symbols:
            Lowercase Binance symbols, e.g. ``["dogeusdt", "shibusdt"]``.
        interval:
            Kline interval, e.g. ``"1m"``, ``"5m"``, ``"1h"``.
        """
        for sym in symbols:
            stream = f"{sym.lower()}@kline_{interval}"
            if stream not in self._subscriptions:
                self._subscriptions.append(stream)
                logger.debug("Subscribed to stream: {}", stream)

    def subscribe_tickers(self, symbols: list[str]) -> None:
        """Queue 24-hr mini-ticker subscriptions."""
        for sym in symbols:
            stream = f"{sym.lower()}@miniTicker"
            if stream not in self._subscriptions:
                self._subscriptions.append(stream)
                logger.debug("Subscribed to stream: {}", stream)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the combined WebSocket stream and listen indefinitely.

        Automatically reconnects with exponential back-off on failures.
        """
        if not self._subscriptions:
            logger.warning("No subscriptions — nothing to stream.")
            return

        self._running = True
        url = self._build_url()
        logger.info("Starting WebSocket feed — {}", url)

        while self._running:
            try:
                await self._connect_and_listen(url)
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.InvalidStatusCode,
                OSError,
            ) as exc:
                if not self._running:
                    break
                logger.warning(
                    "WebSocket disconnected ({}). Reconnecting in {:.1f}s …",
                    exc,
                    self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, _MAX_RECONNECT_DELAY
                )
            except asyncio.CancelledError:
                logger.info("WebSocket feed cancelled.")
                break

        await self._close_ws()
        logger.info("WebSocket feed stopped.")

    async def stop(self) -> None:
        """Signal the feed to stop gracefully."""
        self._running = False
        await self._close_ws()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_url(self) -> str:
        """Construct the combined-streams URL."""
        streams = "/".join(self._subscriptions)
        return f"{self._base_url}/stream?streams={streams}"

    async def _connect_and_listen(self, url: str) -> None:
        """Connect, launch heartbeat, and dispatch messages."""
        async with websockets.connect(
            url,
            ping_interval=_HEARTBEAT_INTERVAL,
            ping_timeout=_HEARTBEAT_TIMEOUT,
        ) as ws:
            self._ws = ws
            self._reconnect_delay = _INITIAL_RECONNECT_DELAY  # reset on success
            self._last_message_time = time.monotonic()
            logger.info("WebSocket connected.")

            # Heartbeat monitor task
            heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

            try:
                async for raw_msg in ws:
                    self._last_message_time = time.monotonic()
                    await self._dispatch(raw_msg)
            finally:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task

    async def _dispatch(self, raw_msg: str | bytes) -> None:
        """Parse JSON and invoke registered callbacks."""
        try:
            msg: dict = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.warning("Non-JSON message received — skipping.")
            return

        # Combined-stream envelope
        data: dict = msg.get("data", msg)
        event_type: str = data.get("e", "")

        # Fire raw callbacks
        for cb in self._raw_callbacks:
            try:
                await cb(data)
            except Exception:
                logger.opt(exception=True).error("Error in raw callback")

        # Kline event
        if event_type == "kline":
            kline_data = self._normalise_kline(data)
            for cb in self._kline_callbacks:
                try:
                    await cb(kline_data)
                except Exception:
                    logger.opt(exception=True).error("Error in kline callback")

        # Mini-ticker event
        elif event_type == "24hrMiniTicker":
            for cb in self._ticker_callbacks:
                try:
                    await cb(data)
                except Exception:
                    logger.opt(exception=True).error("Error in ticker callback")

    @staticmethod
    def _normalise_kline(data: dict) -> dict[str, Any]:
        """Flatten the nested Binance kline payload."""
        k = data.get("k", {})
        return {
            "symbol": k.get("s", ""),
            "interval": k.get("i", ""),
            "open_time": k.get("t"),
            "close_time": k.get("T"),
            "open": float(k.get("o", 0)),
            "high": float(k.get("h", 0)),
            "low": float(k.get("l", 0)),
            "close": float(k.get("c", 0)),
            "volume": float(k.get("v", 0)),
            "is_closed": k.get("x", False),
        }

    async def _heartbeat_monitor(self) -> None:
        """Close the socket if no message arrives within the timeout."""
        while self._running:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            elapsed = time.monotonic() - self._last_message_time
            if elapsed > _HEARTBEAT_INTERVAL + _HEARTBEAT_TIMEOUT:
                logger.warning(
                    "No message for {:.0f}s — forcing reconnect.", elapsed
                )
                await self._close_ws()
                return

    async def _close_ws(self) -> None:
        """Safely close the WebSocket if open."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

