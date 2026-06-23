"""
AI Trade Bot — Async data collector.

Fetches OHLCV candle data from:
  • Binance (crypto altcoins) via CCXT async
  • Alpaca Markets (US equities) via alpaca-py

Collected data is normalised into pandas DataFrames and optionally
persisted through the :class:`Storage` layer.

Usage::

    from config.settings import settings
    from src.data.collector import DataCollector

    collector = DataCollector(settings)
    df = await collector.fetch_crypto_ohlcv("DOGE/USDT", "5m", limit=200)
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Optional

import ccxt.async_support as ccxt_async
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from loguru import logger

from config.settings import Settings
from src.data.storage import Storage
from src.net import verified_ssl_context

# ── Timeframe mapping helpers ───────────────────────────────────────────

_ALPACA_TF_MAP: dict[str, TimeFrame] = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "4h": TimeFrame(4, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}


class DataCollector:
    """Unified async data collector for crypto and stock markets.

    Parameters
    ----------
    settings:
        Application settings (API keys, pairs, etc.).
    storage:
        Optional :class:`Storage` instance.  When provided, every fetch
        automatically persists the returned candles to SQLite.
    """

    def __init__(
        self,
        settings: Settings,
        storage: Optional[Storage] = None,
    ) -> None:
        self._settings = settings
        self._storage = storage

        # ── Binance (CCXT async) ────────────────────────────────────
        exchange_opts: dict = {
            "apiKey": settings.binance.api_key or None,
            "secret": settings.binance.secret_key or None,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if settings.binance.testnet:
            exchange_opts["sandbox"] = True

        self._exchange: ccxt_async.binance = ccxt_async.binance(exchange_opts)
        logger.info(
            "CCXT Binance exchange created (testnet={})",
            settings.binance.testnet,
        )

        # ── Alpaca (sync SDK — wrapped in executor) ─────────────────
        alpaca_key = settings.alpaca.api_key or None
        alpaca_secret = settings.alpaca.secret_key or None
        if alpaca_key and alpaca_secret:
            self._stock_client: Optional[StockHistoricalDataClient] = (
                StockHistoricalDataClient(
                    api_key=alpaca_key,
                    secret_key=alpaca_secret,
                )
            )
            # NOTE: On a network with TLS interception (corporate proxy / AV SSL
            # inspection) requests to data.alpaca.markets may fail with
            # "unable to get local issuer certificate". The secure fix is to let
            # Python trust the OS certificate store, e.g. `pip install truststore`
            # then `import truststore; truststore.inject_into_ssl()` at startup.
            # We do NOT disable TLS verification here.
            logger.info("Alpaca StockHistoricalDataClient initialised.")
        else:
            self._stock_client = None
            logger.warning(
                "Alpaca API keys not set — stock data collection disabled."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully close the CCXT exchange connection."""
        await self._exchange.close()
        logger.info("CCXT exchange connection closed.")

    # ------------------------------------------------------------------
    # Crypto (Binance via CCXT)
    # ------------------------------------------------------------------

    async def fetch_crypto_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for a crypto pair from Binance (handles pagination for >1000)."""
        logger.debug("Fetching {} {} candles for {}", limit, timeframe, symbol)
        import aiohttp
        import asyncio

        binance_symbol = symbol.replace("/", "")
        all_raw = []
        remaining = limit
        end_time = None

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=verified_ssl_context())) as session:
            while remaining > 0:
                fetch_amount = min(remaining, 1000)
                url = (
                    f"https://api.binance.com/api/v3/klines"
                    f"?symbol={binance_symbol}&interval={timeframe}&limit={fetch_amount}"
                )
                if end_time:
                    url += f"&endTime={end_time}"
                
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            logger.error("Binance HTTP {} for {}", resp.status, symbol)
                            break
                        chunk = await resp.json()
                        if not chunk:
                            break
                        
                        all_raw = chunk + all_raw
                        remaining -= len(chunk)
                        
                        # Set end_time to the open time of the earliest candle fetched minus 1 ms
                        end_time = chunk[0][0] - 1
                        
                        # Be polite to the API
                        await asyncio.sleep(0.2)
                except Exception as exc:
                    logger.error("Error fetching {} {}: {}", symbol, timeframe, exc)
                    break

        raw = all_raw[-limit:] if len(all_raw) > limit else all_raw


        if not raw:
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            df = pd.DataFrame(
                raw,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ],
            )
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        logger.info(
            "Fetched {} candles for {} [{}]", len(df), symbol, timeframe
        )

        # Persist if storage is available
        if self._storage is not None:
            await self._storage.save_candles(df, symbol=symbol, timeframe=timeframe)

        return df

    # ------------------------------------------------------------------
    # Stocks (Alpaca)
    # ------------------------------------------------------------------

    async def fetch_stock_bars(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch historical stock bars from Alpaca.

        Parameters
        ----------
        symbol:
            US equity ticker, e.g. ``"AAPL"``.
        timeframe:
            Candle interval string (see ``_ALPACA_TF_MAP``).
        limit:
            Number of bars to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``open``, ``high``, ``low``,
            ``close``, ``volume``.
        """
        if self._stock_client is None:
            logger.error(
                "Stock data collection disabled — Alpaca API keys not configured."
            )
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        tf = _ALPACA_TF_MAP.get(timeframe)
        if tf is None:
            logger.error("Unsupported Alpaca timeframe: {}", timeframe)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        end = dt.datetime.now(dt.timezone.utc)
        # Rough start estimate — slightly over-fetch, then trim
        start = end - dt.timedelta(days=max(limit // 78 + 2, 5))

        # IEX feed works on Alpaca's free tier; the default SIP feed rejects
        # recent data without a paid subscription ("subscription does not permit
        # querying recent SIP data").
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
            feed=DataFeed.IEX,
        )

        logger.debug("Fetching {} {} bars for {}", limit, timeframe, symbol)
        try:
            # alpaca-py is synchronous — run in executor
            loop = asyncio.get_running_loop()
            bars = await loop.run_in_executor(
                None, self._stock_client.get_stock_bars, request
            )
        except Exception as exc:
            logger.error("Alpaca error fetching {} {}: {}", symbol, timeframe, exc)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        records: list[dict] = []
        # BarSet.__contains__ does NOT check symbols (``symbol in bars`` is False
        # even when data exists), so index directly and fall back to ``.data``.
        try:
            bar_set = bars[symbol]
        except (KeyError, TypeError):
            data = getattr(bars, "data", {})
            bar_set = data.get(symbol, []) if isinstance(data, dict) else []
        for bar in bar_set:
            records.append(
                {
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
            )

        df = pd.DataFrame(
            records,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        logger.info("Fetched {} bars for {} [{}]", len(df), symbol, timeframe)

        if self._storage is not None:
            await self._storage.save_candles(df, symbol=symbol, timeframe=timeframe)

        return df

    # ------------------------------------------------------------------
    # Batch fetch
    # ------------------------------------------------------------------

    async def fetch_multiple(
        self,
        symbols: list[str],
        timeframe: str = "5m",
        limit: int = 200,
        *,
        asset_type: str = "crypto",
    ) -> dict[str, pd.DataFrame]:
        """Fetch candles for several symbols concurrently.

        Parameters
        ----------
        symbols:
            List of trading pairs / tickers.
        timeframe:
            Common candle interval.
        limit:
            Candle count per symbol.
        asset_type:
            ``"crypto"`` (default) or ``"stock"``.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of symbol → DataFrame.
        """
        fetch_fn = (
            self.fetch_crypto_ohlcv if asset_type == "crypto"
            else self.fetch_stock_bars
        )

        tasks = [fetch_fn(sym, timeframe, limit) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, pd.DataFrame] = {}
        for sym, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.error("Failed to fetch {}: {}", sym, res)
                output[sym] = pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
            else:
                output[sym] = res

        logger.info(
            "Batch fetch complete — {}/{} symbols succeeded.",
            sum(1 for v in output.values() if not v.empty),
            len(symbols),
        )
        return output
