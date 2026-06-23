"""
AI Trade Bot — Lightweight news feed (Claude-free).

Fetches recent news *headlines* for a symbol so the sentiment layer can turn
them into a -1..+1 score. Designed to be cheap and dependency-light:

  * Crypto — keyless public RSS feeds (CoinDesk, Cointelegraph), parsed with
    the standard-library XML parser. Optionally CryptoPanic if a free token is
    configured (``CRYPTOPANIC_TOKEN``).
  * Stocks — Alpaca News API, reusing the existing Alpaca credentials.

Everything degrades gracefully: any network/parse error returns an empty list,
so the caller simply falls back to the Fear & Greed index (existing behaviour).

No LLM / Claude calls are made here — this is plain HTTP + keyword scoring
downstream.

Usage
-----
>>> from src.data.news_feed import NewsFeed
>>> feed = NewsFeed()
>>> headlines = await feed.fetch_crypto_news("DOGE/USDT")
>>> headlines = await feed.fetch_stock_news("AAPL")
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import aiohttp
from loguru import logger

from src.net import verified_ssl_context

# Keyless, public RSS feeds. Title-only parsing keeps it robust across feeds.
_CRYPTO_RSS_FEEDS: tuple[str, ...] = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
)

# Map the quote-stripped base asset to extra keywords for headline filtering.
_COIN_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "eth", "ether"),
    "SOL": ("solana", "sol"),
    "AVAX": ("avalanche", "avax"),
    "XRP": ("ripple", "xrp"),
    "ADA": ("cardano", "ada"),
    "DOGE": ("dogecoin", "doge"),
}

_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=10)


class NewsFeed:
    """Async news headline fetcher for crypto and stocks.

    Parameters
    ----------
    cryptopanic_token:
        Optional CryptoPanic API token (free tier). When set, crypto news is
        sourced from CryptoPanic (per-currency filtering) instead of RSS.
    alpaca_key, alpaca_secret:
        Optional Alpaca credentials for the stock news API. When omitted, stock
        news returns an empty list.
    """

    def __init__(
        self,
        cryptopanic_token: str = "",
        alpaca_key: str = "",
        alpaca_secret: str = "",
    ) -> None:
        self._cryptopanic_token = cryptopanic_token or ""
        self._alpaca_key = alpaca_key or ""
        self._alpaca_secret = alpaca_secret or ""

    # ------------------------------------------------------------------ crypto

    async def fetch_crypto_news(
        self, symbol: str, limit: int = 20
    ) -> list[str]:
        """Return recent crypto headlines relevant to *symbol*.

        ``symbol`` may be a CCXT pair (``"DOGE/USDT"``) or a bare base asset.
        Returns at most *limit* headlines; empty list on any failure.
        """
        base = symbol.split("/")[0].upper()

        if self._cryptopanic_token:
            headlines = await self._fetch_cryptopanic(base, limit)
            if headlines:
                return headlines
            # fall through to RSS if CryptoPanic returned nothing

        return await self._fetch_rss(base, limit)

    async def _fetch_cryptopanic(self, base: str, limit: int) -> list[str]:
        """Fetch from CryptoPanic (free tier) filtered by currency."""
        url = (
            "https://cryptopanic.com/api/v1/posts/"
            f"?auth_token={self._cryptopanic_token}&currencies={base}&public=true"
        )
        try:
            async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning("[news] CryptoPanic HTTP {}", resp.status)
                        return []
                    data = await resp.json(content_type=None)
            results = data.get("results", []) if isinstance(data, dict) else []
            return [str(r.get("title", "")).strip() for r in results[:limit] if r.get("title")]
        except Exception as exc:
            logger.warning("[news] CryptoPanic fetch failed: {}", exc)
            return []

    async def _fetch_rss(self, base: str, limit: int) -> list[str]:
        """Fetch and merge keyless RSS feeds, filtering titles by coin name."""
        aliases = _COIN_ALIASES.get(base, (base.lower(),))
        titles: list[str] = []
        try:
            async with aiohttp.ClientSession(
                timeout=_HTTP_TIMEOUT,
                connector=aiohttp.TCPConnector(ssl=verified_ssl_context()),
            ) as session:
                for feed_url in _CRYPTO_RSS_FEEDS:
                    titles.extend(await self._fetch_one_rss(session, feed_url))
        except Exception as exc:
            logger.warning("[news] RSS fetch failed: {}", exc)
            return []

        # Filter to headlines mentioning the coin; if none match, fall back to
        # the broad market headlines (still useful as a general-sentiment proxy).
        matched = [t for t in titles if any(a in t.lower() for a in aliases)]
        selected = matched or titles
        return selected[:limit]

    @staticmethod
    async def _fetch_one_rss(
        session: aiohttp.ClientSession, url: str
    ) -> list[str]:
        """Fetch a single RSS feed and return its item titles."""
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.debug("[news] RSS HTTP {} for {}", resp.status, url)
                    return []
                text = await resp.text()
        except Exception as exc:
            logger.debug("[news] RSS error for {}: {}", url, exc)
            return []

        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            logger.debug("[news] RSS parse error for {}: {}", url, exc)
            return []

        # RSS 2.0: channel/item/title ; be namespace-tolerant via tag suffix.
        titles: list[str] = []
        for elem in root.iter():
            if elem.tag.split("}")[-1] == "title" and elem.text:
                titles.append(elem.text.strip())
        # First title is usually the channel name — drop it if there are items.
        return titles[1:] if len(titles) > 1 else titles

    # ------------------------------------------------------------------ stocks

    async def fetch_stock_news(
        self, symbol: str, limit: int = 20
    ) -> list[str]:
        """Return recent stock headlines for *symbol* via the Alpaca News API.

        Returns an empty list when Alpaca credentials are not configured or on
        any error.
        """
        if not (self._alpaca_key and self._alpaca_secret):
            return []

        try:
            import asyncio

            from alpaca.data.historical.news import NewsClient
            from alpaca.data.requests import NewsRequest
        except Exception as exc:  # ImportError or SDK layout change
            logger.debug("[news] Alpaca news SDK unavailable: {}", exc)
            return []

        try:
            client = NewsClient(
                api_key=self._alpaca_key,
                secret_key=self._alpaca_secret,
            )
            request = NewsRequest(symbols=symbol, limit=limit)
            loop = asyncio.get_running_loop()
            news = await loop.run_in_executor(None, client.get_news, request)
        except Exception as exc:
            logger.warning("[news] Alpaca news fetch failed for {}: {}", symbol, exc)
            return []

        return self._extract_alpaca_headlines(news, limit)

    @staticmethod
    def _extract_alpaca_headlines(news: Any, limit: int) -> list[str]:
        """Pull headline strings out of an Alpaca news response object."""
        items = getattr(news, "data", None)
        if isinstance(items, dict):
            # NewsSet.data maps symbol -> list[News]; flatten the values.
            flat: list[Any] = []
            for v in items.values():
                flat.extend(v)
            items = flat
        elif items is None:
            items = news if isinstance(news, list) else []

        headlines: list[str] = []
        for art in items[:limit]:
            headline = getattr(art, "headline", None)
            if headline:
                headlines.append(str(headline).strip())
        return headlines

    # --------------------------------------------------------------- dispatch

    async def fetch_news(self, symbol: str, limit: int = 20) -> list[str]:
        """Fetch news for *symbol*, routing crypto vs stock by symbol format."""
        if "/" in symbol:
            return await self.fetch_crypto_news(symbol, limit)
        return await self.fetch_stock_news(symbol, limit)
