"""
AI Trade Bot — Sentiment risk gate (Claude-free).

A conservative news/sentiment filter applied *after* the strategy and risk
checks, just before an order is placed. It is a **risk gate, not an alpha
source**: it only blocks trades that fight strongly adverse news/market mood
(e.g. don't BUY into very bearish sentiment), and never initiates trades.

Sentiment is computed with the existing keyword :class:`SentimentAnalyzer` and
the Crypto Fear & Greed index — no LLM/Claude calls. Headlines come from
:class:`NewsFeed` (keyless RSS / Alpaca). Per-symbol results are cached with a
TTL so we don't hit the network every tick.

Two modes (config-driven, opt-in):
  * ``gate=False`` (observation): only logs the score, never blocks.
  * ``gate=True``  (enforcing): vetoes trades against strong adverse sentiment.

Usage
-----
>>> gate = SentimentGate(threshold=0.5, enforce=False)
>>> allow, reason, score = await gate.check(signal)
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.data.news_feed import NewsFeed
from src.ml.sentiment import SentimentAnalyzer
from src.strategy.base import Signal, TradeAction


class SentimentGate:
    """News/sentiment veto applied before order execution.

    Parameters
    ----------
    threshold:
        Absolute score in ``[0, 1]`` beyond which sentiment is "strong". A BUY
        is vetoed when score < ``-threshold``; a SELL when score > ``+threshold``.
    enforce:
        When ``True`` the gate actually blocks trades; when ``False`` it runs in
        observation mode (logs + annotates the signal but always allows).
    cache_ttl:
        Seconds to cache a symbol's sentiment score (default 1800 = 30 min).
    news_feed, analyzer:
        Optional injected collaborators (mainly for tests). Created with sane
        defaults otherwise.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        enforce: bool = False,
        cache_ttl: float = 1800.0,
        news_feed: NewsFeed | None = None,
        analyzer: SentimentAnalyzer | None = None,
    ) -> None:
        self.threshold = threshold
        self.enforce = enforce
        self.cache_ttl = cache_ttl
        self._news = news_feed or NewsFeed()
        self._analyzer = analyzer or SentimentAnalyzer()
        # symbol -> (timestamp, score)
        self._cache: dict[str, tuple[float, float]] = {}
        logger.info(
            "SentimentGate ready — enforce={} threshold={:.2f} ttl={}s",
            enforce, threshold, int(cache_ttl),
        )

    async def score(self, symbol: str) -> float:
        """Return the cached or freshly-computed sentiment score for *symbol*.

        Score is in ``[-1, 1]`` (-1 = very bearish, +1 = very bullish). Any
        failure yields a neutral-ish score from the Fear & Greed fallback, so
        this never raises.
        """
        now = time.monotonic()
        cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < self.cache_ttl:
            return cached[1]

        try:
            headlines = await self._news.fetch_news(symbol)
        except Exception:
            logger.exception("[sentiment] News fetch failed for {}", symbol)
            headlines = []

        try:
            value = await self._analyzer.get_market_sentiment(symbol, headlines)
        except Exception:
            logger.exception("[sentiment] Scoring failed for {}", symbol)
            value = 0.0

        self._cache[symbol] = (now, value)
        return value

    async def check(self, signal: Signal) -> tuple[bool, str, float]:
        """Evaluate *signal* against current sentiment.

        Returns ``(allow, reason, score)``. In observation mode ``allow`` is
        always ``True``. The score is written to ``signal.metadata['sentiment']``
        so the dashboard/logs can surface it.
        """
        value = await self.score(signal.symbol)

        # Annotate the signal (metadata is a plain dict on the frozen Signal).
        try:
            signal.metadata["sentiment"] = round(value, 4)
        except Exception:
            pass

        adverse = (
            (signal.action == TradeAction.BUY and value < -self.threshold)
            or (signal.action == TradeAction.SELL and value > self.threshold)
        )

        if not adverse:
            return True, "sentiment ok", value

        reason = (
            f"adverse sentiment {value:+.2f} vs {signal.action.value} "
            f"(threshold {self.threshold:.2f})"
        )
        if self.enforce:
            logger.info("[sentiment] VETO {} — {}", signal.symbol, reason)
            return False, reason, value

        logger.info(
            "[sentiment] (observe) would veto {} — {}", signal.symbol, reason
        )
        return True, f"observe: {reason}", value
