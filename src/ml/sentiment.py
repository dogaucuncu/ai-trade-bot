"""
AI Trade Bot — Market sentiment analysis.

Provides a lightweight keyword-based sentiment analyser suitable for
micro-capital accounts where heavy transformer models are impractical.
Also integrates the *Crypto Fear & Greed Index* from alternative.me.

A ``USE_TRANSFORMER`` flag is stubbed for future FinBERT upgrade.

Usage::

    from src.ml.sentiment import SentimentAnalyzer

    sa = SentimentAnalyzer()
    score, label = sa.analyze_text("Bitcoin rallies to new highs!")
    fgi = await sa.fetch_crypto_fear_greed_index()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import aiohttp
from loguru import logger

# Future flag — set to True when hardware allows FinBERT inference.
USE_TRANSFORMER: bool = False


# ── Keyword dictionaries ───────────────────────────────────────────────

_POSITIVE_KEYWORDS: set[str] = {
    "bullish", "surge", "rally", "breakout", "pump", "gain", "buy",
    "moon", "all-time-high", "ath", "upgrade", "partnership",
    "adoption", "accumulate", "soar", "rocket", "profit", "boom",
    "recovery", "outperform", "growth", "milestone", "optimism",
    "uptrend", "long",
}

_NEGATIVE_KEYWORDS: set[str] = {
    "bearish", "crash", "dump", "sell", "plunge", "hack", "ban",
    "regulation", "lawsuit", "bankrupt", "fear", "panic", "scam",
    "fraud", "delist", "downgrade", "liquidation", "rug", "exploit",
    "collapse", "decline", "loss", "fud", "short", "recession",
    "warning", "investigation", "default",
}


# =====================================================================
# Sentiment result
# =====================================================================


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Immutable result from a sentiment analysis call."""

    score: float  # -1.0 (bearish) to 1.0 (bullish)
    label: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]


# =====================================================================
# Main analyser
# =====================================================================


class SentimentAnalyzer:
    """Keyword-based market sentiment analyser.

    Resource-friendly alternative to transformer models — suitable for
    a $50 micro-capital trading setup.

    Parameters
    ----------
    positive_keywords:
        Extra positive keywords to merge in.
    negative_keywords:
        Extra negative keywords to merge in.
    """

    def __init__(
        self,
        positive_keywords: set[str] | None = None,
        negative_keywords: set[str] | None = None,
    ) -> None:
        self._positive = _POSITIVE_KEYWORDS | (positive_keywords or set())
        self._negative = _NEGATIVE_KEYWORDS | (negative_keywords or set())
        logger.info(
            "SentimentAnalyzer ready — {} pos / {} neg keywords",
            len(self._positive),
            len(self._negative),
        )

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze_text(self, text: str) -> tuple[float, str]:
        """Analyse a single piece of text for market sentiment.

        Parameters
        ----------
        text:
            News headline, tweet, or short paragraph.

        Returns
        -------
        tuple[float, str]
            ``(sentiment_score, label)``  where *sentiment_score* ∈ [-1, 1]
            and *label* is ``"POSITIVE"`` / ``"NEGATIVE"`` / ``"NEUTRAL"``.
        """
        if USE_TRANSFORMER:
            return self._analyze_transformer(text)

        words = set(re.findall(r"[a-z\-]+", text.lower()))

        pos_hits = words & self._positive
        neg_hits = words & self._negative

        pos_count = len(pos_hits)
        neg_count = len(neg_hits)
        total = pos_count + neg_count

        if total == 0:
            return 0.0, "NEUTRAL"

        # Score: normalised difference — ranges in [-1, 1]
        raw = (pos_count - neg_count) / total
        score = max(-1.0, min(1.0, raw))

        if score > 0.1:
            label = "POSITIVE"
        elif score < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        logger.debug(
            "Sentiment: {:.2f} ({}) — pos={}, neg={} | '{}'",
            score, label, pos_hits or "∅", neg_hits or "∅",
            text[:80],
        )
        return score, label

    def analyze_headlines(
        self, headlines: list[str]
    ) -> tuple[float, str]:
        """Aggregate sentiment across multiple headlines.

        Parameters
        ----------
        headlines:
            List of headline strings.

        Returns
        -------
        tuple[float, str]
            ``(average_score, overall_label)``
        """
        if not headlines:
            logger.warning("No headlines provided — returning NEUTRAL.")
            return 0.0, "NEUTRAL"

        scores: list[float] = []
        for headline in headlines:
            score, _ = self.analyze_text(headline)
            scores.append(score)

        avg = sum(scores) / len(scores)
        avg = max(-1.0, min(1.0, avg))

        if avg > 0.1:
            label = "POSITIVE"
        elif avg < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        logger.info(
            "Aggregate sentiment over {} headlines: {:.3f} ({})",
            len(headlines), avg, label,
        )
        return avg, label

    # ------------------------------------------------------------------
    # Crypto Fear & Greed Index
    # ------------------------------------------------------------------

    async def fetch_crypto_fear_greed_index(self) -> int:
        """Fetch the latest Crypto Fear & Greed Index (0-100).

        Source: https://alternative.me/crypto/fear-and-greed-index/

        Returns
        -------
        int
            Index value.  0 = Extreme Fear, 100 = Extreme Greed.
            Returns 50 (neutral) on failure.
        """
        url = "https://api.alternative.me/fng/?limit=1&format=json"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Fear & Greed API returned HTTP {}",
                            resp.status,
                        )
                        return 50
                    data = await resp.json(content_type=None)

            value = int(data["data"][0]["value"])
            classification = data["data"][0].get("value_classification", "Unknown")
            logger.info(
                "Fear & Greed Index: {} ({})", value, classification
            )
            return value

        except Exception as exc:
            logger.error("Failed to fetch Fear & Greed Index: {}", exc)
            return 50

    # ------------------------------------------------------------------
    # Combined sentiment
    # ------------------------------------------------------------------

    async def get_market_sentiment(
        self,
        symbol: str,
        headlines: list[str] | None = None,
    ) -> float:
        """Compute a combined market sentiment score for *symbol*.

        Merges keyword-based headline sentiment (if provided) with the
        Crypto Fear & Greed Index.

        Parameters
        ----------
        symbol:
            Trading pair or ticker (used for logging).
        headlines:
            Optional list of recent news headlines for the symbol.

        Returns
        -------
        float
            Combined sentiment in ``[-1.0, 1.0]``.
        """
        # Headline sentiment
        if headlines:
            headline_score, _ = self.analyze_headlines(headlines)
        else:
            headline_score = 0.0

        # Fear & Greed (normalise 0-100 → -1..+1)
        fgi = await self.fetch_crypto_fear_greed_index()
        fgi_score = (fgi - 50) / 50.0  # -1.0 (extreme fear) to +1.0 (greed)

        # Weighted blend — headlines 60 %, FGI 40 %
        if headlines:
            combined = headline_score * 0.6 + fgi_score * 0.4
        else:
            # No headlines — rely entirely on FGI
            combined = fgi_score

        combined = max(-1.0, min(1.0, combined))

        logger.info(
            "[{}] Market sentiment — headline={:.2f}, fgi={} ({:.2f}), "
            "combined={:.2f}",
            symbol, headline_score, fgi, fgi_score, combined,
        )
        return combined

    # ------------------------------------------------------------------
    # Future transformer stub
    # ------------------------------------------------------------------

    @staticmethod
    def _analyze_transformer(text: str) -> tuple[float, str]:
        """Placeholder for HuggingFace FinBERT sentiment analysis.

        Enable by setting ``USE_TRANSFORMER = True`` at module level
        and installing ``transformers`` + ``torch``.  Currently raises
        ``NotImplementedError``.
        """
        # When ready, load with:
        #   from transformers import pipeline
        #   classifier = pipeline("sentiment-analysis",
        #                         model="ProsusAI/finbert")
        #   result = classifier(text)[0]
        #   ...
        raise NotImplementedError(
            "FinBERT transformer backend not yet implemented. "
            "Set USE_TRANSFORMER = False to use keyword-based analysis."
        )
