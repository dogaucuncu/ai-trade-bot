"""
AI Trade Bot — Prediction service.

Combines LSTM model predictions with sentiment analysis to produce a
unified directional forecast.  Falls back to technical-indicator-only
analysis when no trained ML model is available.

Usage::

    from src.ml.predictor import PredictionService

    service = PredictionService(settings)
    result = await service.get_prediction("DOGE/USDT", "5m")
    print(result.direction, result.confidence)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import Settings, settings as default_settings
from src.data.collector import DataCollector
from src.indicators.technical import TechnicalIndicators
from src.ml.lstm_model import LSTMPredictor
from src.ml.sentiment import SentimentAnalyzer


# ── Constants ───────────────────────────────────────────────────────────

_MODELS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "models"

# Weighted blend: LSTM 70 %, Sentiment 30 %
_LSTM_WEIGHT: float = 0.70
_SENTIMENT_WEIGHT: float = 0.30

# Cache TTL in seconds (avoid redundant predictions within this window)
_CACHE_TTL: int = 60


# =====================================================================
# Prediction result
# =====================================================================


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Immutable container for a single prediction.

    Attributes
    ----------
    symbol:
        Trading pair / ticker.
    direction:
        ``"UP"``, ``"DOWN"``, or ``"SIDEWAYS"``.
    confidence:
        Combined confidence score in ``[0.0, 1.0]``.
    lstm_confidence:
        Raw LSTM softmax confidence (or 0.0 if unavailable).
    sentiment_score:
        Sentiment component in ``[-1.0, 1.0]``.
    timestamp:
        UTC time of prediction.
    source:
        Description of how the prediction was derived.
    metadata:
        Extra debugging information.
    """

    symbol: str
    direction: str
    confidence: float
    lstm_confidence: float
    sentiment_score: float
    timestamp: datetime
    source: str = "lstm+sentiment"
    metadata: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Prediction service
# =====================================================================


class PredictionService:
    """Unified prediction service merging ML and sentiment signals.

    Parameters
    ----------
    settings:
        Application settings.
    models_dir:
        Root directory containing saved models.
    cache_ttl:
        Seconds to cache a prediction per ``(symbol, timeframe)`` key.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        models_dir: Path | str | None = None,
        cache_ttl: int = _CACHE_TTL,
    ) -> None:
        self._settings = settings or default_settings
        self._models_dir = Path(models_dir) if models_dir else _MODELS_DIR
        self._cache_ttl = cache_ttl

        # Lazy-loaded components
        self._collector: DataCollector | None = None
        self._sentiment: SentimentAnalyzer = SentimentAnalyzer()

        # Model cache: (symbol, timeframe) → LSTMPredictor
        self._predictors: dict[tuple[str, str], LSTMPredictor] = {}

        # Prediction cache: (symbol, timeframe) → (PredictionResult, ts)
        self._cache: dict[tuple[str, str], tuple[PredictionResult, float]] = {}

        logger.info(
            "PredictionService initialised (models_dir={}, cache_ttl={}s)",
            self._models_dir, self._cache_ttl,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release resources."""
        if self._collector is not None:
            await self._collector.close()
            self._collector = None

    async def _get_collector(self) -> DataCollector:
        if self._collector is None:
            self._collector = DataCollector(self._settings)
        return self._collector

    # ------------------------------------------------------------------
    # Main prediction API
    # ------------------------------------------------------------------

    async def get_prediction(
        self,
        symbol: str,
        timeframe: str = "5m",
        headlines: list[str] | None = None,
    ) -> PredictionResult:
        """Generate a combined prediction for *symbol*.

        1. Try the trained LSTM model.
        2. Fetch sentiment score.
        3. Blend them: ``0.70 * lstm + 0.30 * sentiment``.
        4. Fall back to technical-indicator heuristics if LSTM unavailable.

        Parameters
        ----------
        symbol:
            Trading pair / ticker.
        timeframe:
            Candle interval.
        headlines:
            Optional recent news headlines for sentiment analysis.

        Returns
        -------
        PredictionResult
        """
        cache_key = (symbol, timeframe)

        # ── Check cache ──────────────────────────────────────────────
        cached = self._cache.get(cache_key)
        if cached is not None:
            result, ts = cached
            if time.time() - ts < self._cache_ttl:
                logger.debug(
                    "Cache hit for {}:{} (age={:.0f}s)",
                    symbol, timeframe, time.time() - ts,
                )
                return result

        # ── Fetch recent data ────────────────────────────────────────
        collector = await self._get_collector()
        is_crypto = "/" in symbol

        if is_crypto:
            df = await collector.fetch_crypto_ohlcv(symbol, timeframe, limit=200)
        else:
            df = await collector.fetch_stock_bars(symbol, timeframe, limit=200)

        if df.empty or len(df) < 30:
            logger.warning(
                "Insufficient data for {} — returning neutral prediction.",
                symbol,
            )
            result = PredictionResult(
                symbol=symbol,
                direction="SIDEWAYS",
                confidence=0.0,
                lstm_confidence=0.0,
                sentiment_score=0.0,
                timestamp=datetime.now(timezone.utc),
                source="insufficient_data",
            )
            self._cache[cache_key] = (result, time.time())
            return result

        # Add indicators
        df = TechnicalIndicators.add_all_indicators(df)

        # ── LSTM prediction ──────────────────────────────────────────
        lstm_direction: str | None = None
        lstm_confidence: float = 0.0
        lstm_available = False

        predictor = self._load_predictor(symbol, timeframe)
        if predictor is not None:
            try:
                lstm_direction, lstm_confidence = predictor.predict(df)
                lstm_available = True
                logger.info(
                    "[{}] LSTM → {} ({:.2%})",
                    symbol, lstm_direction, lstm_confidence,
                )
            except Exception as exc:
                logger.warning("[{}] LSTM prediction failed: {}", symbol, exc)

        # ── Sentiment ────────────────────────────────────────────────
        try:
            sentiment_score = await self._sentiment.get_market_sentiment(
                symbol, headlines=headlines,
            )
        except Exception as exc:
            logger.warning("[{}] Sentiment fetch failed: {}", symbol, exc)
            sentiment_score = 0.0

        # ── Combine or fallback ──────────────────────────────────────
        if lstm_available and lstm_direction is not None:
            result = self._blend_prediction(
                symbol, lstm_direction, lstm_confidence, sentiment_score,
            )
        else:
            # Fallback: technical-indicator heuristic
            result = self._technical_fallback(symbol, df, sentiment_score)

        # Cache result
        self._cache[cache_key] = (result, time.time())
        logger.info(
            "[{}] Prediction → {} (confidence={:.2%}, source={})",
            symbol, result.direction, result.confidence, result.source,
        )
        return result

    # ------------------------------------------------------------------
    # Blending
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_prediction(
        symbol: str,
        lstm_direction: str,
        lstm_confidence: float,
        sentiment_score: float,
    ) -> PredictionResult:
        """Combine LSTM output with sentiment into a final prediction.

        The LSTM provides a class label + softmax confidence.
        Sentiment is in [-1, 1].

        The combined confidence is::

            conf = LSTM_WEIGHT * lstm_confidence
                 + SENTIMENT_WEIGHT * |sentiment| * agreement_factor

        where *agreement_factor* is 1.0 if sentiment agrees with the
        LSTM direction and 0.5 otherwise (penalty for conflict).
        """
        # Convert sentiment to implied direction
        if sentiment_score > 0.15:
            sent_dir = "UP"
        elif sentiment_score < -0.15:
            sent_dir = "DOWN"
        else:
            sent_dir = "SIDEWAYS"

        # Agreement multiplier
        agrees = (lstm_direction == sent_dir) or sent_dir == "SIDEWAYS"
        agreement = 1.0 if agrees else 0.5

        combined_conf = (
            _LSTM_WEIGHT * lstm_confidence
            + _SENTIMENT_WEIGHT * abs(sentiment_score) * agreement
        )
        combined_conf = max(0.0, min(1.0, combined_conf))

        # If strong disagreement, reduce confidence further
        if not agrees and abs(sentiment_score) > 0.5:
            # Sentiment strongly opposes LSTM — downgrade to SIDEWAYS
            # if LSTM confidence is mediocre
            if lstm_confidence < 0.6:
                final_dir = "SIDEWAYS"
                combined_conf *= 0.6
            else:
                final_dir = lstm_direction
        else:
            final_dir = lstm_direction

        return PredictionResult(
            symbol=symbol,
            direction=final_dir,
            confidence=combined_conf,
            lstm_confidence=lstm_confidence,
            sentiment_score=sentiment_score,
            timestamp=datetime.now(timezone.utc),
            source="lstm+sentiment",
            metadata={
                "lstm_direction": lstm_direction,
                "sentiment_direction": sent_dir,
                "agreement": agrees,
            },
        )

    # ------------------------------------------------------------------
    # Technical fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _technical_fallback(
        symbol: str,
        df: pd.DataFrame,
        sentiment_score: float,
    ) -> PredictionResult:
        """Derive a direction purely from technical indicators.

        A simple voting system counts bullish / bearish signals from
        RSI, MACD, Bollinger Band position, and EMA crossover.
        """
        score = 0.0
        votes = 0

        # RSI
        if "rsi_14" in df.columns:
            rsi = df["rsi_14"].iloc[-1]
            if not pd.isna(rsi):
                votes += 1
                if rsi < 30:
                    score += 1.0  # oversold → bullish
                elif rsi > 70:
                    score -= 1.0  # overbought → bearish

        # MACD histogram
        if "macd_histogram" in df.columns:
            hist = df["macd_histogram"].iloc[-1]
            if not pd.isna(hist):
                votes += 1
                if hist > 0:
                    score += 0.7
                else:
                    score -= 0.7

        # Bollinger %B
        if "bb_pband" in df.columns:
            pband = df["bb_pband"].iloc[-1]
            if not pd.isna(pband):
                votes += 1
                if pband < 0.2:
                    score += 0.8  # near lower band
                elif pband > 0.8:
                    score -= 0.8  # near upper band

        # EMA crossover (9 vs 21)
        if "ema_9" in df.columns and "ema_21" in df.columns:
            ema9 = df["ema_9"].iloc[-1]
            ema21 = df["ema_21"].iloc[-1]
            if not pd.isna(ema9) and not pd.isna(ema21):
                votes += 1
                if ema9 > ema21:
                    score += 0.6
                else:
                    score -= 0.6

        # Factor in sentiment
        score += sentiment_score * 0.5
        votes += 1

        if votes == 0:
            direction = "SIDEWAYS"
            confidence = 0.0
        else:
            avg = score / votes
            if avg > 0.2:
                direction = "UP"
            elif avg < -0.2:
                direction = "DOWN"
            else:
                direction = "SIDEWAYS"
            confidence = min(abs(avg), 1.0) * 0.6  # cap at 60 % for heuristic

        return PredictionResult(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            lstm_confidence=0.0,
            sentiment_score=sentiment_score,
            timestamp=datetime.now(timezone.utc),
            source="technical_fallback",
            metadata={"indicator_score": score, "votes": votes},
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_predictor(
        self, symbol: str, timeframe: str
    ) -> LSTMPredictor | None:
        """Load a trained LSTM predictor from disk (cached)."""
        key = (symbol, timeframe)
        if key in self._predictors:
            return self._predictors[key]

        safe_name = symbol.replace("/", "_").replace("\\", "_")
        model_dir = self._models_dir / f"{safe_name}_{timeframe}"

        if not model_dir.exists():
            logger.debug(
                "No trained model found at {} — will use fallback.",
                model_dir,
            )
            return None

        try:
            predictor = LSTMPredictor()
            predictor.load_model(model_dir)
            self._predictors[key] = predictor
            logger.info("Loaded LSTM model for {}:{}", symbol, timeframe)
            return predictor
        except Exception as exc:
            logger.error(
                "Failed to load model for {}:{} — {}", symbol, timeframe, exc
            )
            return None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Flush the prediction cache."""
        self._cache.clear()
        logger.debug("Prediction cache cleared.")

    def invalidate(self, symbol: str, timeframe: str) -> None:
        """Remove a specific entry from the cache."""
        key = (symbol, timeframe)
        self._cache.pop(key, None)
        logger.debug("Cache invalidated for {}:{}", symbol, timeframe)
