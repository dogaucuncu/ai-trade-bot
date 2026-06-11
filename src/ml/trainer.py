"""
AI Trade Bot — Model training pipeline.

Orchestrates end-to-end training of LSTM models for each trading symbol.
Uses :class:`DataCollector` to fetch historical candles and
:class:`TechnicalIndicators` to enrich features before feeding them
into :class:`LSTMPredictor`.

Usage::

    from src.ml.trainer import ModelTrainer

    trainer = ModelTrainer(settings)
    await trainer.train_lstm("DOGE/USDT", "5m", lookback_days=90)
    await trainer.retrain_all_models()
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import Settings, settings as default_settings
from src.data.collector import DataCollector
from src.indicators.technical import TechnicalIndicators
from src.ml.lstm_model import (
    DIRECTION_LABELS,
    LSTMPredictor,
    TrainingHistory,
    _add_features,
    _generate_labels,
)


# ── Constants ───────────────────────────────────────────────────────────

_MODELS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "models"

# How many candles to fetch per timeframe-minute when requesting N days.
_CANDLES_PER_DAY: dict[str, int] = {
    "1m": 1440,
    "5m": 288,
    "15m": 96,
    "1h": 24,
    "4h": 6,
    "1d": 1,
}


# =====================================================================
# Training metrics result
# =====================================================================


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Metrics from a single evaluation run."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    classification_report: str
    extra: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Model trainer
# =====================================================================


class ModelTrainer:
    """End-to-end LSTM training pipeline.

    Parameters
    ----------
    settings:
        Application settings (API keys, symbol lists, etc.).
    models_dir:
        Root directory for saved models.  Defaults to ``<project>/models/``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        models_dir: Path | str | None = None,
    ) -> None:
        self._settings = settings or default_settings
        self._models_dir = Path(models_dir) if models_dir else _MODELS_DIR
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._collector: DataCollector | None = None
        logger.info("ModelTrainer ready — models_dir={}", self._models_dir)

    # ------------------------------------------------------------------
    # Data collector lifecycle
    # ------------------------------------------------------------------

    async def _get_collector(self) -> DataCollector:
        """Lazy-initialise the data collector."""
        if self._collector is None:
            self._collector = DataCollector(self._settings)
        return self._collector

    async def close(self) -> None:
        """Close the underlying data collector connection."""
        if self._collector is not None:
            await self._collector.close()
            self._collector = None

    # ------------------------------------------------------------------
    # Train a single symbol
    # ------------------------------------------------------------------

    async def train_lstm(
        self,
        symbol: str,
        timeframe: str = "5m",
        lookback_days: int = 90,
        epochs: int = 50,
        lr: float = 0.001,
        lookback: int = 60,
        batch_size: int = 32,
    ) -> TrainingHistory:
        """Train (or retrain) an LSTM model for a single symbol.

        Parameters
        ----------
        symbol:
            Trading pair / ticker, e.g. ``"DOGE/USDT"`` or ``"AAPL"``.
        timeframe:
            Candle interval (must exist in ``_CANDLES_PER_DAY``).
        lookback_days:
            How many days of historical data to fetch.
        epochs:
            Training epochs.
        lr:
            Learning rate.
        lookback:
            LSTM input sequence length (bars).
        batch_size:
            Mini-batch size.

        Returns
        -------
        TrainingHistory
            Training and validation losses / accuracies per epoch.
        """
        logger.info(
            "Training LSTM for {} on {} — {} days of data",
            symbol, timeframe, lookback_days,
        )

        # ── 1. Fetch data ────────────────────────────────────────────
        df = await self._fetch_historical(symbol, timeframe, lookback_days)
        if df.empty or len(df) < lookback + 20:
            logger.error(
                "Not enough data for {} (got {} rows). Aborting.",
                symbol, len(df),
            )
            return TrainingHistory()

        # ── 2. Add technical indicators ──────────────────────────────
        df = TechnicalIndicators.add_all_indicators(df)
        logger.info(
            "Indicator-enriched DataFrame: {} rows × {} cols",
            len(df), len(df.columns),
        )

        # ── 3. Prepare data & train ──────────────────────────────────
        predictor = LSTMPredictor()
        train_loader, val_loader = predictor.prepare_data(
            df, lookback=lookback, batch_size=batch_size,
        )

        history = predictor.train(
            train_loader, val_loader, epochs=epochs, lr=lr,
        )

        # ── 4. Save model ────────────────────────────────────────────
        model_path = self._model_path(symbol, timeframe)
        predictor.save_model(model_path)
        logger.info("Model saved to {}", model_path)

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictor: LSTMPredictor,
        test_df: pd.DataFrame,
        lookback: int = 60,
    ) -> EvaluationMetrics:
        """Evaluate a trained model on held-out test data.

        Parameters
        ----------
        predictor:
            A trained :class:`LSTMPredictor`.
        test_df:
            DataFrame with OHLCV + indicator columns.
        lookback:
            Sequence length matching the predictor's training.

        Returns
        -------
        EvaluationMetrics
        """
        df = _add_features(test_df)
        labels = _generate_labels(df)

        y_true: list[int] = []
        y_pred: list[int] = []

        # Slide over test set one bar at a time
        for end in range(lookback, len(df) - 1):
            window = df.iloc[end - lookback : end + 1]
            true_label = int(labels.iloc[end])
            if pd.isna(true_label):
                continue

            try:
                direction, _ = predictor.predict(window, lookback=lookback)
            except Exception:
                continue

            pred_label = {"UP": 0, "DOWN": 1, "SIDEWAYS": 2}[direction]
            y_true.append(true_label)
            y_pred.append(pred_label)

        if not y_true:
            logger.warning("No valid predictions during evaluation.")
            return EvaluationMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1=0.0,
                classification_report="No data",
            )

        target_names = [DIRECTION_LABELS[i] for i in sorted(DIRECTION_LABELS)]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0,
        )

        metrics = EvaluationMetrics(
            accuracy=acc, precision=prec, recall=rec, f1=f1,
            classification_report=report,
        )
        logger.info(
            "Evaluation — accuracy={:.2%}, precision={:.2%}, "
            "recall={:.2%}, f1={:.2%}",
            acc, prec, rec, f1,
        )
        logger.debug("Classification report:\n{}", report)
        return metrics

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    async def walk_forward_validation(
        self,
        symbol: str,
        timeframe: str = "5m",
        lookback_days: int = 90,
        n_splits: int = 5,
        lookback: int = 60,
        epochs: int = 30,
    ) -> list[EvaluationMetrics]:
        """Time-series walk-forward cross-validation.

        The data is divided chronologically into *n_splits + 1* folds.
        For each iteration *i*, folds 0..i are used for training and
        fold *i+1* for testing.

        Parameters
        ----------
        symbol:
            Trading pair / ticker.
        timeframe:
            Candle interval.
        lookback_days:
            Total historical data to fetch.
        n_splits:
            Number of train/test splits.
        lookback:
            Sequence length.
        epochs:
            Training epochs per split.

        Returns
        -------
        list[EvaluationMetrics]
            One metrics object per split.
        """
        logger.info(
            "Walk-forward validation for {} — {} splits", symbol, n_splits,
        )

        df = await self._fetch_historical(symbol, timeframe, lookback_days)
        if df.empty:
            logger.error("No data — aborting walk-forward validation.")
            return []

        df = TechnicalIndicators.add_all_indicators(df)

        total_rows = len(df)
        fold_size = total_rows // (n_splits + 1)

        if fold_size < lookback + 20:
            logger.error(
                "Fold size {} too small for lookback={} — reduce n_splits "
                "or increase lookback_days.",
                fold_size, lookback,
            )
            return []

        all_metrics: list[EvaluationMetrics] = []

        for split in range(n_splits):
            train_end = (split + 1) * fold_size
            test_end = min(train_end + fold_size, total_rows)

            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]

            logger.info(
                "Split {}/{} — train 0:{}, test {}:{}",
                split + 1, n_splits, train_end, train_end, test_end,
            )

            predictor = LSTMPredictor()
            try:
                train_loader, val_loader = predictor.prepare_data(
                    train_df, lookback=lookback,
                )
                predictor.train(train_loader, val_loader, epochs=epochs)
            except ValueError as exc:
                logger.warning("Split {} skipped: {}", split + 1, exc)
                continue

            metrics = self.evaluate(predictor, test_df, lookback=lookback)
            all_metrics.append(metrics)
            logger.info(
                "Split {}/{} — accuracy={:.2%}, f1={:.2%}",
                split + 1, n_splits, metrics.accuracy, metrics.f1,
            )

        # Summary
        if all_metrics:
            avg_acc = np.mean([m.accuracy for m in all_metrics])
            avg_f1 = np.mean([m.f1 for m in all_metrics])
            logger.info(
                "Walk-forward complete — avg_accuracy={:.2%}, avg_f1={:.2%}",
                avg_acc, avg_f1,
            )
        else:
            logger.warning("No valid splits completed.")

        return all_metrics

    # ------------------------------------------------------------------
    # Batch retrain
    # ------------------------------------------------------------------

    async def retrain_all_models(
        self,
        timeframe: str = "5m",
        lookback_days: int = 90,
        epochs: int = 50,
    ) -> dict[str, TrainingHistory]:
        """Retrain LSTM models for every configured symbol.

        Iterates over all Binance default pairs and Alpaca default
        symbols defined in the application settings.

        Returns
        -------
        dict[str, TrainingHistory]
            Mapping of symbol → training history.

        Note
        ----
        Suggested schedule: weekly retraining via APScheduler or a
        cron job::

            # crontab example
            0 2 * * 1  cd /path/to/project && python -m src.ml.trainer
        """
        symbols: list[str] = list(self._settings.binance.default_pairs) + list(
            self._settings.alpaca.default_symbols
        )

        logger.info(
            "Retraining all models — {} symbols, tf={}, days={}",
            len(symbols), timeframe, lookback_days,
        )

        results: dict[str, TrainingHistory] = {}
        for symbol in symbols:
            try:
                history = await self.train_lstm(
                    symbol, timeframe,
                    lookback_days=lookback_days,
                    epochs=epochs,
                )
                results[symbol] = history
                logger.info("✓ {} retrained successfully.", symbol)
            except Exception as exc:
                logger.error("✗ {} training failed: {}", symbol, exc)
                results[symbol] = TrainingHistory()

        succeeded = sum(
            1 for h in results.values() if h.train_losses
        )
        logger.info(
            "Retrain complete — {}/{} models succeeded.",
            succeeded, len(symbols),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _model_path(self, symbol: str, timeframe: str) -> Path:
        """Return the directory path for a specific model."""
        safe_name = symbol.replace("/", "_").replace("\\", "_")
        return self._models_dir / f"{safe_name}_{timeframe}"

    async def _fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int,
    ) -> pd.DataFrame:
        """Fetch historical candle data in batches.

        Binance limits OHLCV requests to 1 000 candles.  This method
        issues multiple sequential requests to cover ``lookback_days``.
        """
        collector = await self._get_collector()
        candles_per_day = _CANDLES_PER_DAY.get(timeframe, 288)
        total_candles = candles_per_day * lookback_days

        # Determine asset type from symbol format
        is_crypto = "/" in symbol

        if is_crypto:
            df = await collector.fetch_crypto_ohlcv(
                symbol, timeframe, limit=total_candles,
            )
        else:
            df = await collector.fetch_stock_bars(
                symbol, timeframe, limit=total_candles,
            )

        if df is None or df.empty:
            logger.warning("No historical data fetched for {}", symbol)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(
            "Historical data for {} — {} rows ({} days requested)",
            symbol, len(df), lookback_days,
        )
        return df


# =====================================================================
# CLI entry-point for manual / cron retraining
# =====================================================================

if __name__ == "__main__":
    async def _main() -> None:
        default_settings.configure_logging()
        trainer = ModelTrainer()
        try:
            await trainer.retrain_all_models()
        finally:
            await trainer.close()

    asyncio.run(_main())
