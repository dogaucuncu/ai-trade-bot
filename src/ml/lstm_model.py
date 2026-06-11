"""
AI Trade Bot — LSTM model for price direction prediction.

A two-layer LSTM network classifies the next-bar direction as
**UP**, **DOWN**, or **SIDEWAYS** based on normalised OHLCV data
enriched with standard technical indicators.

Usage::

    from src.ml.lstm_model import LSTMPredictor

    predictor = LSTMPredictor()
    train_ds, val_ds = predictor.prepare_data(df, lookback=60)
    history = predictor.train(train_ds, epochs=50)
    direction, confidence = predictor.predict(recent_df)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


# ── Direction labels ────────────────────────────────────────────────────
DIRECTION_LABELS: dict[int, str] = {0: "UP", 1: "DOWN", 2: "SIDEWAYS"}
LABEL_TO_IDX: dict[str, int] = {v: k for k, v in DIRECTION_LABELS.items()}

# Thresholds for labelling (0.3 % moves — to overcome 0.2% round-trip fees)
UP_THRESHOLD: float = 1.003
DOWN_THRESHOLD: float = 0.997


# =====================================================================
# PyTorch module
# =====================================================================


class PricePredictorLSTM(nn.Module):
    """Two-layer LSTM with a fully-connected classifier head.

    Parameters
    ----------
    input_size:
        Number of input features per time-step.
    hidden_size:
        LSTM hidden state dimension (default 128).
    num_layers:
        Stacked LSTM layers (default 2).
    dropout:
        Dropout probability between LSTM layers (default 0.2).
    num_classes:
        Output classes — UP / DOWN / SIDEWAYS (default 3).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, seq_len, features)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.
        """
        # h0, c0 default to zeros
        lstm_out, _ = self.lstm(x)
        # Use the output of the last time-step
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


# =====================================================================
# Feature engineering helpers
# =====================================================================

_FEATURE_COLUMNS: list[str] = [
    "open", "high", "low", "close", "volume",
    "rsi_14",
    "macd_line", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower", "bb_bandwidth",
    "ema_9", "ema_21",
    "atr_14",
    "volume_ratio",
]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight technical features needed by the LSTM.

    The caller may already have indicator columns from
    :class:`TechnicalIndicators`; this function fills any gaps.
    """
    df = df.copy()

    # -- RSI ---------------------------------------------------------------
    if "rsi_14" not in df.columns:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # -- MACD --------------------------------------------------------------
    if "macd_line" not in df.columns:
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    # -- Bollinger Bands ---------------------------------------------------
    if "bb_upper" not in df.columns:
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_middle"] = sma20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / sma20.replace(
            0, np.nan
        )

    # -- EMAs --------------------------------------------------------------
    if "ema_9" not in df.columns:
        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    if "ema_21" not in df.columns:
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

    # -- ATR ---------------------------------------------------------------
    if "atr_14" not in df.columns:
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.ewm(span=14, adjust=False).mean()

    # -- Volume ratio ------------------------------------------------------
    if "volume_ratio" not in df.columns:
        vol_sma = df["volume"].rolling(20).mean().replace(0, np.nan)
        df["volume_ratio"] = df["volume"] / vol_sma

    return df


def _generate_labels(df: pd.DataFrame) -> pd.Series:
    """Create direction labels based on next-bar close price.

    Returns
    -------
    pd.Series
        Integer labels — 0 = UP, 1 = DOWN, 2 = SIDEWAYS.
    """
    next_close = df["close"].shift(-1)
    current_close = df["close"]

    labels = pd.Series(LABEL_TO_IDX["SIDEWAYS"], index=df.index, dtype=np.int64)
    labels[next_close > current_close * UP_THRESHOLD] = LABEL_TO_IDX["UP"]
    labels[next_close < current_close * DOWN_THRESHOLD] = LABEL_TO_IDX["DOWN"]
    return labels


# =====================================================================
# High-level predictor wrapper
# =====================================================================


@dataclass
class TrainingHistory:
    """Container for training metrics per epoch."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)


class LSTMPredictor:
    """High-level wrapper around :class:`PricePredictorLSTM`.

    Handles feature engineering, data preparation, training loop,
    prediction, and model persistence.

    Parameters
    ----------
    hidden_size:
        LSTM hidden units.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability.
    device:
        ``"cuda"`` or ``"cpu"``; auto-detected if ``None``.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model: PricePredictorLSTM | None = None
        self._scaler: MinMaxScaler = MinMaxScaler()
        self._feature_columns: list[str] = []
        self._is_fitted: bool = False

        logger.info(
            "LSTMPredictor initialised (hidden={}, layers={}, device={})",
            hidden_size,
            num_layers,
            self.device,
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        lookback: int = 60,
        val_split: float = 0.2,
        batch_size: int = 32,
    ) -> tuple[DataLoader, DataLoader]:
        """Transform a raw OHLCV DataFrame into training & validation loaders.

        Parameters
        ----------
        df:
            DataFrame with ``open, high, low, close, volume`` columns.
        lookback:
            Number of past bars per input sequence.
        val_split:
            Fraction of data reserved for validation (chronological).
        batch_size:
            Mini-batch size.

        Returns
        -------
        tuple[DataLoader, DataLoader]
            ``(train_loader, val_loader)``
        """
        logger.info(
            "Preparing data — {} rows, lookback={}, val_split={}",
            len(df), lookback, val_split,
        )

        # Feature engineering
        df = _add_features(df)
        labels = _generate_labels(df)

        # Determine available features
        available = [c for c in _FEATURE_COLUMNS if c in df.columns]
        self._feature_columns = available
        logger.debug("Using {} features: {}", len(available), available)

        # Drop rows where features or label are NaN
        feature_df = df[available].copy()
        valid_mask = feature_df.notna().all(axis=1) & labels.notna()
        feature_df = feature_df.loc[valid_mask].reset_index(drop=True)
        labels = labels.loc[valid_mask].reset_index(drop=True)

        if len(feature_df) < lookback + 10:
            raise ValueError(
                f"Not enough valid rows ({len(feature_df)}) for lookback={lookback}. "
                f"Need at least {lookback + 10}."
            )

        # Fit scaler on all data (no look-ahead bias in scaling is a
        # simplification — acceptable for this scale of project)
        scaled = self._scaler.fit_transform(feature_df.values)
        self._is_fitted = True

        # Build sequences
        X, y = self._build_sequences(scaled, labels.values, lookback)

        # Chronological split
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(
            "Data split — train: {} sequences, val: {} sequences",
            len(X_train), len(X_val),
        )

        train_ds = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), torch.LongTensor(y_val)
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=False  # keep time-order
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Initialise model now that we know input_size
        self._model = PricePredictorLSTM(
            input_size=len(available),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        logger.debug("Model created with input_size={}", len(available))

        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> TrainingHistory:
        """Run the training loop.

        Parameters
        ----------
        train_loader:
            Training data loader.
        val_loader:
            Validation data loader (optional).
        epochs:
            Number of epochs.
        lr:
            Learning rate for Adam optimiser.

        Returns
        -------
        TrainingHistory
            Per-epoch losses and validation accuracy.
        """
        if self._model is None:
            raise RuntimeError("Call prepare_data() before train().")

        self._model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        history = TrainingHistory()

        logger.info("Starting training — {} epochs, lr={}", epochs, lr)

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batches += 1

            avg_train_loss = running_loss / max(batches, 1)
            history.train_losses.append(avg_train_loss)

            # Validation
            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_loader(
                    val_loader, criterion
                )
                history.val_losses.append(val_loss)
                history.val_accuracies.append(val_acc)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch {}/{} — train_loss: {:.4f}, val_loss: {:.4f}, "
                    "val_acc: {:.2%}",
                    epoch, epochs, avg_train_loss, val_loss, val_acc,
                )

        logger.info(
            "Training complete — final val_acc: {:.2%}",
            history.val_accuracies[-1] if history.val_accuracies else 0.0,
        )
        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, recent_data: pd.DataFrame, lookback: int = 60
    ) -> tuple[str, float]:
        """Predict direction from the most recent bars.

        Parameters
        ----------
        recent_data:
            DataFrame with at least ``lookback`` rows and OHLCV columns.
        lookback:
            Sequence length the model expects.

        Returns
        -------
        tuple[str, float]
            ``(direction, confidence)`` where direction is one of
            ``"UP"``, ``"DOWN"``, ``"SIDEWAYS"`` and confidence ∈ [0, 1].
        """
        if self._model is None or not self._is_fitted:
            raise RuntimeError("Model not trained — call train() first.")

        self._model.eval()

        # Feature engineering
        df = _add_features(recent_data)
        available = [c for c in self._feature_columns if c in df.columns]
        feature_df = df[available].dropna()

        if len(feature_df) < lookback:
            logger.warning(
                "Not enough rows for prediction ({} < {}), returning SIDEWAYS",
                len(feature_df), lookback,
            )
            return "SIDEWAYS", 0.0

        # Take the last `lookback` rows
        tail = feature_df.iloc[-lookback:]
        scaled = self._scaler.transform(tail.values)

        X = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        direction = DIRECTION_LABELS[predicted_idx]

        logger.info(
            "Prediction: {} (confidence={:.2%}) — probs={}",
            direction,
            confidence,
            {DIRECTION_LABELS[i]: f"{p:.3f}" for i, p in enumerate(probs)},
        )
        return direction, confidence

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str | Path) -> None:
        """Save the trained model, scaler, and metadata to disk.

        Parameters
        ----------
        path:
            Directory (or file prefix) where artifacts are stored.
        """
        if self._model is None:
            raise RuntimeError("No model to save.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Model weights
        model_path = path / "lstm_model.pt"
        torch.save(self._model.state_dict(), model_path)

        # Scaler
        scaler_path = path / "scaler.npy"
        np.savez(
            scaler_path,
            scale=self._scaler.scale_,
            min=self._scaler.min_,
            data_min=self._scaler.data_min_,
            data_max=self._scaler.data_max_,
            data_range=self._scaler.data_range_,
        )

        # Metadata
        meta: dict[str, Any] = {
            "feature_columns": self._feature_columns,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "input_size": len(self._feature_columns),
        }
        meta_path = path / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info("Model saved to {}", path)

    def load_model(self, path: str | Path) -> None:
        """Load a previously saved model from disk.

        Parameters
        ----------
        path:
            Directory containing ``lstm_model.pt``, ``scaler.npy.npz``,
            and ``metadata.json``.
        """
        path = Path(path)

        # Metadata
        meta_path = path / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._feature_columns = meta["feature_columns"]
        self.hidden_size = meta["hidden_size"]
        self.num_layers = meta["num_layers"]
        self.dropout = meta["dropout"]

        # Rebuild model
        input_size = meta["input_size"]
        self._model = PricePredictorLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        model_path = path / "lstm_model.pt"
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

        # Scaler
        scaler_path = path / "scaler.npy.npz"
        data = np.load(scaler_path)
        self._scaler = MinMaxScaler()
        self._scaler.scale_ = data["scale"]
        self._scaler.min_ = data["min"]
        self._scaler.data_min_ = data["data_min"]
        self._scaler.data_max_ = data["data_max"]
        self._scaler.data_range_ = data["data_range"]
        # Mark scaler as fitted
        self._scaler.n_features_in_ = input_size
        self._is_fitted = True

        logger.info(
            "Model loaded from {} (input_size={}, features={})",
            path, input_size, self._feature_columns,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sequences(
        features: np.ndarray,
        labels: np.ndarray,
        lookback: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a window over features/labels to create sequences."""
        X: list[np.ndarray] = []
        y: list[int] = []
        for i in range(lookback, len(features)):
            X.append(features[i - lookback : i])
            y.append(int(labels[i]))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def _evaluate_loader(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Evaluate loss and accuracy on a data loader."""
        assert self._model is not None
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)

                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        self._model.train()
        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
