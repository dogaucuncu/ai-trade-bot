#!/usr/bin/env python3
"""
Train & save LSTM models for the configured crypto pairs.

Defaults are tuned for this project's realities:
  * timeframe 15m, ~20k bars (~7 months) — enough samples in a recent,
    roughly single-regime window without over-reaching into stale history.
  * SMALL model (hidden=48, 1 layer) — a big model overfits these sample
    sizes (see src/ml/trainer.py train_lstm docstring).
  * Stationary feature set + class weights are applied automatically inside
    src/ml/lstm_model.py.

IMPORTANT: train accuracy is NOT proof of profit. Before trusting a model,
gate it through the honest out-of-sample test:

    venv/Scripts/python.exe -m backtest.walkforward_ml --all --tf 15m

Usage::

    venv/Scripts/python.exe train_model.py                 # all config pairs
    venv/Scripts/python.exe train_model.py --symbol SOL/USDT
    venv/Scripts/python.exe train_model.py --tf 15m --days 210 --epochs 25
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from config.settings import settings
from src.ml.trainer import ModelTrainer


async def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM models per coin")
    parser.add_argument("--symbol", default=None,
                        help="Train a single pair (default: all config pairs)")
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--days", type=int, default=210,
                        help="History to fetch in days (15m*210d ~= 20k bars)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--layers", type=int, default=1)
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else list(settings.binance.default_pairs)

    logger.info(
        "Training {} model(s) — tf={} days={} (hidden={}, layers={}, epochs={})",
        len(symbols), args.tf, args.days, args.hidden, args.layers, args.epochs,
    )

    trainer = ModelTrainer()
    results: dict[str, float] = {}
    try:
        for symbol in symbols:
            try:
                history = await trainer.train_lstm(
                    symbol=symbol,
                    timeframe=args.tf,
                    lookback_days=args.days,
                    epochs=args.epochs,
                    lookback=args.lookback,
                    batch_size=args.batch,
                    hidden_size=args.hidden,
                    num_layers=args.layers,
                )
                val_acc = (
                    history.val_accuracies[-1] if history.val_accuracies else 0.0
                )
                results[symbol] = val_acc
                logger.info("[OK] {} trained — final val_acc={:.2%}", symbol, val_acc)
            except Exception:
                logger.exception("[FAIL] {} training failed", symbol)
                results[symbol] = float("nan")
    finally:
        await trainer.close()

    print("\n" + "=" * 50)
    print("  TRAINING SUMMARY (val_acc; 3-class random = 33%)")
    print("=" * 50)
    for sym, acc in results.items():
        print(f"  {sym:12s} : {acc:.2%}")
    print("=" * 50)
    print("  NOTE: val_acc != profit. Verify edge with:")
    print(f"    python -m backtest.walkforward_ml --all --tf {args.tf}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(main())
