"""
Robustness test — is a coin's walk-forward edge real, or luck?

A single positive backtest means little, especially after screening several
coins. Two cheap, honest stress tests separate signal from noise:

1. **Seed stability** — re-run the SAME config with different random seeds.
   LSTM training is stochastic (weight init, optimiser). If the net return
   swings wildly across seeds (e.g. +18% / -20% / +3%), the "edge" is just
   one lucky initialisation. A real edge stays consistently positive.

2. **Threshold sensitivity** — vary the confidence threshold with a FIXED
   seed. A robust edge degrades gracefully as you change the knob; an overfit
   one appears at exactly one setting and collapses elsewhere.

Run::

    venv/Scripts/python.exe -m backtest.robustness --symbol DOGE/USDT --tf 15m

This is a diagnostic, not a deployment step. Treat a coin as trustworthy only
if it survives BOTH tests.
"""

from __future__ import annotations

import argparse
import random
import sys
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from loguru import logger

from backtest.walkforward_ml import fetch_klines, run_walk_forward


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run(df, *, symbol, tf, threshold, seed, args) -> Any:
    """One full walk-forward run with a fixed seed; returns the overall report."""
    _set_seed(seed)
    res = run_walk_forward(
        df.copy(),
        symbol=symbol, timeframe=tf,
        train_size=args.train, test_size=args.test,
        epochs=args.epochs, lookback=args.lookback,
        hidden_size=args.hidden, num_layers=args.layers,
        fee_rate=args.fee, slippage=args.slippage,
        confidence_threshold=threshold,
        stop_pct=args.stop, tp_pct=args.tp, exposure=args.exposure,
        initial_capital=args.capital,
    )
    return res.get("overall")


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward robustness test")
    p.add_argument("--symbol", default="DOGE/USDT")
    p.add_argument("--tf", default="15m")
    p.add_argument("--candles", type=int, default=9000)
    p.add_argument("--train", type=int, default=5000)
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--stop", type=float, default=0.015)
    p.add_argument("--tp", type=float, default=0.03)
    p.add_argument("--exposure", type=float, default=0.95)
    p.add_argument("--capital", type=float, default=50.0)
    p.add_argument("--seeds", default="1,2,3,7,42,123")
    p.add_argument("--thresholds", default="0.35,0.40,0.45,0.50")
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # quiet: this runs many trainings

    print(f"Fetching {args.candles} candles for {args.symbol} ({args.tf})...",
          flush=True)
    df = fetch_klines(args.symbol, args.tf, args.candles)
    if df.empty or len(df) < args.train + args.test + 250:
        print(f"Not enough data ({len(df)} rows).")
        return

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]

    # ── 1. Seed stability (threshold fixed at 0.40) ──────────────────────
    print(f"\n[1/2] Seed stability — {len(seeds)} seeds @ threshold 0.40", flush=True)
    seed_rows = []
    for s in seeds:
        r = _run(df, symbol=args.symbol, tf=args.tf, threshold=0.40, seed=s, args=args)
        if r is not None:
            seed_rows.append((s, r))
            print(f"  seed {s:>4d}: net {r.total_return_pct:+7.2f}%  "
                  f"excess {r.excess_vs_hold_pct:+7.2f}%  PF {r.profit_factor:>5.2f}  "
                  f"trades {r.total_trades}", flush=True)

    # ── 2. Threshold sensitivity (seed fixed at 42) ──────────────────────
    print(f"\n[2/2] Threshold sensitivity — seed 42", flush=True)
    thr_rows = []
    for t in thresholds:
        r = _run(df, symbol=args.symbol, tf=args.tf, threshold=t, seed=42, args=args)
        if r is not None:
            thr_rows.append((t, r))
            print(f"  thr {t:.2f}: net {r.total_return_pct:+7.2f}%  "
                  f"excess {r.excess_vs_hold_pct:+7.2f}%  PF {r.profit_factor:>5.2f}  "
                  f"trades {r.total_trades}", flush=True)

    # ── Verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ROBUSTNESS VERDICT -- {args.symbol} {args.tf}")
    print("=" * 60)
    if seed_rows:
        nets = [r.total_return_pct for _, r in seed_rows]
        pfs = [r.profit_factor for _, r in seed_rows if r.profit_factor != float("inf")]
        pos = sum(1 for n in nets if n > 0)
        print(f"  Seed net%   : mean {mean(nets):+.2f}  std {pstdev(nets):.2f}  "
              f"min {min(nets):+.2f}  max {max(nets):+.2f}")
        print(f"  Profitable  : {pos}/{len(nets)} seeds positive")
        if pfs:
            print(f"  PF mean     : {mean(pfs):.2f}")
    if thr_rows:
        tpos = sum(1 for _, r in thr_rows if r.total_return_pct > 0)
        print(f"  Thresholds  : {tpos}/{len(thr_rows)} settings positive")
    print("-" * 60)
    print("  Trustworthy edge = most seeds positive (low std) AND")
    print("  most thresholds positive. Otherwise: likely noise/overfit.")
    print("=" * 60)


if __name__ == "__main__":
    main()
