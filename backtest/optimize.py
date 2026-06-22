"""
Parameter optimization (grid search) with honest in-sample / out-of-sample split.

Sweeps a strategy's config parameters, ranks combinations by **in-sample** edge
(profit factor, then Sharpe), then re-runs the best ones on a held-out
**out-of-sample** window. The IS→OOS comparison is the overfit check: a combo
that looks great in-sample but collapses out-of-sample was curve-fit, not real.

This deliberately mirrors the project's honesty discipline (see
``backtest/strategy_eval.py`` and ``backtest/robustness.py``): profit factor is
the sizing-independent edge signal, and a good backtest is necessary but not
sufficient — forward-testing in paper remains the real proof.

Run::

    venv/Scripts/python.exe -m backtest.optimize --strategy mean_reversion --symbol AVAX/USDT --tf 15m
    venv/Scripts/python.exe -m backtest.optimize --strategy breakout --symbol SOL/USDT --tf 1h --candles 4000
    venv/Scripts/python.exe -m backtest.optimize --strategy vwap_reversion --symbol DOGE/USDT --tf 15m --top 5
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
from typing import Any

from loguru import logger

from backtest.backtester import Backtester
from backtest.walkforward_ml import fetch_klines
from src.strategy.breakout import BreakoutStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.scalping import ScalpingStrategy
from src.strategy.vwap_reversion import VWAPReversionStrategy

# Strategy registry: name -> class
STRATEGIES: dict[str, type] = {
    "scalping": ScalpingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum": MomentumStrategy,
    "breakout": BreakoutStrategy,
    "vwap_reversion": VWAPReversionStrategy,
}

# Parameter grids per strategy. Kept small on purpose — wide grids invite
# overfitting and the IS/OOS split is meant to catch it, not enable it.
GRIDS: dict[str, dict[str, list[Any]]] = {
    "mean_reversion": {
        "z_entry": [1.5, 2.0, 2.5],
        "stop_loss_pct": [0.01, 0.015, 0.02],
        "target_pct": [0.015, 0.02, 0.03],
    },
    "breakout": {
        "channel_period": [10, 20, 30],
        "stop_loss_pct": [0.015, 0.02, 0.03],
        "target_pct": [0.02, 0.03, 0.05],
    },
    "vwap_reversion": {
        "vwap_period": [14, 20, 30],
        "band_pct": [0.01, 0.015, 0.02],
        "target_pct": [0.01, 0.015, 0.02],
    },
    "scalping": {
        "rsi_oversold": [25, 30],
        "rsi_overbought": [70, 75],
        "target_pct": [0.004, 0.006, 0.008],
    },
    "momentum": {
        "atr_multiplier": [1.5, 2.0, 2.5],
        "stop_loss_pct": [0.015, 0.02],
        "target_pct": [0.03, 0.04, 0.05],
    },
}


def _pf_str(pf: float) -> str:
    return "inf" if pf == float("inf") else f"{pf:.2f}"


async def _backtest(bt_kwargs: dict, cls: type, config: dict, symbol: str, tf: str, df) -> Any:
    """Run one backtest on a pre-loaded data slice."""
    bt = Backtester(**bt_kwargs)
    return await bt.run(
        strategy=cls(config=config),
        symbol=symbol,
        timeframe=tf,
        start_date=str(df.index[0])[:10],
        end_date=str(df.index[-1])[:10],
        data=df,
    )


async def main() -> None:
    p = argparse.ArgumentParser(description="Strategy parameter optimization (grid search)")
    p.add_argument("--strategy", required=True, choices=sorted(GRIDS))
    p.add_argument("--symbol", required=True, help="e.g. AVAX/USDT")
    p.add_argument("--tf", default="15m", help="timeframe (default 15m)")
    p.add_argument("--candles", type=int, default=4000)
    p.add_argument("--capital", type=float, default=1000.0)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--train", type=float, default=0.70, help="in-sample fraction")
    p.add_argument("--min-trades", type=int, default=5, help="min in-sample trades to qualify")
    p.add_argument("--top", type=int, default=5, help="top combos to validate OOS")
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    symbol = args.symbol.upper()
    cls = STRATEGIES[args.strategy]
    grid = GRIDS[args.strategy]

    df = fetch_klines(symbol, args.tf, args.candles)
    if df.empty or len(df) < 120:
        print(f"Not enough data for {symbol} {args.tf} (got {len(df)} bars).")
        return

    split = int(len(df) * args.train)
    is_df, oos_df = df.iloc[:split], df.iloc[split:]

    bt_kwargs = dict(
        fee_rate=args.fee, slippage_rate=args.slippage, initial_capital=args.capital
    )

    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(
        f"Optimizing {args.strategy} on {symbol} {args.tf}: {len(combos)} combos | "
        f"IS {len(is_df)} bars / OOS {len(oos_df)} bars "
        f"(fee {args.fee:.2%}, slip {args.slippage:.2%})",
        flush=True,
    )

    # ── In-sample sweep ──────────────────────────────────────────────────
    scored: list[tuple[dict, Any]] = []
    for combo in combos:
        config = dict(zip(keys, combo))
        config["timeframe"] = args.tf
        try:
            res = await _backtest(bt_kwargs, cls, config, symbol, args.tf, is_df)
        except Exception as exc:
            logger.warning("Combo {} failed: {}", config, exc)
            continue
        scored.append((config, res))

    qualified = [(c, r) for c, r in scored if r.total_trades >= args.min_trades]
    qualified.sort(key=lambda cr: (cr[1].profit_factor, cr[1].sharpe_ratio), reverse=True)

    if not qualified:
        print(
            f"\nNo combo reached >={args.min_trades} in-sample trades. "
            "Try more candles or a looser grid."
        )
        return

    # ── Out-of-sample validation of the top combos ───────────────────────
    print("\n" + "=" * 78)
    print(f"  TOP {min(args.top, len(qualified))} BY IN-SAMPLE PROFIT FACTOR -- validated out-of-sample")
    print("=" * 78)
    print(f"  {'params':<46} {'IS_PF':>6} {'IS_tr':>6} {'OOS_PF':>7} {'OOS_tr':>7}")
    print("-" * 78)

    overfit_flags: list[str] = []
    for config, is_res in qualified[: args.top]:
        try:
            oos_res = await _backtest(bt_kwargs, cls, config, symbol, args.tf, oos_df)
        except Exception as exc:
            logger.warning("OOS for {} failed: {}", config, exc)
            continue

        params = {k: config[k] for k in keys}
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(
            f"  {params_str:<46} {_pf_str(is_res.profit_factor):>6} "
            f"{is_res.total_trades:>6} {_pf_str(oos_res.profit_factor):>7} "
            f"{oos_res.total_trades:>7}"
        )

        # Overfit signal: strong in-sample edge that does not survive OOS.
        if is_res.profit_factor > 1.2 and oos_res.profit_factor < 1.0:
            overfit_flags.append(params_str)

    print("-" * 78)
    best_cfg = {k: qualified[0][0][k] for k in keys}
    print(f"  Best in-sample params: {best_cfg}")
    if overfit_flags:
        print("\n  (!) OVERFIT WARNING -- these looked good in-sample but failed OOS:")
        for f in overfit_flags:
            print(f"      {f}")
    print(
        "\n  Pick a combo that holds PF>1 in BOTH windows. Then forward-test it\n"
        "  in paper before enabling live -- a backtest is necessary, not sufficient."
    )
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
