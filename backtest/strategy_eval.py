"""
Rule-based strategy evaluation across coins.

The LSTM only earned its keep on DOGE. The rule-based strategies
(scalping, mean-reversion, momentum) are coin-agnostic and need no training,
so this script measures each one honestly on every configured coin, on the
timeframe it was designed for, with realistic fees + slippage, and compares
the net result to simply buying and holding.

Unlike the ML walk-forward there is no train/test split here — rule strategies
have nothing to fit. The honest risk instead is *parameter* overfitting (the
RSI/z-score/target thresholds were hand-picked), so a profitable backtest is
necessary-but-not-sufficient evidence; forward-testing in paper is still the
real proof.

Run::

    venv/Scripts/python.exe -m backtest.strategy_eval            # all config coins
    venv/Scripts/python.exe -m backtest.strategy_eval --symbols SOL/USDT,DOGE/USDT
    venv/Scripts/python.exe -m backtest.strategy_eval --candles 4000 --capital 1000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from loguru import logger

from backtest.backtester import Backtester
from backtest.metrics import buy_and_hold_return
from backtest.walkforward_ml import fetch_klines
from config.settings import settings
from src.strategy.breakout import BreakoutStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy
from src.strategy.scalping import ScalpingStrategy
from src.strategy.vwap_reversion import VWAPReversionStrategy

# Each rule strategy on the timeframe it was designed for.
STRATEGIES: dict[str, tuple[str, type]] = {
    "scalping": ("5m", ScalpingStrategy),
    "mean_reversion": ("15m", MeanReversionStrategy),
    "momentum": ("1h", MomentumStrategy),
    "breakout": ("1h", BreakoutStrategy),
    "vwap_reversion": ("15m", VWAPReversionStrategy),
}


async def _eval_one(
    bt_kwargs: dict, strat_name: str, tf: str, strat_cls: type,
    symbol: str, candles: int,
) -> dict[str, Any] | None:
    """Backtest one strategy on one coin; return a summary row."""
    df = fetch_klines(symbol, tf, candles)
    if df.empty or len(df) < 60:
        logger.warning("No data for {} {} — skipping", symbol, tf)
        return None

    bt = Backtester(**bt_kwargs)
    result = await bt.run(
        strategy=strat_cls(),
        symbol=symbol,
        timeframe=tf,
        start_date=str(df.index[0])[:10],
        end_date=str(df.index[-1])[:10],
        data=df,
    )
    hold_pct = buy_and_hold_return(df["close"].tolist()) * 100
    return {
        "strategy": strat_name,
        "symbol": symbol,
        "trades": result.total_trades,
        "net": result.total_return,
        "hold": hold_pct,
        "excess": result.total_return - hold_pct,
        "sharpe": result.sharpe_ratio,
        "pf": result.profit_factor,
        "maxdd": result.max_drawdown,
        "win": result.win_rate,
    }


async def main() -> None:
    p = argparse.ArgumentParser(description="Rule-based strategy evaluation")
    p.add_argument("--symbols", default=None, help="comma list; default = config pairs")
    p.add_argument("--candles", type=int, default=3000)
    p.add_argument("--capital", type=float, default=1000.0)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else list(settings.binance.default_pairs)
    )
    bt_kwargs = dict(
        fee_rate=args.fee, slippage_rate=args.slippage,
        initial_capital=args.capital,
    )

    print(f"Evaluating {len(STRATEGIES)} rule strategies x {len(symbols)} coins "
          f"(capital ${args.capital:.0f}, fee {args.fee:.2%}, slip {args.slippage:.2%})",
          flush=True)

    rows: list[dict[str, Any]] = []
    for strat_name, (tf, strat_cls) in STRATEGIES.items():
        print(f"\n[{strat_name} @ {tf}]", flush=True)
        for sym in symbols:
            row = await _eval_one(bt_kwargs, strat_name, tf, strat_cls, sym, args.candles)
            if row is None:
                continue
            rows.append(row)
            pf = "inf" if row["pf"] == float("inf") else f"{row['pf']:.2f}"
            print(f"  {sym:12s} trades {row['trades']:>4d}  net {row['net']:+7.2f}%  "
                  f"hold {row['hold']:+7.2f}%  excess {row['excess']:+7.2f}%  "
                  f"PF {pf:>5s}  DD {row['maxdd']:.1f}%", flush=True)

    # ── Verdict ──────────────────────────────────────────────────────────
    # Profit factor is the sizing-INDEPENDENT edge signal: gross win $ / gross
    # loss $ per trade, net of fees. Capital % return is distorted here because
    # the PositionSizer deploys little capital (equity barely moves), so PF +
    # win rate are the honest read of whether a strategy actually has edge.
    print("\n" + "=" * 70)
    print("  RULE-STRATEGY VERDICT -- edge = Profit Factor (sizing-independent)")
    print("=" * 70)
    winners = [r for r in rows if r["pf"] > 1.0 and r["trades"] >= 10]
    if winners:
        print("  PF > 1.0 (real per-trade edge after fees) with >=10 trades:")
        for r in sorted(winners, key=lambda x: x["pf"], reverse=True):
            pf = "inf" if r["pf"] == float("inf") else f"{r['pf']:.2f}"
            print(f"    {r['strategy']:14s} {r['symbol']:12s} "
                  f"PF {pf:>5s}  win {r['win']:.0f}%  trades {r['trades']}  "
                  f"net {r['net']:+.2f}%")
    else:
        print("  None. No rule strategy clears PF>1 with a usable sample on any coin.")
    print("-" * 70)
    print(f"  {len(winners)}/{len(rows)} strategy-coin combos have a real edge (PF>1).")
    print("  NOTE: net % is suppressed by conservative position sizing; PF is the")
    print("        honest edge signal. Few trades = noise; params can overfit.")
    print("  Real proof = forward-test the PF>1 winners in paper.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
