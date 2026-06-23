"""
Honest walk-forward backtest for the LSTM model.

This module replaces the misleading ``backtest_ml.py`` script, which had
three fatal flaws that made its results meaningless:

1. **In-sample testing** — it tested on the very candles the model was
   trained on, so the model had already "seen the answers".
2. **Look-ahead scaling** — the scaler was fit on the whole dataset
   (now fixed in ``src/ml/lstm_model.py``).
3. **Per-bar position flipping** — it could reverse the position every
   single candle, churning fees with no holding logic.

What this does instead
----------------------
* **Walk-forward folds** — the series is split into consecutive
  (train → test) windows. A *fresh* model is trained on each train
  window and evaluated ONLY on the immediately following, never-seen
  test window. The folds roll forward through time.
* **Realistic costs** — taker fee + slippage on every fill.
* **No churn** — a position is held until a *confident opposite* signal,
  a stop-loss, or a take-profit. Low-confidence bars do nothing.
* **Honest metrics** — net-of-fee return, timeframe-aware Sharpe/Sortino,
  max drawdown, profit factor, and — crucially — the **excess return
  versus simply buying and holding** over the same out-of-sample period.

Run it::

    venv/Scripts/python.exe -m backtest.walkforward_ml --symbol DOGE/USDT \
        --tf 1h --candles 4000 --train 1200 --test 400 --epochs 15

Use ``--fast`` for a quick smoke run (small model, few epochs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.metrics import PerformanceReport  # noqa: E402
from src.indicators.technical import TechnicalIndicators  # noqa: E402
from src.ml.lstm_model import LSTMPredictor  # noqa: E402
from src.net import verified_ssl_context  # noqa: E402

_RESULTS_DIR = _PROJECT_ROOT / "backtest" / "results"
_INDICATOR_WARMUP = 250  # rows of history fed to predict() so indicators are valid


# =========================================================================
# Data fetching (Binance public klines, with pagination > 1000)
# =========================================================================

def fetch_klines(symbol: str, interval: str, total: int) -> pd.DataFrame:
    """Fetch up to *total* recent klines from Binance, paginating by 1000.

    Returns a DataFrame indexed by UTC timestamp with OHLCV columns.
    """
    binance_symbol = symbol.replace("/", "")
    # Verified TLS via the OS trust store (truststore) — no MITM exposure.
    ctx = verified_ssl_context()

    rows: list[list[Any]] = []
    end_time: int | None = None
    remaining = total

    while remaining > 0:
        limit = min(1000, remaining)
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={binance_symbol}&interval={interval}&limit={limit}"
        )
        if end_time is not None:
            url += f"&endTime={end_time}"

        with urllib.request.urlopen(url, timeout=20, context=ctx) as resp:
            batch = json.load(resp)

        if not batch:
            break

        rows = batch + rows  # prepend older data
        remaining -= len(batch)
        # next request ends just before the oldest candle we have
        end_time = batch[0][0] - 1
        if len(batch) < limit:
            break
        time.sleep(0.25)  # be polite to the public API

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df


# =========================================================================
# Single out-of-sample test window simulation
# =========================================================================

class _Position:
    __slots__ = ("side", "entry", "qty", "stop", "tp", "entry_fee")

    def __init__(self, side, entry, qty, stop, tp, entry_fee):
        self.side = side          # "LONG" or "SHORT"
        self.entry = entry
        self.qty = qty
        self.stop = stop
        self.tp = tp
        self.entry_fee = entry_fee


def simulate_window(
    predictor: LSTMPredictor,
    full: pd.DataFrame,
    test_start: int,
    test_end: int,
    *,
    timeframe: str,
    capital: float,
    fee_rate: float,
    slippage: float,
    confidence_threshold: float,
    stop_pct: float,
    tp_pct: float,
    exposure: float,
    lookback: int,
) -> dict[str, Any]:
    """Trade a single out-of-sample window bar-by-bar.

    The model only ever sees data up to the current bar. Fills include
    slippage; entries and exits each pay ``fee_rate``. A position is flipped
    only on a confident opposite signal, otherwise held to SL/TP.
    """
    pos: _Position | None = None
    equity_curve: list[float] = [capital]
    trade_pnls: list[float] = []
    total_fees = 0.0

    def close_position(exit_price: float) -> None:
        nonlocal capital, pos, total_fees
        assert pos is not None
        if pos.side == "LONG":
            realized = (exit_price - pos.entry) * pos.qty
        else:
            realized = (pos.entry - exit_price) * pos.qty
        exit_fee = exit_price * pos.qty * fee_rate
        capital += realized - exit_fee
        total_fees += exit_fee
        trade_pnls.append(realized - pos.entry_fee - exit_fee)
        pos = None

    def open_position(side: str, price: float) -> None:
        nonlocal capital, pos, total_fees
        fill = price * (1 + slippage) if side == "LONG" else price * (1 - slippage)
        notional = capital * exposure
        qty = notional / fill
        entry_fee = notional * fee_rate
        capital -= entry_fee
        total_fees += entry_fee
        if side == "LONG":
            stop = fill * (1 - stop_pct)
            tp = fill * (1 + tp_pct)
        else:
            stop = fill * (1 + stop_pct)
            tp = fill * (1 - tp_pct)
        pos = _Position(side, fill, qty, stop, tp, entry_fee)

    for t in range(test_start, test_end):
        bar = full.iloc[t]
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])

        # --- 1. intrabar SL/TP on the open position ----------------------
        if pos is not None:
            if pos.side == "LONG":
                if low <= pos.stop:
                    close_position(pos.stop)
                elif high >= pos.tp:
                    close_position(pos.tp)
            else:
                if high >= pos.stop:
                    close_position(pos.stop)
                elif low <= pos.tp:
                    close_position(pos.tp)

        # --- 2. model signal using data up to *and including* bar t ------
        window = full.iloc[max(0, t - _INDICATOR_WARMUP):t + 1]
        try:
            direction, conf = predictor.predict(window, lookback=lookback)
        except Exception:
            direction, conf = "SIDEWAYS", 0.0
        if conf < confidence_threshold:
            direction = "SIDEWAYS"

        # --- 3. act on the signal ----------------------------------------
        if pos is None:
            if direction == "UP":
                open_position("LONG", close)
            elif direction == "DOWN":
                open_position("SHORT", close)
        else:
            opposite = (pos.side == "LONG" and direction == "DOWN") or (
                pos.side == "SHORT" and direction == "UP"
            )
            if opposite:
                exit_fill = close * (1 - slippage) if pos.side == "LONG" else close * (1 + slippage)
                close_position(exit_fill)
                open_position("LONG" if direction == "UP" else "SHORT", close)

        # --- 4. mark-to-market equity ------------------------------------
        if pos is not None:
            if pos.side == "LONG":
                unreal = (close - pos.entry) * pos.qty
            else:
                unreal = (pos.entry - close) * pos.qty
        else:
            unreal = 0.0
        equity_curve.append(capital + unreal)

    # force-close at the last test bar
    if pos is not None:
        close_position(float(full.iloc[test_end - 1]["close"]))
        equity_curve.append(capital)

    close_prices = full.iloc[test_start:test_end]["close"].tolist()
    report = PerformanceReport.from_run(
        timeframe=timeframe,
        initial_capital=equity_curve[0],
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
        total_fees=total_fees,
        close_prices=close_prices,
    )
    return {
        "report": report,
        "final_capital": capital,
        "equity_curve": equity_curve,
        "trade_pnls": trade_pnls,
    }


# =========================================================================
# Walk-forward driver
# =========================================================================

def run_walk_forward(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    train_size: int,
    test_size: int,
    epochs: int,
    lookback: int,
    hidden_size: int,
    num_layers: int,
    fee_rate: float,
    slippage: float,
    confidence_threshold: float,
    stop_pct: float,
    tp_pct: float,
    exposure: float,
    initial_capital: float,
) -> dict[str, Any]:
    """Train/evaluate the model across rolling out-of-sample folds."""
    # Pre-compute indicators once on the full series (each value uses only
    # its own past, so this introduces no leak).
    df = TechnicalIndicators.add_all_indicators(df.copy())
    n = len(df)

    fold_reports: list[PerformanceReport] = []
    all_trade_pnls: list[float] = []
    combined_equity: list[float] = [initial_capital]
    capital = initial_capital
    start = 0
    fold_idx = 0

    while start + train_size + test_size <= n:
        fold_idx += 1
        train_df = df.iloc[start:start + train_size].reset_index(drop=True)
        test_start = start + train_size
        test_end = min(test_start + test_size, n)

        logger.info(
            "── Fold {} — train rows [{}:{}]  test rows [{}:{}] ──",
            fold_idx, start, start + train_size, test_start, test_end,
        )

        # Fresh model per fold — scaler fits on train only (leak-free).
        predictor = LSTMPredictor(
            hidden_size=hidden_size, num_layers=num_layers, dropout=0.2,
        )
        try:
            train_loader, val_loader = predictor.prepare_data(
                train_df, lookback=lookback, val_split=0.15, batch_size=64,
            )
            history = predictor.train(train_loader, val_loader, epochs=epochs)
            val_acc = history.val_accuracies[-1] if history.val_accuracies else 0.0
            logger.info("Fold {} trained — val_acc={:.2%}", fold_idx, val_acc)
        except Exception as exc:
            logger.warning("Fold {} training failed: {} — skipping", fold_idx, exc)
            start += test_size
            continue

        sim = simulate_window(
            predictor, df, test_start, test_end,
            timeframe=timeframe, capital=capital,
            fee_rate=fee_rate, slippage=slippage,
            confidence_threshold=confidence_threshold,
            stop_pct=stop_pct, tp_pct=tp_pct, exposure=exposure,
            lookback=lookback,
        )
        report: PerformanceReport = sim["report"]
        fold_reports.append(report)
        all_trade_pnls.extend(sim["trade_pnls"])
        logger.info("Fold {} result:\n{}", fold_idx, report.summary())

        # Each fold's sim starts with the previous fold's ending capital, so
        # the equity curves simply concatenate into one continuous series.
        combined_equity.extend(sim["equity_curve"][1:])
        capital = sim["final_capital"]

        start += test_size

    if not fold_reports:
        logger.error("No completed folds — need more candles or smaller windows.")
        return {"folds": [], "overall": None}

    # Overall out-of-sample report over the concatenated test periods.
    first_test = train_size
    last_test = train_size + test_size * len(fold_reports)
    oos_close = df.iloc[first_test:min(last_test, n)]["close"].tolist()

    overall = PerformanceReport.from_run(
        timeframe=timeframe,
        initial_capital=initial_capital,
        equity_curve=combined_equity,
        trade_pnls=all_trade_pnls,
        total_fees=sum(r.total_fees for r in fold_reports),
        close_prices=oos_close,
    )

    return {"folds": fold_reports, "overall": overall, "combined_equity": combined_equity}


# =========================================================================
# CLI
# =========================================================================

def _save(symbol: str, timeframe: str, result: dict[str, Any]) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overall = result["overall"]
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": asdict(overall) if overall else None,
        "folds": [asdict(r) for r in result["folds"]],
    }
    path = _RESULTS_DIR / f"walkforward_{symbol.replace('/', '_')}_{timeframe}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _resolve_symbols(args) -> list[str]:
    """Decide which symbols to evaluate from the CLI args."""
    if args.all:
        from config.settings import settings
        return list(settings.binance.default_pairs)
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    return [args.symbol]


def _run_one(args, symbol: str) -> dict | None:
    """Fetch data and run the walk-forward for a single symbol."""
    logger.info("Fetching {} candles for {} ({})...", args.candles, symbol, args.tf)
    df = fetch_klines(symbol, args.tf, args.candles)
    if df.empty or len(df) < args.train + args.test + _INDICATOR_WARMUP:
        logger.warning("Not enough data for {} ({} rows) — skipping.", symbol, len(df))
        return None
    logger.info("Fetched {} candles for {}  [{} -> {}]", len(df), symbol, df.index[0], df.index[-1])

    result = run_walk_forward(
        df,
        symbol=symbol, timeframe=args.tf,
        train_size=args.train, test_size=args.test,
        epochs=args.epochs, lookback=args.lookback,
        hidden_size=args.hidden, num_layers=args.layers,
        fee_rate=args.fee, slippage=args.slippage,
        confidence_threshold=args.threshold,
        stop_pct=args.stop, tp_pct=args.tp, exposure=args.exposure,
        initial_capital=args.capital,
    )
    return result if result.get("overall") is not None else None


def main() -> None:
    p = argparse.ArgumentParser(description="Honest walk-forward ML backtest")
    p.add_argument("--symbol", default="DOGE/USDT")
    p.add_argument("--symbols", default=None,
                   help="comma-separated list, e.g. SOL/USDT,AVAX/USDT")
    p.add_argument("--all", action="store_true",
                   help="evaluate every pair in settings.binance.default_pairs")
    p.add_argument("--tf", default="1h")
    p.add_argument("--candles", type=int, default=4000)
    p.add_argument("--train", type=int, default=1200)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--stop", type=float, default=0.015)
    p.add_argument("--tp", type=float, default=0.03)
    p.add_argument("--exposure", type=float, default=0.95)
    p.add_argument("--capital", type=float, default=50.0)
    p.add_argument("--fast", action="store_true",
                   help="quick smoke run (small data, few epochs)")
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.fast:
        args.candles, args.train, args.test = 1400, 800, 250
        args.epochs, args.hidden, args.layers = 6, 32, 1

    symbols = _resolve_symbols(args)

    # ── Single symbol: detailed per-fold report ──────────────────────────
    if len(symbols) == 1:
        result = _run_one(args, symbols[0])
        if result is None:
            logger.error("No completed result for {}.", symbols[0])
            return
        overall = result["overall"]
        print("\n" + "=" * 56)
        print(f"  WALK-FORWARD (OUT-OF-SAMPLE) -- {symbols[0]} {args.tf}")
        print(f"  {len(result['folds'])} folds | fee={args.fee:.2%} slippage={args.slippage:.2%}")
        print("=" * 56)
        print(overall.summary())
        print("\n  Per-fold excess vs buy&hold:")
        for i, r in enumerate(result["folds"], 1):
            print(f"    Fold {i}: net {r.total_return_pct:+.2f}%  | "
                  f"hold {r.buy_and_hold_pct:+.2f}%  | "
                  f"excess {r.excess_vs_hold_pct:+.2f}%  | "
                  f"trades {r.total_trades}")
        print(f"\n  Saved -> {_save(symbols[0], args.tf, result)}")
        return

    # ── Multiple symbols: comparison table + verdict ─────────────────────
    rows: list[tuple[str, Any]] = []
    for sym in symbols:
        result = _run_one(args, sym)
        if result is not None:
            rows.append((sym, result["overall"]))
            _save(sym, args.tf, result)

    if not rows:
        logger.error("No symbols produced a completed result.")
        return

    print("\n" + "=" * 78)
    print(f"  MULTI-COIN WALK-FORWARD (OUT-OF-SAMPLE) -- {args.tf}  "
          f"fee={args.fee:.2%} slip={args.slippage:.2%}")
    print("=" * 78)
    print(f"  {'Coin':12s} {'Trades':>6s} {'Net%':>8s} {'Hold%':>8s} "
          f"{'Excess%':>8s} {'Sharpe':>7s} {'PF':>6s} {'MaxDD%':>7s}")
    print("  " + "-" * 74)
    beat = 0
    for sym, r in rows:
        pf = "inf" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
        # "Real" edge = positive net AND beats buy&hold AND actually trades.
        if r.total_return_pct > 0 and r.excess_vs_hold_pct > 0 and r.total_trades >= 5:
            beat += 1
        print(f"  {sym:12s} {r.total_trades:>6d} {r.total_return_pct:>8.2f} "
              f"{r.buy_and_hold_pct:>8.2f} {r.excess_vs_hold_pct:>8.2f} "
              f"{r.sharpe:>7.2f} {pf:>6s} {r.max_drawdown_pct:>7.2f}")
    print("  " + "-" * 74)
    print(f"  Verdict: {beat}/{len(rows)} coins show a positive, hold-beating edge "
          f"(>=5 trades).")
    print("  Reminder: few trades => noise. Treat single-fold wins with suspicion.")
    print("=" * 78)


if __name__ == "__main__":
    main()
