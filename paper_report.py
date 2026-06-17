#!/usr/bin/env python3
"""
Paper / live performance report.

Reads the equity snapshots and closed trades the running bot persists to the
database and prints an honest performance summary using the same metric
functions as the backtester (``backtest.metrics``). Run it any time while or
after a paper session::

    venv/Scripts/python.exe paper_report.py

Equity snapshots are taken once per engine tick (~1 minute), so risk ratios
are annualised on a 1-minute basis. Buy & hold is not shown here because a
paper session can span several symbols at once.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger  # noqa: E402

from backtest.metrics import (  # noqa: E402
    expectancy,
    max_drawdown,
    profit_factor,
    returns_from_equity,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from config.settings import settings  # noqa: E402
from src.data.storage import Storage  # noqa: E402


async def build_report(db_url: str | None = None) -> str:
    """Return a formatted performance report string from persisted data."""
    storage = Storage(db_url or settings.db_url)
    await storage.init_db()
    try:
        curve = await storage.get_equity_curve()
        trades = await storage.get_trades(limit=1_000_000)
    finally:
        await storage.close()

    if not curve:
        return (
            "No equity snapshots yet. Start a paper session with "
            "`python main.py` and let it run."
        )

    equity = [c["equity"] for c in curve]
    rets = returns_from_equity(equity)
    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]

    initial, final = equity[0], equity[-1]
    total_ret = ((final - initial) / initial * 100) if initial else 0.0
    pf = profit_factor(pnls)
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"

    return (
        "=" * 56 + "\n"
        f"  PAPER/LIVE PERFORMANCE ({curve[-1]['mode']})\n"
        f"  {curve[0]['timestamp']} -> {curve[-1]['timestamp']}\n"
        f"  {len(curve)} equity snapshots (~1m each)\n"
        + "=" * 56 + "\n"
        f"  Initial -> Final : ${initial:.2f} -> ${final:.2f}\n"
        f"  Total Return     : {total_ret:+.2f}%\n"
        f"  Sharpe / Sortino : {sharpe_ratio(rets, '1m'):.2f} / "
        f"{sortino_ratio(rets, '1m'):.2f}  (1m basis)\n"
        f"  Max Drawdown     : {max_drawdown(equity) * 100:.2f}%\n"
        f"  Closed trades    : {len(pnls)}\n"
        f"  Win Rate         : {win_rate(pnls) * 100:.1f}%\n"
        f"  Profit Factor    : {pf_s}\n"
        f"  Expectancy/trade : ${expectancy(pnls):.4f}\n"
        + "=" * 56
    )


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    print(asyncio.run(build_report()))


if __name__ == "__main__":
    main()
