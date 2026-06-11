"""
AI Trade Bot — Paper Trading + Dashboard birlikte başlatma scripti.

Kullanım:
    python run_bot.py              # Paper trading (varsayılan)
    python run_bot.py --dry-run   # Sadece sinyal üret, işlem yapma
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger
from config.settings import settings


async def main(dry_run: bool = False) -> None:
    """Bot engine + dashboard'u paralel başlat."""

    from src.bot.engine import BotEngine
    from dashboard.app import state as dashboard_state, app as dashboard_app
    import uvicorn

    # ── 1. Bot engine oluştur ────────────────────────────────────────────
    crypto_symbols = list(settings.binance.default_pairs)
    stock_symbols  = list(settings.alpaca.default_symbols)

    engine = BotEngine(
        capital=settings.initial_capital,
        crypto_symbols=crypto_symbols,
        stock_symbols=stock_symbols,
        dry_run=dry_run,
    )

    logger.info("Initialising bot engine...")
    await engine.initialize()

    # ── 2. Dashboard'a engine'i bağla ────────────────────────────────────
    dashboard_state.attach_engine(engine)
    logger.info("Engine attached to dashboard.")

    # ── 3. Uvicorn'u arka planda başlat ─────────────────────────────────
    uv_config = uvicorn.Config(
        app=dashboard_app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="warning",   # uvicorn çıktısını minimumda tut
    )
    uv_server = uvicorn.Server(uv_config)

    mode_str = "DRY-RUN" if dry_run else "PAPER TRADING"
    logger.info("=" * 55)
    logger.info("  🤖  AI Trade Bot  —  {}", mode_str)
    logger.info("  💰  Capital : ${:.2f}  (Crypto ${:.2f}  |  Stocks ${:.2f})",
                settings.initial_capital,
                settings.crypto_capital,
                settings.stock_capital)
    logger.info("  📊  Dashboard : http://{}:{}", settings.dashboard_host, settings.dashboard_port)
    logger.info("  📈  Pairs     : {}", ", ".join(crypto_symbols[:4]) + "...")
    logger.info("  🛡️  Mode      : Binance Testnet + Alpaca Paper")
    logger.info("  ⚡  Press Ctrl+C to stop gracefully")
    logger.info("=" * 55)

    # ── 4. Her ikisini asyncio.gather ile paralel çalıştır ───────────────
    try:
        await asyncio.gather(
            uv_server.serve(),   # Dashboard (port 8000)
            engine.run(),        # Bot main loop (her 60 sn bir tick)
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown signal received...")
    finally:
        await engine.shutdown()
        logger.info("Bot stopped cleanly. Goodbye! 👋")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Trade Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sinyal üret ama gerçek işlem yapma (test modu)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(dry_run=args.dry_run))
    except KeyboardInterrupt:
        pass
