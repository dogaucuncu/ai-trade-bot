#!/usr/bin/env python3
"""
AI Trade Bot — Main entry point.

Loads configuration, verifies all imports, prints a startup banner, and
(in a future iteration) starts the bot engine event loop.

Usage::

    python main.py                   # paper mode, default verbosity
    python main.py --mode live       # live mode (requires real keys)
    python main.py -v                # verbose / DEBUG logging
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the project root is on sys.path so relative imports resolve
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Trade Bot — crypto & stock micro-trading engine",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default=None,
        help="Trading mode (overrides TRADING_MODE env var)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def _print_banner(settings) -> None:  # noqa: ANN001
    """Print a pretty startup banner."""
    from loguru import logger

    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║              🤖  AI TRADE BOT  v0.1.0  🤖                  ║
    ║         Crypto (Binance) + Stocks (Alpaca)                  ║
    ║         Micro-capital · AI-powered signals                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    logger.opt(raw=True).info(banner)
    logger.info("Mode          : {}", settings.trading_mode.upper())
    logger.info("Capital       : ${:.2f} total", settings.initial_capital)
    logger.info("  Crypto      : ${:.2f} ({:.0%})", settings.crypto_capital, settings.crypto_allocation)
    logger.info("  Stocks      : ${:.2f} ({:.0%})", settings.stock_capital, settings.stock_allocation)
    logger.info("Binance pairs : {}", ", ".join(settings.binance.default_pairs))
    logger.info("Alpaca symbols: {}", ", ".join(settings.alpaca.default_symbols))
    logger.info("Fractional    : {}", settings.fractional_shares)
    logger.info("Timeframes    : {}", ", ".join(settings.timeframes))
    logger.info("Database      : {}", settings.db_url)
    logger.info("Risk per trade: {:.0%} (~${:.2f})", settings.risk.max_risk_per_trade, settings.initial_capital * settings.risk.max_risk_per_trade)
    logger.info("Max positions : {} (crypto={}, stock={})", settings.risk.max_open_positions, settings.risk.max_crypto_positions, settings.risk.max_stock_positions)
    logger.info("Max drawdown  : {:.0%}", settings.risk.max_drawdown)


def _verify_imports() -> bool:
    """Try importing every core module; return True if all succeed."""
    from loguru import logger

    modules = [
        ("config.settings", "Settings"),
        ("src.data.collector", "DataCollector"),
        ("src.data.storage", "Storage"),
        ("src.data.websocket_feed", "WebSocketFeed"),
        ("src.indicators.technical", "TechnicalIndicators"),
    ]

    all_ok = True
    for mod_path, cls_name in modules:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            getattr(mod, cls_name)
            logger.info("  ✓ {}.{}", mod_path, cls_name)
        except Exception as exc:
            logger.error("  ✗ {}.{} — {}", mod_path, cls_name, exc)
            all_ok = False

    return all_ok


async def _async_main(settings) -> None:  # noqa: ANN001
    """Async bootstrap: initialise DB, verify connectivity."""
    from loguru import logger
    from src.data.storage import Storage

    # Ensure data directory exists
    data_dir = settings.project_root / "data"
    data_dir.mkdir(exist_ok=True)

    storage = Storage(settings.db_url)
    await storage.init_db()
    logger.info("Database ready.")

    from src.bot.engine import BotEngine
    from dashboard.app import app, state as dashboard_state
    import uvicorn
    import ssl
    import certifi
    import os
    
    # Fix SSL certificate issues on Windows
    os.environ['SSL_CERT_FILE'] = certifi.where()
    
    # In paper mode, we do a dry run (no actual orders sent to exchange unless using testnet)
    is_dry_run = (settings.trading_mode == "paper")
    
    engine = BotEngine(
        capital=settings.initial_capital,
        dry_run=is_dry_run
    )
    await engine.initialize()
    
    # Attach engine to dashboard state
    dashboard_state.attach_engine(engine)
    
    logger.info(f"All systems nominal. Starting bot engine loop (dry_run={is_dry_run})...")
    logger.info(f"Starting dashboard at http://{settings.dashboard_host}:{settings.dashboard_port}")
    
    config = uvicorn.Config(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="warning"
    )
    server = uvicorn.Server(config)
    
    try:
        # Run both the bot engine and the dashboard API server concurrently
        await asyncio.gather(
            engine.run(),
            server.serve(),
            dashboard_state.start_broadcast_loop()
        )
    except asyncio.CancelledError:
        pass
    finally:
        await engine.shutdown()
        await storage.close()


def main() -> None:
    """Synchronous entry point."""
    args = _parse_args()

    # ── Import settings (triggers .env loading) ─────────────────────
    from config.settings import settings

    # CLI overrides
    if args.mode:
        settings.trading_mode = args.mode
    if args.verbose:
        settings.log_level = "DEBUG"

    settings.configure_logging()

    from loguru import logger

    _print_banner(settings)

    # ── Verify imports ──────────────────────────────────────────────
    logger.info("Verifying module imports …")
    if not _verify_imports():
        logger.error(
            "Some modules failed to import. Install dependencies:\n"
            "  pip install -r requirements.txt"
        )
        sys.exit(1)

    logger.info("All modules imported successfully.")

    # ── Run async bootstrap ─────────────────────────────────────────
    try:
        asyncio.run(_async_main(settings))
    except KeyboardInterrupt:
        logger.info("Shutdown requested — bye! 👋")


if __name__ == "__main__":
    main()
