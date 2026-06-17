"""
AI Trade Bot — Centralised configuration.

All settings are loaded from environment variables (via a .env file) with
sensible defaults for paper-trading mode.  Import the singleton ``settings``
object anywhere in the project::

    from config.settings import settings
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from loguru import logger

# ── locate the project root (.env lives next to this package) ────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / "config" / ".env"

if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
    logger.info("Loaded environment from {}", _ENV_PATH)
else:
    # Also try project-root .env as a fallback
    _fallback = _PROJECT_ROOT / ".env"
    if _fallback.exists():
        load_dotenv(_fallback)
        logger.info("Loaded environment from {}", _fallback)
    else:
        logger.warning(
            "No .env file found — using system environment variables only."
        )


def _env(key: str, default: str = "") -> str:
    """Return an environment variable or *default*."""
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    """Parse a boolean-ish environment variable."""
    return _env(key, str(default)).lower() in {"1", "true", "yes"}


def _env_float(key: str, default: float = 0.0) -> float:
    """Parse a float environment variable."""
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int = 0) -> int:
    """Parse an integer environment variable."""
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _crypto_pairs_override() -> dict:
    """Parse the ``CRYPTO_SYMBOLS`` env var into a ``default_pairs`` override.

    Accepts a comma-separated list, e.g. ``CRYPTO_SYMBOLS=SOL/USDT,BTC/USDT``.
    Returns ``{}`` when unset so :class:`BinanceSettings` keeps its own default.
    """
    raw = _env("CRYPTO_SYMBOLS", "")
    pairs = tuple(s.strip().upper() for s in raw.split(",") if s.strip())
    return {"default_pairs": pairs} if pairs else {}


# =====================================================================
# Nested setting groups
# =====================================================================

@dataclass(frozen=True, slots=True)
class BinanceSettings:
    """Binance exchange credentials and preferences."""

    api_key: str = ""
    secret_key: str = ""
    testnet: bool = True
    # Single source of truth for which crypto pairs the bot trades.
    # Curated to liquid, currently-listed pairs that scale from micro
    # ($100) to larger ($1000+) capital. Dropped from the old list:
    #   SHIB/PEPE (meme, extreme volatility — risky for micro capital)
    #   MATIC     (migrated to POL; Binance delisting/rebrand risk)
    # When scaling up, BTC/USDT and ETH/USDT are the natural additions
    # (deepest liquidity, lowest slippage). Override via CRYPTO_SYMBOLS env.
    default_pairs: tuple[str, ...] = (
        "SOL/USDT",
        "AVAX/USDT",
        "XRP/USDT",
        "ADA/USDT",
        "DOGE/USDT",
    )


@dataclass(frozen=True, slots=True)
class AlpacaSettings:
    """Alpaca Markets credentials and preferences."""

    api_key: str = ""
    secret_key: str = ""
    paper: bool = True
    default_symbols: tuple[str, ...] = (
        "AAPL",
        "MSFT",
        "TSLA",
        "NVDA",
        "AMD",
        "META",
    )


@dataclass(frozen=True, slots=True)
class RiskSettings:
    """Risk-management parameters (micro-account friendly)."""

    max_risk_per_trade: float = 0.02        # 2 % of capital per trade (~$1)
    daily_loss_limit: float = 0.03          # 3 % max daily drawdown
    max_open_positions: int = 3             # reduced for $50 capital
    max_crypto_positions: int = 2           # 2 crypto slots
    max_stock_positions: int = 1            # 1 stock slot
    max_portfolio_exposure: float = 0.30    # 30 % of capital in market
    max_drawdown: float = 0.15              # 15 % absolute max drawdown


@dataclass(frozen=True, slots=True)
class SmtpSettings:
    """SMTP configuration for e-mail notifications."""

    host: str = "smtp.gmail.com"
    port: int = 587
    username: str = ""
    password: str = ""
    from_addr: str = ""
    to_addr: str = ""


# =====================================================================
# Top-level settings
# =====================================================================

@dataclass(slots=True)
class Settings:
    """Application-wide configuration singleton.

    Every field falls back to a safe default suitable for paper trading
    with ~$50 starting capital.  Override values via environment variables
    or a ``config/.env`` file.
    """

    # -- General ----------------------------------------------------------
    trading_mode: Literal["paper", "live"] = "paper"
    initial_capital: float = 50.0
    log_level: str = "INFO"
    project_root: Path = _PROJECT_ROOT

    # -- Capital allocation ($50 split) -----------------------------------
    crypto_allocation: float = 0.75         # 75 % → ~$37.50 for altcoins
    stock_allocation: float = 0.25          # 25 % → ~$12.50 for equities
    min_trade_amount_crypto: float = 5.0    # Binance min ≈ $5-$10
    min_trade_amount_stock: float = 5.0     # Alpaca fractional min
    fractional_shares: bool = True          # Alpaca fractional-share orders

    # -- Timeframes -------------------------------------------------------
    timeframes: tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h")

    # -- Persistence ------------------------------------------------------
    db_url: str = f"sqlite+aiosqlite:///{_PROJECT_ROOT / 'data' / 'tradebot.db'}"
    redis_url: str = "redis://localhost:6379/0"

    # -- Dashboard --------------------------------------------------------
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000

    # -- Sub-configs (populated by ``from_env``) --------------------------
    binance: BinanceSettings = field(default_factory=BinanceSettings)
    alpaca: AlpacaSettings = field(default_factory=AlpacaSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    smtp: SmtpSettings = field(default_factory=SmtpSettings)

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        """Build a ``Settings`` instance from current environment variables."""
        trading_mode_raw = _env("TRADING_MODE", "paper").lower()
        trading_mode: Literal["paper", "live"] = (
            "live" if trading_mode_raw == "live" else "paper"
        )

        return cls(
            trading_mode=trading_mode,
            initial_capital=_env_float("INITIAL_CAPITAL", 50.0),
            crypto_allocation=_env_float("CRYPTO_ALLOCATION", 0.75),
            stock_allocation=_env_float("STOCK_ALLOCATION", 0.25),
            min_trade_amount_crypto=_env_float("MIN_TRADE_CRYPTO", 5.0),
            min_trade_amount_stock=_env_float("MIN_TRADE_STOCK", 5.0),
            fractional_shares=_env_bool("FRACTIONAL_SHARES", default=True),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            db_url=_env(
                "DATABASE_URL",
                f"sqlite+aiosqlite:///{_PROJECT_ROOT / 'data' / 'tradebot.db'}",
            ),
            redis_url=_env("REDIS_URL", "redis://localhost:6379/0"),
            dashboard_host=_env("DASHBOARD_HOST", "127.0.0.1"),
            dashboard_port=_env_int("DASHBOARD_PORT", 8000),
            binance=BinanceSettings(
                api_key=_env("BINANCE_API_KEY"),
                secret_key=_env("BINANCE_SECRET_KEY"),
                testnet=_env_bool("BINANCE_TESTNET", default=True),
                **_crypto_pairs_override(),
            ),
            alpaca=AlpacaSettings(
                api_key=_env("ALPACA_API_KEY"),
                secret_key=_env("ALPACA_SECRET_KEY"),
                paper=_env_bool("ALPACA_PAPER", default=True),
            ),
            risk=RiskSettings(),
            smtp=SmtpSettings(
                host=_env("SMTP_HOST", "smtp.gmail.com"),
                port=_env_int("SMTP_PORT", 587),
                username=_env("SMTP_USERNAME"),
                password=_env("SMTP_PASSWORD"),
                from_addr=_env("SMTP_FROM"),
                to_addr=_env("SMTP_TO"),
            ),
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def configure_logging(self) -> None:
        """Set up *loguru* with the configured log level."""
        logger.remove()  # remove default stderr handler
        logger.add(
            sys.stderr,
            level=self.log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        logger.add(
            str(log_dir / "tradebot_{time:YYYY-MM-DD}.log"),
            level=self.log_level,
            rotation="00:00",
            retention="30 days",
            compression="zip",
        )
        logger.info(
            "Logging configured — level={}, mode={}",
            self.log_level,
            self.trading_mode,
        )

    @property
    def crypto_capital(self) -> float:
        """Capital allocated to crypto trading."""
        return self.initial_capital * self.crypto_allocation

    @property
    def stock_capital(self) -> float:
        """Capital allocated to stock trading."""
        return self.initial_capital * self.stock_allocation

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Settings(mode={self.trading_mode!r}, capital=${self.initial_capital:.2f} "
            f"[crypto=${self.crypto_capital:.2f}/stocks=${self.stock_capital:.2f}], "
            f"binance_testnet={self.binance.testnet}, alpaca_paper={self.alpaca.paper})"
        )


# ── Module-level singleton ──────────────────────────────────────────────
settings: Settings = Settings.from_env()
