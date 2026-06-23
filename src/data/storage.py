"""
AI Trade Bot — Async SQLite storage layer.

Uses SQLAlchemy 2.x async engine with aiosqlite to persist candle data,
executed trades, and generated signals.

Usage::

    from src.data.storage import Storage

    storage = Storage(db_url="sqlite+aiosqlite:///data/tradebot.db")
    await storage.init_db()
    await storage.save_candles(df, symbol="DOGE/USDT", timeframe="5m")
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Sequence

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    delete,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


# =====================================================================
# ORM models
# =====================================================================

class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Candle(Base):
    """OHLCV candle row."""

    __tablename__ = "candles"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(30), nullable=False, index=True)
    timeframe: str = Column(String(5), nullable=False, index=True)
    timestamp: dt.datetime = Column(DateTime, nullable=False, index=True)
    open: float = Column(Float, nullable=False)
    high: float = Column(Float, nullable=False)
    low: float = Column(Float, nullable=False)
    close: float = Column(Float, nullable=False)
    volume: float = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
    )

    def __repr__(self) -> str:
        return (
            f"<Candle {self.symbol} {self.timeframe} "
            f"{self.timestamp} C={self.close}>"
        )


class Trade(Base):
    """Executed-trade record."""

    __tablename__ = "trades"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(30), nullable=False, index=True)
    side: str = Column(String(4), nullable=False)          # buy | sell
    price: float = Column(Float, nullable=False)
    quantity: float = Column(Float, nullable=False)
    timestamp: dt.datetime = Column(DateTime, nullable=False, index=True)
    strategy: str = Column(String(60), nullable=False, default="manual")
    pnl: float = Column(Float, nullable=True)               # realised P&L
    status: str = Column(String(20), nullable=False, default="filled")

    def __repr__(self) -> str:
        return (
            f"<Trade {self.side.upper()} {self.quantity} {self.symbol} "
            f"@ {self.price} pnl={self.pnl}>"
        )


class Signal(Base):
    """Strategy signal record."""

    __tablename__ = "signals"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(30), nullable=False, index=True)
    strategy: str = Column(String(60), nullable=False)
    signal_type: str = Column(String(10), nullable=False)   # buy | sell | hold
    confidence: float = Column(Float, nullable=True)
    timestamp: dt.datetime = Column(DateTime, nullable=False, index=True)
    signal_metadata: str = Column("metadata", Text, nullable=True)  # JSON blob

    def __repr__(self) -> str:
        return (
            f"<Signal {self.signal_type.upper()} {self.symbol} "
            f"conf={self.confidence:.2f}>"
        )


class EquitySnapshot(Base):
    """Point-in-time equity/balance snapshot for the live/paper run.

    Recording these lets us reconstruct an equity curve and compute honest
    performance metrics over a paper-trading session (see
    ``backtest.metrics.PerformanceReport``).
    """

    __tablename__ = "equity_snapshots"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: dt.datetime = Column(DateTime, nullable=False, index=True)
    equity: float = Column(Float, nullable=False)            # balance + unrealised
    balance: float = Column(Float, nullable=False)           # realised cash
    daily_pnl: float = Column(Float, nullable=False, default=0.0)
    open_positions: int = Column(Integer, nullable=False, default=0)
    mode: str = Column(String(10), nullable=False, default="paper")  # paper|live

    def __repr__(self) -> str:
        return f"<EquitySnapshot {self.timestamp} eq={self.equity:.2f}>"


class OpenPosition(Base):
    """Currently-open position, persisted so a restart can recover state.

    Rows are inserted when a position opens and deleted when it closes, so
    the table always reflects the live set of open positions.
    """

    __tablename__ = "open_positions"

    position_id: str = Column(String(40), primary_key=True)
    symbol: str = Column(String(30), nullable=False, index=True)
    side: str = Column(String(4), nullable=False)            # BUY | SELL
    entry_price: float = Column(Float, nullable=False)
    quantity: float = Column(Float, nullable=False)
    stop_loss: float = Column(Float, nullable=False)
    take_profit: float = Column(Float, nullable=False)
    entry_time: dt.datetime = Column(DateTime, nullable=False)
    strategy: str = Column(String(60), nullable=False, default="manual")
    value_usd: float = Column(Float, nullable=False, default=0.0)
    # Exchange order id of the resting protective stop-loss (live mode). Persisted
    # so a restart can cancel it before closing — avoids an orphaned opposite fill.
    stop_order_id: str = Column(String(40), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<OpenPosition {self.side} {self.quantity} {self.symbol} "
            f"@ {self.entry_price}>"
        )


# =====================================================================
# Storage service
# =====================================================================

class Storage:
    """Async persistence layer backed by SQLite (via aiosqlite).

    Parameters
    ----------
    db_url:
        SQLAlchemy async connection string, e.g.
        ``sqlite+aiosqlite:///data/tradebot.db``.
    echo:
        If ``True``, emit SQL statements to the log.
    """

    def __init__(self, db_url: str, *, echo: bool = False) -> None:
        self._engine = create_async_engine(db_url, echo=echo)
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Storage initialised — {}", db_url)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """Create all tables if they don't already exist, then run migrations."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(self._migrate_schema)
        logger.info("Database tables verified / created.")

    @staticmethod
    def _migrate_schema(conn) -> None:  # noqa: ANN001 — sync SQLAlchemy connection
        """Lightweight, idempotent schema migrations for existing databases.

        ``create_all`` never alters existing tables, so columns added after a DB
        was first created must be patched in by hand. SQLite supports
        ``ALTER TABLE ... ADD COLUMN`` for nullable columns.
        """
        from sqlalchemy import text

        # open_positions.stop_order_id (added for live-restart stop recovery)
        cols = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(open_positions)"))
        }
        if "stop_order_id" not in cols:
            conn.execute(
                text("ALTER TABLE open_positions ADD COLUMN stop_order_id VARCHAR(40)")
            )
            logger.info("[storage] Migrated: added open_positions.stop_order_id")

    async def close(self) -> None:
        """Dispose of the engine connection pool."""
        await self._engine.dispose()
        logger.info("Storage engine disposed.")

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    async def save_candles(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Upsert OHLCV candles from a DataFrame.

        Parameters
        ----------
        df:
            Must contain columns: timestamp, open, high, low, close, volume.
        symbol:
            Trading pair / ticker, e.g. ``"DOGE/USDT"`` or ``"AAPL"``.
        timeframe:
            Candle period, e.g. ``"5m"``, ``"1h"``.

        Returns
        -------
        int
            Number of new rows inserted (duplicates are skipped).
        """
        inserted = 0
        async with self._session_factory() as session:
            for _, row in df.iterrows():
                ts = pd.Timestamp(row["timestamp"]).to_pydatetime()

                # Check for existing candle (upsert logic)
                existing = await session.execute(
                    select(Candle).where(
                        Candle.symbol == symbol,
                        Candle.timeframe == timeframe,
                        Candle.timestamp == ts,
                    )
                )
                if existing.scalar_one_or_none() is not None:
                    continue

                session.add(
                    Candle(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
                )
                inserted += 1
            await session.commit()

        logger.debug(
            "Saved {}/{} candles for {} [{}]",
            inserted,
            len(df),
            symbol,
            timeframe,
        )
        return inserted

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        since: Optional[dt.datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve stored candles as a DataFrame.

        Parameters
        ----------
        symbol:
            Trading pair / ticker.
        timeframe:
            Candle period.
        limit:
            Maximum rows to return (most recent first).
        since:
            Optional lower-bound timestamp filter.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, open, high, low, close, volume.
        """
        stmt = (
            select(Candle)
            .where(Candle.symbol == symbol, Candle.timeframe == timeframe)
            .order_by(Candle.timestamp.desc())
            .limit(limit)
        )
        if since is not None:
            stmt = stmt.where(Candle.timestamp >= since)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows: Sequence[Candle] = result.scalars().all()

        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        data = [
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in reversed(rows)  # chronological order
        ]
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    async def save_trade(
        self,
        *,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        timestamp: dt.datetime | None = None,
        strategy: str = "manual",
        pnl: float | None = None,
        status: str = "filled",
    ) -> int:
        """Persist a single executed trade and return its row id."""
        trade = Trade(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=timestamp or dt.datetime.now(dt.timezone.utc),
            strategy=strategy,
            pnl=pnl,
            status=status,
        )
        async with self._session_factory() as session:
            session.add(trade)
            await session.commit()
            trade_id: int = trade.id  # populated after commit

        logger.info("Trade #{} saved — {} {} {} @ {}", trade_id, side, quantity, symbol, price)
        return trade_id

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return recent trades as a list of dicts."""
        stmt = select(Trade).order_by(Trade.timestamp.desc()).limit(limit)
        if symbol:
            stmt = stmt.where(Trade.symbol == symbol)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows: Sequence[Trade] = result.scalars().all()

        return [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "price": t.price,
                "quantity": t.quantity,
                "timestamp": t.timestamp,
                "strategy": t.strategy,
                "pnl": t.pnl,
                "status": t.status,
            }
            for t in rows
        ]

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    async def save_signal(
        self,
        *,
        symbol: str,
        strategy: str,
        signal_type: str,
        confidence: float = 0.0,
        timestamp: dt.datetime | None = None,
        signal_metadata: str | None = None,
    ) -> int:
        """Persist a strategy signal and return its row id."""
        sig = Signal(
            symbol=symbol,
            strategy=strategy,
            signal_type=signal_type,
            confidence=confidence,
            timestamp=timestamp or dt.datetime.now(dt.timezone.utc),
            signal_metadata=signal_metadata,
        )
        async with self._session_factory() as session:
            session.add(sig)
            await session.commit()
            sig_id: int = sig.id

        logger.debug(
            "Signal #{} saved — {} {} (conf={:.2f})",
            sig_id,
            signal_type,
            symbol,
            confidence,
        )
        return sig_id

    # ------------------------------------------------------------------
    # Equity snapshots
    # ------------------------------------------------------------------

    async def save_equity_snapshot(
        self,
        *,
        equity: float,
        balance: float,
        daily_pnl: float = 0.0,
        open_positions: int = 0,
        mode: str = "paper",
        timestamp: dt.datetime | None = None,
    ) -> int:
        """Persist an equity/balance snapshot and return its row id."""
        snap = EquitySnapshot(
            timestamp=timestamp or dt.datetime.now(dt.timezone.utc),
            equity=equity,
            balance=balance,
            daily_pnl=daily_pnl,
            open_positions=open_positions,
            mode=mode,
        )
        async with self._session_factory() as session:
            session.add(snap)
            await session.commit()
            snap_id: int = snap.id
        return snap_id

    async def get_equity_curve(
        self,
        limit: int = 100_000,
        since: Optional[dt.datetime] = None,
    ) -> list[dict]:
        """Return equity snapshots in chronological order."""
        stmt = select(EquitySnapshot).order_by(EquitySnapshot.timestamp.asc()).limit(limit)
        if since is not None:
            stmt = stmt.where(EquitySnapshot.timestamp >= since)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows: Sequence[EquitySnapshot] = result.scalars().all()

        return [
            {
                "timestamp": r.timestamp,
                "equity": r.equity,
                "balance": r.balance,
                "daily_pnl": r.daily_pnl,
                "open_positions": r.open_positions,
                "mode": r.mode,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Open positions (state persistence / recovery)
    # ------------------------------------------------------------------

    async def save_open_position(
        self,
        *,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        entry_time: dt.datetime,
        strategy: str = "manual",
        value_usd: float = 0.0,
        stop_order_id: str | None = None,
    ) -> None:
        """Insert or replace an open-position row."""
        async with self._session_factory() as session:
            # Replace any stale row with the same id (idempotent upsert).
            await session.execute(
                delete(OpenPosition).where(OpenPosition.position_id == position_id)
            )
            session.add(
                OpenPosition(
                    position_id=position_id,
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_time=entry_time,
                    strategy=strategy,
                    value_usd=value_usd,
                    stop_order_id=stop_order_id,
                )
            )
            await session.commit()
        logger.debug("Open position {} persisted ({} {})", position_id, side, symbol)

    async def delete_open_position(self, position_id: str) -> None:
        """Remove an open-position row once it has been closed."""
        async with self._session_factory() as session:
            await session.execute(
                delete(OpenPosition).where(OpenPosition.position_id == position_id)
            )
            await session.commit()
        logger.debug("Open position {} removed", position_id)

    async def get_open_positions(self) -> list[dict]:
        """Return all persisted open positions (for restart recovery)."""
        async with self._session_factory() as session:
            result = await session.execute(select(OpenPosition))
            rows: Sequence[OpenPosition] = result.scalars().all()

        return [
            {
                "position_id": r.position_id,
                "symbol": r.symbol,
                "side": r.side,
                "entry_price": r.entry_price,
                "quantity": r.quantity,
                "stop_loss": r.stop_loss,
                "take_profit": r.take_profit,
                "entry_time": r.entry_time,
                "strategy": r.strategy,
                "value_usd": r.value_usd,
                "stop_order_id": r.stop_order_id,
            }
            for r in rows
        ]
