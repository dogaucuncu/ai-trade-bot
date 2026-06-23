"""
AI Trade Bot — FastAPI web dashboard.

Provides a real-time web UI and REST API for monitoring and controlling
the trading bot.  Endpoints cover bot status, account information, open
positions, trade history, strategy signals, and performance statistics.

A WebSocket endpoint at ``/ws`` pushes live updates to connected clients.

Usage
-----
Run standalone for development::

    uvicorn dashboard.app:app --host 127.0.0.1 --port 8000 --reload

Or import and mount within the main application.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request

# ── Project imports ─────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import settings  # noqa: E402


# =========================================================================
# Auth / origin policy
# =========================================================================
# Control-endpoint token. Use the configured value, else mint a per-process
# random one. The local UI receives it via the rendered page (same-origin), so
# it keeps working; cross-site/CSRF callers cannot read it and are rejected.
DASHBOARD_TOKEN: str = settings.dashboard_token or secrets.token_urlsafe(32)
if not settings.dashboard_token:
    logger.warning(
        "[dashboard] No DASHBOARD_TOKEN configured — generated an ephemeral "
        "token for this run. The local UI is authorized automatically; "
        "external callers are blocked. Set DASHBOARD_TOKEN in config/.env for "
        "a stable token."
    )

# Origins the browser UI may legitimately come from (same host, bound port).
_ALLOWED_ORIGINS: list[str] = [
    f"http://{settings.dashboard_host}:{settings.dashboard_port}",
    f"http://127.0.0.1:{settings.dashboard_port}",
    f"http://localhost:{settings.dashboard_port}",
]
# Host header values accepted (DNS-rebinding mitigation). Starlette compares the
# hostname without the port.
_ALLOWED_HOSTS: list[str] = list(
    {settings.dashboard_host, "127.0.0.1", "localhost"}
)

# Symbols/timeframes the chart endpoint will fetch. The symbol is interpolated
# into the Binance REST URL, so constrain it to the configured universe rather
# than passing arbitrary user input through.
_ALLOWED_SYMBOLS: frozenset[str] = frozenset(
    set(settings.binance.default_pairs) | set(settings.alpaca.default_symbols)
)
_ALLOWED_TIMEFRAMES: frozenset[str] = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
)


def require_token(
    x_dashboard_token: str | None = Header(default=None),
) -> None:
    """Reject state-changing calls that lack the correct dashboard token."""
    if not secrets.compare_digest(x_dashboard_token or "", DASHBOARD_TOKEN):
        raise HTTPException(
            status_code=403, detail="Invalid or missing dashboard token"
        )


# =========================================================================
# WebSocket connection manager
# =========================================================================

class ConnectionManager:
    """Track active WebSocket connections and broadcast updates."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("[ws] Client connected — {} total", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("[ws] Client disconnected — {} remaining", len(self._connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a JSON message to every connected client."""
        payload = json.dumps(message, default=str)
        stale: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)

    @property
    def client_count(self) -> int:
        return len(self._connections)


ws_manager = ConnectionManager()


# =========================================================================
# Shared state — populated when the bot engine is attached
# =========================================================================

class DashboardState:
    """Mutable state container shared between dashboard and bot engine."""

    def __init__(self) -> None:
        self.engine: Any = None
        self.start_time: datetime = datetime.now(timezone.utc)
        self.recent_signals: list[dict[str, Any]] = []
        self.recent_trades: list[dict[str, Any]] = []
        self.daily_equity_curve: list[dict[str, Any]] = []
        self._broadcast_task: asyncio.Task[None] | None = None

    def attach_engine(self, engine: Any) -> None:
        """Attach a live :class:`BotEngine` instance."""
        self.engine = engine
        logger.info("[dashboard] Engine attached")

    async def start_broadcast_loop(self) -> None:
        """Periodically push status to WebSocket clients."""
        while True:
            try:
                if ws_manager.client_count > 0:
                    await ws_manager.broadcast(self.get_status())
            except Exception:
                logger.exception("[dashboard] Broadcast error")
            await asyncio.sleep(2.0)

    # -- snapshot builders ------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Build a combined status snapshot."""
        engine = self.engine
        now = datetime.now(timezone.utc)
        uptime_s = (now - self.start_time).total_seconds()

        if engine is not None:
            return {
                "type": "status_update",
                "running": engine._running,
                "mode": settings.trading_mode,
                "uptime_seconds": uptime_s,
                "tick": engine._tick_count,
                "circuit_breaker": engine.circuit_breaker.state.value,
                "open_positions": len(engine._positions),
                "daily_pnl": engine.risk_manager.get_daily_pnl(),
                "timestamp": now.isoformat(),
            }
        return {
            "type": "status_update",
            "running": False,
            "mode": settings.trading_mode,
            "uptime_seconds": uptime_s,
            "tick": 0,
            "circuit_breaker": "NORMAL",
            "open_positions": 0,
            "daily_pnl": 0.0,
            "timestamp": now.isoformat(),
        }

    def add_signal(self, signal_dict: dict[str, Any]) -> None:
        """Record a signal (kept in memory, capped at 200)."""
        self.recent_signals.insert(0, signal_dict)
        self.recent_signals = self.recent_signals[:200]

    def add_trade(self, trade_dict: dict[str, Any]) -> None:
        """Record a trade (kept in memory, capped at 200)."""
        self.recent_trades.insert(0, trade_dict)
        self.recent_trades = self.recent_trades[:200]

    def record_equity(self, balance: float) -> None:
        """Append a point to the equity curve."""
        self.daily_equity_curve.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "balance": balance,
        })
        # Keep last 1 440 points (~24 h at 1-min ticks)
        self.daily_equity_curve = self.daily_equity_curve[-1440:]


state = DashboardState()


# =========================================================================
# Lifespan — start background broadcast loop
# =========================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):  # noqa: ANN201
    """Start/stop background tasks alongside the FastAPI app."""
    state._broadcast_task = asyncio.create_task(state.start_broadcast_loop())
    logger.info("[dashboard] Broadcast loop started")
    yield
    if state._broadcast_task:
        state._broadcast_task.cancel()
        try:
            await state._broadcast_task
        except asyncio.CancelledError:
            pass
    logger.info("[dashboard] Broadcast loop stopped")


# =========================================================================
# FastAPI application
# =========================================================================

app = FastAPI(
    title="AI Trade Bot Dashboard",
    version="0.1.0",
    lifespan=lifespan,
)

# -- Host / CORS hardening ------------------------------------------------
# Reject requests with an unexpected Host header (DNS-rebinding defense).
app.add_middleware(TrustedHostMiddleware, allowed_hosts=_ALLOWED_HOSTS)

# Lock CORS to the dashboard's own origin. No wildcard, no credentials — the
# old ["*"] + allow_credentials let any website read account/position data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Dashboard-Token"],
)

# -- Static files / templates --------------------------------------------
_DASHBOARD_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _DASHBOARD_DIR / "static"
_TEMPLATE_DIR = _DASHBOARD_DIR / "templates"

_STATIC_DIR.mkdir(parents=True, exist_ok=True)
_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


# =========================================================================
# Routes — pages
# =========================================================================

@app.get("/")
async def index(request: Request):
    """Render the main dashboard page."""
    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "version": "0.1.0",
            "dashboard_token": DASHBOARD_TOKEN,
        },
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# =========================================================================
# Routes — REST API
# =========================================================================

@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Bot status: running/stopped, mode, uptime."""
    return state.get_status()


@app.get("/api/account")
async def api_account() -> dict[str, Any]:
    """Account balance, total P&L, daily P&L."""
    engine = state.engine
    if engine is None:
        return {
            "balance": settings.initial_capital,
            "initial_capital": settings.initial_capital,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_pct": 0.0,
            "crypto_capital": settings.crypto_capital,
            "stock_capital": settings.stock_capital,
        }

    try:
        balance = await engine._get_balance()
    except Exception:
        balance = engine.capital

    daily_pnl = engine.risk_manager.get_daily_pnl()
    total_pnl = balance - settings.initial_capital

    return {
        "balance": round(balance, 2),
        "initial_capital": settings.initial_capital,
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(
            (total_pnl / settings.initial_capital) * 100
            if settings.initial_capital > 0 else 0.0, 2,
        ),
        "daily_pnl": round(daily_pnl, 2),
        "daily_pnl_pct": round(
            (daily_pnl / settings.initial_capital) * 100
            if settings.initial_capital > 0 else 0.0, 2,
        ),
        "crypto_capital": round(settings.crypto_capital, 2),
        "stock_capital": round(settings.stock_capital, 2),
    }


@app.get("/api/positions")
async def api_positions() -> list[dict[str, Any]]:
    """List open positions with unrealised P&L."""
    engine = state.engine
    if engine is None:
        return []

    positions: list[dict[str, Any]] = []
    for pid, pos in engine._positions.items():
        positions.append({
            "position_id": pid,
            "symbol": pos.symbol,
            "side": pos.side.value,
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "unrealized_pnl": round(pos.unrealized_pnl, 4),
            "entry_time": pos.entry_time.isoformat(),
            "strategy": pos.strategy_name,
        })
    return positions


@app.get("/api/chart_data")
async def api_chart_data(symbol: str = "DOGE/USDT", timeframe: str = "1h", limit: int = 300) -> list[dict[str, Any]]:
    """Return historical OHLCV data for TradingView Lightweight Charts."""
    # Constrain inputs: symbol must be in the configured universe, timeframe must
    # be a known interval, limit is clamped. The symbol flows into a Binance URL.
    if symbol not in _ALLOWED_SYMBOLS or timeframe not in _ALLOWED_TIMEFRAMES:
        logger.warning(
            "[dashboard] chart_data rejected unknown symbol/timeframe: {} {}",
            symbol, timeframe,
        )
        return []
    limit = max(1, min(int(limit), 1000))

    engine = state.engine
    if engine is None:
        return []

    try:
        # Fetch the dataframe from the live engine
        df = await engine._fetch_data(symbol, timeframe, limit=limit)
        if df is None or df.empty:
            return []

        # Convert to lightweight-charts compatible format
        # { time: 1610000000, open: 1, high: 2, low: 0, close: 1.5 }
        # time must be unix timestamp in seconds
        
        result = []
        for _, row in df.iterrows():
            ts = int(row["timestamp"].timestamp())
            result.append({
                "time": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"])
            })
        return result
    except Exception as e:
        logger.error(f"[dashboard] Failed to fetch chart data: {e}")
        return []


async def _fetch_db_trades(limit: int = 200) -> list[dict[str, Any]]:
    """Return recent trades from the engine's DB (source of truth).

    The engine persists every closed position via ``storage.save_trade``; the
    dashboard reads from there so history survives restarts. Falls back to the
    in-memory list only if no engine/storage is attached. Maps ``side`` →
    ``action`` so the frontend's existing rendering works unchanged.
    """
    engine = state.engine
    storage = getattr(engine, "storage", None) if engine is not None else None
    if storage is not None:
        try:
            rows = await storage.get_trades(limit=limit)
            return [
                {
                    "timestamp": r["timestamp"],
                    "symbol": r["symbol"],
                    "side": r["side"],
                    "action": (r["side"] or "").upper(),
                    "price": r["price"],
                    "quantity": r["quantity"],
                    "pnl": r["pnl"] or 0.0,
                    "strategy": r["strategy"],
                    "status": r["status"],
                }
                for r in rows
            ]
        except Exception:
            logger.exception("[dashboard] DB trade fetch failed — using in-memory")
    return list(state.recent_trades)


@app.get("/api/trades")
async def api_trades() -> list[dict[str, Any]]:
    """Recent trade history (last 50) — from the persisted DB."""
    trades = await _fetch_db_trades(limit=50)
    return trades[:50]


@app.get("/api/signals")
async def api_signals() -> list[dict[str, Any]]:
    """Recent signals from strategies (last 50)."""
    return state.recent_signals[:50]


async def _fetch_db_equity_curve(limit: int = 2000) -> list[dict[str, Any]]:
    """Return the equity curve from the engine's DB (falls back to in-memory)."""
    engine = state.engine
    storage = getattr(engine, "storage", None) if engine is not None else None
    if storage is not None:
        try:
            return await storage.get_equity_curve(limit=limit)
        except Exception:
            logger.exception("[dashboard] DB equity fetch failed — using in-memory")
    return state.daily_equity_curve


@app.get("/api/performance")
async def api_performance() -> dict[str, Any]:
    """Daily / weekly / monthly performance statistics (from the persisted DB)."""
    trades = await _fetch_db_trades(limit=1000)
    equity_curve = await _fetch_db_equity_curve()
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "equity_curve": equity_curve,
            "strategy_stats": {},
        }

    pnls = [t.get("pnl", 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_wins = sum(wins) if wins else 0.0
    total_losses = abs(sum(losses)) if losses else 0.0

    # Per-strategy breakdown
    strategy_stats: dict[str, dict[str, Any]] = {}
    for t in trades:
        s = t.get("strategy", "unknown")
        if s not in strategy_stats:
            strategy_stats[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
        strategy_stats[s]["trades"] += 1
        pnl = t.get("pnl", 0.0)
        strategy_stats[s]["pnl"] += pnl
        if pnl > 0:
            strategy_stats[s]["wins"] += 1

    for s_data in strategy_stats.values():
        total = s_data["trades"]
        s_data["win_rate"] = round(
            (s_data["wins"] / total) * 100 if total > 0 else 0.0, 1,
        )
        s_data["pnl"] = round(s_data["pnl"], 4)

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(
            (len(wins) / len(trades)) * 100 if trades else 0.0, 1,
        ),
        "total_pnl": round(sum(pnls), 4),
        "avg_win": round(
            (total_wins / len(wins)) if wins else 0.0, 4,
        ),
        "avg_loss": round(
            (total_losses / len(losses)) if losses else 0.0, 4,
        ),
        "profit_factor": round(
            (total_wins / total_losses) if total_losses > 0 else 0.0, 2,
        ),
        "best_trade": round(max(pnls) if pnls else 0.0, 4),
        "worst_trade": round(min(pnls) if pnls else 0.0, 4),
        "equity_curve": equity_curve,
        "strategy_stats": strategy_stats,
    }


# =========================================================================
# Routes — bot controls
# =========================================================================

@app.post("/api/bot/start")
async def api_bot_start(_: None = Depends(require_token)) -> dict[str, Any]:
    """Start the bot engine."""
    engine = state.engine
    if engine is None:
        return {"success": False, "message": "No engine attached to dashboard"}

    if engine._running:
        return {"success": False, "message": "Bot is already running"}

    # Launch the engine in a background task
    asyncio.create_task(engine.run())
    state.start_time = datetime.now(timezone.utc)

    await ws_manager.broadcast({
        "type": "notification",
        "title": "Bot Started",
        "message": f"Trading bot started in {settings.trading_mode} mode",
        "level": "success",
    })

    logger.info("[dashboard] Bot started via API")
    return {"success": True, "message": "Bot started"}


@app.post("/api/bot/stop")
async def api_bot_stop(_: None = Depends(require_token)) -> dict[str, Any]:
    """Stop the bot engine gracefully."""
    engine = state.engine
    if engine is None:
        return {"success": False, "message": "No engine attached to dashboard"}

    if not engine._running:
        return {"success": False, "message": "Bot is not running"}

    await engine.shutdown()

    await ws_manager.broadcast({
        "type": "notification",
        "title": "Bot Stopped",
        "message": "Trading bot stopped gracefully",
        "level": "warning",
    })

    logger.info("[dashboard] Bot stopped via API")
    return {"success": True, "message": "Bot stopped"}


@app.post("/api/bot/emergency-stop")
async def api_emergency_stop(_: None = Depends(require_token)) -> dict[str, Any]:
    """Trigger the circuit breaker — close all positions immediately."""
    engine = state.engine
    if engine is None:
        return {"success": False, "message": "No engine attached to dashboard"}

    engine.circuit_breaker.trigger_emergency_stop(reason="Dashboard emergency stop")

    await ws_manager.broadcast({
        "type": "notification",
        "title": "🚨 EMERGENCY STOP",
        "message": "Circuit breaker triggered — closing all positions",
        "level": "error",
    })

    logger.critical("[dashboard] EMERGENCY STOP triggered via API")
    return {"success": True, "message": "Emergency stop triggered"}


# =========================================================================
# WebSocket — real-time updates
# =========================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time dashboard updates."""
    # CORS does not apply to WebSockets, so a malicious page could otherwise
    # open ws://127.0.0.1/ws and read live account/P&L data. Browsers always
    # send an Origin header on WS handshakes; reject any that isn't ours.
    # (Non-browser clients omit Origin and are allowed.)
    origin = ws.headers.get("origin")
    if origin is not None and origin not in _ALLOWED_ORIGINS:
        await ws.close(code=1008)
        logger.warning("[ws] Rejected connection from origin {}", origin)
        return

    await ws_manager.connect(ws)
    try:
        # Send initial state on connect
        await ws.send_text(json.dumps(state.get_status(), default=str))

        # Keep connection alive; listen for client messages
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                elif msg_type == "get_status":
                    await ws.send_text(
                        json.dumps(state.get_status(), default=str)
                    )
                else:
                    logger.debug("[ws] Unknown message type: {}", msg_type)

            except json.JSONDecodeError:
                logger.warning("[ws] Invalid JSON from client")

    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)
        logger.exception("[ws] WebSocket error")


# =========================================================================
# Startup helper — attach engine and run uvicorn
# =========================================================================

def run_dashboard(engine: Any | None = None) -> None:
    """Convenience function to start the dashboard with an optional engine.

    Parameters
    ----------
    engine : BotEngine | None
        The live bot engine to attach.
    """
    import uvicorn

    if engine is not None:
        state.attach_engine(engine)

    logger.info(
        "[dashboard] Starting on {}:{}",
        settings.dashboard_host,
        settings.dashboard_port,
    )
    uvicorn.run(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="info",
    )
