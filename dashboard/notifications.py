"""
Notification manager — email alerts, web-push, and rate limiting.

Provides asynchronous email notifications (via ``aiosmtplib``), an in-memory
web-push queue served through the dashboard WebSocket, and rate-limiting to
avoid spamming the trader's inbox.

Usage
-----
>>> from dashboard.notifications import NotificationManager
>>> nm = NotificationManager()
>>> await nm.send_trade_notification(trade_dict)
>>> await nm.send_daily_summary(stats_dict)
>>> await nm.send_emergency_alert("Max drawdown exceeded")
"""

from __future__ import annotations

import asyncio
import html
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import settings  # noqa: E402


# =========================================================================
# Data classes
# =========================================================================

@dataclass(slots=True)
class WebPushNotification:
    """A web-push notification queued for delivery via WebSocket."""

    title: str
    message: str
    notification_type: Literal["info", "success", "warning", "error"] = "info"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, str]:
        return {
            "type": "notification",
            "title": self.title,
            "message": self.message,
            "level": self.notification_type,
            "timestamp": self.timestamp,
        }


# =========================================================================
# HTML email templates
# =========================================================================

_BASE_STYLE = """
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #0a0e27;
           color: #e2e8f0; margin: 0; padding: 20px; }
    .container { max-width: 600px; margin: 0 auto;
                 background: rgba(255,255,255,0.05);
                 border-radius: 16px; padding: 30px;
                 border: 1px solid rgba(255,255,255,0.1); }
    .header { text-align: center; padding-bottom: 20px;
              border-bottom: 1px solid rgba(255,255,255,0.1); }
    .header h1 { color: #7c3aed; margin: 0; font-size: 24px; }
    .header p  { color: #94a3b8; margin: 5px 0 0; }
    .content   { padding: 20px 0; }
    .stat-row  { display: flex; justify-content: space-between;
                 padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .stat-label { color: #94a3b8; }
    .stat-value { font-weight: 600; }
    .positive   { color: #10b981; }
    .negative   { color: #ef4444; }
    .footer     { text-align: center; padding-top: 20px;
                  color: #64748b; font-size: 12px; }
    .alert-box  { padding: 15px; border-radius: 8px; margin: 15px 0; }
    .alert-success { background: rgba(16,185,129,0.15); border: 1px solid #10b981; }
    .alert-warning { background: rgba(245,158,11,0.15); border: 1px solid #f59e0b; }
    .alert-error   { background: rgba(239,68,68,0.15);  border: 1px solid #ef4444; }
    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    th    { text-align: left; color: #94a3b8; padding: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1); }
    td    { padding: 8px; }
</style>
"""


def _trade_email_html(trade: dict[str, Any]) -> str:
    """Build an HTML email body for a single trade notification."""
    pnl = trade.get("pnl", 0.0)
    pnl_class = "positive" if pnl >= 0 else "negative"
    pnl_sign = "+" if pnl >= 0 else ""
    action = trade.get("action", "UNKNOWN")
    symbol = html.escape(str(trade.get("symbol", "N/A")))
    price = trade.get("price", 0.0)
    quantity = trade.get("quantity", 0.0)
    strategy = html.escape(str(trade.get("strategy", "N/A")))
    timestamp = trade.get("timestamp", datetime.now(timezone.utc).isoformat())

    return f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
    <div class="container">
        <div class="header">
            <h1>🤖 Trade Alert</h1>
            <p>{timestamp}</p>
        </div>
        <div class="content">
            <div class="stat-row">
                <span class="stat-label">Action</span>
                <span class="stat-value">{html.escape(str(action))}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Symbol</span>
                <span class="stat-value">{symbol}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Price</span>
                <span class="stat-value">${price:.4f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Quantity</span>
                <span class="stat-value">{quantity:.6f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Strategy</span>
                <span class="stat-value">{strategy}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">P&L</span>
                <span class="stat-value {pnl_class}">{pnl_sign}${pnl:.4f}</span>
            </div>
        </div>
        <div class="footer">AI Trade Bot — Automated Trading System</div>
    </div>
    </body></html>"""


def _daily_summary_html(stats: dict[str, Any]) -> str:
    """Build an HTML email body for the end-of-day summary."""
    total_pnl = stats.get("total_pnl", 0.0)
    pnl_class = "positive" if total_pnl >= 0 else "negative"
    pnl_sign = "+" if total_pnl >= 0 else ""
    win_rate = stats.get("win_rate", 0.0)
    total_trades = stats.get("total_trades", 0)
    best = stats.get("best_trade", 0.0)
    worst = stats.get("worst_trade", 0.0)
    balance = stats.get("balance", settings.initial_capital)

    return f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
    <div class="container">
        <div class="header">
            <h1>📊 Daily Summary</h1>
            <p>{datetime.now(timezone.utc).strftime('%Y-%m-%d')}</p>
        </div>
        <div class="content">
            <div class="stat-row">
                <span class="stat-label">Total P&L</span>
                <span class="stat-value {pnl_class}">{pnl_sign}${total_pnl:.4f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Current Balance</span>
                <span class="stat-value">${balance:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Win Rate</span>
                <span class="stat-value">{win_rate:.1f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Trades</span>
                <span class="stat-value">{total_trades}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Best Trade</span>
                <span class="stat-value positive">+${best:.4f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Worst Trade</span>
                <span class="stat-value negative">${worst:.4f}</span>
            </div>
        </div>
        <div class="footer">AI Trade Bot — Automated Trading System</div>
    </div>
    </body></html>"""


def _emergency_email_html(reason: str) -> str:
    """Build an HTML email body for an emergency alert."""
    safe_reason = html.escape(reason)
    return f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
    <div class="container">
        <div class="header">
            <h1>🚨 EMERGENCY ALERT</h1>
            <p>{datetime.now(timezone.utc).isoformat()}</p>
        </div>
        <div class="content">
            <div class="alert-box alert-error">
                <strong>Critical Alert</strong><br>
                {safe_reason}
            </div>
            <p style="color:#94a3b8; text-align:center;">
                The circuit breaker has been triggered. All positions are being
                closed immediately. Manual review required.
            </p>
        </div>
        <div class="footer">AI Trade Bot — Automated Trading System</div>
    </div>
    </body></html>"""


# =========================================================================
# NotificationManager
# =========================================================================

class NotificationManager:
    """Unified notification hub — email, web-push, and rate limiting.

    Parameters
    ----------
    max_emails_per_hour : int
        Rate limit for outbound emails (default 10).
    email_enabled : bool
        Set ``False`` to skip email sending entirely (e.g. in tests).

    Examples
    --------
    >>> nm = NotificationManager()
    >>> await nm.send_trade_notification({"symbol": "SOL/USDT", ...})
    >>> await nm.push_notification("Heads up", "Something happened")
    """

    def __init__(
        self,
        max_emails_per_hour: int = 10,
        email_enabled: bool | None = None,
    ) -> None:
        self.max_emails_per_hour = max_emails_per_hour
        self.email_enabled: bool = (
            email_enabled if email_enabled is not None
            else bool(settings.smtp.username and settings.smtp.password)
        )

        # Rate-limit tracking: timestamps of sent emails
        self._email_timestamps: deque[float] = deque(maxlen=max_emails_per_hour)

        # Async email queue
        self._email_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._queue_task: asyncio.Task[None] | None = None

        # Web-push notification buffer (served via WebSocket)
        self._push_buffer: deque[dict[str, str]] = deque(maxlen=100)

        logger.info(
            "NotificationManager initialised — email={} rate_limit={}/h",
            self.email_enabled,
            max_emails_per_hour,
        )

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Start the background email-sending worker."""
        if self._queue_task is None or self._queue_task.done():
            self._queue_task = asyncio.create_task(self._email_worker())
            logger.info("[notify] Email worker started")

    async def stop(self) -> None:
        """Stop the background email worker gracefully."""
        if self._queue_task and not self._queue_task.done():
            self._queue_task.cancel()
            try:
                await self._queue_task
            except asyncio.CancelledError:
                pass
            logger.info("[notify] Email worker stopped")

    # ---------------------------------------------------------------- email

    async def send_email(self, subject: str, body_html: str) -> bool:
        """Send an HTML email via SMTP (queued, rate-limited).

        Parameters
        ----------
        subject : str
            Email subject line.
        body_html : str
            Full HTML body.

        Returns
        -------
        bool
            ``True`` if the email was queued (may still fail on send).
        """
        if not self.email_enabled:
            logger.debug("[notify] Email disabled — skipping: {}", subject)
            return False

        if not self._check_rate_limit():
            logger.warning(
                "[notify] Email rate limit reached ({}/h) — dropping: {}",
                self.max_emails_per_hour, subject,
            )
            return False

        await self._email_queue.put((subject, body_html))
        logger.debug("[notify] Email queued: {}", subject)
        return True

    async def send_trade_notification(self, trade: dict[str, Any]) -> None:
        """Send a formatted trade-alert email and push notification.

        Parameters
        ----------
        trade : dict
            Trade dict with keys: symbol, action, price, quantity, pnl,
            strategy, timestamp.
        """
        pnl = trade.get("pnl", 0.0)
        symbol = trade.get("symbol", "N/A")
        action = trade.get("action", "TRADE")
        pnl_sign = "+" if pnl >= 0 else ""

        subject = f"Trade: {action} {symbol} — {pnl_sign}${pnl:.4f}"
        body = _trade_email_html(trade)
        await self.send_email(subject, body)

        level: Literal["info", "success", "warning", "error"] = (
            "success" if pnl >= 0 else "warning"
        )
        await self.push_notification(
            title=f"{action} {symbol}",
            message=f"P&L: {pnl_sign}${pnl:.4f}",
            notification_type=level,
        )

    async def send_daily_summary(self, stats: dict[str, Any]) -> None:
        """Send an end-of-day performance summary email.

        Parameters
        ----------
        stats : dict
            Performance stats (total_pnl, win_rate, total_trades, etc.).
        """
        total_pnl = stats.get("total_pnl", 0.0)
        pnl_sign = "+" if total_pnl >= 0 else ""

        subject = (
            f"Daily Summary — {pnl_sign}${total_pnl:.4f} | "
            f"{stats.get('total_trades', 0)} trades"
        )
        body = _daily_summary_html(stats)
        await self.send_email(subject, body)

        await self.push_notification(
            title="📊 Daily Summary",
            message=f"P&L: {pnl_sign}${total_pnl:.4f} — "
                    f"{stats.get('total_trades', 0)} trades",
            notification_type="info",
        )

    async def send_emergency_alert(self, reason: str) -> None:
        """Send an immediate emergency alert email (bypasses rate limit).

        Parameters
        ----------
        reason : str
            Human-readable reason for the emergency.
        """
        subject = f"🚨 EMERGENCY — {reason}"
        body = _emergency_email_html(reason)

        # Bypass rate limit for emergencies
        if self.email_enabled:
            await self._email_queue.put((subject, body))
            logger.critical("[notify] Emergency email queued: {}", reason)

        await self.push_notification(
            title="🚨 EMERGENCY",
            message=reason,
            notification_type="error",
        )

    # ---------------------------------------------------------------- web-push

    async def push_notification(
        self,
        title: str,
        message: str,
        notification_type: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        """Queue a web-push notification for delivery via WebSocket.

        Parameters
        ----------
        title : str
            Notification title.
        message : str
            Notification body text.
        notification_type : str
            One of ``'info'``, ``'success'``, ``'warning'``, ``'error'``.
        """
        notif = WebPushNotification(
            title=title,
            message=message,
            notification_type=notification_type,
        )
        self._push_buffer.append(notif.to_dict())

        # Also broadcast immediately via the dashboard WebSocket manager
        try:
            from dashboard.app import ws_manager
            await ws_manager.broadcast(notif.to_dict())
        except Exception:
            logger.debug("[notify] WebSocket broadcast unavailable")

        logger.info(
            "[notify] Push: [{}] {} — {}",
            notification_type.upper(), title, message,
        )

    def get_recent_notifications(
        self, limit: int = 50
    ) -> list[dict[str, str]]:
        """Return the most recent web-push notifications.

        Parameters
        ----------
        limit : int
            Maximum number of notifications to return.

        Returns
        -------
        list[dict]
            Notification dicts, newest first.
        """
        items = list(self._push_buffer)
        items.reverse()
        return items[:limit]

    # ---------------------------------------------------------------- internals

    def _check_rate_limit(self) -> bool:
        """Return ``True`` if we haven't exceeded the hourly email cap."""
        now = time.monotonic()
        # Purge timestamps older than 1 hour
        while self._email_timestamps and (now - self._email_timestamps[0]) > 3600:
            self._email_timestamps.popleft()
        return len(self._email_timestamps) < self.max_emails_per_hour

    async def _email_worker(self) -> None:
        """Background coroutine that drains the email queue."""
        while True:
            try:
                subject, body_html = await self._email_queue.get()
                await self._send_smtp(subject, body_html)
                self._email_timestamps.append(time.monotonic())
                self._email_queue.task_done()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[notify] Email worker error")
                await asyncio.sleep(5.0)

    async def _send_smtp(self, subject: str, body_html: str) -> None:
        """Actually send an email via SMTP using aiosmtplib."""
        try:
            import aiosmtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
        except ImportError:
            logger.error(
                "[notify] aiosmtplib not installed — pip install aiosmtplib"
            )
            return

        smtp_cfg = settings.smtp
        if not smtp_cfg.from_addr or not smtp_cfg.to_addr:
            logger.warning("[notify] SMTP from/to not configured — skipping")
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_cfg.from_addr
        msg["To"] = smtp_cfg.to_addr

        # Plain-text fallback
        plain = (
            f"{subject}\n\n"
            "This email requires an HTML-capable email client.\n"
            "— AI Trade Bot"
        )
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        try:
            await aiosmtplib.send(
                msg,
                hostname=smtp_cfg.host,
                port=smtp_cfg.port,
                username=smtp_cfg.username,
                password=smtp_cfg.password,
                start_tls=True,
            )
            logger.info("[notify] Email sent: {}", subject)
        except Exception as exc:
            logger.error("[notify] SMTP send failed: {} — {}", subject, exc)
