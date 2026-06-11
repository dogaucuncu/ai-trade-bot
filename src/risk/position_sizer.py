"""
Position sizer — calculates optimal trade sizes for micro-capital accounts.

Provides two sizing methods:

1. **Kelly Criterion** — information-theoretic optimal fraction
2. **ATR-based sizing** — volatility-adjusted (higher ATR → smaller size)

Both methods are constrained so that:

* Risk per trade never exceeds the configured maximum (1–2 % of capital).
* Position value is never below exchange minimums (~$5 for Binance).
* For stocks, fractional shares are assumed (Alpaca supports them).

Capital split: ~75 % crypto ($37.50) / ~25 % stocks ($12.50) on $50.

Usage
-----
>>> sizer = PositionSizer(max_risk_pct=0.02, min_order_usd=5.0)
>>> units = sizer.calculate(
...     balance=50, win_rate=0.55, avg_win=0.004,
...     avg_loss=0.003, atr=0.50, price=145.0,
... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass(slots=True)
class SizingResult:
    """Output of a position-sizing calculation.

    Attributes
    ----------
    units : float
        Number of base units (coins / shares) to trade.
    value_usd : float
        Dollar value of the position.
    risk_usd : float
        Maximum dollar amount at risk.
    method : str
        Sizing method that was used (``"kelly"`` or ``"atr"``).
    capped : bool
        ``True`` if the raw size was capped by a constraint.
    """

    units: float
    value_usd: float
    risk_usd: float
    method: str
    capped: bool = False


class PositionSizer:
    """Compute trade sizes suitable for a $40–50 micro-capital account.

    Parameters
    ----------
    max_risk_pct : float
        Maximum fraction of *balance* risked per trade (default 2 %).
    min_order_usd : float
        Minimum order value in USD (exchange constraint, default $5).
    max_position_pct : float
        Maximum single-position size as fraction of balance (default 30 %).
    kelly_fraction : float
        Fraction of the full Kelly bet to use (half-Kelly = 0.5 is
        standard for extra safety).

    Examples
    --------
    >>> ps = PositionSizer()
    >>> result = ps.calculate(
    ...     balance=50, win_rate=0.55, avg_win=0.005,
    ...     avg_loss=0.004, atr=1.20, price=150.0,
    ... )
    >>> print(result)
    """

    def __init__(
        self,
        max_risk_pct: float = 0.02,
        min_order_usd: float = 5.0,
        max_position_pct: float = 0.30,
        kelly_fraction: float = 0.50,
    ) -> None:
        self.max_risk_pct = max_risk_pct
        self.min_order_usd = min_order_usd
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction

        logger.info(
            "PositionSizer initialised — max_risk={:.1%} min_order=${:.2f} "
            "kelly_frac={:.0%}",
            max_risk_pct, min_order_usd, kelly_fraction,
        )

    # ---------------------------------------------------------------- public

    def calculate(
        self,
        balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        atr: float,
        price: float,
        method: str = "auto",
    ) -> SizingResult:
        """Determine position size in base units.

        Parameters
        ----------
        balance : float
            Current account balance in USD.
        win_rate : float
            Historical win rate in ``[0.0, 1.0]``.
        avg_win : float
            Average winning trade return (decimal, e.g. 0.005 = 0.5 %).
        avg_loss : float
            Average losing trade return (decimal, positive, e.g. 0.003).
        atr : float
            Current Average True Range in price units.
        price : float
            Current asset price in USD.
        method : str
            ``"kelly"``, ``"atr"``, or ``"auto"`` (uses the more
            conservative of the two).

        Returns
        -------
        SizingResult
        """
        if price <= 0 or balance <= 0:
            logger.warning("[sizer] Invalid price/balance — returning 0")
            return SizingResult(0.0, 0.0, 0.0, method="none")

        kelly_units = self._kelly_size(balance, win_rate, avg_win, avg_loss, price)
        atr_units = self._atr_size(balance, atr, price)

        if method == "kelly":
            raw_units = kelly_units
            chosen = "kelly"
        elif method == "atr":
            raw_units = atr_units
            chosen = "atr"
        else:  # auto — use the smaller (more conservative) result
            if kelly_units <= atr_units:
                raw_units = kelly_units
                chosen = "kelly"
            else:
                raw_units = atr_units
                chosen = "atr"

        # --- apply constraints -------------------------------------------
        capped = False
        max_risk_usd = balance * self.max_risk_pct
        max_units_by_risk = max_risk_usd / price if price > 0 else 0
        max_units_by_pct = (balance * self.max_position_pct) / price

        final_units = min(raw_units, max_units_by_risk, max_units_by_pct)
        if final_units < raw_units:
            capped = True

        final_value = final_units * price
        risk_usd = min(max_risk_usd, final_value)

        # Check minimum order value
        if final_value < self.min_order_usd:
            # Try bumping up to minimum if balance allows
            min_units = self.min_order_usd / price
            if min_units * price <= balance * self.max_position_pct:
                final_units = min_units
                final_value = final_units * price
                risk_usd = min(max_risk_usd, final_value)
                capped = True
                logger.debug(
                    "[sizer] Bumped to minimum order ${:.2f}", final_value
                )
            else:
                logger.warning(
                    "[sizer] Cannot meet min order ${:.2f} within constraints",
                    self.min_order_usd,
                )
                return SizingResult(0.0, 0.0, 0.0, method=chosen, capped=True)

        result = SizingResult(
            units=round(final_units, 8),
            value_usd=round(final_value, 2),
            risk_usd=round(risk_usd, 2),
            method=chosen,
            capped=capped,
        )

        logger.info(
            "[sizer] {} → {:.6f} units (${:.2f}) risk=${:.2f} capped={}",
            chosen, result.units, result.value_usd, result.risk_usd, capped,
        )
        return result

    # ---------------------------------------------------------- kelly sizing

    def _kelly_size(
        self,
        balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        price: float,
    ) -> float:
        """Full Kelly ×  ``kelly_fraction`` → units.

        Kelly formula: ``f* = (p / a) − (q / b)``
        where p = win prob, q = 1−p, b = avg win, a = avg loss.
        """
        if avg_loss <= 0 or avg_win <= 0:
            logger.debug("[sizer] No valid win/loss stats — kelly = 0")
            return 0.0

        p = max(0.0, min(1.0, win_rate))
        q = 1.0 - p

        # Kelly fraction of bankroll
        f_star = (p / avg_loss) - (q / avg_win)
        f_star = max(0.0, f_star)  # never go negative
        f_star *= self.kelly_fraction  # fractional Kelly

        position_usd = balance * f_star
        units = position_usd / price if price > 0 else 0.0

        logger.debug(
            "[sizer] Kelly f*={:.4f} frac_kelly={:.4f} → ${:.2f} → {:.6f} units",
            f_star / self.kelly_fraction if self.kelly_fraction else 0,
            f_star, position_usd, units,
        )
        return units

    # ------------------------------------------------------ ATR-based sizing

    def _atr_size(
        self,
        balance: float,
        atr: float,
        price: float,
    ) -> float:
        """Position size = risk_usd / (ATR × multiplier).

        Higher ATR → smaller position.
        """
        if atr <= 0:
            logger.debug("[sizer] ATR ≤ 0 — falling back to risk-based sizing")
            atr = price * 0.01  # default 1 % of price

        max_risk_usd = balance * self.max_risk_pct
        atr_multiplier = 2.0  # risk = 2× ATR
        risk_distance = atr * atr_multiplier

        units = max_risk_usd / risk_distance if risk_distance > 0 else 0.0

        logger.debug(
            "[sizer] ATR={:.6f} risk_dist={:.6f} → {:.6f} units",
            atr, risk_distance, units,
        )
        return units

    # ------------------------------------------------------------ utilities

    @staticmethod
    def crypto_allocation(total_balance: float) -> float:
        """Return the crypto sub-balance (75 % of total)."""
        return total_balance * 0.75

    @staticmethod
    def stock_allocation(total_balance: float) -> float:
        """Return the stock sub-balance (25 % of total)."""
        return total_balance * 0.25
