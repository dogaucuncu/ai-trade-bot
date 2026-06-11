"""
Ensemble strategy — weighted consensus of multiple sub-strategies.

Collects signals from a list of child strategies, applies per-strategy
weights, and emits a consensus signal only when combined confidence
exceeds a minimum threshold.

Default weights
---------------
* scalping:       0.30
* mean_reversion: 0.30
* momentum:       0.40

Weights can be adjusted dynamically based on recent performance
(see :meth:`update_weights`).

Usage
-----
>>> from src.strategy.scalping import ScalpingStrategy
>>> from src.strategy.mean_reversion import MeanReversionStrategy
>>> from src.strategy.momentum import MomentumStrategy
>>> ensemble = EnsembleStrategy(
...     strategies=[ScalpingStrategy(), MeanReversionStrategy(), MomentumStrategy()]
... )
>>> signal = ensemble.analyze(df)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from loguru import logger

from src.strategy.base import BaseStrategy, Position, Signal, TradeAction


class EnsembleStrategy(BaseStrategy):
    """Weighted-voting ensemble of multiple trading strategies.

    Parameters
    ----------
    strategies : list[BaseStrategy]
        Child strategies whose signals are aggregated.
    weights : dict[str, float] | None
        ``{strategy_name: weight}`` override. Defaults are applied for
        unknown names (weight = 0.25).
    min_confidence : float
        Minimum aggregated confidence to produce a non-HOLD signal.
    config : dict | None
        Additional config overrides.

    Examples
    --------
    >>> ensemble = EnsembleStrategy(strategies=[ScalpingStrategy()])
    >>> sig = ensemble.analyze(df)
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "scalping": 0.30,
        "mean_reversion": 0.30,
        "momentum": 0.40,
    }

    def __init__(
        self,
        strategies: list[BaseStrategy],
        weights: dict[str, float] | None = None,
        min_confidence: float = 0.60,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name="ensemble", config=config)
        self.strategies = strategies
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.min_confidence = min_confidence
        # Performance tracker: strategy_name -> list[float] of recent PnL %
        self._recent_pnl: dict[str, list[float]] = defaultdict(list)
        self._max_pnl_history = 20  # rolling window per strategy

        logger.info(
            "Ensemble created with {} strategies, weights={}",
            len(strategies),
            self.weights,
        )

    # ------------------------------------------------------------------ core

    def analyze(self, df: pd.DataFrame) -> Signal:
        """Collect signals from all sub-strategies, perform weighted voting.

        The action is chosen by majority-weighted vote.  Confidence is the
        weighted average of the per-strategy confidences for the winning
        action.
        """
        symbol: str = df.attrs.get("symbol", "UNKNOWN")

        if not self.validate_dataframe(df):
            return self._hold_signal(symbol)

        # --- collect signals ----------------------------------------------
        signals: list[Signal] = []
        for strat in self.strategies:
            try:
                sig = strat.analyze(df)
                signals.append(sig)
            except Exception:
                logger.exception("[ensemble] {} raised during analyze", strat.name)

        if not signals:
            return self._hold_signal(symbol)

        # --- weighted voting ----------------------------------------------
        action_scores: dict[TradeAction, float] = defaultdict(float)
        action_confidences: dict[TradeAction, list[tuple[float, float]]] = defaultdict(list)

        for sig in signals:
            w = self.weights.get(sig.strategy_name, 0.25)
            action_scores[sig.action] += w * sig.confidence
            action_confidences[sig.action].append((w, sig.confidence))

        # Choose action with highest weighted score
        best_action = max(action_scores, key=action_scores.get)  # type: ignore[arg-type]

        if best_action == TradeAction.HOLD:
            return self._hold_signal(symbol)

        # Weighted average confidence for the winning action
        pairs = action_confidences[best_action]
        total_weight = sum(w for w, _ in pairs)
        if total_weight == 0:
            return self._hold_signal(symbol)
        agg_confidence = sum(w * c for w, c in pairs) / total_weight

        if agg_confidence < self.min_confidence:
            logger.debug(
                "[ensemble] Consensus confidence {:.2f} < threshold {:.2f} — HOLD",
                agg_confidence, self.min_confidence,
            )
            return self._hold_signal(symbol)

        # --- build consensus signal ---------------------------------------
        # Use the signal from the highest-weighted agreeing strategy for SL/TP
        agreeing = [
            s for s in signals if s.action == best_action and s.action != TradeAction.HOLD
        ]
        if not agreeing:
            return self._hold_signal(symbol)

        agreeing.sort(
            key=lambda s: self.weights.get(s.strategy_name, 0.25) * s.confidence,
            reverse=True,
        )
        best_signal = agreeing[0]

        meta: dict[str, Any] = {
            "sub_signals": {
                s.strategy_name: {
                    "action": s.action.value,
                    "confidence": s.confidence,
                    "weight": self.weights.get(s.strategy_name, 0.25),
                }
                for s in signals
            },
            "action_scores": {a.value: round(v, 4) for a, v in action_scores.items()},
            "current_weights": dict(self.weights),
        }

        consensus = Signal(
            symbol=symbol,
            action=best_action,
            confidence=round(min(1.0, agg_confidence), 4),
            stop_loss_price=best_signal.stop_loss_price,
            take_profit_price=best_signal.take_profit_price,
            strategy_name=self.name,
            metadata=meta,
        )

        logger.info(
            "[ensemble] {} signal — {} conf={:.2f} (from {} strategies agreeing)",
            best_action.value, symbol, consensus.confidence, len(agreeing),
        )
        return consensus

    # ---------------------------------------------------------------- gates

    def should_enter(self, df: pd.DataFrame, signal: Signal) -> bool:
        """Enter only when at least two sub-strategies agree on the action."""
        if signal.action == TradeAction.HOLD:
            return False
        if signal.confidence < self.min_confidence:
            return False

        sub = signal.metadata.get("sub_signals", {})
        agreeing = sum(
            1 for v in sub.values() if v.get("action") == signal.action.value
        )
        if agreeing < 2:
            logger.debug("[ensemble] Only {} strategies agree — need ≥ 2", agreeing)
            return False

        return True

    def should_exit(self, df: pd.DataFrame, position: Position) -> bool:
        """Exit if majority of sub-strategies recommend exit."""
        exit_votes = 0
        total = len(self.strategies)
        for strat in self.strategies:
            try:
                if strat.should_exit(df, position):
                    exit_votes += 1
            except Exception:
                logger.exception("[ensemble] {} raised during should_exit", strat.name)

        should = exit_votes > total / 2
        if should:
            logger.info(
                "[ensemble] Majority exit ({}/{}) for {}", exit_votes, total, position.symbol
            )
        return should

    # ------------------------------------------------------- dynamic weights

    def record_strategy_result(
        self, strategy_name: str, pnl_pct: float
    ) -> None:
        """Record a trade result to feed the dynamic weight adjustment.

        Parameters
        ----------
        strategy_name : str
            Name of the strategy that produced the trade.
        pnl_pct : float
            Percentage return of the closed trade.
        """
        history = self._recent_pnl[strategy_name]
        history.append(pnl_pct)
        if len(history) > self._max_pnl_history:
            history.pop(0)

    def update_weights(self) -> None:
        """Re-balance strategy weights based on recent performance.

        Uses a simple Sharpe-like ratio: ``mean(pnl) / std(pnl)`` per
        strategy.  Strategies with no history retain their current weight.
        The weights are normalised to sum to 1.0.
        """
        import numpy as np

        raw: dict[str, float] = {}
        for name, history in self._recent_pnl.items():
            if len(history) < 3:
                raw[name] = self.weights.get(name, 0.25)
                continue
            arr = np.array(history)
            std = float(np.std(arr))
            if std == 0:
                raw[name] = float(np.mean(arr))
            else:
                raw[name] = max(0.05, float(np.mean(arr) / std))

        # Include strategies that have no PnL history yet
        for name in self.weights:
            if name not in raw:
                raw[name] = self.weights[name]

        total = sum(raw.values())
        if total > 0:
            self.weights = {k: v / total for k, v in raw.items()}

        logger.info("[ensemble] Updated weights: {}", self.weights)
