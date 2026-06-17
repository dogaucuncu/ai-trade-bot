from __future__ import annotations

import pandas as pd
from loguru import logger
from datetime import datetime, timezone

from src.strategy.base import BaseStrategy, Signal, TradeAction, Position
from src.ml.predictor import PredictionService

class MLStrategy(BaseStrategy):
    """
    ML-based strategy that uses the LSTM PredictionService.
    Takes a pre-fetched dataframe, passes it to the loaded LSTM model,
    and returns a BUY/SELL signal based on the predicted direction.
    """

    def __init__(self, name: str, config: dict | None = None) -> None:
        super().__init__(name, config)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.50)
        # SL/TP as fractions of entry price. Defaults match the config that
        # showed edge in backtest/walkforward_ml.py (1.5% stop / 3% target),
        # not the old wide 5%/10% which was never validated.
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.015)
        self.take_profit_pct = self.config.get("take_profit_pct", 0.03)
        self.predictor_service = PredictionService()
        logger.info(
            "MLStrategy '{}' initialized — threshold {:.2%}, SL {:.1%}, TP {:.1%}",
            self.name, self.confidence_threshold,
            self.stop_loss_pct, self.take_profit_pct,
        )

    def analyze(self, df: pd.DataFrame) -> Signal:
        """Analyze the dataframe using the ML model."""
        symbol = df.attrs.get("symbol")
        timeframe = df.attrs.get("timeframe", "1h")
        
        if not symbol:
            logger.warning("[MLStrategy] DataFrame is missing 'symbol' attribute.")
            return self._hold_signal("UNKNOWN")

        if not self.validate_dataframe(df):
            return self._hold_signal(symbol)

        current_price = df["close"].iloc[-1]
        
        # We bypass get_prediction() to avoid re-fetching data,
        # and directly use the predictor if available.
        predictor = self.predictor_service._load_predictor(symbol, timeframe)
        
        if not predictor:
            logger.warning(f"[MLStrategy] No ML model found for {symbol} {timeframe}")
            return self._hold_signal(symbol)

        try:
            direction, confidence = predictor.predict(df)
        except Exception as e:
            logger.warning(f"[MLStrategy] Prediction failed for {symbol}: {e}")
            return self._hold_signal(symbol)

        logger.debug(f"[MLStrategy] {symbol} {timeframe} -> {direction} ({confidence:.2%})")

        action = TradeAction.HOLD
        
        if confidence > self.confidence_threshold:
            if direction == "UP":
                action = TradeAction.BUY
            elif direction == "DOWN":
                action = TradeAction.SELL

        # SL/TP from config (default 1.5% / 3% — the validated values).
        if action == TradeAction.BUY:
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        else:
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)

        return Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "current_price": current_price,
                "direction": direction,
                "timeframe": timeframe
            }
        )

    def should_enter(self, df: pd.DataFrame, signal: Signal) -> bool:
        """Confirm entry. ML model handles all logic, so we just return True."""
        return True

    def should_exit(self, df: pd.DataFrame, position: Position) -> bool:
        """
        Check if we should exit an existing position early.
        We ask the ML model for the latest prediction.
        If we are LONG and it says DOWN with high confidence, exit early.
        """
        symbol = position.symbol
        timeframe = df.attrs.get("timeframe", "1h")
        
        predictor = self.predictor_service._load_predictor(symbol, timeframe)
        if not predictor:
            return False
            
        try:
            direction, confidence = predictor.predict(df)
        except Exception:
            return False
            
        if confidence > self.confidence_threshold:
            if position.side == TradeAction.BUY and direction == "DOWN":
                logger.info(f"[{self.name}] Early ML exit signal for LONG {symbol}")
                return True
            elif position.side == TradeAction.SELL and direction == "UP":
                logger.info(f"[{self.name}] Early ML exit signal for SHORT {symbol}")
                return True
                
        return False
