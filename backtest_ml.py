"""
DEPRECATED — DO NOT TRUST THE RESULTS OF THIS SCRIPT.

This backtest is misleading for three reasons and is kept only for reference:

1. IN-SAMPLE: it evaluates the model on (largely) the same DOGE/USDT 15m
   candles the model was trained on, so the model has "seen the answers".
2. PER-BAR CHURN: it can reverse the position every single candle.
3. (Historically) it relied on a scaler fit with look-ahead bias.

Use the honest, leak-free walk-forward engine instead:

    venv/Scripts/python.exe -m backtest.walkforward_ml --symbol DOGE/USDT --tf 1h

See backtest/walkforward_ml.py and backtest/metrics.py.
"""

import asyncio
import pandas as pd
from pathlib import Path
from loguru import logger
import sqlite3

from src.ml.lstm_model import LSTMPredictor
from src.indicators.technical import TechnicalIndicators

DB_PATH = Path("f:/Trade bot/data/tradebot.db")
MODEL_DIR = Path("f:/Trade bot/models/DOGE_USDT_15m")

async def main():
    logger.warning(
        "DEPRECATED in-sample backtest. Results are NOT trustworthy. "
        "Use: python -m backtest.walkforward_ml  (see module docstring)."
    )
    logger.info("Starting ML Futures Backtest on DOGE/USDT (15m)...")

    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        return

    from src.data.collector import DataCollector
    from config.settings import settings
    
    collector = DataCollector(settings)
    df = await collector.fetch_crypto_ohlcv('DOGE/USDT', '15m', limit=5760)
    await collector.close()

    if df is None or df.empty:
        logger.error("No data found for DOGE/USDT 1h")
        return
    
    # Ensure correct types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    logger.info(f"Loaded {len(df)} candles from database.")

    # Drop duplicates if any
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 2. Add Indicators
    logger.info("Calculating technical indicators...")
    df = TechnicalIndicators.add_all_indicators(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Data ready for backtest. Rows: {len(df)}")

    # 3. Load Model
    predictor = LSTMPredictor()
    predictor.load_model(MODEL_DIR)

    # 4. Futures Backtest Variables
    capital = 50.0
    position = None  # None, "LONG", "SHORT"
    entry_price = 0.0
    position_size = 0.0
    fee_rate = 0.001 # 0.1% per trade (0.2% round trip)
    
    total_trades = 0
    winning_trades = 0
    total_fees = 0.0
    
    trade_log = []

    logger.info("Starting simulation...")
    
    lookback = 60
    
    # We step through the dataframe row by row.
    # To avoid the slow loop over 4000 rows, we will vectorize prediction if possible, 
    # but LSTM requires sliding windows. Since predicting 4000 rows one-by-one takes time,
    # we'll just do it in a loop and print progress.
    
    for i in range(300, len(df)):
        if i % 500 == 0:
            logger.info(f"Simulated {i}/{len(df)} candles...")
            
        current_idx = i - 1
        current_candle = df.iloc[current_idx]
        current_price = current_candle['close']
        current_time = current_candle['timestamp']
        
        # Next candle is what happens after we make a decision
        next_candle = df.iloc[i]
        next_price = next_candle['close']
        
        # Predict based on data up to current_idx (needs 300 rows for EMA200 to leave 60 valid rows)
        window = df.iloc[i-300 : i]
        
        try:
            direction, confidence = predictor.predict(window)
        except Exception as e:
            continue
            
        if confidence < 0.50:
            direction = "SIDEWAYS" # Override if low confidence

        # Futures Logic:
        if direction == "UP":
            if position == "SHORT":
                # Close SHORT
                pnl = (entry_price - current_price) * position_size
                fee = current_price * position_size * fee_rate
                capital += pnl - fee
                total_fees += fee
                total_trades += 1
                if pnl > 0: winning_trades += 1
                position = None
                
            if position == None:
                # Open LONG
                position_size = (capital * 0.95) / current_price # Use 95% of capital to be safe
                fee = current_price * position_size * fee_rate
                capital -= fee
                total_fees += fee
                entry_price = current_price
                position = "LONG"
                
        elif direction == "DOWN":
            if position == "LONG":
                # Close LONG
                pnl = (current_price - entry_price) * position_size
                fee = current_price * position_size * fee_rate
                capital += pnl - fee
                total_fees += fee
                total_trades += 1
                if pnl > 0: winning_trades += 1
                position = None
                
            if position == None:
                # Open SHORT
                position_size = (capital * 0.95) / current_price
                fee = current_price * position_size * fee_rate
                capital -= fee
                total_fees += fee
                entry_price = current_price
                position = "SHORT"
                
        elif direction == "SIDEWAYS":
            # Close any open positions to avoid holding risk during sideways markets
            if position == "LONG":
                pnl = (current_price - entry_price) * position_size
                fee = current_price * position_size * fee_rate
                capital += pnl - fee
                total_fees += fee
                total_trades += 1
                if pnl > 0: winning_trades += 1
                position = None
            elif position == "SHORT":
                pnl = (entry_price - current_price) * position_size
                fee = current_price * position_size * fee_rate
                capital += pnl - fee
                total_fees += fee
                total_trades += 1
                if pnl > 0: winning_trades += 1
                position = None

    # Close any open position at the very end
    if position == "LONG":
        current_price = df.iloc[-1]['close']
        pnl = (current_price - entry_price) * position_size
        capital += pnl
    elif position == "SHORT":
        current_price = df.iloc[-1]['close']
        pnl = (entry_price - current_price) * position_size
        capital += pnl

    logger.info("=========================================")
    logger.info("BACKTEST RESULTS (FUTURES)")
    logger.info("=========================================")
    logger.info(f"Initial Capital: $50.00")
    logger.info(f"Final Capital:   ${capital:.2f}")
    logger.info(f"Net Profit:      ${capital - 50.0:.2f} ({((capital/50.0)-1)*100:.2f}%)")
    logger.info(f"Total Trades:    {total_trades}")
    if total_trades > 0:
        logger.info(f"Win Rate:        {(winning_trades/total_trades)*100:.2f}%")
    logger.info(f"Total Fees Paid: ${total_fees:.2f}")
    logger.info("=========================================")

if __name__ == "__main__":
    # Disable spammy predictor logs during loop
    logger.remove()
    import sys
    logger.add(sys.stderr, level="INFO")
    
    asyncio.run(main())
