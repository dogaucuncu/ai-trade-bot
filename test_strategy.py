"""Live strategy signal test — fetch real data, compute indicators, generate signals."""
import asyncio
from config.settings import settings
from src.data.collector import DataCollector
from src.indicators.technical import TechnicalIndicators as TI
from src.strategy.scalping import ScalpingStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumStrategy

async def test_strategies():
    print("=== Live Strategy Signal Test ===\n")
    
    collector = DataCollector(settings)
    
    # Fetch enough data for indicators (need ~200 candles for EMA200)
    symbol = "DOGE/USDT"
    print(f"Fetching 300 x 5m candles for {symbol}...")
    df = await collector.fetch_crypto_ohlcv(symbol, "5m", limit=300)
    
    if df is None or df.empty:
        print("No data!")
        await collector.close()
        return
    
    print(f"Got {len(df)} candles ({df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']})")
    
    # Add all technical indicators
    print("\nComputing indicators...")
    df = TI.add_all_indicators(df)
    print(f"DataFrame now has {len(df.columns)} columns, {len(df)} rows")
    
    # Show latest indicator values
    last = df.iloc[-1]
    print(f"\n--- {symbol} Latest Values ---")
    print(f"  Close:       ${last['close']:.6f}")
    if 'rsi_14' in df.columns:
        print(f"  RSI(14):     {last['rsi_14']:.1f}")
    if 'macd_line' in df.columns:
        print(f"  MACD:        {last['macd_line']:.8f}")
    if 'bb_upper' in df.columns:
        print(f"  BB Upper:    ${last['bb_upper']:.6f}")
        print(f"  BB Lower:    ${last['bb_lower']:.6f}")
    if 'ema_9' in df.columns:
        print(f"  EMA(9):      ${last['ema_9']:.6f}")
        print(f"  EMA(21):     ${last['ema_21']:.6f}")
    if 'atr_14' in df.columns:
        print(f"  ATR(14):     ${last['atr_14']:.8f}")
    
    # Test each strategy
    strategies = [
        ("Scalping", ScalpingStrategy()),
        ("Mean Reversion", MeanReversionStrategy()),
        ("Momentum", MomentumStrategy()),
    ]
    
    print(f"\n--- Strategy Signals for {symbol} ---")
    for name, strategy in strategies:
        try:
            signal = strategy.analyze(df.copy())
            action = signal.action.value if hasattr(signal.action, 'value') else str(signal.action)
            print(f"  {name:20s} => {action:6s} (confidence: {signal.confidence:.1%})")
            if signal.stop_loss_price:
                print(f"  {'':20s}    SL: ${signal.stop_loss_price:.6f}  TP: ${signal.take_profit_price:.6f}")
        except Exception as e:
            print(f"  {name:20s} => ERROR: {e}")
    
    await collector.close()
    print("\n=== Test Complete ===")

asyncio.run(test_strategies())
