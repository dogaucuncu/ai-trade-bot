"""Quick API connectivity test — fetches live market data."""
import asyncio
from config.settings import settings
from src.data.collector import DataCollector

async def test_connectivity():
    print("=== API Connectivity Test ===\n")
    
    collector = DataCollector(settings)
    
    # Test 1: Binance — fetch DOGE/USDT candles
    print("[1] Binance DOGE/USDT (5m candles, last 10)...")
    try:
        df = await collector.fetch_crypto_ohlcv("DOGE/USDT", "5m", limit=10)
        if df is not None and not df.empty:
            print(f"    SUCCESS! Got {len(df)} candles")
            print(f"    Latest: {df.iloc[-1]['timestamp']} | Close: ${df.iloc[-1]['close']:.6f} | Vol: {df.iloc[-1]['volume']:.0f}")
        else:
            print("    No data returned (empty)")
    except Exception as e:
        print(f"    FAILED: {e}")
    
    # Test 2: Binance — fetch XRP/USDT
    print("\n[2] Binance XRP/USDT (1h candles, last 5)...")
    try:
        df = await collector.fetch_crypto_ohlcv("XRP/USDT", "1h", limit=5)
        if df is not None and not df.empty:
            print(f"    SUCCESS! Got {len(df)} candles")
            print(f"    Latest: {df.iloc[-1]['timestamp']} | Close: ${df.iloc[-1]['close']:.4f}")
        else:
            print("    No data returned (empty)")
    except Exception as e:
        print(f"    FAILED: {e}")

    # Test 3: Multiple symbols
    print("\n[3] Batch fetch: PEPE, SOL, ADA (5m, last 5)...")
    try:
        results = await collector.fetch_multiple(
            symbols=["PEPE/USDT", "SOL/USDT", "ADA/USDT"],
            timeframe="5m",
            limit=5
        )
        for sym, df in results.items():
            if df is not None and not df.empty:
                print(f"    {sym}: {len(df)} candles | Close: ${df.iloc[-1]['close']}")
            else:
                print(f"    {sym}: No data")
    except Exception as e:
        print(f"    FAILED: {e}")

    # Cleanup
    await collector.close()
    print("\n=== Test Complete ===")

asyncio.run(test_connectivity())
