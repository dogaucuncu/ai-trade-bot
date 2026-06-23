import asyncio
import argparse
from datetime import datetime
import pandas as pd
import aiohttp

from backtest.backtester import Backtester
from src.strategy.ensemble import EnsembleStrategy
from src.net import verified_ssl_context

async def fetch_historical_data(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch historical data from Binance for backtesting."""
    print(f"Fetching {limit} candles for {symbol} ({timeframe})...")
    
    binance_symbol = symbol.replace("/", "")
    tf_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d",
    }
    interval = tf_map.get(timeframe, "1h")
    
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={binance_symbol}&interval={interval}&limit={limit}"
    )
    
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=verified_ssl_context())) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Failed to fetch data: HTTP {resp.status}")
                return pd.DataFrame()
            raw = await resp.json()
            
    df = pd.DataFrame(
        raw,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
        
    print(f"Successfully fetched {len(df)} candles.")
    return df

async def main():
    parser = argparse.ArgumentParser(description="Run Strategy Backtest")
    parser.add_argument("--symbol", type=str, default="DOGE/USDT")
    parser.add_argument("--tf", type=str, default="1m")
    args = parser.parse_args()

    # 1. Fetch real historical data
    df = await fetch_historical_data(args.symbol, args.tf, limit=1000)
    if df.empty:
        return

    # 2. Add indicators (TechnicalIndicators requires a 'timestamp' column normally, 
    # but the strategy does it inside `analyze()` on the dataframe directly)
    # The strategies expect the DataFrame to just have open,high,low,close,volume 
    # and they add indicators themselves.

    from src.strategy.scalping import ScalpingStrategy
    from src.strategy.mean_reversion import MeanReversionStrategy
    from src.strategy.momentum import MomentumStrategy

    strategy = EnsembleStrategy(
        strategies=[
            ScalpingStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy()
        ]
    )
    
    start_date = str(df.index[0])[:10]
    end_date = str(df.index[-1])[:10]


    
    bt = Backtester(initial_capital=50.0, fee_rate=0.001)
    result = await bt.run(
        strategy=strategy,
        symbol=args.symbol,
        timeframe=args.tf,
        start_date=start_date,
        end_date=end_date,
        data=df
    )
    
    # 5. Generate Report
    bt.plot_results(result)
    print("\nBacktest complete! HTML report saved to backtest/results/")

if __name__ == "__main__":
    asyncio.run(main())
