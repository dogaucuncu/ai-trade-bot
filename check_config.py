"""Quick config verification script."""
from config.settings import settings

print("=== Config Check ===")
print(f"Mode: {settings.trading_mode}")
print(f"Capital: ${settings.initial_capital}")

bk = settings.binance.api_key
bs = settings.binance.secret_key
ak = settings.alpaca.api_key
als = settings.alpaca.secret_key

bk_ok = bk and bk != "your_binance_api_key_here"
bs_ok = bs and bs != "your_binance_secret_key_here"
ak_ok = ak and ak != "your_alpaca_api_key_here"
as_ok = als and als != "your_alpaca_secret_key_here"

print(f"Binance API Key:  {'SET' if bk_ok else 'NOT SET'}")
print(f"Binance Secret:   {'SET' if bs_ok else 'NOT SET'}")
print(f"Binance Testnet:  {settings.binance.testnet}")
print(f"Alpaca API Key:   {'SET' if ak_ok else 'NOT SET'}")
print(f"Alpaca Secret:    {'SET' if as_ok else 'NOT SET'}")
print(f"Alpaca Paper:     {settings.alpaca.paper}")
print()

if not all([bk_ok, bs_ok, ak_ok, as_ok]):
    missing = []
    if not bk_ok: missing.append("BINANCE_API_KEY")
    if not bs_ok: missing.append("BINANCE_SECRET_KEY")
    if not ak_ok: missing.append("ALPACA_API_KEY")
    if not as_ok: missing.append("ALPACA_SECRET_KEY")
    print(f"MISSING: {', '.join(missing)}")
    print("Please edit config/.env and fill in the missing keys.")
else:
    print("All API keys configured!")
