import requests
import numpy as np

BTC_SOURCES = [
    # CoinGecko (free, sometimes rate-limited)
    {
        "url": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
        "extract": lambda j: float(j["bitcoin"]["usd"]),
    },
    # Binance (no key needed, very reliable)
    {
        "url": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        "extract": lambda j: float(j["price"]),
    },
    # CoinCap (no key needed)
    {
        "url": "https://api.coincap.io/v2/assets/bitcoin",
        "extract": lambda j: float(j["data"]["priceUsd"]),
    },
]

# Seed for the simulated fallback price (keeps it stable between calls)
_sim_price = {"value": 67000.0}

def fetch_btc_price() -> float:
    """
    Try three free price APIs in order.
    If all fail (offline / rate-limited), return a simulated price
    so the live stream keeps running in demo mode.
    """
    for source in BTC_SOURCES:
        try:
            resp = requests.get(
                source["url"],
                timeout=5,
                headers={"User-Agent": "TrustLayer/1.0"},
            )
            resp.raise_for_status()
            return source["extract"](resp.json())
        except Exception:
            continue   # try next source

    # All APIs failed → simulate a realistic BTC price walk
    rng = np.random.default_rng()
    _sim_price["value"] = _sim_price["value"] * (1 + rng.normal(0, 0.002))
    return round(_sim_price["value"], 2)
