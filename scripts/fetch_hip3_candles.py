#!/usr/bin/env python3
"""
Fetch real HIP-3 OHLCV candle history from Hyperliquid API.

HIP-3 markets live on separate perp DEX deployers (xyz, flx, km, cash, vntl, hyna).
This script fetches daily 1d candles for all 17 verified HIP-3 assets from each
asset's on-chain launch date to the reference date, then caches the result as
CSV in data/candles/.

Usage:  python3 scripts/fetch_hip3_candles.py
"""
import sys, time, json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hyperliquid_hip3_client import HIP3_LAUNCH_DATES, REFERENCE_DATE
from fetch_hip3_funding import HIP3_MARKET_NAMES

BASE_URL = "https://api.hyperliquid.xyz/info"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "candles"


def post_info(payload, retries=4):
    """POST /info with exponential backoff for transient API hiccups."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(BASE_URL, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


def fetch_candles(coin: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch OHLCV candles for a HIP-3 market, paginating 90-day chunks.

    The API caps each call at ~500 candles. For daily candles that means
    ~500 days per request, so we typically only need one chunk; we still
    paginate to be safe for long histories.
    """
    all_rows = []
    chunk_ms = 90 * 24 * 3600 * 1000  # 90-day chunks
    cur_start = start_ms

    while cur_start < end_ms:
        cur_end = min(cur_start + chunk_ms, end_ms)
        data = post_info({
            "type": "candleSnapshot",
            "req": {
                "coin": coin, "interval": interval,
                "startTime": cur_start, "endTime": cur_end,
            },
        })
        if isinstance(data, list) and data:
            all_rows.extend(data)
        cur_start = cur_end
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["t"].astype(int), unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
          .sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"])
          .reset_index(drop=True)
    )
    return df


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("  Fetching real HIP-3 OHLCV candles from Hyperliquid")
    print("=" * 80)

    end_ms = int(REFERENCE_DATE.replace(tzinfo=timezone.utc).timestamp() * 1000)

    summary = []
    for asset, market in HIP3_MARKET_NAMES.items():
        launch = HIP3_LAUNCH_DATES[asset]
        # Start a bit earlier to include the launch day fully
        start_ms = int((launch - timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp() * 1000)
        print(f"\n  {asset:6s} ({market}) — launch {launch.date()} ...", end=" ", flush=True)

        try:
            df = fetch_candles(market, "1d", start_ms, end_ms)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if df.empty:
            print("no data")
            continue

        path = CACHE_DIR / f"{asset}.csv"
        df.to_csv(path, index=False, float_format="%.8f")

        ret = (df["close"].iloc[-1] / df["open"].iloc[0] - 1) * 100
        vol_ann = float(np.std(np.diff(np.log(df["close"].values)), ddof=1) * np.sqrt(365) * 100)
        print(
            f"{len(df):>3d} bars  "
            f"price ${df['close'].iloc[-1]:>9.2f}  "
            f"return {ret:+7.1f}%  vol {vol_ann:4.0f}%"
        )
        summary.append({
            "asset": asset, "market": market, "n_bars": len(df),
            "first_ts": df["timestamp"].iloc[0], "last_ts": df["timestamp"].iloc[-1],
            "first_price": float(df["open"].iloc[0]),
            "last_price": float(df["close"].iloc[-1]),
            "total_return_pct": ret, "ann_vol_pct": vol_ann,
        })

    sdf = pd.DataFrame(summary)
    sdf.to_csv(CACHE_DIR / "_summary.csv", index=False, float_format="%.4f")
    print("\n" + "=" * 80)
    print(f"  Cached {len(sdf)} assets → {CACHE_DIR}")
    print("=" * 80)
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
