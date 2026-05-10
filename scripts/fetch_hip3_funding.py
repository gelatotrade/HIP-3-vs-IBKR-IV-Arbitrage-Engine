#!/usr/bin/env python3
"""
Fetch real HIP-3 funding rate history from Hyperliquid API.

HIP-3 markets live on separate perp DEX deployers (xyz, flx, km, cash, vntl, hyna).
Each deployer sets per-asset funding_multiplier and funding_interest_rate that
modify the hourly funding calculation.

This script fetches hourly funding history for all 17 verified HIP-3 assets,
paginating backwards from now to each asset's launch date, and caches the
result as CSV in data/funding_rates/.

Usage:  python3 scripts/fetch_hip3_funding.py
"""
import sys, time, json, os
from datetime import datetime, timezone
from pathlib import Path

import requests
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hyperliquid_hip3_client import HIP3_LAUNCH_DATES, REFERENCE_DATE

BASE_URL = "https://api.hyperliquid.xyz/info"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "funding_rates"

# Map our assets → HIP-3 market identifiers (primary deployer)
# Verified on-chain (see perpDexs endpoint). QQQ = xyz:XYZ100 (Nasdaq-100 proxy).
# OIL = xyz:CL (WTI crude). SPY = xyz:SP500. Others use asset's xyz:TICKER.
HIP3_MARKET_NAMES = {
    "QQQ":    "xyz:XYZ100",
    "NVDA":   "xyz:NVDA",
    "TSLA":   "xyz:TSLA",
    "PLTR":   "xyz:PLTR",
    "AMZN":   "xyz:AMZN",
    "GOOGL":  "xyz:GOOGL",
    "MSFT":   "xyz:MSFT",
    "META":   "xyz:META",
    "AAPL":   "xyz:AAPL",
    "COIN":   "xyz:COIN",
    "MSTR":   "xyz:MSTR",
    "AMD":    "xyz:AMD",
    "NFLX":   "xyz:NFLX",
    "GOLD":   "xyz:GOLD",
    "SILVER": "xyz:SILVER",
    "OIL":    "xyz:CL",
    "SPY":    "xyz:SP500",
}


def post_info(payload, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.post(BASE_URL, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_funding_history(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch paginated hourly funding history for a HIP-3 market.

    API returns max 500 entries per call. Page forward by advancing startTime
    past the last returned entry until we reach end_ms.
    """
    all_rows = []
    cur_start = start_ms

    while cur_start < end_ms:
        data = post_info({
            "type": "fundingHistory",
            "coin": coin,
            "startTime": cur_start,
        })
        if not isinstance(data, list) or len(data) == 0:
            break
        all_rows.extend(data)
        last_ts = max(int(d["time"]) for d in data)
        if last_ts <= cur_start:
            break
        cur_start = last_ts + 1
        time.sleep(0.15)  # rate-limit buffer

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["premium"] = pd.to_numeric(df["premium"], errors="coerce")
    df = (
        df.sort_values("time")
          .drop_duplicates(subset=["time"])
          .reset_index(drop=True)
    )
    return df


def aggregate_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Convert hourly funding rates into daily aggregate (sum of 24 hours).

    Hyperliquid settles funding every hour; each hour pays 1/8 of computed 8h rate.
    Daily aggregate = sum of all hourly fundingRate values that day (UTC).
    """
    if hourly.empty:
        return pd.DataFrame()
    df = hourly.copy()
    df["date"] = df["time"].dt.tz_convert("UTC").dt.floor("1d")
    daily = (
        df.groupby("date")
          .agg(funding_rate=("fundingRate", "sum"),
               funding_mean_hourly=("fundingRate", "mean"),
               n_hours=("fundingRate", "size"),
               premium_mean=("premium", "mean"))
          .reset_index()
          .rename(columns={"date": "timestamp"})
    )
    return daily


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("  Fetching real HIP-3 funding rate history from Hyperliquid")
    print("=" * 80)

    end_ms = int(REFERENCE_DATE.replace(tzinfo=timezone.utc).timestamp() * 1000)

    summary = []
    for asset, market in HIP3_MARKET_NAMES.items():
        launch = HIP3_LAUNCH_DATES[asset]
        start_ms = int(launch.replace(tzinfo=timezone.utc).timestamp() * 1000)
        print(f"\n  {asset:6s} ({market}) — launch {launch.date()} ...", end=" ", flush=True)

        try:
            hourly = fetch_funding_history(market, start_ms, end_ms)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if hourly.empty:
            print("no data")
            continue

        daily = aggregate_daily(hourly)
        hourly_path = CACHE_DIR / f"{asset}_hourly.csv"
        daily_path = CACHE_DIR / f"{asset}_daily.csv"
        hourly.to_csv(hourly_path, index=False, float_format="%.10f")
        daily.to_csv(daily_path, index=False, float_format="%.10f")

        mean_daily_bps = daily["funding_rate"].mean() * 10000
        std_daily_bps = daily["funding_rate"].std() * 10000
        print(
            f"{len(hourly):>5d} hourly / {len(daily):>4d} daily   "
            f"mean={mean_daily_bps:+.2f} bp/day  std={std_daily_bps:.2f} bp"
        )
        summary.append({
            "asset": asset, "market": market,
            "n_hourly": len(hourly), "n_daily": len(daily),
            "mean_daily_bps": mean_daily_bps, "std_daily_bps": std_daily_bps,
            "min_daily_bps": daily["funding_rate"].min() * 10000,
            "max_daily_bps": daily["funding_rate"].max() * 10000,
        })

    sdf = pd.DataFrame(summary)
    sdf.to_csv(CACHE_DIR / "_summary.csv", index=False, float_format="%.4f")
    print("\n" + "=" * 80)
    print(f"  Cached {len(sdf)} assets → {CACHE_DIR}")
    print("=" * 80)
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
