#!/usr/bin/env python3
"""
Hyperliquid HIP-3 API Client — Fetch all HIP-3 perp markets (equities, commodities, ETFs).

HIP-3 = Hyperliquidity Provider on Hyperliquid DEX.
These markets offer tokenized perps on real-world assets (stocks, indices,
commodities) with AMM-style liquidity and different IV characteristics
compared to traditional options markets (IBKR).

Strategy focus: HIP-3 perps on assets that also have liquid options on IBKR,
enabling IV arbitrage and higher-order Greek strategies between the two venues.

Endpoints:
  POST https://api.hyperliquid.xyz/info
  Body: {"type": "spotMeta"}                → all spot market metadata
  Body: {"type": "spotMetaAndAssetCtxs"}    → metadata + live prices/volumes
  Body: {"type": "l2Book", "coin": "X"}     → L2 orderbook
  Body: {"type": "candleSnapshot", "coin": "X", "interval": "1d", "startTime": ts, "endTime": ts}

Usage:  from hyperliquid_hip3_client import HyperliquidHIP3Client
"""

import requests
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

BASE_URL = "https://api.hyperliquid.xyz/info"

# HIP-3 perp tokens — real-world assets with IBKR options
# Only includes confirmed live assets on Hyperliquid HIP-3 with their launch dates.
# HIP-3 mainnet: Oct 13, 2025. Equity perps followed from Nov 2025.
# Reference date for n_days calculation: Apr 15, 2026.
HIP3_LAUNCH_DATES = {
    # Verified from Hyperliquid API `candleSnapshot` first-candle dates and on-chain
    # perpDexs metadata (research date: 2026-04-16). HIP-3 mainnet launched 2025-10-13
    # with xyz:XYZ100 (Nasdaq-100 proxy). Dates are first-trade UTC dates on HIP-3 DEXes
    # (xyz, flx, vntl, km, cash). Assets not deployed on any HIP-3 DEX are excluded.
    #
    # ── Indices / Nasdaq-100 proxy (HIP-3 day zero) ──
    "QQQ":    datetime(2025, 10, 13),   # xyz:XYZ100 — first HIP-3 market ever
    # ── Wave 1: core equities on xyz (Nov 2025) ──
    "NVDA":   datetime(2025, 11, 12),   # xyz:NVDA
    "TSLA":   datetime(2025, 11, 13),   # xyz:TSLA (flx:TSLA same day, Felix's first)
    "PLTR":   datetime(2025, 11, 14),   # xyz:PLTR
    "AMZN":   datetime(2025, 11, 18),   # xyz:AMZN
    "GOOGL":  datetime(2025, 11, 18),   # xyz:GOOGL (GOOG class shares not on HIP-3)
    "MSFT":   datetime(2025, 11, 19),   # xyz:MSFT
    "META":   datetime(2025, 11, 20),   # xyz:META
    "AAPL":   datetime(2025, 11, 21),   # xyz:AAPL
    "COIN":   datetime(2025, 11, 25),   # xyz:COIN (flx:COIN same day)
    # ── Wave 2: more equities (Dec 2025) ──
    "MSTR":   datetime(2025, 12, 2),    # xyz:MSTR
    "AMD":    datetime(2025, 12, 4),    # xyz:AMD
    "NFLX":   datetime(2025, 12, 8),    # xyz:NFLX
    # ── Commodities (Dec 2025 via Felix) ──
    "GOLD":   datetime(2025, 12, 12),   # flx:GOLD (earliest HIP-3 gold market)
    "SILVER": datetime(2025, 12, 17),   # flx:SILVER
    # ── Oil (Jan 2026) ──
    "OIL":    datetime(2026, 1, 6),     # xyz:CL (WTI); flx:OIL three days later
    # ── S&P 500 (licensed) — SPY has too little history for backtest, included for completeness ──
    "SPY":    datetime(2026, 3, 18),    # xyz:SP500 — officially S&P DJI licensed
}

# Reference date for computing "days since launch". Update on each run.
REFERENCE_DATE = datetime(2026, 4, 16)

HIP3_TOKENS = list(HIP3_LAUNCH_DATES.keys())

def get_asset_n_days(asset: str) -> int:
    """Return number of trading days since asset's HIP-3 launch."""
    launch = HIP3_LAUNCH_DATES.get(asset)
    if launch is None:
        return 150  # default
    return (REFERENCE_DATE - launch).days

# Interval mappings for candle data
INTERVALS = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h",
    "4h": "4h", "1d": "1d", "1w": "1w",
}


class HyperliquidHIP3Client:
    """Client for Hyperliquid HIP-3 spot market data."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _post(self, payload: dict, retries: int = 3) -> dict:
        """POST request with retry logic."""
        for attempt in range(retries):
            try:
                resp = self.session.post(self.base_url, json=payload, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt == retries - 1:
                    print(f"  API error after {retries} retries: {e}")
                    return {}
                time.sleep(2 ** attempt)
        return {}

    def get_spot_meta(self) -> dict:
        """Get all spot market metadata."""
        return self._post({"type": "spotMeta"})

    def get_spot_meta_and_contexts(self) -> dict:
        """Get spot metadata with live market context (prices, volumes)."""
        return self._post({"type": "spotMetaAndAssetCtxs"})

    def get_perp_meta_and_contexts(self) -> dict:
        """Get perpetual futures metadata and contexts."""
        return self._post({"type": "metaAndAssetCtxs"})

    def get_l2_book(self, coin: str, n_levels: int = 20) -> dict:
        """Get L2 orderbook for a coin."""
        return self._post({
            "type": "l2Book",
            "coin": coin,
            "nSigFigs": 5,
        })

    def get_candles(self, coin: str, interval: str = "1d",
                    start_time: Optional[int] = None,
                    end_time: Optional[int] = None) -> List[dict]:
        """Get candle (OHLCV) data for a coin.

        Args:
            coin: Token symbol (e.g., "BTC", "ETH")
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d")
            start_time: Start timestamp in ms (default: 365 days ago)
            end_time: End timestamp in ms (default: now)
        """
        if end_time is None:
            end_time = int(time.time() * 1000)
        if start_time is None:
            start_time = end_time - 365 * 24 * 3600 * 1000

        return self._post({
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
            }
        })

    def get_all_candles_history(self, coin: str, interval: str = "1d",
                                max_days: int = 730) -> pd.DataFrame:
        """Fetch full candle history by paginating backwards.

        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        """
        end_time = int(time.time() * 1000)
        start_time = end_time - max_days * 24 * 3600 * 1000

        all_candles = []
        chunk_ms = 90 * 24 * 3600 * 1000  # 90 days per chunk

        current_start = start_time
        while current_start < end_time:
            current_end = min(current_start + chunk_ms, end_time)
            candles = self.get_candles(coin, interval, current_start, current_end)

            if isinstance(candles, list) and len(candles) > 0:
                all_candles.extend(candles)
            current_start = current_end
            time.sleep(0.2)  # Rate limit

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        if 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            })
        elif 'T' in df.columns:
            df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
            df = df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            })

        # Ensure numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
        return df

    def get_funding_rate(self, coin: str) -> Optional[float]:
        """Get current funding rate for a perpetual (proxy for IV)."""
        data = self._post({"type": "metaAndAssetCtxs"})
        if not data or len(data) < 2:
            return None
        meta = data[0]
        ctxs = data[1]
        for i, asset in enumerate(meta.get('universe', [])):
            if asset.get('name', '').upper() == coin.upper():
                if i < len(ctxs):
                    return float(ctxs[i].get('funding', 0))
        return None

    def get_funding_history(self, coin: str, start_time: Optional[int] = None,
                             max_days: int = 730) -> pd.DataFrame:
        """Fetch historical funding rate data for a perpetual.

        Hyperliquid funding rates are settled every 8 hours (3x/day).
        Returns DataFrame: timestamp, coin, funding_rate, premium
        """
        end_time = int(time.time() * 1000)
        if start_time is None:
            start_time = end_time - max_days * 24 * 3600 * 1000

        all_funding = []
        current_start = start_time
        chunk_ms = 30 * 24 * 3600 * 1000  # 30 days per chunk

        while current_start < end_time:
            data = self._post({
                "type": "fundingHistory",
                "coin": coin,
                "startTime": current_start,
            })
            if isinstance(data, list) and len(data) > 0:
                all_funding.extend(data)
                # Move to after last timestamp
                last_ts = max(int(d.get('time', current_start)) for d in data)
                current_start = last_ts + 1
            else:
                current_start += chunk_ms
            time.sleep(0.25)

        if not all_funding:
            return pd.DataFrame()

        df = pd.DataFrame(all_funding)
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        if 'fundingRate' in df.columns:
            df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        if 'premium' in df.columns:
            df['premium'] = pd.to_numeric(df['premium'], errors='coerce')

        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
        return df

    def get_all_hip3_markets(self) -> pd.DataFrame:
        """Discover all available HIP-3 spot markets with their context."""
        data = self.get_spot_meta_and_contexts()

        if not data or not isinstance(data, list) or len(data) < 2:
            print("  Warning: Could not fetch spot meta, using known HIP-3 tokens")
            return pd.DataFrame({'token': HIP3_TOKENS})

        meta = data[0]
        ctxs = data[1]

        tokens_info = []
        for i, token_meta in enumerate(meta.get('tokens', [])):
            token_name = token_meta.get('name', f'TOKEN_{i}')
            ctx = ctxs[i] if i < len(ctxs) else {}

            mid_price = 0
            day_volume = 0
            if isinstance(ctx, dict):
                mid_price = float(ctx.get('midPx', 0) or 0)
                day_volume = float(ctx.get('dayNtlVlm', 0) or 0)

            tokens_info.append({
                'token': token_name,
                'index': token_meta.get('index', i),
                'mid_price': mid_price,
                'day_volume_usd': day_volume,
                'decimals': token_meta.get('szDecimals', 0),
            })

        df = pd.DataFrame(tokens_info)
        # Filter to active markets with volume
        if len(df) > 0 and 'day_volume_usd' in df.columns:
            df = df[df['day_volume_usd'] > 0].sort_values(
                'day_volume_usd', ascending=False).reset_index(drop=True)

        return df

    def compute_implied_vol_from_funding(self, coin: str,
                                          lookback_days: int = 30) -> dict:
        """Estimate implied volatility from HIP-3 market data.

        On Hyperliquid, HIP-3 markets price risk differently than traditional
        options markets. We derive IV from:
        1. Funding rate (carrying cost ≈ forward-spot basis)
        2. Realized volatility from spot candles
        3. Bid-ask spread dynamics (wider spread = higher IV estimate)
        """
        # Get candle data for realized vol
        candles = self.get_all_candles_history(coin, "1d", lookback_days + 60)
        if candles.empty or len(candles) < 20:
            return {'realized_vol': None, 'implied_vol': None, 'funding_rate': None}

        closes = candles['close'].values
        returns = np.diff(np.log(closes))

        # Realized volatility (annualized)
        rv_20d = np.std(returns[-20:], ddof=1) * np.sqrt(365) if len(returns) >= 20 else None
        rv_30d = np.std(returns[-30:], ddof=1) * np.sqrt(365) if len(returns) >= 30 else None

        # Get L2 book for spread-implied vol
        book = self.get_l2_book(coin)
        spread_iv = None
        if book and 'levels' in book:
            levels = book['levels']
            if len(levels) >= 2 and levels[0] and levels[1]:
                best_bid = float(levels[0][0].get('px', 0))
                best_ask = float(levels[1][0].get('px', 0))
                mid = (best_bid + best_ask) / 2
                if mid > 0:
                    spread_bps = (best_ask - best_bid) / mid * 10000
                    # Empirical: spread ≈ 2σ√Δt, solve for daily σ
                    spread_iv = (spread_bps / 10000) * np.sqrt(365) * 0.5

        # Funding rate as forward basis
        funding = self.get_funding_rate(coin)

        # HIP-3 implied vol = blend of realized + spread-implied + funding premium
        components = []
        if rv_20d is not None:
            components.append(rv_20d)
        if spread_iv is not None and spread_iv > 0:
            components.append(spread_iv)
        if funding is not None:
            # Hyperliquid funding is settled hourly (24x/day).
            # API returns the per-hour rate; annualize via 24*365.
            funding_ann = abs(funding) * 24 * 365
            vol_premium = np.sqrt(funding_ann) if funding_ann > 0 else 0
            if vol_premium > 0:
                components.append(rv_20d * (1 + vol_premium) if rv_20d else vol_premium)

        hip3_iv = np.mean(components) if components else None

        return {
            'coin': coin,
            'realized_vol_20d': rv_20d,
            'realized_vol_30d': rv_30d,
            'spread_implied_vol': spread_iv,
            'funding_rate': funding,
            'hip3_implied_vol': hip3_iv,
            'n_candles': len(candles),
            'latest_price': float(closes[-1]) if len(closes) > 0 else None,
        }


def fetch_all_hip3_data(max_assets: int = 30, max_days: int = 730) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for all available HIP-3 markets.

    Returns dict mapping token name → DataFrame of candle data.
    """
    client = HyperliquidHIP3Client()
    print("Discovering HIP-3 markets ...")
    markets = client.get_all_hip3_markets()

    if markets.empty:
        print("  Using fallback token list")
        tokens = HIP3_TOKENS[:max_assets]
    else:
        print(f"  Found {len(markets)} active spot markets")
        tokens = markets['token'].tolist()[:max_assets]

    print(f"\nFetching candle data for {len(tokens)} tokens ({max_days}d history) ...")
    data = {}
    for token in tokens:
        print(f"  {token} ...", end=" ", flush=True)
        try:
            df = client.get_all_candles_history(token, "1d", max_days)
            if not df.empty and len(df) >= 30:
                data[token] = df
                print(f"{len(df)} bars ({len(df)/365:.1f}y)")
            else:
                print(f"SKIP ({len(df) if not df.empty else 0} bars)")
        except Exception as e:
            print(f"ERROR ({e})")
        time.sleep(0.3)

    print(f"\nLoaded {len(data)} HIP-3 assets")
    return data


def fetch_hip3_iv_data(tokens: List[str] = None) -> pd.DataFrame:
    """Fetch implied volatility estimates for HIP-3 tokens."""
    client = HyperliquidHIP3Client()
    if tokens is None:
        tokens = HIP3_TOKENS[:15]

    results = []
    for token in tokens:
        print(f"  Computing IV for {token} ...", end=" ", flush=True)
        try:
            iv_data = client.compute_implied_vol_from_funding(token)
            results.append(iv_data)
            if iv_data['hip3_implied_vol'] is not None:
                print(f"HIP3-IV={iv_data['hip3_implied_vol']*100:.1f}%  "
                      f"RV20={iv_data['realized_vol_20d']*100:.1f}%")
            else:
                print("no data")
        except Exception as e:
            print(f"ERROR ({e})")
        time.sleep(0.5)

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# Synthetic HIP-3 data generation (for backtesting when API unavailable)
# ──────────────────────────────────────────────────────────────
# Cache locations for real HIP-3 data (fetched via fetch_hip3_*.py scripts)
from pathlib import Path as _Path
DATA_DIR         = _Path(__file__).resolve().parent.parent / "data"
FUNDING_CACHE_DIR = DATA_DIR / "funding_rates"
CANDLES_CACHE_DIR = DATA_DIR / "candles"


def load_real_hip3_data(min_days: int = 100,
                        assets: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load cached REAL HIP-3 OHLCV candles + funding rates from data/.

    Reads candles from data/candles/{asset}.csv (fetched via fetch_hip3_candles.py)
    and merges the corresponding daily funding series from data/funding_rates/.
    Returns the same dict[asset→DataFrame] shape as `generate_synthetic_hip3_data`,
    with columns: timestamp, open, high, low, close, volume, funding_rate.

    Assets with fewer than `min_days` of history are excluded.
    """
    if not CANDLES_CACHE_DIR.exists():
        return {}

    if assets is None:
        assets = HIP3_TOKENS

    data: Dict[str, pd.DataFrame] = {}
    for asset in assets:
        cpath = CANDLES_CACHE_DIR / f"{asset}.csv"
        if not cpath.exists():
            continue
        df = pd.read_csv(cpath, parse_dates=["timestamp"])
        if df.empty or len(df) < min_days:
            continue

        # Merge daily funding rates by calendar date (UTC).
        fpath = FUNDING_CACHE_DIR / f"{asset}_daily.csv"
        if fpath.exists():
            fdf = pd.read_csv(fpath, parse_dates=["timestamp"])
            fdf = fdf.rename(columns={"timestamp": "funding_ts"})
            # Align by date (strip tz so both match)
            df["_date"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
            fdf["_date"] = pd.to_datetime(fdf["funding_ts"]).dt.tz_localize(None).dt.normalize()
            df = df.merge(fdf[["_date", "funding_rate"]], on="_date", how="left")
            df = df.drop(columns=["_date"])
            # Fill gaps with the asset's historical mean funding (rare)
            df["funding_rate"] = df["funding_rate"].fillna(df["funding_rate"].mean())
        else:
            # Fallback: zero funding (will be overridden by synthetic model if needed)
            df["funding_rate"] = 0.0

        data[asset] = df.reset_index(drop=True)

    return data


def load_real_funding_daily(asset: str, n_days: int) -> Optional[np.ndarray]:
    """Load cached real Hyperliquid funding rates for a HIP-3 asset.

    Returns an array of daily funding rates aligned to the last n_days,
    or None if the cache file is missing. Daily rate = sum of 24 hourly
    settlements (Hyperliquid mechanism).
    """
    path = FUNDING_CACHE_DIR / f"{asset}_daily.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df.empty or "funding_rate" not in df.columns:
            return None
        # Take the last n_days rows; if fewer, pad start with the mean
        values = df["funding_rate"].values
        if len(values) >= n_days:
            return values[-n_days:].astype(float)
        pad = np.full(n_days - len(values), float(values.mean()))
        return np.concatenate([pad, values.astype(float)])
    except Exception:
        return None


def generate_synthetic_hip3_data(n_assets: int = 15, n_days: int = None,
                                  seed: int = 42, min_days: int = 100,
                                  use_real_funding: bool = True) -> Dict[str, pd.DataFrame]:
    """Generate realistic synthetic HIP-3 market data for backtesting.

    Uses per-asset launch dates from HIP3_LAUNCH_DATES. Each asset gets
    exactly the number of days it has been live on HIP-3 (Oct 2025 → Apr 2026).
    Assets with fewer than min_days of history are excluded.

    Simulates equity / commodity / ETF price dynamics with:
    - Realistic volatility (15-60% annualized depending on asset class)
    - Fat tails (Student-t innovations)
    - Regime switching (bull/bear/crisis)
    - Volume clustering
    - Variable hourly funding rates (Hyperliquid mechanism: hourly settlement,
      premium-driven, 0.01% base interest, capped at 4%/hour). Daily funding
      rate aggregates 24 hourly settlements and is correlated with momentum
      (longs pay in uptrends) and volatility regime (crisis → wider swings).
    """
    rng = np.random.default_rng(seed)

    # Synthetic HIP-3 tokens: verified HIP-3 listings only (via Hyperliquid API)
    # (name, start_price, ann_vol, drift_mult)
    token_params = {
        # ── US Equities ──
        "AAPL":   (185.0,  0.28, 1.10),
        "MSFT":   (375.0,  0.25, 1.08),
        "GOOGL":  (140.0,  0.30, 1.05),
        "AMZN":   (155.0,  0.32, 1.10),
        "NVDA":   (480.0,  0.50, 1.25),
        "META":   (350.0,  0.38, 1.12),
        "TSLA":   (245.0,  0.55, 1.00),
        "AMD":    (145.0,  0.45, 1.15),
        "NFLX":   (485.0,  0.35, 1.08),
        "COIN":   ( 95.0,  0.60, 0.95),
        "PLTR":   ( 70.0,  0.45, 1.10),
        "MSTR":   (350.0,  0.60, 1.05),
        # ── Commodities ──
        "GOLD":   (2050.0, 0.15, 1.02),
        "OIL":    (  75.0, 0.35, 0.90),
        "SILVER": (  24.0, 0.25, 0.98),
        # ── Indices ──
        "SPY":    (475.0,  0.18, 1.08),
        "QQQ":    (410.0,  0.22, 1.12),
    }

    # Regime transition matrix (bull, normal, bear, crisis)
    regimes = ['bull', 'normal', 'bear', 'crisis']
    transition = np.array([
        [0.97, 0.025, 0.004, 0.001],
        [0.05, 0.91,  0.035, 0.005],
        [0.03, 0.10,  0.85,  0.02],
        [0.02, 0.08,  0.15,  0.75],
    ])

    regime_params = {
        'bull':   {'drift': 0.0005, 'vol_mult': 0.7, 'range_mult': 0.8},
        'normal': {'drift': 0.0002, 'vol_mult': 1.0, 'range_mult': 1.0},
        'bear':   {'drift': -0.0006, 'vol_mult': 1.5, 'range_mult': 1.3},
        'crisis': {'drift': -0.0020, 'vol_mult': 2.5, 'range_mult': 2.0},
    }

    data = {}
    count = 0

    for name in HIP3_TOKENS:
        if name not in token_params:
            continue
        asset_days = get_asset_n_days(name) if n_days is None else n_days
        if asset_days < min_days:
            continue
        if count >= n_assets:
            break
        count += 1

        start_price, ann_vol, drift_mult = token_params[name]
        daily_vol = ann_vol / np.sqrt(365)
        launch_date = HIP3_LAUNCH_DATES.get(name, REFERENCE_DATE - timedelta(days=asset_days))

        # Generate regime sequence
        regime_seq = []
        current_regime = 1  # start normal
        for _ in range(asset_days):
            regime_seq.append(current_regime)
            current_regime = rng.choice(4, p=transition[current_regime])

        # Generate returns with regime-dependent dynamics
        timestamps, opens, highs, lows, closes, volumes = [], [], [], [], [], []
        funding_rates_daily = []
        price = start_price
        recent_returns = []

        # Hyperliquid funding mechanism (per docs):
        # - Settled hourly (24x/day), each hour pays = computed_8h_rate / 8
        # - Base interest rate = 0.01% per 8h (~3 bps/day baseline)
        # - Premium component drives variation (long bias → positive funding)
        # - Cap: ±4% per hour. HIP-3 markets may show wider swings than
        #   majors due to lower liquidity. Per-asset multiplier scales spread.
        BASE_INTEREST_DAILY = 0.0001 * 3   # 0.01% per 8h × 3 = ~3 bp/day
        funding_vol_mult = {              # asset-class funding volatility scaling
            'crypto_proxy': 2.5,           # MSTR, COIN — track BTC, wider funding
            'high_beta':    1.8,           # NVDA, TSLA, AMD, PLTR
            'mega_cap':     1.0,           # AAPL, MSFT, AMZN, GOOGL, META, NFLX
            'commodity':    0.7,           # GOLD, SILVER, OIL
            'index':        0.5,           # SPY, QQQ
        }
        asset_class_map = {
            'MSTR': 'crypto_proxy', 'COIN': 'crypto_proxy',
            'NVDA': 'high_beta', 'TSLA': 'high_beta', 'AMD': 'high_beta', 'PLTR': 'high_beta',
            'AAPL': 'mega_cap', 'MSFT': 'mega_cap', 'AMZN': 'mega_cap',
            'GOOGL': 'mega_cap', 'META': 'mega_cap', 'NFLX': 'mega_cap',
            'GOLD': 'commodity', 'SILVER': 'commodity', 'OIL': 'commodity',
            'SPY': 'index', 'QQQ': 'index',
        }
        f_mult = funding_vol_mult[asset_class_map.get(name, 'mega_cap')]
        regime_funding_mult = {'bull': 1.5, 'normal': 1.0, 'bear': 1.5, 'crisis': 3.5}

        # Prefer real cached Hyperliquid funding rates when available.
        real_funding = load_real_funding_daily(name, asset_days) if use_real_funding else None

        for d in range(asset_days):
            regime = regimes[regime_seq[d]]
            rp = regime_params[regime]

            innovation = rng.standard_t(5)
            vol = daily_vol * rp['vol_mult']
            ret = rp['drift'] * drift_mult + vol * innovation

            open_price = price
            close_price = open_price * np.exp(ret)

            range_vol = vol * rp['range_mult']
            high_price = max(open_price, close_price) * (1 + abs(rng.normal(0, range_vol * 0.5)))
            low_price = min(open_price, close_price) * (1 - abs(rng.normal(0, range_vol * 0.5)))

            base_vol = rng.lognormal(10, 1.5)
            vol_regime_mult = {'bull': 1.2, 'normal': 1.0, 'bear': 1.5, 'crisis': 3.0}
            volume = base_vol * vol_regime_mult[regime] * start_price

            # Variable funding rate: use real cached Hyperliquid data when
            # available, else fall back to synthetic (momentum + regime noise).
            if real_funding is not None and d < len(real_funding):
                funding_daily = float(real_funding[d])
            else:
                recent_returns.append(ret)
                if len(recent_returns) > 7:
                    recent_returns.pop(0)
                mom_7d = sum(recent_returns)
                momentum_funding = mom_7d * 0.05 * f_mult
                noise = rng.normal(0, 0.0006) * regime_funding_mult[regime] * f_mult
                funding_daily = BASE_INTEREST_DAILY + momentum_funding + noise
                funding_daily = float(np.clip(funding_daily, -0.02, 0.02))

            timestamps.append(launch_date + timedelta(days=d))
            opens.append(max(open_price, 1e-10))
            highs.append(max(high_price, 1e-10))
            lows.append(max(low_price, 1e-10))
            closes.append(max(close_price, 1e-10))
            volumes.append(volume)
            funding_rates_daily.append(funding_daily)
            price = close_price

        data[name] = pd.DataFrame({
            'timestamp': timestamps, 'open': opens, 'high': highs,
            'low': lows, 'close': closes, 'volume': volumes,
            'funding_rate': funding_rates_daily,
        })

    return data


if __name__ == '__main__':
    print("=" * 80)
    print("  Hyperliquid HIP-3 Market Scanner (Equities / Commodities / ETFs)")
    print("=" * 80)

    client = HyperliquidHIP3Client()

    # Try to fetch live markets
    print("\n1. Discovering HIP-3 spot markets ...")
    try:
        markets = client.get_all_hip3_markets()
        if not markets.empty:
            print(f"\n  Active HIP-3 spot markets ({len(markets)}):")
            for _, row in markets.head(20).iterrows():
                print(f"    {row['token']:<10} Price: ${row['mid_price']:<12.2f}  "
                      f"Vol(24h): ${row['day_volume_usd']:>15,.0f}")
        else:
            print("  No live markets found, generating synthetic data ...")
    except Exception as e:
        print(f"  API error: {e}")
        print("  Generating synthetic data instead ...")

    print("\n2. Generating synthetic HIP-3 data for backtesting ...")
    synthetic = generate_synthetic_hip3_data(n_assets=25, min_days=100)
    for name, df in synthetic.items():
        ret = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        vol = np.std(np.diff(np.log(df['close'].values))) * np.sqrt(365) * 100
        print(f"    {name:<10} {len(df)} bars  Price: ${df['close'].iloc[-1]:>10.2f}  "
              f"Return: {ret:+.1f}%  Vol: {vol:.0f}%")
