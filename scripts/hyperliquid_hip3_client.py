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
    # ── Wave 1: Core equities (Nov 9-13, 2025) ──
    "AAPL":   datetime(2025, 11, 9),    # 157 days
    "MSFT":   datetime(2025, 11, 12),   # 154 days
    "NVDA":   datetime(2025, 11, 13),   # 153 days
    "TSLA":   datetime(2025, 11, 13),   # 153 days
    # ── Wave 2: More equities (late Nov 2025) ──
    "GOOG":   datetime(2025, 11, 20),   # 146 days
    "AMZN":   datetime(2025, 11, 20),   # 146 days
    "META":   datetime(2025, 11, 22),   # 144 days
    "AMD":    datetime(2025, 11, 25),   # 141 days
    "NFLX":   datetime(2025, 11, 25),   # 141 days
    "COIN":   datetime(2025, 11, 28),   # 138 days
    # ── Wave 3: Extended equities (Dec 2025) ──
    "PLTR":   datetime(2025, 12, 5),    # 131 days
    "MSTR":   datetime(2025, 12, 5),    # 131 days
    "GME":    datetime(2025, 12, 8),    # 128 days
    "UBER":   datetime(2025, 12, 15),   # 121 days
    "SQ":     datetime(2025, 12, 15),   # 121 days
    "SHOP":   datetime(2025, 12, 18),   # 118 days
    "ARM":    datetime(2025, 12, 20),   # 116 days
    "SMCI":   datetime(2025, 12, 22),   # 114 days
    "NKE":    datetime(2025, 12, 22),   # 114 days
    "SNOW":   datetime(2025, 12, 20),   # 116 days
    # ── Commodities (Dec 2025) ──
    "GOLD":   datetime(2025, 12, 10),   # 126 days
    "OIL":    datetime(2025, 12, 12),   # 124 days
    "SILVER": datetime(2025, 12, 15),   # 121 days
    # ── ETFs & Indices (Mar 2026) ──
    "SPY":    datetime(2026, 3, 18),    # 28 days
    "QQQ":    datetime(2026, 3, 20),    # 26 days
}

REFERENCE_DATE = datetime(2026, 4, 15)

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
            # Annualize 8h funding rate → vol premium
            funding_ann = abs(funding) * 3 * 365  # 3x per day
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
def generate_synthetic_hip3_data(n_assets: int = 15, n_days: int = None,
                                  seed: int = 42, min_days: int = 100) -> Dict[str, pd.DataFrame]:
    """Generate realistic synthetic HIP-3 market data for backtesting.

    Uses per-asset launch dates from HIP3_LAUNCH_DATES. Each asset gets
    exactly the number of days it has been live on HIP-3 (Oct 2025 → Apr 2026).
    Assets with fewer than min_days of history are excluded.

    Simulates equity / commodity / ETF price dynamics with:
    - Realistic volatility (15-60% annualized depending on asset class)
    - Fat tails (Student-t innovations)
    - Regime switching (bull/bear/crisis)
    - Volume clustering
    """
    rng = np.random.default_rng(seed)

    # Synthetic HIP-3 tokens: equities, commodities, ETFs
    # (name, start_price, ann_vol, drift_mult)
    token_params = {
        # ── US Equities ──
        "AAPL":   (185.0,  0.28, 1.10),
        "MSFT":   (375.0,  0.25, 1.08),
        "GOOG":   (140.0,  0.30, 1.05),
        "AMZN":   (155.0,  0.32, 1.10),
        "NVDA":   (480.0,  0.50, 1.25),
        "META":   (350.0,  0.38, 1.12),
        "TSLA":   (245.0,  0.55, 1.00),
        "AMD":    (145.0,  0.45, 1.15),
        "NFLX":   (485.0,  0.35, 1.08),
        "COIN":   ( 95.0,  0.60, 0.95),
        # ── Extended equities ──
        "PLTR":   ( 70.0,  0.45, 1.10),
        "MSTR":   (350.0,  0.60, 1.05),
        "GME":    ( 28.0,  0.55, 0.90),
        "UBER":   ( 75.0,  0.35, 1.08),
        "SQ":     ( 80.0,  0.45, 1.05),
        "SHOP":   ( 90.0,  0.40, 1.10),
        "ARM":    (145.0,  0.45, 1.15),
        "SMCI":   ( 40.0,  0.65, 1.00),
        "NKE":    ( 72.0,  0.30, 0.95),
        "SNOW":   (170.0,  0.40, 1.05),
        # ── Commodities ──
        "GOLD":   (2050.0, 0.15, 1.02),
        "OIL":    (  75.0, 0.35, 0.90),
        "SILVER": (  24.0, 0.25, 0.98),
        # ── ETFs / Indices ──
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
        price = start_price

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

            timestamps.append(launch_date + timedelta(days=d))
            opens.append(max(open_price, 1e-10))
            highs.append(max(high_price, 1e-10))
            lows.append(max(low_price, 1e-10))
            closes.append(max(close_price, 1e-10))
            volumes.append(volume)
            price = close_price

        data[name] = pd.DataFrame({
            'timestamp': timestamps, 'open': opens, 'high': highs,
            'low': lows, 'close': closes, 'volume': volumes,
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
