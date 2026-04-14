#!/usr/bin/env python3
"""
Hyperliquid HIP-3 API Client — Fetch all HIP-3 stock/spot markets.

HIP-3 = Hyperliquidity Provider for spot tokens on Hyperliquid DEX.
These markets have AMM-style liquidity with different IV characteristics
compared to traditional options markets (IBKR).

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

# Known HIP-3 spot tokens (stock/equity-linked and major tokens)
# HIP-3 markets include tokenized equities and high-liquidity spot pairs
HIP3_TOKENS = [
    # Tokenized equities / stock-like HIP-3 markets
    "PURR", "HFUN", "JEFF", "TRUMP", "MELANIA",
    # Major crypto with HIP-3 liquidity
    "BTC", "ETH", "SOL", "AVAX", "ARB", "OP", "SUI", "APT",
    "DOGE", "PEPE", "WIF", "BONK", "ONDO", "TIA", "SEI",
    "INJ", "PYTH", "JTO", "JUP", "W", "STRK", "DYM",
    "MANTA", "ALT", "PIXEL", "PORTAL", "AEVO", "ENA",
]

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
            funding_ann = abs(funding) * 3 * 365  # 3x per day, 365 days
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
def generate_synthetic_hip3_data(n_assets: int = 15, n_days: int = 730,
                                  seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Generate realistic synthetic HIP-3 market data for backtesting.

    Simulates crypto-like price dynamics with:
    - Higher volatility than equities (50-150% annualized)
    - Fat tails (Student-t innovations)
    - Regime switching (bull/bear/crisis)
    - Volume clustering
    """
    rng = np.random.default_rng(seed)

    # Synthetic HIP-3 tokens with realistic parameters
    token_params = [
        ("PURR",    1.2,   1.20, 0.85),  # (name, start_price, ann_vol, drift_mult)
        ("HFUN",    0.05,  1.50, 0.70),
        ("JEFF",    3.50,  0.95, 0.90),
        ("BTC",     42000, 0.65, 1.10),
        ("ETH",     2200,  0.75, 1.05),
        ("SOL",     95,    0.90, 1.15),
        ("AVAX",    32,    0.85, 0.95),
        ("ARB",     1.15,  1.00, 0.90),
        ("OP",      2.80,  0.95, 0.92),
        ("SUI",     1.45,  1.10, 1.00),
        ("APT",     8.50,  0.88, 0.95),
        ("DOGE",    0.08,  1.05, 0.80),
        ("PEPE",    0.000001, 1.60, 0.75),
        ("WIF",     0.50,  1.40, 0.85),
        ("ONDO",    0.70,  1.00, 1.05),
    ]

    # Regime transition matrix (bull, normal, bear, crisis)
    regimes = ['bull', 'normal', 'bear', 'crisis']
    transition = np.array([
        [0.96, 0.03, 0.008, 0.002],
        [0.04, 0.90, 0.05,  0.01],
        [0.02, 0.08, 0.87,  0.03],
        [0.01, 0.05, 0.14,  0.80],
    ])

    regime_params = {
        'bull':   {'drift': 0.0015, 'vol_mult': 0.7, 'range_mult': 0.8},
        'normal': {'drift': 0.0003, 'vol_mult': 1.0, 'range_mult': 1.0},
        'bear':   {'drift': -0.0010, 'vol_mult': 1.4, 'range_mult': 1.3},
        'crisis': {'drift': -0.0035, 'vol_mult': 2.5, 'range_mult': 2.0},
    }

    data = {}
    tokens_to_use = token_params[:n_assets]

    for name, start_price, ann_vol, drift_mult in tokens_to_use:
        daily_vol = ann_vol / np.sqrt(365)

        # Generate regime sequence
        regime_seq = []
        current_regime = 1  # start normal
        for _ in range(n_days):
            regime_seq.append(current_regime)
            current_regime = rng.choice(4, p=transition[current_regime])

        # Generate returns with regime-dependent dynamics
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        price = start_price
        base_date = datetime(2023, 1, 1)

        for d in range(n_days):
            regime = regimes[regime_seq[d]]
            rp = regime_params[regime]

            # Student-t innovations (df=5 for fat tails)
            innovation = rng.standard_t(5)
            vol = daily_vol * rp['vol_mult']
            ret = rp['drift'] * drift_mult + vol * innovation

            open_price = price
            close_price = open_price * np.exp(ret)

            # Intraday range
            range_vol = vol * rp['range_mult']
            intraday_range = abs(rng.normal(0, range_vol * 2))
            high_price = max(open_price, close_price) * (1 + abs(rng.normal(0, range_vol * 0.5)))
            low_price = min(open_price, close_price) * (1 - abs(rng.normal(0, range_vol * 0.5)))

            # Volume with clustering
            base_vol = rng.lognormal(10, 1.5)
            vol_regime_mult = {'bull': 1.2, 'normal': 1.0, 'bear': 1.5, 'crisis': 3.0}
            volume = base_vol * vol_regime_mult[regime] * start_price

            timestamps.append(base_date + timedelta(days=d))
            opens.append(max(open_price, 1e-10))
            highs.append(max(high_price, 1e-10))
            lows.append(max(low_price, 1e-10))
            closes.append(max(close_price, 1e-10))
            volumes.append(volume)

            price = close_price

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
        })
        data[name] = df

    return data


if __name__ == '__main__':
    print("=" * 80)
    print("  Hyperliquid HIP-3 Market Scanner")
    print("=" * 80)

    client = HyperliquidHIP3Client()

    # Try to fetch live markets
    print("\n1. Discovering HIP-3 spot markets ...")
    try:
        markets = client.get_all_hip3_markets()
        if not markets.empty:
            print(f"\n  Active HIP-3 spot markets ({len(markets)}):")
            for _, row in markets.head(20).iterrows():
                print(f"    {row['token']:<10} Price: ${row['mid_price']:<12.6f}  "
                      f"Vol(24h): ${row['day_volume_usd']:>15,.0f}")
        else:
            print("  No live markets found, generating synthetic data ...")
    except Exception as e:
        print(f"  API error: {e}")
        print("  Generating synthetic data instead ...")

    print("\n2. Generating synthetic HIP-3 data for backtesting ...")
    synthetic = generate_synthetic_hip3_data(n_assets=15, n_days=730)
    for name, df in synthetic.items():
        ret = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        vol = np.std(np.diff(np.log(df['close'].values))) * np.sqrt(365) * 100
        print(f"    {name:<10} {len(df)} bars  Price: ${df['close'].iloc[-1]:.6f}  "
              f"Return: {ret:+.1f}%  Vol: {vol:.0f}%")
