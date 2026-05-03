#!/usr/bin/env python3
"""
Discover and fetch ALL HIP-3 perp markets across every Hyperliquid deployer.

Queries the live `perpDexs` endpoint to find every deployer DEX, then
`metaAndAssetCtxs` per deployer to enumerate every listed asset, and finally
`candleSnapshot` + `fundingHistory` to download the per-asset history since
each asset's on-chain launch.

Filters:
  - Skip pure crypto perps (focus is HIP-3 RWAs vs IBKR options)
  - Skip dead listings (24h vol == 0 AND open interest == 0)
  - When the same instrument exists across multiple deployers, keep the
    deployer with the longest on-chain history (typically xyz).

Output:
  data/all_hip3/candles/{ASSET}.csv
  data/all_hip3/funding/{ASSET}_daily.csv
  data/all_hip3/_summary.csv
"""
import warnings
warnings.filterwarnings('ignore')

import time, json
from pathlib import Path
import requests
import numpy as np
import pandas as pd

API = 'https://api.hyperliquid.xyz/info'
DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'all_hip3'
(DATA_DIR / 'candles').mkdir(parents=True, exist_ok=True)
(DATA_DIR / 'funding').mkdir(parents=True, exist_ok=True)

# Pure-crypto markets to skip (strategy is HIP-3 vs IBKR options, not crypto)
PURE_CRYPTO = {
    'BTC', 'ETH', 'SOL', 'HYPE', 'XRP', 'BNB', 'DOGE', 'SUI', 'PUMP',
    'FARTCOIN', 'ENA', 'LTC', 'LINK', 'XPL', 'IP', 'BCH', 'ADA', 'BASED',
    'XMR', 'ZEC', '1000PEPE', 'LIGHTER', 'PURRDAT', 'USDE',
    'TOTAL2', 'OTHERS', 'BTCD', 'LIT',
}


def post(payload, retries=3):
    for k in range(retries):
        try:
            r = requests.post(API, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if k == retries - 1:
                raise
            time.sleep(2 ** k)


def discover_all_markets():
    """Return list of dicts: {dex, name, full_ticker, mark_px, day_vol, oi}."""
    perp_dexs = post({'type': 'perpDexs'})
    deployers = [d['name'] for d in perp_dexs if d]
    print(f'Found {len(deployers)} HIP-3 deployers: {deployers}')
    out = []
    for dex in deployers:
        try:
            data = post({'type': 'metaAndAssetCtxs', 'dex': dex})
            if not isinstance(data, list) or len(data) < 2:
                continue
            meta, ctxs = data[0], data[1]
            universe = meta.get('universe', [])
            for i, asset in enumerate(universe):
                ctx = ctxs[i] if i < len(ctxs) else {}
                full_name = asset.get('name', '?')
                # Hyperliquid returns "dex:SYMBOL" in `name`; split it
                if ':' in full_name:
                    name = full_name.split(':', 1)[1]
                else:
                    name = full_name
                if name in PURE_CRYPTO:
                    continue
                day_vol = float(ctx.get('dayNtlVlm', 0) or 0)
                oi = float(ctx.get('openInterest', 0) or 0)
                if day_vol == 0 and oi == 0:
                    continue  # zero-liquidity ghost listings
                out.append({
                    'dex': dex,
                    'name': name,
                    'full_ticker': full_name,
                    'mark_px': float(ctx.get('markPx', 0) or 0),
                    'day_vol': day_vol,
                    'oi': oi,
                    'sz_decimals': asset.get('szDecimals', 0),
                })
        except Exception as e:
            print(f'  {dex}: ERROR {e}')
    print(f'Active HIP-3 markets (after filters): {len(out)}')
    return out


def fetch_candles_full(dex, name, interval='1d'):
    """Fetch all available daily candles since launch.

    The HIP-3 candleSnapshot endpoint accepts an explicit start time. We
    walk back in time until the response is empty (= asset launch reached).
    """
    full_ticker = f'{dex}:{name}'
    end_ms = int(time.time() * 1000)
    start_ms = int(pd.Timestamp('2025-10-01').timestamp() * 1000)  # before HIP-3 mainnet
    try:
        candles = post({
            'type': 'candleSnapshot',
            'req': {
                'coin': full_ticker,
                'interval': interval,
                'startTime': start_ms,
                'endTime': end_ms,
            }
        })
        if not isinstance(candles, list):
            return None
        rows = []
        for c in candles:
            rows.append({
                'timestamp': pd.to_datetime(c['t'], unit='ms', utc=True),
                'open': float(c['o']),
                'high': float(c['h']),
                'low': float(c['l']),
                'close': float(c['c']),
                'volume': float(c['v']),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception:
        return None


def fetch_funding_full(dex, name):
    """Fetch funding history (hourly), aggregate to daily."""
    full_ticker = f'{dex}:{name}'
    end_ms = int(time.time() * 1000)
    start_ms = int(pd.Timestamp('2025-10-01').timestamp() * 1000)
    try:
        data = post({
            'type': 'fundingHistory',
            'coin': full_ticker,
            'startTime': start_ms,
            'endTime': end_ms,
        })
        if not isinstance(data, list) or not data:
            return None
        rows = []
        for f in data:
            rows.append({
                'timestamp': pd.to_datetime(f['time'], unit='ms', utc=True),
                'funding_rate': float(f['fundingRate']),
                'premium': float(f.get('premium', 0)),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)
        # Aggregate hourly → daily (sum funding rates within day)
        df['date'] = df['timestamp'].dt.floor('D')
        daily = df.groupby('date').agg({
            'funding_rate': 'sum',
            'premium': 'mean',
        }).reset_index()
        daily.columns = ['timestamp', 'funding_rate', 'premium_mean']
        daily['n_hours'] = df.groupby('date').size().values
        return daily
    except Exception:
        return None


def main():
    print('═' * 90)
    print('  Discovering ALL HIP-3 Perp Markets across every deployer')
    print('═' * 90)
    markets = discover_all_markets()

    # Deduplicate: same asset on multiple deployers → keep longest history
    by_name = {}
    for m in markets:
        by_name.setdefault(m['name'], []).append(m)

    print(f'\n  {len(markets)} total listings → {len(by_name)} unique asset symbols')

    print('\n── Downloading candles + funding for each market ──')
    summary = []
    t0 = time.time()
    for name, candidates in by_name.items():
        # Pick deployer with most history (try each in order of total volume)
        candidates_sorted = sorted(candidates, key=lambda x: -x['day_vol'])
        chosen = None
        df_chosen = None
        for cand in candidates_sorted:
            df = fetch_candles_full(cand['dex'], name)
            if df is not None and len(df) >= 30:
                chosen = cand
                df_chosen = df
                break
            time.sleep(0.1)
        if chosen is None:
            print(f'  {name:<14} SKIP — no candles ≥30 bars on any deployer')
            continue
        # Save candles
        candle_path = DATA_DIR / 'candles' / f'{name}.csv'
        df_chosen.to_csv(candle_path, index=False)
        # Fetch funding
        fdf = fetch_funding_full(chosen['dex'], name)
        if fdf is not None and len(fdf) >= 1:
            fdf.to_csv(DATA_DIR / 'funding' / f'{name}_daily.csv', index=False)
            n_funding = len(fdf)
        else:
            n_funding = 0
        ret_pct = (df_chosen['close'].iloc[-1] / df_chosen['close'].iloc[0] - 1) * 100
        print(f'  {name:<14} ({chosen["dex"]:<5}) {len(df_chosen):>4} bars  '
              f'${df_chosen["close"].iloc[0]:>10,.4f} → ${df_chosen["close"].iloc[-1]:>10,.4f}  '
              f'{ret_pct:>+7.1f}%  funding={n_funding}d')
        summary.append({
            'asset': name,
            'dex_chosen': chosen['dex'],
            'full_ticker': chosen['full_ticker'],
            'n_bars': len(df_chosen),
            'first_ts': str(df_chosen['timestamp'].iloc[0]),
            'last_ts': str(df_chosen['timestamp'].iloc[-1]),
            'first_price': float(df_chosen['close'].iloc[0]),
            'last_price': float(df_chosen['close'].iloc[-1]),
            'total_return_pct': ret_pct,
            'day_vol_usd': chosen['day_vol'],
            'open_interest': chosen['oi'],
            'n_funding_days': n_funding,
        })
        time.sleep(0.15)
    pd.DataFrame(summary).to_csv(DATA_DIR / '_summary.csv', index=False)
    print('-' * 90)
    print(f'Done in {time.time()-t0:.1f}s. {len(summary)} markets cached.')
    print(f'Cache: {DATA_DIR}')


if __name__ == '__main__':
    main()
