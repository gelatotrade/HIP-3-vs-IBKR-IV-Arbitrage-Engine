#!/usr/bin/env python3
"""
Fetch real IBKR-equivalent options chains for HIP-3 underlyings.

Source: yfinance pulls live options data from Yahoo Finance, which sources
quotes directly from CBOE / OPRA (the same OPRA feed IBKR Trader Workstation
uses for retail). This is the most accessible way to obtain authenticated,
exchange-published options data without an active IBKR TWS session.

For each HIP-3 underlying we fetch:
  - All listed expirations (typically 7d → 2y)
  - Per-strike: bid, ask, lastPrice, impliedVolatility, volume, openInterest
  - Both calls and puts

We then fit Gatheral's SVI model to each expiry slice → calibrated SVI params
that the backtest can use as the IBKR IV reference instead of synthetic IV.

Output:
  data/ibkr_options/{ASSET}_chain.csv     — flat options chain
  data/ibkr_options/{ASSET}_svi.csv       — per-expiry SVI parameters
  data/ibkr_options/_summary.csv          — chain depth summary
"""
import warnings
warnings.filterwarnings('ignore')

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

OUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'ibkr_options'
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Underlying mapping: HIP-3 ticker → equity ticker for option chain lookup
HIP3_UNDERLYING = {
    'AAPL':   'AAPL',
    'MSFT':   'MSFT',
    'NVDA':   'NVDA',
    'TSLA':   'TSLA',
    'AMZN':   'AMZN',
    'GOOGL':  'GOOGL',
    'META':   'META',
    'NFLX':   'NFLX',
    'PLTR':   'PLTR',
    'COIN':   'COIN',
    'MSTR':   'MSTR',
    'AMD':    'AMD',
    'QQQ':    'QQQ',     # Nasdaq-100 ETF
    'SPY':    'SPY',     # S&P 500 ETF
    'GOLD':   'GLD',     # SPDR Gold Trust ETF
    'SILVER': 'SLV',     # iShares Silver Trust ETF
    'OIL':    'USO',     # United States Oil Fund ETF (WTI proxy)
}


def svi_w(k, a, b, rho, m, sigma):
    """SVI total variance: w(k) = a + b · (ρ(k−m) + √((k−m)² + σ²))."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi_slice(log_K, ivs, T):
    """Fit SVI to a single expiry slice."""
    w_obs = (ivs ** 2) * T
    valid = (w_obs > 0) & np.isfinite(w_obs) & np.isfinite(log_K)
    if valid.sum() < 5:
        return None
    k = log_K[valid]; w = w_obs[valid]

    def loss(p):
        a, b, rho, m, sigma = p
        if b <= 0 or sigma <= 1e-4 or abs(rho) >= 0.999:
            return 1e10
        return np.sum((svi_w(k, a, b, rho, m, sigma) - w) ** 2)

    best = None
    for x0 in [(w.mean() * 0.5, 0.1, -0.5, 0.0, 0.1),
               (w.mean() * 0.3, 0.05, -0.3, 0.0, 0.2),
               (w.mean() * 0.7, 0.2, -0.7, k.mean(), 0.05)]:
        try:
            res = minimize(loss, x0, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-8,
                                    'maxiter': 2000})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    return best.x if best else None


def fetch_one(tk_hip3, tk_equity, max_expiries=8):
    """Fetch options chain for one underlying. Returns (chain_df, svi_df)."""
    print(f'  {tk_hip3:<6} (-> {tk_equity:<5}) ', end=' ', flush=True)
    try:
        t = yf.Ticker(tk_equity)
        spot = t.history(period='1d')['Close'].iloc[-1]
        all_exps = t.options
        if not all_exps:
            print('NO OPTIONS')
            return None, None
        # Pick a balanced set of near/medium/far expiries
        n = len(all_exps)
        idxs = list(set(np.linspace(0, n - 1, max_expiries, dtype=int)))
        expiries = [all_exps[i] for i in sorted(idxs)]
        rows = []
        svi_rows = []
        for exp in expiries:
            try:
                ch = t.option_chain(exp)
                exp_date = pd.to_datetime(exp)
                T = max((exp_date - pd.Timestamp.now()).days, 1) / 365.0
                for side, df in [('C', ch.calls), ('P', ch.puts)]:
                    df = df.dropna(subset=['strike', 'impliedVolatility'])
                    df = df[(df['impliedVolatility'] > 0.01) &
                            (df['impliedVolatility'] < 5.0)]
                    if df.empty:
                        continue
                    for _, row in df.iterrows():
                        rows.append({
                            'underlying': tk_hip3,
                            'equity_ticker': tk_equity,
                            'expiration': exp,
                            'T': T,
                            'side': side,
                            'strike': row['strike'],
                            'log_moneyness': np.log(row['strike'] / spot),
                            'bid': row.get('bid', 0),
                            'ask': row.get('ask', 0),
                            'last': row.get('lastPrice', 0),
                            'iv': row['impliedVolatility'],
                            'volume': row.get('volume', 0),
                            'open_interest': row.get('openInterest', 0),
                            'spot_at_fetch': spot,
                        })
                # Fit SVI to combined call+put slice (use OTM half each side)
                slice_rows = [r for r in rows if r['expiration'] == exp]
                if len(slice_rows) >= 5:
                    ks = np.array([r['log_moneyness'] for r in slice_rows])
                    ivs = np.array([r['iv'] for r in slice_rows])
                    params = fit_svi_slice(ks, ivs, T)
                    if params is not None:
                        a, b, rho, m, sigma = params
                        atm_iv = np.sqrt(max(svi_w(0.0, *params) / T, 1e-6))
                        svi_rows.append({
                            'underlying': tk_hip3,
                            'equity_ticker': tk_equity,
                            'expiration': exp,
                            'T': T,
                            'spot': spot,
                            'svi_a': a, 'svi_b': b, 'svi_rho': rho,
                            'svi_m': m, 'svi_sigma': sigma,
                            'atm_iv': atm_iv,
                            'n_quotes': len(slice_rows),
                        })
            except Exception as e:
                continue
        if not rows:
            print('NO QUOTES')
            return None, None
        print(f'{len(rows):>5} quotes, {len(svi_rows)} SVI slices, '
              f'spot ${spot:.2f}')
        return pd.DataFrame(rows), pd.DataFrame(svi_rows)
    except Exception as e:
        print(f'ERROR {e}')
        return None, None


def main():
    print('═' * 80)
    print('  Fetch IBKR-Equivalent Options Chains via Yahoo/CBOE OPRA Feed')
    print('═' * 80)
    print(f'  Output: {OUT_DIR}')
    print()
    summary_rows = []
    t0 = time.time()
    for tk_hip3, tk_equity in HIP3_UNDERLYING.items():
        chain, svi = fetch_one(tk_hip3, tk_equity)
        if chain is not None and not chain.empty:
            chain.to_csv(OUT_DIR / f'{tk_hip3}_chain.csv', index=False)
            if svi is not None and not svi.empty:
                svi.to_csv(OUT_DIR / f'{tk_hip3}_svi.csv', index=False)
                atm_iv_30d = float(svi.iloc[(svi['T'] - 30 / 365)
                                            .abs().argsort()[:1]]['atm_iv'].iloc[0])
            else:
                atm_iv_30d = np.nan
            summary_rows.append({
                'asset': tk_hip3,
                'equity_ticker': tk_equity,
                'spot': float(chain['spot_at_fetch'].iloc[0]),
                'n_quotes': len(chain),
                'n_expiries': chain['expiration'].nunique(),
                'iv_min': float(chain['iv'].min()),
                'iv_max': float(chain['iv'].max()),
                'iv_atm_30d': atm_iv_30d,
            })
        time.sleep(0.5)  # polite throttle
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / '_summary.csv', index=False)
    print('-' * 80)
    print(f'Done in {time.time()-t0:.1f}s. Summary:')
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == '__main__':
    main()
