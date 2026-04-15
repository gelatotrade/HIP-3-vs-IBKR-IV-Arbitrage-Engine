#!/usr/bin/env python3
"""
HIP-3 Market Backtest — Walk-Forward with IV Arbitrage & Greeks Strategies.

Architecture:
  1. BASE: Long/short HIP-3 spot tokens based on regime + momentum
  2. OVERLAY: IV arbitrage (HIP-3 IV vs IBKR IV vol spread)
  3. GREEKS: Second/third order Greek strategies ensemble
  4. REGIME: Controls position sizing, spread width, strategy selection

Walk-forward: 60% train / 40% test
Statistical validation: Sharpe t-test, block bootstrap, permutation test, deflated SR

Usage:  python3 scripts/hip3_backtest.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys, time
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperliquid_hip3_client import generate_synthetic_hip3_data
from iv_arbitrage_engine import IVArbitrageEngine
from greeks_strategies import GreeksStrategyEngine, classify_regime

# ── Config ──
TRAIN_PCT      = 0.60
MIN_TRAIN_BARS = 60
MIN_TEST_BARS  = 30

PARAM_GRID = {
    'n_levels':       [8, 12, 18],
    'level_step_bps': [15, 30, 50],
    'order_size':     [0.01, 0.02],
    'crisis_vol':     [0.25, 0.35, 0.50],
    'crisis_trim':    [0.15, 0.30],
    'ema_len':        [5, 10],
    'iv_arb_weight':  [0.3, 0.5],
}

HL_MAKER_FEE_BPS  = 0.2
HL_TAKER_FEE_BPS  = 0.5
HL_FUNDING_BPS    = 1.0
ADVERSE_SEL       = 0.40

N_BOOTSTRAP    = 3000
BLOCK_SIZE     = 15
N_PERMUTATIONS = 3000
SIGNIFICANCE   = 0.05

OUT_DIR = Path(__file__).resolve().parent.parent / 'results'


def run_hip3_backtest(opens, highs, lows, closes, params, hip3_iv, ibkr_iv):
    n_levels   = params['n_levels']
    step_bps   = params['level_step_bps']
    order_sz   = params['order_size']
    crisis_vol = params['crisis_vol']
    crisis_trim= params['crisis_trim']
    ema_len    = params['ema_len']
    iv_weight  = params['iv_arb_weight']

    N = len(closes)
    ema = np.empty(N); ema[0] = closes[0]
    a = 2.0 / (ema_len + 1)
    for i in range(1, N):
        ema[i] = a * closes[i] + (1 - a) * ema[i-1]

    rets = np.zeros(N)
    rets[1:] = np.diff(np.log(closes))

    daily_pnl   = np.zeros(N)
    daily_bench  = np.zeros(N)
    total_fills  = 0
    total_spread = 0.0
    total_iv_pnl = 0.0

    for i in range(1, N):
        regime = classify_regime(rets, i, crisis_vol)

        if regime == 'CRISIS':
            base, spread_mult, size_mult = 0.5 - crisis_trim, 3.0, 0.3
        elif regime == 'CAUTIOUS':
            base, spread_mult, size_mult = 0.7, 2.0, 0.5
        elif regime == 'RECOVERY':
            base, spread_mult, size_mult = 1.05, 1.3, 1.1
        elif regime == 'BULL':
            base, spread_mult, size_mult = 1.10, 0.8, 1.2
        else:
            base, spread_mult, size_mult = 1.0, 1.0, 1.0

        fair = ema[i-1]
        lo, hi = lows[i], highs[i]

        bar_spread_pnl = 0.0
        bar_fills = 0
        sz = order_sz * size_mult
        daily_range_bps = (hi - lo) / fair * 10_000 if fair > 0 else 0

        for lv in range(1, n_levels + 1):
            offset = fair * step_bps * lv * spread_mult / 10_000
            level_dist_bps = step_bps * lv * spread_mult
            bid, ask = fair - offset, fair + offset

            repeats = max(1, min(int(daily_range_bps / (2 * max(level_dist_bps, 1))), 6))

            if lo <= bid:
                capture = sz * repeats * offset / fair * (1.0 - ADVERSE_SEL)
                bar_spread_pnl += capture - sz * repeats * HL_MAKER_FEE_BPS / 10_000
                bar_fills += repeats
            if hi >= ask:
                capture = sz * repeats * offset / fair * (1.0 - ADVERSE_SEL)
                bar_spread_pnl += capture - sz * repeats * HL_MAKER_FEE_BPS / 10_000
                bar_fills += repeats

        iv_pnl = 0.0
        if i < len(hip3_iv) and i < len(ibkr_iv) and hip3_iv[i] > 0 and ibkr_iv[i] > 0:
            vol_spread = hip3_iv[i] - ibkr_iv[i]
            iv_signal = -np.sign(vol_spread) * min(abs(vol_spread) * 2, 0.10)
            iv_pnl = iv_signal * abs(rets[i]) * iv_weight
            iv_pnl -= abs(iv_signal) * HL_FUNDING_BPS / 10_000

        daily_pnl[i]  = base * rets[i] + bar_spread_pnl + iv_pnl
        daily_bench[i] = rets[i]
        total_fills  += bar_fills
        total_spread += bar_spread_pnl
        total_iv_pnl += iv_pnl

    return daily_pnl, daily_bench, total_fills, total_spread, total_iv_pnl


def compute_metrics(s, b, fills, spread_pnl, iv_pnl):
    n = len(s)
    if n < 30:
        return None
    ann = 365
    ann_s = np.mean(s) * ann
    ann_b = np.mean(b) * ann
    vol_s = np.std(s, ddof=1) * np.sqrt(ann)
    vol_b = np.std(b, ddof=1) * np.sqrt(ann)
    sh_s = ann_s / vol_s if vol_s > 1e-10 else 0
    sh_b = ann_b / vol_b if vol_b > 1e-10 else 0

    cum_s = np.exp(np.cumsum(s))
    dd_s = (np.maximum.accumulate(cum_s) - cum_s) / np.maximum.accumulate(cum_s)
    cum_b = np.exp(np.cumsum(b))
    dd_b = (np.maximum.accumulate(cum_b) - cum_b) / np.maximum.accumulate(cum_b)

    down = s[s < 0]
    dv = np.std(down, ddof=1) * np.sqrt(ann) if len(down) > 2 else vol_s
    sortino = ann_s / dv if dv > 1e-10 else 0
    alpha = ann_s - ann_b
    tr = np.std(s - b, ddof=1) * np.sqrt(ann)
    ir = alpha / tr if tr > 1e-10 else 0
    calmar = ann_s / dd_s.max() if dd_s.max() > 1e-10 else 0
    calmar_b = ann_b / dd_b.max() if dd_b.max() > 1e-10 else 0

    return {
        'ann_ret': ann_s, 'bench_ret': ann_b, 'alpha': alpha,
        'sharpe': sh_s, 'sharpe_bench': sh_b,
        'sortino': sortino, 'calmar': calmar, 'calmar_bench': calmar_b,
        'max_dd': dd_s.max(), 'max_dd_bench': dd_b.max(),
        'info_ratio': ir,
        'total_ret': np.exp(np.sum(s)) - 1,
        'bench_total': np.exp(np.sum(b)) - 1,
        'n_bars': n, 'total_fills': fills,
        'fills_per_day': fills / n,
        'spread_ann': spread_pnl / n * ann,
        'iv_arb_ann': iv_pnl / n * ann,
    }


def walk_forward(o, h, l, c, hip3_iv, ibkr_iv):
    N = len(c)
    split = int(N * TRAIN_PCT)
    if split < MIN_TRAIN_BARS or (N - split) < MIN_TEST_BARS:
        return None

    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))

    best_score, best_params = -999, dict(zip(keys, combos[0]))
    for combo in combos:
        params = dict(zip(keys, combo))
        s, b, fl, sp, iv = run_hip3_backtest(
            o[:split], h[:split], l[:split], c[:split],
            params, hip3_iv[:split], ibkr_iv[:split])
        m = compute_metrics(s, b, fl, sp, iv)
        if m:
            score = m['sharpe']*0.3 + m['calmar']*0.2 + m['alpha']*5.0 + m['iv_arb_ann']*3.0
            if score > best_score:
                best_score, best_params = score, params

    s, b, fl, sp, iv = run_hip3_backtest(
        o[split:], h[split:], l[split:], c[split:],
        best_params, hip3_iv[split:], ibkr_iv[split:])
    m = compute_metrics(s, b, fl, sp, iv)
    if m is None:
        return None
    m.update(best_params)
    m['train_score'] = best_score
    m['train_bars']  = split
    m['test_bars']   = N - split
    m['strat_returns'] = s
    m['bench_returns'] = b
    return m


# ── Statistical tests ──
def sharpe_ttest(r):
    n = len(r)
    if n < 20: return {'p_value': 1.0, 'significant': False}
    mu = np.mean(r)*365; sig = np.std(r,ddof=1)*np.sqrt(365)
    sr = mu/sig if sig>1e-10 else 0
    rho = np.corrcoef(r[:-1],r[1:])[0,1] if len(r)>1 else 0
    eta = max(1+2*rho, 0.5)
    se = np.sqrt(eta/n)*np.sqrt(1+0.5*sr**2)
    t = sr/se if se>1e-10 else 0
    p = 1-sp_stats.norm.cdf(t)
    return {'p_value':p, 'significant':p<SIGNIFICANCE, 'sharpe':sr}

def block_bootstrap(r):
    n=len(r); rng=np.random.default_rng(42)
    nb=(n//BLOCK_SIZE)+1; off=np.arange(BLOCK_SIZE)
    shs=np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        st=rng.integers(0,n,size=nb)
        ix=(st[:,None]+off[None,:]).ravel()%n
        s_=r[ix[:n]]; mu=np.mean(s_)*365; sig=np.std(s_,ddof=1)*np.sqrt(365)
        shs[b]=mu/sig if sig>1e-10 else 0
    return {'ci_lo':np.percentile(shs,2.5),'ci_hi':np.percentile(shs,97.5),
            'p_value':np.mean(shs<=0),'significant':np.percentile(shs,2.5)>0}

def perm_test(s,b):
    obs=np.mean(s)-np.mean(b); rng=np.random.default_rng(42)
    d=s-b; signs=rng.choice([-1.,1.],size=(N_PERMUTATIONS,len(d)))
    pd_=(signs*d[None,:]).mean(axis=1); p=np.mean(pd_>=obs)
    return {'alpha_ann':obs*365,'p_value':p,'significant':p<SIGNIFICANCE}

def deflated_sr(sr,nt,no):
    if nt<=1: return {'dsr':1.0,'significant':True}
    em=np.sqrt(2*np.log(nt))*(1-0.5772/np.sqrt(2*np.log(nt)))
    se=np.sqrt((1+0.5*sr**2)/no)
    if se<1e-10: return {'dsr':0,'significant':False}
    z=(sr-em)/se; p=sp_stats.norm.cdf(z)
    return {'dsr':p,'significant':p>0.95}


def run_greeks_backtest(prices, hip3_iv, ibkr_iv):
    engine = GreeksStrategyEngine()
    results = engine.run_all_strategies(prices, hip3_iv, ibkr_iv)
    ensemble_pnl = engine.ensemble_strategy(results)
    return results, ensemble_pnl


def main():
    print('='*90)
    print('  HIP-3 MARKET BACKTEST — IV Arbitrage + Greeks Strategies')
    print('  HIP-3 vs IBKR | Walk-forward: 60%/40% | Regime-adaptive')
    print('='*90)

    print('\nGenerating HIP-3 synthetic market data (per-asset history since HIP-3 launch) ...')
    data = generate_synthetic_hip3_data(n_assets=25, min_days=100)

    arb_engine = IVArbitrageEngine()
    nc = len(list(product(*PARAM_GRID.values())))
    print(f'Grid: {nc} combos | Hyperliquid fees | IV arb + Greeks overlay')

    results = []; curves = {}; greeks_results_all = {}
    t0 = time.time()

    for tk, df in data.items():
        print(f'\n  {tk} ...', end=' ', flush=True)
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        if len(c) < MIN_TRAIN_BARS + MIN_TEST_BARS:
            print(f'SKIP ({len(c)} bars)'); continue

        hip3_iv = arb_engine.compute_hip3_implied_vol(c)
        ibkr_iv = arb_engine.compute_ibkr_atm_iv(c)

        wf = walk_forward(o, h, l, c, hip3_iv, ibkr_iv)
        if wf is None:
            print('SKIP'); continue

        sr = wf.pop('strat_returns'); br = wf.pop('bench_returns')

        split = int(len(c) * TRAIN_PCT)
        gk_results, gk_ensemble = run_greeks_backtest(c[split:], hip3_iv[split:], ibkr_iv[split:])
        greeks_results_all[tk] = gk_results

        min_len = min(len(sr), len(gk_ensemble))
        combined = sr[:min_len] + gk_ensemble[:min_len] * 0.3
        curves[tk] = (combined, br[:min_len])

        st1=sharpe_ttest(sr); st2=block_bootstrap(sr)
        st3=perm_test(sr,br); st4=deflated_sr(wf['sharpe'],nc,wf['test_bars'])

        row = {'asset':tk, **wf}
        row['sr_pval']=st1['p_value']; row['sr_sig']=st1['significant']
        row['boot_ci_lo']=st2['ci_lo']; row['boot_ci_hi']=st2['ci_hi']
        row['boot_pval']=st2['p_value']; row['boot_sig']=st2['significant']
        row['perm_pval']=st3['p_value']; row['perm_sig']=st3['significant']
        row['dsr']=st4['dsr']; row['dsr_sig']=st4['significant']

        best_greek = max(gk_results.values(), key=lambda x: x.sharpe)
        row['best_greek_strategy'] = best_greek.name
        row['best_greek_sharpe'] = best_greek.sharpe
        row['greeks_ensemble_ret'] = float(np.sum(gk_ensemble))
        results.append(row)

        pv = f'p=[{st1["p_value"]:.4f} {st2["p_value"]:.4f} {st3["p_value"]:.4f}]'
        print(f'Alpha={wf["alpha"]*100:+.1f}%  Sharpe={wf["sharpe"]:.2f}  '
              f'IVarb={wf["iv_arb_ann"]*100:.1f}%  '
              f'Greek={best_greek.name}({best_greek.sharpe:.2f})  {pv}')

    if not results: sys.exit('No results.')
    df_res = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / 'hip3_backtest_results.csv'
    df_res.to_csv(csv, index=False, float_format='%.6f')

    print('\n'+'='*140)
    print('  OUT-OF-SAMPLE RESULTS — HIP-3 IV Arbitrage + Greeks Strategies')
    print('='*140)

    def fmt_p(p):
        if p < 0.001: return '<.001*'
        if p < 0.05: return f'{p:.3f}*'
        return f'{p:.3f} '

    print(f'\n{"Asset":<8} {"Alpha":>7} {"Sharpe":>7} {"S.Bn":>6} '
          f'{"Calmar":>7} {"MaxDD":>6} {"DD.Bn":>6} '
          f'{"IVarb":>6} {"Fil/d":>5} '
          f'{"BestGreek":<16} {"GkSR":>5} '
          f'{"p(SR)":>7} {"p(Bt)":>7} {"p(Pm)":>7}')
    print('-'*130)

    pa = 0
    for _, r in df_res.sort_values('alpha', ascending=False).iterrows():
        if r['alpha'] > 0: pa += 1
        print(f'{r["asset"]:<8} {r["alpha"]*100:>+6.1f}% {r["sharpe"]:>7.2f} '
              f'{r["sharpe_bench"]:>6.2f} '
              f'{r["calmar"]:>7.2f} {r["max_dd"]*100:>5.1f}% {r["max_dd_bench"]*100:>5.1f}% '
              f'{r["iv_arb_ann"]*100:>5.1f}% {r["fills_per_day"]:>5.0f} '
              f'{r["best_greek_strategy"]:<16} {r["best_greek_sharpe"]:>5.2f} '
              f'{fmt_p(r["sr_pval"]):>7} {fmt_p(r["boot_pval"]):>7} '
              f'{fmt_p(r["perm_pval"]):>7}')

    print(f'\n── Summary ──')
    print(f'  Positive alpha:    {pa}/{len(df_res)}')
    print(f'  Mean alpha:        {df_res["alpha"].mean()*100:+.1f}%')
    print(f'  Mean Sharpe:       {df_res["sharpe"].mean():.2f}  (bench: {df_res["sharpe_bench"].mean():.2f})')
    print(f'  Mean Calmar:       {df_res["calmar"].mean():.2f}')
    print(f'  Mean MaxDD:        {df_res["max_dd"].mean()*100:.1f}%  (bench: {df_res["max_dd_bench"].mean()*100:.1f}%)')
    print(f'  Mean IV arb/yr:    {df_res["iv_arb_ann"].mean()*100:.1f}%')
    print(f'  Mean fills/day:    {df_res["fills_per_day"].mean():.0f}')
    print(f'  Time: {time.time()-t0:.1f}s | CSV: {csv}')

    return df_res, curves, greeks_results_all


if __name__ == '__main__':
    df_res, curves, greeks_all = main()
