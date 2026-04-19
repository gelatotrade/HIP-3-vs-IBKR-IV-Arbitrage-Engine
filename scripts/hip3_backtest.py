#!/usr/bin/env python3
"""
HIP-3 Market Backtest — Purged K-Fold CV with IV Arbitrage & Greeks Strategies.

Architecture:
  1. BASE: Long/short HIP-3 spot tokens based on regime + momentum
  2. OVERLAY: IV arbitrage (HIP-3 IV vs IBKR IV vol spread)
  3. GREEKS: Second/third order Greek strategies ensemble
  4. REGIME: Controls position sizing, spread width, strategy selection

Methodology: Purged expanding-window K-fold CV (Lopez de Prado 2018, Ch.7)
  - Expanding training window with purge + embargo gap at fold boundaries
  - Hansen's SPA test (2005) for multiple-testing correction across 432 param combos
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
  - Fold-level Sharpe distribution for robustness assessment

Statistical validation: paired t-test (t-distribution), Sharpe t-test (Lo 2002),
  block bootstrap, permutation test, deflated SR, Hansen's SPA

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

from hyperliquid_hip3_client import (
    generate_synthetic_hip3_data, load_real_hip3_data,
)
from iv_arbitrage_engine import IVArbitrageEngine
from greeks_strategies import GreeksStrategyEngine, classify_regime

# ── Config ──
PURGE_BARS     = 2
EMBARGO_BARS   = 3
MIN_TRAIN_BARS = 40
MIN_TEST_BARS  = 15

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
ADVERSE_SEL       = 0.40
# Hyperliquid funding is settled HOURLY (24x/day). Real funding is variable
# per bar (passed via funding_rates argument). Fallback below is the daily
# baseline if no rate provided: 0.01% per 8h = ~3 bp/day.
HL_FUNDING_FALLBACK_DAILY = 0.0003

N_BOOTSTRAP    = 3000
BLOCK_SIZE     = 15
N_PERMUTATIONS = 3000
SIGNIFICANCE   = 0.05

OUT_DIR = Path(__file__).resolve().parent.parent / 'results'


def run_hip3_backtest(opens, highs, lows, closes, params, hip3_iv, ibkr_iv,
                      funding_rates=None):
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

    daily_pnl    = np.zeros(N)
    daily_bench  = np.zeros(N)
    total_fills  = 0
    total_spread = 0.0
    total_iv_pnl = 0.0
    total_funding= 0.0

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

        # Variable funding rate (Hyperliquid hourly settlement, aggregated daily).
        # Long position pays funding when rate > 0; short receives.
        if funding_rates is not None and i < len(funding_rates):
            f_rate = float(funding_rates[i])
        else:
            f_rate = HL_FUNDING_FALLBACK_DAILY

        iv_pnl = 0.0
        iv_signal = 0.0
        if i < len(hip3_iv) and i < len(ibkr_iv) and hip3_iv[i] > 0 and ibkr_iv[i] > 0:
            vol_spread = hip3_iv[i] - ibkr_iv[i]
            iv_signal = -np.sign(vol_spread) * min(abs(vol_spread) * 2, 0.10)
            iv_pnl = iv_signal * abs(rets[i]) * iv_weight
            # IV arb leg pays funding proportional to position size and direction
            iv_pnl -= iv_signal * f_rate * iv_weight

        # Base position pays funding too: long perp pays when funding>0, short receives.
        # Sign convention: positive base means long, so funding cost = +base * f_rate.
        base_funding_cost = base * f_rate

        daily_pnl[i]  = base * rets[i] + bar_spread_pnl + iv_pnl - base_funding_cost
        daily_bench[i] = rets[i]
        total_fills  += bar_fills
        total_spread += bar_spread_pnl
        total_iv_pnl += iv_pnl
        total_funding += base_funding_cost

    return daily_pnl, daily_bench, total_fills, total_spread, total_iv_pnl


def compute_metrics(s, b, fills, spread_pnl, iv_pnl):
    n = len(s)
    if n < min(MIN_TEST_BARS, 15):
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


def purged_kfold_cv(o, h, l, c, hip3_iv, ibkr_iv, funding_rates=None):
    """Purged expanding-window K-fold CV (Lopez de Prado 2018, Ch.7).

    For each fold k, train on [0, fold_start - purge - embargo) and test on fold k.
    Purge+embargo gaps prevent information leakage at fold boundaries.
    All 432 param combos evaluated on each test fold for Hansen's SPA test.
    Variable funding rates (Hyperliquid hourly settlement) applied to base
    and IV-arb position legs.
    """
    N = len(c)
    fold_size = max(N // 5, 20)
    boundaries = list(range(0, N, fold_size))
    if boundaries[-1] != N:
        boundaries.append(N)

    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    n_combos = len(combos)

    all_oos_strat = []
    all_oos_bench = []
    fold_sharpes = []
    fold_params = []
    total_fills = 0
    total_spread = 0.0
    total_iv_pnl = 0.0
    n_folds_used = 0
    first_test_start = None
    combo_oos_excess = [[] for _ in range(n_combos)]

    for k in range(1, len(boundaries) - 1):
        test_start = boundaries[k]
        test_end = boundaries[k + 1]
        test_len = test_end - test_start
        train_end = max(0, test_start - PURGE_BARS - EMBARGO_BARS)

        if train_end < MIN_TRAIN_BARS or test_len < MIN_TEST_BARS:
            continue

        if first_test_start is None:
            first_test_start = test_start

        best_score, best_idx = -999, 0
        for ci, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            f_train = funding_rates[:train_end] if funding_rates is not None else None
            s, b, fl, sp, iv = run_hip3_backtest(
                o[:train_end], h[:train_end], l[:train_end], c[:train_end],
                params, hip3_iv[:train_end], ibkr_iv[:train_end], f_train)
            m = compute_metrics(s, b, fl, sp, iv)
            if m:
                score = m['sharpe']*0.3 + m['calmar']*0.2 + m['alpha']*5.0 + m['iv_arb_ann']*3.0
                if score > best_score:
                    best_score, best_idx = score, ci

        f_test = funding_rates[test_start:test_end] if funding_rates is not None else None
        for ci, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            s_c, b_c, _, _, _ = run_hip3_backtest(
                o[test_start:test_end], h[test_start:test_end],
                l[test_start:test_end], c[test_start:test_end],
                params, hip3_iv[test_start:test_end], ibkr_iv[test_start:test_end], f_test)
            combo_oos_excess[ci].extend(list(s_c - b_c))

        best_params = dict(zip(keys, combos[best_idx]))
        s, b, fl, sp, iv = run_hip3_backtest(
            o[test_start:test_end], h[test_start:test_end],
            l[test_start:test_end], c[test_start:test_end],
            best_params, hip3_iv[test_start:test_end], ibkr_iv[test_start:test_end], f_test)

        all_oos_strat.extend(list(s))
        all_oos_bench.extend(list(b))
        total_fills += fl
        total_spread += sp
        total_iv_pnl += iv
        n_folds_used += 1
        fold_params.append(best_params)

        if len(s) >= 10:
            mu = np.mean(s) * 365
            sig = np.std(s, ddof=1) * np.sqrt(365)
            fold_sharpes.append(mu / sig if sig > 1e-10 else 0)

    if n_folds_used == 0 or len(all_oos_strat) < MIN_TEST_BARS:
        return None

    sr = np.array(all_oos_strat)
    br = np.array(all_oos_bench)
    m = compute_metrics(sr, br, total_fills, total_spread, total_iv_pnl)
    if m is None:
        return None

    m.update(fold_params[-1])
    m['train_score'] = best_score
    m['train_bars'] = train_end
    m['test_bars'] = len(sr)
    m['strat_returns'] = sr
    m['bench_returns'] = br
    m['n_folds'] = n_folds_used
    m['fold_sharpes'] = fold_sharpes
    m['sharpe_std'] = np.std(fold_sharpes, ddof=1) if len(fold_sharpes) > 1 else 0.0
    m['oos_pct'] = len(sr) / N
    m['combo_oos_excess'] = combo_oos_excess
    m['first_test_start'] = first_test_start
    return m


# ── Statistical tests ──
def ttest_excess(s, b):
    """Paired t-test on excess returns (strategy - benchmark).
    H0: mean(excess) = 0,  H1: mean(excess) > 0  (one-sided).
    Uses scipy t-distribution with n-1 degrees of freedom.
    """
    excess = s - b
    n = len(excess)
    if n < 10:
        return {'t_stat': 0.0, 'df': n-1, 'p_value': 1.0, 'significant': False}
    t_stat, p_two = sp_stats.ttest_1samp(excess, 0.0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    return {'t_stat': float(t_stat), 'df': n-1,
            'p_value': float(p_one), 'significant': p_one < SIGNIFICANCE}

def ttest_sharpe(r):
    """Sharpe-ratio t-test with Lo (2002) autocorrelation-adjusted SE.
    Uses t-distribution (not normal) for proper finite-sample inference.
    H0: SR = 0,  H1: SR > 0  (one-sided).
    """
    n = len(r)
    if n < 20:
        return {'t_stat': 0.0, 'df': n-1, 'p_value': 1.0, 'significant': False, 'sharpe': 0.0}
    mu = np.mean(r)*365; sig = np.std(r, ddof=1)*np.sqrt(365)
    sr = mu/sig if sig > 1e-10 else 0
    # Lo (2002) autocorrelation adjustment
    rho = np.corrcoef(r[:-1], r[1:])[0, 1] if len(r) > 1 else 0
    eta = max(1 + 2*rho, 0.5)
    se = np.sqrt(eta/n) * np.sqrt(1 + 0.5*sr**2)
    t = sr/se if se > 1e-10 else 0
    # t-distribution with n-1 df (not normal)
    p = 1 - sp_stats.t.cdf(t, df=n-1)
    return {'t_stat': float(t), 'df': n-1,
            'p_value': float(p), 'significant': p < SIGNIFICANCE, 'sharpe': sr}

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


def _stationary_bootstrap_indices(n, avg_block, rng):
    indices = []
    idx = rng.integers(0, n)
    while len(indices) < n:
        indices.append(idx % n)
        if rng.random() < 1.0 / avg_block:
            idx = rng.integers(0, n)
        else:
            idx += 1
    return np.array(indices[:n])


def hansen_spa_test(combo_oos_excess, n_boot=N_BOOTSTRAP):
    """Hansen (2005) Superior Predictive Ability test (consistent version).
    Tests H0: no parameter combination outperforms the benchmark
    after correcting for multiple testing across all 432 combos.
    Uses stationary bootstrap (Politis & Romano 1994).
    """
    valid = [np.array(r) for r in combo_oos_excess if len(r) >= 10]
    if len(valid) < 2:
        return {'spa_pval': 1.0, 'spa_sig': False}

    n = min(len(r) for r in valid)
    D = np.array([r[:n] for r in valid])
    d_bar = D.mean(axis=1)
    d_std = D.std(axis=1, ddof=1)
    d_std = np.maximum(d_std, 1e-10)

    T_obs = np.max(np.sqrt(n) * d_bar / d_std)
    mu_center = np.maximum(d_bar, 0)

    avg_block = max(int(n ** (1/3)), 2)
    rng = np.random.default_rng(42)

    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        indices = _stationary_bootstrap_indices(n, avg_block, rng)
        D_boot = D[:, indices]
        boot_means = D_boot.mean(axis=1)
        boot_stds = D_boot.std(axis=1, ddof=1)
        boot_stds = np.maximum(boot_stds, 1e-10)
        boot_stats[b] = np.max(np.sqrt(n) * (boot_means - mu_center) / boot_stds)

    p_value = np.mean(boot_stats >= T_obs)
    return {'spa_pval': float(p_value), 'spa_sig': p_value < SIGNIFICANCE}


def run_greeks_backtest(prices, hip3_iv, ibkr_iv):
    engine = GreeksStrategyEngine()
    results = engine.run_all_strategies(prices, hip3_iv, ibkr_iv)
    ensemble_pnl = engine.ensemble_strategy(results)
    return results, ensemble_pnl


def main():
    print('='*90)
    print('  HIP-3 MARKET BACKTEST — Purged K-Fold CV + IV Arbitrage + Greeks')
    print('  Lopez de Prado (2018) | Hansen SPA (2005) | Regime-adaptive')
    print('='*90)

    # Use REAL Hyperliquid HIP-3 candles + funding rates (cached in data/).
    # Falls back to synthetic generator if the cache is missing.
    data = load_real_hip3_data(min_days=100)
    if data:
        print(f'\nLoaded REAL HIP-3 data (candles + funding) for {len(data)} assets '
              f'from Hyperliquid API cache (data/candles + data/funding_rates).')
    else:
        print('\nReal cache missing — falling back to synthetic HIP-3 market data.')
        print('Run `python3 scripts/fetch_hip3_candles.py` and '
              '`python3 scripts/fetch_hip3_funding.py` first.')
        data = generate_synthetic_hip3_data(n_assets=25, min_days=100)

    arb_engine = IVArbitrageEngine()
    nc = len(list(product(*PARAM_GRID.values())))
    print(f'Grid: {nc} combos | Purge={PURGE_BARS} Embargo={EMBARGO_BARS} | IV arb + Greeks overlay')

    results = []; curves = {}; greeks_results_all = {}
    t0 = time.time()

    for tk, df in data.items():
        print(f'\n  {tk} ({len(df)} bars) ...', end=' ', flush=True)
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        if len(c) < MIN_TRAIN_BARS + MIN_TEST_BARS:
            print(f'SKIP ({len(c)} bars)'); continue

        # Variable funding rate from synthetic data (Hyperliquid hourly mechanism).
        funding = df['funding_rate'].values if 'funding_rate' in df.columns else None
        hip3_iv = arb_engine.compute_hip3_implied_vol(c, funding_rates=funding)
        ibkr_iv = arb_engine.compute_ibkr_atm_iv(c)

        cv = purged_kfold_cv(o, h, l, c, hip3_iv, ibkr_iv, funding_rates=funding)
        if cv is None:
            print('SKIP'); continue

        sr = cv.pop('strat_returns'); br = cv.pop('bench_returns')
        combo_excess = cv.pop('combo_oos_excess')
        fold_sh = cv.pop('fold_sharpes')

        oos_start = cv.pop('first_test_start')
        gk_results, gk_ensemble = run_greeks_backtest(
            c[oos_start:], hip3_iv[oos_start:], ibkr_iv[oos_start:])
        greeks_results_all[tk] = gk_results

        min_len = min(len(sr), len(gk_ensemble))
        combined = sr[:min_len] + gk_ensemble[:min_len] * 0.3
        curves[tk] = (combined, br[:min_len])

        st_t = ttest_excess(sr, br)
        st_sr = ttest_sharpe(sr)
        st2 = block_bootstrap(sr)
        st3 = perm_test(sr, br)
        st4 = deflated_sr(cv['sharpe'], nc, cv['test_bars'])
        st_spa = hansen_spa_test(combo_excess)

        row = {'asset':tk, **cv}
        row['t_stat']=st_t['t_stat']; row['t_df']=st_t['df']
        row['t_pval']=st_t['p_value']; row['t_sig']=st_t['significant']
        row['sr_pval']=st_sr['p_value']; row['sr_sig']=st_sr['significant']
        row['boot_ci_lo']=st2['ci_lo']; row['boot_ci_hi']=st2['ci_hi']
        row['boot_pval']=st2['p_value']; row['boot_sig']=st2['significant']
        row['perm_pval']=st3['p_value']; row['perm_sig']=st3['significant']
        row['dsr']=st4['dsr']; row['dsr_sig']=st4['significant']
        row['spa_pval']=st_spa['spa_pval']; row['spa_sig']=st_spa['spa_sig']
        row['sharpe_std']=cv['sharpe_std']
        row['fold_sharpe_mean']=np.mean(fold_sh) if fold_sh else 0.0

        best_greek = max(gk_results.values(), key=lambda x: x.sharpe)
        row['best_greek_strategy'] = best_greek.name
        row['best_greek_sharpe'] = best_greek.sharpe
        row['greeks_ensemble_ret'] = float(np.sum(gk_ensemble))
        results.append(row)

        fold_str = f'Folds={cv["n_folds"]} OOS={cv["oos_pct"]*100:.0f}%'
        spa_str = f'SPA-p={st_spa["spa_pval"]:.3f}{"*" if st_spa["spa_sig"] else ""}'
        pv = f't({st_t["df"]})={st_t["t_stat"]:.2f} p={st_t["p_value"]:.4f}  {spa_str}'
        print(f'Alpha={cv["alpha"]*100:+.1f}%  Sharpe={cv["sharpe"]:.2f}±{cv["sharpe_std"]:.2f}  '
              f'{fold_str}  {pv}')

    if not results: sys.exit('No results.')
    df_res = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / 'hip3_backtest_results.csv'
    df_res.to_csv(csv, index=False, float_format='%.6f')

    print('\n'+'='*160)
    print('  OUT-OF-SAMPLE RESULTS — Purged K-Fold CV | Hansen SPA | IV Arb + Greeks')
    print('='*160)

    def fmt_p(p):
        if p < 0.001: return '<.001*'
        if p < 0.05: return f'{p:.3f}*'
        return f'{p:.3f} '

    print(f'\n{"Asset":<8} {"Alpha":>7} {"Sharpe":>7} {"SR±":>6} {"Folds":>5} '
          f'{"OOS%":>5} {"Calmar":>7} {"MaxDD":>6} '
          f'{"t-stat":>7} {"p(t)":>7} {"p(SR)":>7} {"p(SPA)":>7} '
          f'{"p(Bt)":>7} {"p(Pm)":>7} {"DSR":>6}')
    print('-'*160)

    pa = 0
    for _, r in df_res.sort_values('alpha', ascending=False).iterrows():
        if r['alpha'] > 0: pa += 1
        print(f'{r["asset"]:<8} {r["alpha"]*100:>+6.1f}% {r["sharpe"]:>7.2f} '
              f'{r["sharpe_std"]:>5.2f}  {r["n_folds"]:>5.0f} '
              f'{r["oos_pct"]*100:>4.0f}% '
              f'{r["calmar"]:>7.2f} {r["max_dd"]*100:>5.1f}% '
              f'{r["t_stat"]:>7.2f} {fmt_p(r["t_pval"]):>7} '
              f'{fmt_p(r["sr_pval"]):>7} {fmt_p(r["spa_pval"]):>7} '
              f'{fmt_p(r["boot_pval"]):>7} {fmt_p(r["perm_pval"]):>7} '
              f'{r["dsr"]:>6.3f}')

    print(f'\n── Summary ──')
    print(f'  Methodology:       Purged K-Fold CV (purge={PURGE_BARS}, embargo={EMBARGO_BARS})')
    print(f'  Positive alpha:    {pa}/{len(df_res)}')
    print(f'  Mean alpha:        {df_res["alpha"].mean()*100:+.1f}%')
    print(f'  Mean Sharpe:       {df_res["sharpe"].mean():.2f} ± {df_res["sharpe_std"].mean():.2f}  '
          f'(bench: {df_res["sharpe_bench"].mean():.2f})')
    print(f'  Mean Calmar:       {df_res["calmar"].mean():.2f}')
    print(f'  Mean MaxDD:        {df_res["max_dd"].mean()*100:.1f}%  '
          f'(bench: {df_res["max_dd_bench"].mean()*100:.1f}%)')
    print(f'  Mean folds:        {df_res["n_folds"].mean():.1f}  '
          f'OOS: {df_res["oos_pct"].mean()*100:.0f}%')
    print(f'  Mean IV arb/yr:    {df_res["iv_arb_ann"].mean()*100:.1f}%')
    print(f'  Mean fills/day:    {df_res["fills_per_day"].mean():.0f}')
    spa_pass = (df_res['spa_sig'] == True).sum()
    print(f'  SPA significant:   {spa_pass}/{len(df_res)}')
    print(f'  Time: {time.time()-t0:.1f}s | CSV: {csv}')

    return df_res, curves, greeks_results_all


if __name__ == '__main__':
    df_res, curves, greeks_all = main()
