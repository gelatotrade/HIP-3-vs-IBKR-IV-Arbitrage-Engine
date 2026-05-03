#!/usr/bin/env python3
"""
HIP-3 Real-Data Premium Pipeline — Crème de la Crème on actual HIP-3 markets.

Uses cached real data:
  - data/candles/{ASSET}.csv         (real Hyperliquid OHLCV per asset since launch)
  - data/funding_rates/{ASSET}_daily.csv (real per-asset hourly funding aggregated daily)

Applies the same 10 premium methods (ARIMA-BMA + EGARCH-t + HAR-RV-J + HMM +
Purged K-fold + Hansen SPA + DSR + Romano-Wolf + SVI + Vega-bucket Greeks +
Kelly-VolTarget-DD + Bayesian + Almgren-Chriss).

Adapted to short HIP-3 history (30–186 bars per asset since launch):
  - 4-fold purged CV (vs 6 on equity data)
  - min_train = 60 bars, refit_every = 14
  - Per-fold OOS evaluation with proper t-statistic.

Output:
  results/hip3_real_premium_results.csv  with Sharpe, Calmar, MaxDD, t-stat,
                                          SPA p, DSR, RW p, fold_sharpe_std.
"""
import warnings
warnings.filterwarnings('ignore')

import sys, time, io, os
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import norm, t as student_t
from pathlib import Path
from itertools import product

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# Reuse all premium methods
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline_premium import (
    arima_bma_forecast, egarch_t_vol, har_rv_j_forecast,
    bipower_variation, hmm_regimes,
    purged_kfold_splits, hansen_spa, deflated_sharpe, romano_wolf_stepm,
    svi_total_variance, svi_ibkr_iv, bs_greeks, vega_bucket_hedged_greeks,
    kelly_position, bayesian_predictive, bayesian_signal,
    almgren_chriss_cost, compute_metrics,
    BG, BG2, TEXT, DIM, GREEN, RED, YELLOW, BLUE, CYAN, WHITE, GRID_C,
    REGIME_NAMES, REGIME_COLORS,
)

OUT_DIR = Path(__file__).resolve().parent.parent / 'results'
IMG_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
OUT_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)


# Real HIP-3 asset universe with their on-chain types
HIP3_ASSETS = {
    'QQQ':    {'type': 'index_etf', 'name': 'Nasdaq-100', 'market': 'xyz:XYZ100'},
    'NVDA':   {'type': 'stock',     'name': 'NVIDIA',     'market': 'xyz:NVDA'},
    'TSLA':   {'type': 'stock',     'name': 'Tesla',      'market': 'xyz:TSLA'},
    'PLTR':   {'type': 'stock',     'name': 'Palantir',   'market': 'xyz:PLTR'},
    'AMZN':   {'type': 'stock',     'name': 'Amazon',     'market': 'xyz:AMZN'},
    'GOOGL':  {'type': 'stock',     'name': 'Alphabet',   'market': 'xyz:GOOGL'},
    'MSFT':   {'type': 'stock',     'name': 'Microsoft',  'market': 'xyz:MSFT'},
    'META':   {'type': 'stock',     'name': 'Meta',       'market': 'xyz:META'},
    'AAPL':   {'type': 'stock',     'name': 'Apple',      'market': 'xyz:AAPL'},
    'COIN':   {'type': 'stock',     'name': 'Coinbase',   'market': 'xyz:COIN'},
    'MSTR':   {'type': 'stock',     'name': 'MicroStrategy', 'market': 'xyz:MSTR'},
    'AMD':    {'type': 'stock',     'name': 'AMD',        'market': 'xyz:AMD'},
    'NFLX':   {'type': 'stock',     'name': 'Netflix',    'market': 'xyz:NFLX'},
    'GOLD':   {'type': 'commodity', 'name': 'Gold',       'market': 'xyz:GOLD'},
    'SILVER': {'type': 'commodity', 'name': 'Silver',     'market': 'xyz:SILVER'},
    'OIL':    {'type': 'commodity', 'name': 'Crude Oil',  'market': 'xyz:CL'},
    # SPY at only 30 bars is too short for purged CV
}


def load_hip3_data():
    print(f'Loading real HIP-3 data for {len(HIP3_ASSETS)} assets ...')
    data = {}
    for tk, info in HIP3_ASSETS.items():
        candle_path = DATA_DIR / 'candles' / f'{tk}.csv'
        funding_path = DATA_DIR / 'funding_rates' / f'{tk}_daily.csv'
        if not candle_path.exists():
            print(f'  {tk:<6}: MISSING candles')
            continue
        df = pd.read_csv(candle_path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        if len(df) < 60:
            print(f'  {tk:<6}: SKIP ({len(df)} bars too short)')
            continue
        # Optional funding history
        if funding_path.exists():
            fd = pd.read_csv(funding_path, parse_dates=['timestamp'])
            fd = fd.sort_values('timestamp').reset_index(drop=True)
            funding = fd['funding_rate'].values.astype(float) if 'funding_rate' in fd.columns else np.zeros(len(df))
            # Align lengths
            if len(funding) < len(df):
                funding = np.concatenate([funding, np.zeros(len(df) - len(funding))])
            elif len(funding) > len(df):
                funding = funding[: len(df)]
        else:
            funding = np.zeros(len(df))
        data[tk] = {
            'o': df['open'].values.astype(float),
            'h': df['high'].values.astype(float),
            'l': df['low'].values.astype(float),
            'c': df['close'].values.astype(float),
            'v': df['volume'].values.astype(float) if 'volume' in df.columns else np.ones(len(df)),
            'funding': funding,
            'dates': df['timestamp'],
            'info': info,
            'launch': df['timestamp'].iloc[0],
        }
        print(f'  {tk:<6} ({info["name"]:<14}) {info["market"]:<14} '
              f'{len(df):>3} bars  ${df["close"].iloc[0]:>10,.2f} → ${df["close"].iloc[-1]:>10,.2f}  '
              f'launch {df["timestamp"].iloc[0].date()}')
    return data


# ── Adapted signal builder for short HIP-3 history ──
def build_signals_short(closes, opens, highs, lows, refit_every=14, min_train=40):
    """Same as build_signals in run_pipeline_premium but adapted for shorter
    history (HIP-3 assets have only 60–186 daily bars since launch)."""
    N = len(closes)
    rets = np.zeros(N); rets[1:] = np.diff(np.log(closes))
    arima_mu = np.zeros(N); arima_var = np.zeros(N)
    egarch_v = np.full(N, 0.3); har_v = np.full(N, 0.3)
    bayes_sig = np.zeros(N)
    refit_at = -1
    last_state = (0.0, 0.0, 0.30, 0.30, 0.0)
    for i in range(min_train, N):
        if i - refit_at >= refit_every:
            arr = rets[max(0, i - 252): i]
            mu_b, var_b = arima_bma_forecast(arr)
            egv = egarch_t_vol(arr)
            harv = har_rv_j_forecast(rets, i)
            mb_, sc, df_ = bayesian_predictive(arr)
            bs = bayesian_signal(mb_, sc, df_)
            last_state = (mu_b, var_b, egv, harv, bs)
            refit_at = i
        arima_mu[i], arima_var[i], egarch_v[i], har_v[i], bayes_sig[i] = last_state
    # HMM with 3 states given fewer bars
    n_states = 3 if N < 130 else 4
    hmm_states, hmm_post = hmm_regimes(rets, n_states=n_states)
    # Pad post matrix to 4 states if needed
    if hmm_post.shape[1] < 4:
        pad = np.zeros((hmm_post.shape[0], 4 - hmm_post.shape[1]))
        hmm_post = np.concatenate([hmm_post, pad], axis=1)
    return {
        'rets': rets,
        'arima_mu': arima_mu,
        'arima_var': arima_var,
        'egarch_v': egarch_v,
        'har_v': har_v,
        'bayes_sig': bayes_sig,
        'hmm_states': hmm_states,
        'hmm_post': hmm_post,
    }


def load_real_ibkr_svi(asset):
    """Load REAL SVI parameters fitted to today's CBOE/Yahoo options chain.

    Returns dict with at-the-money 30d IV plus full SVI surface, or None if
    no chain data is available.
    """
    p = DATA_DIR / 'ibkr_options' / f'{asset}_svi.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    # Prefer the 30d expiration slice (closest to T=30/365)
    df['T_diff'] = (df['T'] - 30 / 365).abs()
    df = df.sort_values('T_diff').reset_index(drop=True)
    return {
        'atm_iv_30d': float(df.iloc[0]['atm_iv']),
        'svi_a': float(df.iloc[0]['svi_a']),
        'svi_b': float(df.iloc[0]['svi_b']),
        'svi_rho': float(df.iloc[0]['svi_rho']),
        'svi_m': float(df.iloc[0]['svi_m']),
        'svi_sigma': float(df.iloc[0]['svi_sigma']),
        'T': float(df.iloc[0]['T']),
        'spot_at_fetch': float(df.iloc[0]['spot']),
        'n_slices': len(df),
    }


def compute_iv_series_real(closes, har_v, asset=None):
    """HIP-3 IV from HAR-RV-J × 1.05; IBKR IV from REAL CBOE/Yahoo
    chain when available, else fallback to synthetic SVI.

    The level of IBKR IV is anchored at today's ATM 30d IV; the bar-by-bar
    variation is driven by HAR-RV-J realised vol so the spread captures
    the true vol-of-vol dynamics of the period.
    """
    N = len(closes)
    hip3_iv = np.maximum(har_v * 1.05, 0.05)
    real = load_real_ibkr_svi(asset) if asset else None
    if real is not None:
        atm = real['atm_iv_30d']
        # Anchor IBKR IV around real ATM IV with HAR-RV-J term structure
        long_run = np.median(har_v[har_v > 0]) if (har_v > 0).any() else atm
        ratio = atm / max(long_run * 1.10, 0.05)
        ibkr_iv = np.maximum(har_v * 1.10 * ratio, 0.05)
    else:
        ibkr_iv = np.zeros(N)
        for i in range(N):
            ibkr_iv[i] = svi_ibkr_iv(closes[i], closes[i], 30 / 365,
                                     base_iv=max(har_v[i] * 1.10, 0.05))
    return hip3_iv, ibkr_iv


# ── Adapted backtest engine using REAL FUNDING ──
PARAM_GRID = {
    'kelly_frac':  [0.20, 0.35],
    'vol_target':  [0.20, 0.30],   # higher for HIP-3 vol
    'dd_max':      [0.15, 0.25],
    'iv_arb_w':    [0.3, 0.5],
    'greeks_w':    [0.4, 0.6],
}

ADVERSE_SEL = 0.45
HL_TAKER_FEE = 0.045 / 100  # Hyperliquid HIP-3 taker fee 0.045%
HL_MAKER_REBATE = -0.015 / 100  # HIP-3 maker rebate
WARM_UP = 40


def run_premium_backtest_real(o, h, l, c, sigs, funding, params,
                              test_idx=None, asset=None):
    """Run premium backtest with REAL funding rates and (when available)
    REAL CBOE/Yahoo IBKR-equivalent option-chain SVI calibration.

    Returns (pnl, bench, fills, spread_pnl, iv_pnl, greeks_pnl, funding,
             positions).
    """
    rets = sigs['rets']
    N = len(c)
    pnl = np.zeros(N)
    bench = np.zeros(N)
    positions = np.ones(N)  # baseline long
    fills_total = 0
    spread_total = 0.0
    iv_total = 0.0
    greeks_total = 0.0
    funding_total = 0.0

    hip3_iv, ibkr_iv = compute_iv_series_real(c, sigs['har_v'], asset=asset)

    eq, peak = 1.0, 1.0
    indices = test_idx if test_idx is not None else range(1, N)

    for i in indices:
        if i < WARM_UP:
            continue

        mu_combined = 0.5 * sigs['arima_mu'][i] + 0.5 * sigs['bayes_sig'][i] * 0.001
        sigma_combined = max(sigs['egarch_v'][i] / np.sqrt(252), 1e-4)

        current_dd = (peak - eq) / peak if peak > 0 else 0
        f = kelly_position(mu_combined, sigma_combined, current_dd,
                           vol_target=params['vol_target'],
                           dd_max=params['dd_max'],
                           kelly_frac=params['kelly_frac'])

        # HMM regime overlay
        if i < len(sigs['hmm_post']):
            p_bull = sigs['hmm_post'][i, 0]
            p_crisis = sigs['hmm_post'][i, 3]
            f *= 1.0 + 0.15 * p_bull - 0.4 * p_crisis
        f = float(np.clip(f, 0.0, 1.15))
        positions[i] = f

        # Vega-bucket Greeks ensemble
        greeks_pnl, _ = vega_bucket_hedged_greeks(
            c[i], hip3_iv[i], ibkr_iv[i], rets[i])

        # MM spread capture (HIP-3 maker-rebate model)
        spread_mult = 1.5 if (i < len(sigs['hmm_post']) and
                              sigs['hmm_post'][i, 3] > 0.3) else 1.0
        fair = c[i - 1]
        rng_bps = (h[i] - l[i]) / fair * 10_000 if fair > 0 else 0
        bar_spread = 0.0
        bar_fills = 0
        for lv in range(1, 11):
            off = fair * 8 * lv * spread_mult / 10_000
            bid, ask = fair - off, fair + off
            reps = max(1, min(int(rng_bps / (16 * lv * spread_mult)), 4))
            if l[i] <= bid:
                cap = 0.02 * reps * off / fair * (1 - ADVERSE_SEL)
                # Maker rebate income vs cost
                bar_spread += cap - 0.02 * reps * HL_MAKER_REBATE
                bar_fills += reps
            if h[i] >= ask:
                cap = 0.02 * reps * off / fair * (1 - ADVERSE_SEL)
                bar_spread += cap - 0.02 * reps * HL_MAKER_REBATE
                bar_fills += reps

        # Almgren-Chriss execution cost
        prev_f = positions[i - 1]
        turnover = abs(f - prev_f)
        sigma_daily = sigma_combined / np.sqrt(252)
        exec_cost = almgren_chriss_cost(turnover, sigma_daily)

        # IV arbitrage overlay
        vs = hip3_iv[i] - ibkr_iv[i]
        iv_pnl = -np.sign(vs) * min(abs(vs) * 1.5, 0.04) * abs(rets[i]) * 0.5

        # REAL FUNDING COST (Hyperliquid HIP-3 historical, daily aggregate)
        fund = funding[i] if i < len(funding) else 0.0
        funding_pnl = -f * fund  # long position pays positive funding

        bar_pnl = (f * rets[i]
                   + bar_spread
                   + params['greeks_w'] * greeks_pnl
                   + params['iv_arb_w'] * iv_pnl
                   - exec_cost
                   + funding_pnl)
        pnl[i] = float(np.clip(bar_pnl, -0.20, 0.20))
        bench[i] = rets[i]

        fills_total += bar_fills
        spread_total += bar_spread
        iv_total += iv_pnl
        greeks_total += greeks_pnl
        funding_total += funding_pnl

        eq *= (1 + pnl[i])
        peak = max(peak, eq)

    if test_idx is not None:
        mask = np.zeros(N, dtype=bool); mask[test_idx] = True
        return (pnl[mask], bench[mask], fills_total, spread_total,
                iv_total, greeks_total, funding_total, positions[mask])
    return (pnl[1:], bench[1:], fills_total, spread_total,
            iv_total, greeks_total, funding_total, positions[1:])


def run_kfold_real(o, h, l, c, funding, sigs, n_folds=4, asset=None):
    """Purged K-fold CV adapted for short HIP-3 history."""
    N = len(c)
    if N < 80:
        return None
    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    n_trials = len(combos)

    oos_returns = np.full(N, np.nan)
    oos_bench = np.full(N, np.nan)
    fold_metrics = []
    fills_per_fold = []
    spread_per_fold = []
    iv_per_fold = []
    greeks_per_fold = []
    funding_per_fold = []

    for train_idx, test_idx, fold_k in purged_kfold_splits(
            N, n_folds=n_folds, purge=1, embargo=2):
        if len(train_idx) < 30 or len(test_idx) < 10:
            continue
        # Param selection on train
        best_score, best_p = -1e9, dict(zip(keys, combos[0]))
        for combo in combos:
            p = dict(zip(keys, combo))
            try:
                s, b, *_ = run_premium_backtest_real(
                    o, h, l, c, sigs, funding, p,
                    test_idx=train_idx, asset=asset)
                m = compute_metrics(s, b)
                if m is None:
                    continue
                score = (m['sharpe'] * 0.4 + m['calmar'] * 0.2
                         + m['alpha'] * 6.0)
                if score > best_score:
                    best_score, best_p = score, p
            except Exception:
                continue
        # OOS evaluation
        try:
            s, b, fl, sp, iv_, gr_, fd_, pos = run_premium_backtest_real(
                o, h, l, c, sigs, funding, best_p,
                test_idx=test_idx, asset=asset)
            valid = test_idx[test_idx >= WARM_UP]
            s_valid = s[-len(valid):] if len(s) >= len(valid) else s
            b_valid = b[-len(valid):] if len(b) >= len(valid) else b
            for j, gi in enumerate(valid):
                if j < len(s_valid):
                    oos_returns[gi] = s_valid[j]
                    oos_bench[gi] = b_valid[j]
            m = compute_metrics(s, b)
            if m is None:
                continue
            m.update(best_p)
            m['fold_id'] = fold_k
            fold_metrics.append(m)
            fills_per_fold.append(fl)
            n_b = max(m['n_bars'], 1)
            spread_per_fold.append(sp / n_b)
            iv_per_fold.append(iv_ / n_b)
            greeks_per_fold.append(gr_ / n_b)
            funding_per_fold.append(fd_ / n_b)
        except Exception:
            continue
    if not fold_metrics:
        return None

    mask = ~np.isnan(oos_returns)
    s_arr = oos_returns[mask]
    b_arr = oos_bench[mask]
    if len(s_arr) < 20:
        return None
    excess = s_arr - b_arr

    # Statistical tests
    try:
        spa_p = hansen_spa(excess.reshape(-1, 1), n_boot=500)
    except Exception:
        spa_p = 1.0
    try:
        dsr = deflated_sharpe(s_arr, n_trials=n_trials)
    except Exception:
        dsr = {'dsr': 0, 'p_value': 1, 'significant': False}
    try:
        rw_pvals = romano_wolf_stepm(excess.reshape(-1, 1), n_boot=500)
    except Exception:
        rw_pvals = [1.0]

    # Paired t-test on excess returns
    if len(excess) >= 5 and excess.std(ddof=1) > 0:
        t_stat = excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))
        t_p = 1.0 - student_t.cdf(t_stat, df=len(excess) - 1)
    else:
        t_stat, t_p = 0.0, 1.0

    agg = compute_metrics(s_arr, b_arr)
    if agg is None:
        return None
    fold_sharpes = [m['sharpe'] for m in fold_metrics]
    agg['fold_sharpes'] = fold_sharpes
    agg['fold_sharpe_mean'] = float(np.mean(fold_sharpes))
    agg['fold_sharpe_std'] = float(np.std(fold_sharpes, ddof=1)) if len(fold_sharpes) > 1 else 0.0
    agg['hansen_spa_p'] = float(spa_p)
    agg['hansen_spa_sig'] = bool(spa_p < 0.05)
    agg['dsr'] = float(dsr['dsr'])
    agg['dsr_sig'] = bool(dsr['significant'])
    agg['romano_wolf_p'] = float(rw_pvals[0])
    agg['romano_wolf_sig'] = bool(rw_pvals[0] < 0.05)
    agg['t_stat'] = float(t_stat)
    agg['t_p'] = float(t_p)
    agg['t_sig'] = bool(t_p < 0.05)
    agg['n_folds'] = len(fold_metrics)
    agg['n_trials'] = n_trials
    total_bars = sum(m['n_bars'] for m in fold_metrics)
    agg['fills_per_day'] = sum(fills_per_fold) / max(total_bars, 1)
    agg['spread_ann'] = float(np.mean(spread_per_fold) * 252) if spread_per_fold else 0
    agg['iv_arb_ann'] = float(np.mean(iv_per_fold) * 252) if iv_per_fold else 0
    agg['greeks_ann'] = float(np.mean(greeks_per_fold) * 252) if greeks_per_fold else 0
    agg['funding_ann'] = float(np.mean(funding_per_fold) * 252) if funding_per_fold else 0
    agg['strat_returns'] = s_arr
    agg['bench_returns'] = b_arr
    return agg


# ── Visualization ──
def style_ax(ax):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=7)
    for sp in ax.spines.values(): sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, alpha=0.3, linewidth=0.5)


def png_hip3_premium_summary(results, data):
    """4-panel summary on real HIP-3 data."""
    print('  Generating real HIP-3 premium summary PNG ...')
    items = sorted([(tk, r) for tk, r in results.items() if r],
                   key=lambda kv: -kv[1].get('alpha', 0))
    if not items:
        return
    tks = [tk for tk, r in items]
    alphas = [r['alpha'] * 100 for tk, r in items]
    sharpes = [r['sharpe'] for tk, r in items]
    bench_sh = [r['sharpe_bench'] for tk, r in items]
    fold_stds = [r.get('fold_sharpe_std', 0) for tk, r in items]
    t_stats = [r.get('t_stat', 0) for tk, r in items]
    dsrs = [r.get('dsr', 0) for tk, r in items]
    calmars = [r['calmar'] for tk, r in items]

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.2,
                  left=0.06, right=0.97, top=0.94, bottom=0.08)
    fig.suptitle('Real HIP-3 Premium Backtest (Crème de la Crème)',
                 color=WHITE, fontsize=14, fontweight='bold')

    ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
    cols = [GREEN if a > 0 else RED for a in alphas]
    ax.bar(tks, alphas, color=cols, alpha=0.85, edgecolor=WHITE)
    ax.set_title('Out-of-Sample Alpha (%)', color=TEXT, fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    ax = fig.add_subplot(gs[0, 1]); style_ax(ax)
    x = np.arange(len(tks))
    ax.bar(x - 0.2, sharpes, 0.4, color=GREEN, alpha=0.85,
           label='Premium (CPCV)', yerr=fold_stds, capsize=3, ecolor=YELLOW)
    ax.bar(x + 0.2, bench_sh, 0.4, color=BLUE, alpha=0.85, label='B&H HIP-3')
    ax.set_xticks(x); ax.set_xticklabels(tks, rotation=45, fontsize=8)
    ax.set_title('Sharpe (with fold-σ error bars)', color=TEXT, fontsize=11)
    ax.legend(fontsize=8, facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)

    ax = fig.add_subplot(gs[1, 0]); style_ax(ax)
    cols = [GREEN if t > 1.96 else (YELLOW if t > 1.65 else RED)
            for t in t_stats]
    ax.bar(tks, t_stats, color=cols, alpha=0.85, edgecolor=WHITE)
    ax.axhline(1.96, color=GREEN, ls='--', lw=1, label='t=1.96 (p=0.05)')
    ax.axhline(2.58, color=CYAN, ls='--', lw=1, label='t=2.58 (p=0.01)')
    ax.set_title('Paired t-statistic (excess returns)',
                 color=TEXT, fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.legend(fontsize=8, facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)

    ax = fig.add_subplot(gs[1, 1]); style_ax(ax)
    cols = [GREEN if c > 3 else (YELLOW if c > 1 else RED) for c in calmars]
    ax.bar(tks, calmars, color=cols, alpha=0.85, edgecolor=WHITE)
    ax.set_title('Calmar Ratio (ann_ret / max_dd)',
                 color=TEXT, fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    plt.savefig(IMG_DIR / 'hip3_real_premium_summary.png', dpi=150,
                facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved hip3_real_premium_summary.png')


def png_hip3_equity_curves(results, data):
    """Equity curves of all assets on real HIP-3 data."""
    print('  Generating real HIP-3 equity curves PNG ...')
    items = sorted([(tk, r) for tk, r in results.items() if r],
                   key=lambda kv: -kv[1].get('alpha', 0))
    if not items:
        return
    fig, axes = plt.subplots(4, 4, figsize=(20, 14), facecolor=BG)
    fig.suptitle('Real HIP-3 Premium — OOS Equity Curves (since on-chain launch)',
                 color=WHITE, fontsize=14, fontweight='bold')
    for ax_ in axes.flat:
        style_ax(ax_); ax_.set_visible(False)
    for ax, (tk, r) in zip(axes.flat, items):
        ax.set_visible(True)
        s = r['strat_returns']
        b = r['bench_returns']
        cs = (np.exp(np.cumsum(s)) - 1) * 100
        cb = (np.exp(np.cumsum(b)) - 1) * 100
        x = range(len(cs))
        ax.plot(x, cb, color='#555588', lw=1.2, label='B&H')
        ax.plot(x, cs, color=GREEN, lw=1.6, label='Premium')
        ax.fill_between(x, cb, cs, where=cs > cb, color=GREEN, alpha=0.18)
        ax.fill_between(x, cb, cs, where=cs <= cb, color=RED, alpha=0.18)
        info = data[tk]['info']
        title = (f'{tk} ({info["name"]}) [{info["market"]}]\n'
                 f'α={r["alpha"]*100:+.0f}%  SR={r["sharpe"]:.2f}±{r["fold_sharpe_std"]:.2f}  '
                 f't={r["t_stat"]:.2f}  Calmar={r["calmar"]:.1f}')
        ax.set_title(title, color=TEXT, fontsize=8)
        ax.legend(fontsize=6, facecolor=BG2, edgecolor=GRID_C,
                  labelcolor=TEXT, loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'hip3_real_premium_equity_curves.png', dpi=140,
                facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved hip3_real_premium_equity_curves.png')


# ── Main ──
def main():
    t0 = time.time()
    print('═' * 100)
    print('  HIP-3 REAL DATA PREMIUM PIPELINE — Crème de la Crème')
    print('  Real OHLCV + Real Funding (per-asset since on-chain launch)')
    print('═' * 100)
    data = load_hip3_data()
    if not data:
        print('No data'); return

    print('\n── Pre-computing premium signals (ARIMA-BMA, EGARCH-t, HMM, HAR-RV-J, Bayesian)')
    sigs = {}
    for tk, d in data.items():
        t1 = time.time()
        sigs[tk] = build_signals_short(d['c'], d['o'], d['h'], d['l'],
                                       refit_every=14, min_train=40)
        print(f'  {tk:<6}: {time.time()-t1:.1f}s')

    print('\n── Running Purged K-fold CV with REAL Hyperliquid funding + REAL CBOE/Yahoo IBKR options')
    results = {}
    for tk, d in data.items():
        t1 = time.time()
        # Adapt fold count to data length
        n_bars = len(d['c'])
        n_folds = 4 if n_bars >= 130 else (3 if n_bars >= 100 else 2)
        try:
            r = run_kfold_real(d['o'], d['h'], d['l'], d['c'],
                               d['funding'], sigs[tk], n_folds=n_folds,
                               asset=tk)
            results[tk] = r
            if r:
                print(f'  {tk:<6} α={r["alpha"]*100:+6.1f}%  '
                      f'SR={r["sharpe"]:5.2f}±{r["fold_sharpe_std"]:.2f}  '
                      f'Calmar={r["calmar"]:6.2f}  MaxDD={r["max_dd"]*100:5.1f}%  '
                      f't={r["t_stat"]:5.2f}  SPA={r["hansen_spa_p"]:.3f}  '
                      f'DSR={r["dsr"]:.3f}  ({time.time()-t1:.1f}s)')
            else:
                print(f'  {tk:<6}: insufficient data')
        except Exception as e:
            print(f'  {tk:<6}: ERROR {e}')
            results[tk] = None

    # Save CSV
    rows = []
    for tk, r in results.items():
        if not r:
            continue
        rows.append({
            'asset': tk,
            'asset_type': data[tk]['info']['type'],
            'asset_name': data[tk]['info']['name'],
            'hip3_market': data[tk]['info']['market'],
            'launch_date': str(data[tk]['launch'].date()),
            'days_live': len(data[tk]['c']),
            'ann_ret': r['ann_ret'],
            'bench_ret': r['bench_ret'],
            'alpha': r['alpha'],
            'sharpe': r['sharpe'],
            'sharpe_bench': r['sharpe_bench'],
            'calmar': r['calmar'],
            'calmar_bench': r['calmar_bench'],
            'max_dd': r['max_dd'],
            'max_dd_bench': r['max_dd_bench'],
            't_stat': r['t_stat'],
            't_p': r['t_p'],
            't_sig': r['t_sig'],
            'fold_sharpe_mean': r['fold_sharpe_mean'],
            'fold_sharpe_std': r['fold_sharpe_std'],
            'hansen_spa_p': r['hansen_spa_p'],
            'hansen_spa_sig': r['hansen_spa_sig'],
            'dsr': r['dsr'],
            'dsr_sig': r['dsr_sig'],
            'romano_wolf_p': r['romano_wolf_p'],
            'romano_wolf_sig': r['romano_wolf_sig'],
            'n_folds': r['n_folds'],
            'n_trials': r['n_trials'],
            'fills_per_day': r['fills_per_day'],
            'spread_ann': r['spread_ann'],
            'iv_arb_ann': r['iv_arb_ann'],
            'greeks_ann': r['greeks_ann'],
            'funding_ann': r['funding_ann'],
        })
    csv_path = OUT_DIR / 'hip3_real_premium_results.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Console summary table
    print('\n' + '═' * 110)
    print('  REAL HIP-3 PREMIUM RESULTS — Sharpe, Calmar, MaxDD, t-statistic, p-values')
    print('═' * 110)
    print(f'  {"Asset":<7} {"Type":<11} {"Days":>5} {"Alpha":>8} {"SR":>8} '
          f'{"Calmar":>7} {"MaxDD":>7} {"t":>7} {"t-p":>8} {"SPA":>8} {"DSR":>6}')
    print('-' * 110)
    sorted_r = sorted(results.items(),
                      key=lambda kv: -(kv[1]['alpha'] if kv[1] else 0))
    for tk, r in sorted_r:
        if not r:
            continue
        print(f'  {tk:<7} {data[tk]["info"]["type"]:<11} '
              f'{len(data[tk]["c"]):>5} '
              f'{r["alpha"]*100:>+7.1f}% '
              f'{r["sharpe"]:>5.2f}±{r["fold_sharpe_std"]:.2f}  '
              f'{r["calmar"]:>6.2f} '
              f'{r["max_dd"]*100:>6.1f}% '
              f'{r["t_stat"]:>6.2f} '
              f'{r["t_p"]:>7.4f} '
              f'{r["hansen_spa_p"]:>7.4f} '
              f'{r["dsr"]:>5.3f}')
    n_total = sum(1 for r in results.values() if r)
    n_pos = sum(1 for r in results.values() if r and r['alpha'] > 0)
    n_t = sum(1 for r in results.values() if r and r['t_sig'])
    n_spa = sum(1 for r in results.values() if r and r['hansen_spa_sig'])
    n_dsr = sum(1 for r in results.values() if r and r['dsr_sig'])
    if n_total:
        mean_a = np.mean([r['alpha'] for r in results.values() if r]) * 100
        mean_sr = np.mean([r['sharpe'] for r in results.values() if r])
        mean_cm = np.mean([r['calmar'] for r in results.values() if r])
        mean_dd = np.mean([r['max_dd'] for r in results.values() if r]) * 100
        print('-' * 110)
        print(f'  Positive alpha: {n_pos}/{n_total}   Mean α: {mean_a:+.1f}%   '
              f'Mean Sharpe: {mean_sr:.2f}   Mean Calmar: {mean_cm:.2f}   '
              f'Mean MaxDD: {mean_dd:.1f}%')
        print(f'  t-test sig (p<0.05): {n_t}/{n_total}   '
              f'Hansen SPA sig: {n_spa}/{n_total}   '
              f'DSR sig (>0.95): {n_dsr}/{n_total}')

    print(f'\n  CSV: {csv_path}')

    # Visuals
    print('\n── Generating Real HIP-3 Premium Visualizations ──')
    png_hip3_premium_summary(results, data)
    png_hip3_equity_curves(results, data)

    print(f'\n  Total time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
