#!/usr/bin/env python3
"""
Run the Crème de la Crème premium pipeline on EVERY active HIP-3 perp market.

Loads cached real Hyperliquid OHLCV + funding from data/all_hip3/, applies
the full 10-method premium stack (ARIMA-BMA, EGARCH-t, HAR-RV-J, HMM,
Purged K-fold CV, Hansen SPA, Deflated SR, Romano-Wolf, SVI, Vega-bucket
Greeks, Kelly-VolTarget-DD, Bayesian, Almgren-Chriss), and reports per-asset
Sharpe, Calmar, MaxDD, t-statistic, Hansen SPA p-value, and DSR.

Output:
  results/hip3_all_premium_results.csv
  docs/img/hip3_all_premium_summary.png
  docs/img/hip3_all_premium_equity_curves.png
"""
import warnings
warnings.filterwarnings('ignore')

import sys, time, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline_premium import (
    arima_bma_forecast, egarch_t_vol, har_rv_j_forecast,
    hmm_regimes, purged_kfold_splits, hansen_spa, deflated_sharpe,
    romano_wolf_stepm, svi_ibkr_iv, vega_bucket_hedged_greeks,
    kelly_position, bayesian_predictive, bayesian_signal,
    almgren_chriss_cost, compute_metrics,
    BG, BG2, TEXT, DIM, GREEN, RED, YELLOW, BLUE, CYAN, WHITE, GRID_C,
)
from run_hip3_premium import (
    build_signals_short, compute_iv_series_real, load_real_ibkr_svi,
    PARAM_GRID, ADVERSE_SEL, HL_TAKER_FEE, HL_MAKER_REBATE, WARM_UP,
)
from scipy.stats import t as student_t
from itertools import product

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'all_hip3'
OUT_DIR = Path(__file__).resolve().parent.parent / 'results'
IMG_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'
OUT_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)


# Asset categorisation for reporting
def categorize(asset):
    if asset in {'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META',
                 'JPM', 'NFLX', 'PLTR', 'COIN', 'MSTR', 'AMD', 'HOOD',
                 'INTC', 'ORCL', 'MU', 'SNDK', 'CRCL', 'COST', 'LLY',
                 'SKHX', 'TSM', 'RIVN', 'BABA', 'SMSN', 'USAR', 'CRWV',
                 'URNM', 'GME', 'HYUNDAI', 'KIOXIA', 'HIMS', 'DKNG',
                 'LITE', 'BX', 'PURRDAT', 'MRVL', 'RKLB', 'BIRD', 'DRAM',
                 'CBRS', 'ZM', 'SOFTBANK', 'BMNR', 'RTX', 'TENCENT',
                 'XIAOMI', 'CAR', 'KWEB'}:
        return 'stock'
    if asset in {'XYZ100', 'SP500', 'KR200', 'JP225', 'EWY', 'EWJ', 'EWZ',
                 'XLE', 'KRW', 'DXY', 'VIX', 'VOL', 'USA500', 'USA100',
                 'US500', 'USTECH', 'USENERGY', 'SMALL2000', 'JPN225',
                 'CBRS', 'USOIL', 'KWEB'}:
        return 'index_etf'
    if asset in {'GOLD', 'SILVER', 'CL', 'OIL', 'BRENTOIL', 'COPPER',
                 'NATGAS', 'URANIUM', 'ALUMINIUM', 'PALLADIUM', 'PLATINUM',
                 'CORN', 'WHEAT', 'TTF', 'GAS', 'WTI', 'GLDMINE',
                 'SOY', 'GOLDJM', 'SILVERJM', 'SEMI', 'SEMIS'}:
        return 'commodity'
    if asset in {'EUR', 'JPY', 'KRW', 'USDE'}:
        return 'fx'
    if asset in {'USBOND'}:
        return 'rates'
    if asset in {'SPACEX', 'OPENAI', 'ANTHROPIC'}:
        return 'pre_ipo'
    if asset in {'MAG7', 'ROBOT', 'INFOTECH', 'NUCLEAR', 'DEFENSE',
                 'ENERGY', 'BIOTECH', 'INFOTECH'}:
        return 'thematic'
    return 'other'


# IBKR underlying mapping (for those with options chains)
IBKR_PROXY = {
    # Stocks - direct
    **{a: a for a in ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL',
                       'META', 'NFLX', 'PLTR', 'COIN', 'MSTR', 'AMD',
                       'HOOD', 'INTC', 'ORCL', 'MU', 'COST', 'LLY',
                       'TSM', 'RIVN', 'BABA', 'GME', 'BX', 'ZM',
                       'MRVL', 'RKLB', 'DKNG', 'CRWV', 'URNM']},
    # Indices/ETFs
    'XYZ100': 'QQQ', 'SP500': 'SPY', 'USA500': 'SPY', 'USA100': 'QQQ',
    'US500': 'SPY', 'USTECH': 'QQQ', 'SMALL2000': 'IWM',
    'KR200': 'EWY', 'JP225': 'EWJ', 'JPN225': 'EWJ',
    'EWY': 'EWY', 'EWJ': 'EWJ', 'EWZ': 'EWZ',
    'XLE': 'XLE', 'KWEB': 'KWEB', 'USENERGY': 'XLE',
    # Commodities
    'GOLD': 'GLD', 'SILVER': 'SLV', 'CL': 'USO', 'OIL': 'USO',
    'BRENTOIL': 'BNO', 'COPPER': 'CPER', 'NATGAS': 'UNG',
    'URANIUM': 'URA', 'PALLADIUM': 'PALL', 'PLATINUM': 'PPLT',
    'WTI': 'USO', 'USOIL': 'USO', 'GLDMINE': 'GDX',
    'CORN': 'CORN', 'WHEAT': 'WEAT',
    # FX
    'EUR': 'FXE', 'JPY': 'FXY', 'KRW': 'FXKR',
    # Thematic baskets approximated by ETFs
    'MAG7': 'XLK', 'SEMIS': 'SOXX', 'SEMI': 'SOXX',
    'INFOTECH': 'XLK', 'NUCLEAR': 'URA', 'DEFENSE': 'ITA',
    'ENERGY': 'XLE', 'BIOTECH': 'XBI', 'ROBOT': 'BOTZ',
}


def load_all_data():
    summary_path = DATA_DIR / '_summary.csv'
    if not summary_path.exists():
        print(f'Run scripts/fetch_all_hip3.py first → {summary_path} missing')
        return {}
    summary = pd.read_csv(summary_path)
    print(f'Loading {len(summary)} HIP-3 markets from cache ...')
    data = {}
    for _, row in summary.iterrows():
        asset = row['asset']
        cp = DATA_DIR / 'candles' / f'{asset}.csv'
        fp = DATA_DIR / 'funding' / f'{asset}_daily.csv'
        if not cp.exists():
            continue
        try:
            df = pd.read_csv(cp, parse_dates=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        except Exception:
            continue
        if len(df) < 60:
            continue
        if fp.exists():
            try:
                fd = pd.read_csv(fp, parse_dates=['timestamp'])
                fd = fd.sort_values('timestamp').reset_index(drop=True)
                funding = fd['funding_rate'].values.astype(float)
            except Exception:
                funding = np.zeros(len(df))
        else:
            funding = np.zeros(len(df))
        if len(funding) < len(df):
            funding = np.concatenate([funding, np.zeros(len(df) - len(funding))])
        elif len(funding) > len(df):
            funding = funding[:len(df)]
        data[asset] = {
            'o': df['open'].values.astype(float),
            'h': df['high'].values.astype(float),
            'l': df['low'].values.astype(float),
            'c': df['close'].values.astype(float),
            'v': (df['volume'].values.astype(float)
                  if 'volume' in df.columns else np.ones(len(df))),
            'funding': funding,
            'dates': df['timestamp'],
            'launch': df['timestamp'].iloc[0],
            'category': categorize(asset),
            'ibkr_proxy': IBKR_PROXY.get(asset, None),
            'full_ticker': row['full_ticker'],
            'dex': row['dex_chosen'],
        }
    print(f'  Loaded {len(data)} markets with ≥60 daily bars')
    return data


def run_premium_backtest_one(o, h, l, c, sigs, funding, params,
                             test_idx=None, ibkr_proxy=None):
    """Same backtest engine as run_hip3_premium but accepts ibkr_proxy
    for SVI lookup."""
    rets = sigs['rets']
    N = len(c)
    pnl = np.zeros(N); bench = np.zeros(N); positions = np.ones(N)
    fills_total = 0; spread_total = 0.0; iv_total = 0.0
    greeks_total = 0.0; funding_total = 0.0

    hip3_iv, ibkr_iv = compute_iv_series_real(c, sigs['har_v'],
                                              asset=ibkr_proxy)

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
        if i < len(sigs['hmm_post']):
            p_bull = sigs['hmm_post'][i, 0]
            p_crisis = sigs['hmm_post'][i, 3]
            f *= 1.0 + 0.15 * p_bull - 0.4 * p_crisis
        f = float(np.clip(f, 0.0, 1.15))
        positions[i] = f
        greeks_pnl, _ = vega_bucket_hedged_greeks(
            c[i], hip3_iv[i], ibkr_iv[i], rets[i])
        spread_mult = 1.5 if (i < len(sigs['hmm_post']) and
                              sigs['hmm_post'][i, 3] > 0.3) else 1.0
        fair = c[i - 1]
        rng_bps = (h[i] - l[i]) / fair * 10_000 if fair > 0 else 0
        bar_spread = 0.0; bar_fills = 0
        for lv in range(1, 11):
            off = fair * 8 * lv * spread_mult / 10_000
            bid, ask = fair - off, fair + off
            reps = max(1, min(int(rng_bps / (16 * lv * spread_mult)), 4))
            if l[i] <= bid:
                cap = 0.02 * reps * off / fair * (1 - ADVERSE_SEL)
                bar_spread += cap - 0.02 * reps * HL_MAKER_REBATE
                bar_fills += reps
            if h[i] >= ask:
                cap = 0.02 * reps * off / fair * (1 - ADVERSE_SEL)
                bar_spread += cap - 0.02 * reps * HL_MAKER_REBATE
                bar_fills += reps
        prev_f = positions[i - 1]
        turnover = abs(f - prev_f)
        sigma_daily = sigma_combined / np.sqrt(252)
        exec_cost = almgren_chriss_cost(turnover, sigma_daily)
        vs = hip3_iv[i] - ibkr_iv[i]
        iv_pnl = -np.sign(vs) * min(abs(vs) * 1.5, 0.04) * abs(rets[i]) * 0.5
        fund = funding[i] if i < len(funding) else 0.0
        funding_pnl = -f * fund
        bar_pnl = (f * rets[i] + bar_spread + params['greeks_w'] * greeks_pnl
                   + params['iv_arb_w'] * iv_pnl - exec_cost + funding_pnl)
        pnl[i] = float(np.clip(bar_pnl, -0.20, 0.20))
        bench[i] = rets[i]
        fills_total += bar_fills; spread_total += bar_spread
        iv_total += iv_pnl; greeks_total += greeks_pnl
        funding_total += funding_pnl
        eq *= (1 + pnl[i]); peak = max(peak, eq)
    if test_idx is not None:
        mask = np.zeros(N, dtype=bool); mask[test_idx] = True
        return (pnl[mask], bench[mask], fills_total, spread_total,
                iv_total, greeks_total, funding_total, positions[mask])
    return (pnl[1:], bench[1:], fills_total, spread_total,
            iv_total, greeks_total, funding_total, positions[1:])


def run_kfold_one(o, h, l, c, funding, sigs, n_folds=4, ibkr_proxy=None):
    N = len(c)
    if N < 80:
        return None
    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    n_trials = len(combos)
    oos_returns = np.full(N, np.nan); oos_bench = np.full(N, np.nan)
    fold_metrics = []
    fills_per_fold = []; spread_per_fold = []
    iv_per_fold = []; greeks_per_fold = []; funding_per_fold = []
    for train_idx, test_idx, fold_k in purged_kfold_splits(
            N, n_folds=n_folds, purge=1, embargo=2):
        if len(train_idx) < 30 or len(test_idx) < 10:
            continue
        best_score, best_p = -1e9, dict(zip(keys, combos[0]))
        for combo in combos:
            p = dict(zip(keys, combo))
            try:
                s, b, *_ = run_premium_backtest_one(
                    o, h, l, c, sigs, funding, p,
                    test_idx=train_idx, ibkr_proxy=ibkr_proxy)
                m = compute_metrics(s, b)
                if m is None: continue
                score = m['sharpe'] * 0.4 + m['calmar'] * 0.2 + m['alpha'] * 6.0
                if score > best_score:
                    best_score, best_p = score, p
            except Exception:
                continue
        try:
            s, b, fl, sp, iv_, gr_, fd_, pos = run_premium_backtest_one(
                o, h, l, c, sigs, funding, best_p,
                test_idx=test_idx, ibkr_proxy=ibkr_proxy)
            valid = test_idx[test_idx >= WARM_UP]
            s_valid = s[-len(valid):] if len(s) >= len(valid) else s
            b_valid = b[-len(valid):] if len(b) >= len(valid) else b
            for j, gi in enumerate(valid):
                if j < len(s_valid):
                    oos_returns[gi] = s_valid[j]
                    oos_bench[gi] = b_valid[j]
            m = compute_metrics(s, b)
            if m is None: continue
            m.update(best_p); m['fold_id'] = fold_k
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
    s_arr = oos_returns[mask]; b_arr = oos_bench[mask]
    if len(s_arr) < 20:
        return None
    excess = s_arr - b_arr
    try: spa_p = hansen_spa(excess.reshape(-1, 1), n_boot=500)
    except Exception: spa_p = 1.0
    try: dsr = deflated_sharpe(s_arr, n_trials=n_trials)
    except Exception: dsr = {'dsr': 0, 'p_value': 1, 'significant': False}
    try: rw_pvals = romano_wolf_stepm(excess.reshape(-1, 1), n_boot=500)
    except Exception: rw_pvals = [1.0]
    if len(excess) >= 5 and excess.std(ddof=1) > 0:
        t_stat = excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))
        t_p = 1.0 - student_t.cdf(t_stat, df=len(excess) - 1)
    else:
        t_stat, t_p = 0.0, 1.0
    agg = compute_metrics(s_arr, b_arr)
    if agg is None: return None
    fold_sharpes = [m['sharpe'] for m in fold_metrics]
    agg.update({
        'fold_sharpes': fold_sharpes,
        'fold_sharpe_mean': float(np.mean(fold_sharpes)),
        'fold_sharpe_std': float(np.std(fold_sharpes, ddof=1))
                           if len(fold_sharpes) > 1 else 0.0,
        'hansen_spa_p': float(spa_p),
        'hansen_spa_sig': bool(spa_p < 0.05),
        'dsr': float(dsr['dsr']),
        'dsr_sig': bool(dsr['significant']),
        'romano_wolf_p': float(rw_pvals[0]),
        'romano_wolf_sig': bool(rw_pvals[0] < 0.05),
        't_stat': float(t_stat), 't_p': float(t_p), 't_sig': bool(t_p < 0.05),
        'n_folds': len(fold_metrics), 'n_trials': n_trials,
    })
    total_bars = sum(m['n_bars'] for m in fold_metrics)
    agg['fills_per_day'] = sum(fills_per_fold) / max(total_bars, 1)
    agg['spread_ann'] = float(np.mean(spread_per_fold) * 252) if spread_per_fold else 0
    agg['iv_arb_ann'] = float(np.mean(iv_per_fold) * 252) if iv_per_fold else 0
    agg['greeks_ann'] = float(np.mean(greeks_per_fold) * 252) if greeks_per_fold else 0
    agg['funding_ann'] = float(np.mean(funding_per_fold) * 252) if funding_per_fold else 0
    agg['strat_returns'] = s_arr; agg['bench_returns'] = b_arr
    return agg


def style_ax(ax):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=7)
    for sp in ax.spines.values(): sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, alpha=0.3, linewidth=0.5)


def png_summary(results, data):
    print('  Generating all-HIP-3 premium summary PNG ...')
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
    calmars = [r['calmar'] for tk, r in items]

    n = len(tks)
    fig = plt.figure(figsize=(max(20, n * 0.25), 14), facecolor=BG)
    gs = GridSpec(2, 2, hspace=0.45, wspace=0.18,
                  left=0.05, right=0.98, top=0.94, bottom=0.10)
    fig.suptitle(f'ALL HIP-3 Premium Backtest — {n} Markets, Crème de la Crème',
                 color=WHITE, fontsize=16, fontweight='bold')

    ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
    cols = [GREEN if a > 0 else RED for a in alphas]
    ax.bar(tks, alphas, color=cols, alpha=0.85, edgecolor=WHITE, linewidth=0.3)
    ax.set_title('Out-of-Sample Alpha (%) — sorted', color=TEXT, fontsize=12)
    ax.tick_params(axis='x', rotation=90, labelsize=6)

    ax = fig.add_subplot(gs[0, 1]); style_ax(ax)
    x = np.arange(n)
    ax.bar(x - 0.2, sharpes, 0.4, color=GREEN, alpha=0.85, label='Premium',
           yerr=fold_stds, capsize=2, ecolor=YELLOW)
    ax.bar(x + 0.2, bench_sh, 0.4, color=BLUE, alpha=0.85, label='B&H')
    ax.set_xticks(x); ax.set_xticklabels(tks, rotation=90, fontsize=6)
    ax.set_title('Sharpe (with fold-σ error bars)', color=TEXT, fontsize=12)
    ax.legend(fontsize=9, facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)

    ax = fig.add_subplot(gs[1, 0]); style_ax(ax)
    cols = [GREEN if t > 1.96 else (YELLOW if t > 1.65 else RED)
            for t in t_stats]
    ax.bar(tks, t_stats, color=cols, alpha=0.85, edgecolor=WHITE, linewidth=0.3)
    ax.axhline(1.96, color=GREEN, ls='--', lw=1, label='t=1.96 (p=0.05)')
    ax.axhline(2.58, color=CYAN, ls='--', lw=1, label='t=2.58 (p=0.01)')
    ax.set_title('Paired t-statistic on excess returns',
                 color=TEXT, fontsize=12)
    ax.tick_params(axis='x', rotation=90, labelsize=6)
    ax.legend(fontsize=9, facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)

    ax = fig.add_subplot(gs[1, 1]); style_ax(ax)
    capped = [min(c, 30) for c in calmars]
    cols = [GREEN if c > 3 else (YELLOW if c > 1 else RED) for c in capped]
    ax.bar(tks, capped, color=cols, alpha=0.85, edgecolor=WHITE, linewidth=0.3)
    ax.set_title('Calmar Ratio (capped at 30)', color=TEXT, fontsize=12)
    ax.tick_params(axis='x', rotation=90, labelsize=6)

    plt.savefig(IMG_DIR / 'hip3_all_premium_summary.png', dpi=140,
                facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print('  Saved hip3_all_premium_summary.png')


def png_equity_curves(results, data, top_n=24):
    print('  Generating all-HIP-3 equity curves PNG ...')
    items = sorted([(tk, r) for tk, r in results.items() if r],
                   key=lambda kv: -kv[1].get('alpha', 0))[:top_n]
    if not items:
        return
    n_cols = 6
    n_rows = (len(items) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 3.2),
                             facecolor=BG)
    fig.suptitle(f'Top {len(items)} HIP-3 Premium OOS Equity Curves',
                 color=WHITE, fontsize=14, fontweight='bold')
    for ax_ in np.array(axes).flat:
        style_ax(ax_); ax_.set_visible(False)
    for ax, (tk, r) in zip(np.array(axes).flat, items):
        ax.set_visible(True)
        s = r['strat_returns']; b = r['bench_returns']
        cs = (np.exp(np.cumsum(s)) - 1) * 100
        cb = (np.exp(np.cumsum(b)) - 1) * 100
        x = range(len(cs))
        ax.plot(x, cb, color='#555588', lw=1.0, label='B&H')
        ax.plot(x, cs, color=GREEN, lw=1.5, label='Premium')
        ax.fill_between(x, cb, cs, where=cs > cb, color=GREEN, alpha=0.18)
        ax.fill_between(x, cb, cs, where=cs <= cb, color=RED, alpha=0.18)
        cat = data[tk]['category']
        title = (f'{tk} ({cat})\nα={r["alpha"]*100:+.0f}%  SR={r["sharpe"]:.2f}  '
                 f't={r["t_stat"]:.1f}  Cal={r["calmar"]:.1f}')
        ax.set_title(title, color=TEXT, fontsize=8)
        ax.legend(fontsize=6, facecolor=BG2, edgecolor=GRID_C,
                  labelcolor=TEXT, loc='upper left')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'hip3_all_premium_equity_curves.png', dpi=130,
                facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print('  Saved hip3_all_premium_equity_curves.png')


def main():
    t0 = time.time()
    print('═' * 100)
    print('  ALL HIP-3 PREMIUM PIPELINE — every active market across all deployers')
    print('═' * 100)
    data = load_all_data()
    if not data:
        return

    print('\n── Pre-computing premium signals (ARIMA-BMA, EGARCH-t, HMM, HAR-RV-J, Bayesian)')
    sigs = {}
    for tk, d in data.items():
        t1 = time.time()
        try:
            sigs[tk] = build_signals_short(d['c'], d['o'], d['h'], d['l'],
                                           refit_every=14, min_train=40)
            print(f'  {tk:<12} ({len(d["c"]):>3} bars) {time.time()-t1:.1f}s')
        except Exception as e:
            print(f'  {tk:<12} signal ERROR: {e}')

    print(f'\n── Running Purged K-fold CV on {len(sigs)} markets')
    print(f'   Real Hyperliquid funding + (where available) real CBOE/Yahoo IBKR options')
    results = {}
    for tk, d in data.items():
        if tk not in sigs:
            continue
        t1 = time.time()
        n_bars = len(d['c'])
        n_folds = 4 if n_bars >= 130 else (3 if n_bars >= 100 else 2)
        try:
            r = run_kfold_one(d['o'], d['h'], d['l'], d['c'],
                              d['funding'], sigs[tk], n_folds=n_folds,
                              ibkr_proxy=d['ibkr_proxy'])
            results[tk] = r
            if r:
                print(f'  {tk:<12} {d["category"]:<10} α={r["alpha"]*100:+6.1f}%  '
                      f'SR={r["sharpe"]:5.2f}±{r["fold_sharpe_std"]:.2f}  '
                      f'Cal={r["calmar"]:6.2f}  DD={r["max_dd"]*100:5.1f}%  '
                      f't={r["t_stat"]:5.2f}  SPA={r["hansen_spa_p"]:.3f}  '
                      f'({time.time()-t1:.1f}s)')
            else:
                print(f'  {tk:<12} insufficient data')
        except Exception as e:
            print(f'  {tk:<12} ERROR {e}')
            results[tk] = None

    rows = []
    for tk, r in results.items():
        if not r: continue
        rows.append({
            'asset': tk,
            'category': data[tk]['category'],
            'full_ticker': data[tk]['full_ticker'],
            'dex': data[tk]['dex'],
            'launch_date': str(data[tk]['launch'].date()),
            'days_live': len(data[tk]['c']),
            'ibkr_proxy': data[tk]['ibkr_proxy'] or '',
            'ann_ret': r['ann_ret'], 'bench_ret': r['bench_ret'],
            'alpha': r['alpha'],
            'sharpe': r['sharpe'], 'sharpe_bench': r['sharpe_bench'],
            'calmar': r['calmar'], 'calmar_bench': r['calmar_bench'],
            'max_dd': r['max_dd'], 'max_dd_bench': r['max_dd_bench'],
            't_stat': r['t_stat'], 't_p': r['t_p'], 't_sig': r['t_sig'],
            'fold_sharpe_mean': r['fold_sharpe_mean'],
            'fold_sharpe_std': r['fold_sharpe_std'],
            'hansen_spa_p': r['hansen_spa_p'],
            'hansen_spa_sig': r['hansen_spa_sig'],
            'dsr': r['dsr'], 'dsr_sig': r['dsr_sig'],
            'romano_wolf_p': r['romano_wolf_p'],
            'romano_wolf_sig': r['romano_wolf_sig'],
            'n_folds': r['n_folds'], 'n_trials': r['n_trials'],
            'fills_per_day': r['fills_per_day'],
            'spread_ann': r['spread_ann'],
            'iv_arb_ann': r['iv_arb_ann'],
            'greeks_ann': r['greeks_ann'],
            'funding_ann': r['funding_ann'],
        })
    csv_path = OUT_DIR / 'hip3_all_premium_results.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Summary table
    print('\n' + '═' * 130)
    print('  ALL HIP-3 PREMIUM RESULTS — Sharpe / Calmar / MaxDD / t-statistic / Hansen SPA / DSR')
    print('═' * 130)
    print(f'  {"Asset":<14} {"Category":<10} {"Days":>5} {"Alpha":>8} '
          f'{"SR":>13} {"Calmar":>7} {"MaxDD":>7} {"t":>6} {"t-p":>7} '
          f'{"SPA":>7} {"DSR":>6}')
    print('-' * 130)
    sorted_r = sorted(results.items(),
                      key=lambda kv: -(kv[1]['alpha'] if kv[1] else -999))
    for tk, r in sorted_r:
        if not r: continue
        print(f'  {tk:<14} {data[tk]["category"]:<10} '
              f'{len(data[tk]["c"]):>5} '
              f'{r["alpha"]*100:>+7.1f}% '
              f'{r["sharpe"]:>5.2f}±{r["fold_sharpe_std"]:.2f}  '
              f'{r["calmar"]:>6.2f} '
              f'{r["max_dd"]*100:>6.1f}% '
              f'{r["t_stat"]:>5.2f} '
              f'{r["t_p"]:>6.4f} '
              f'{r["hansen_spa_p"]:>6.4f} '
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
        print('-' * 130)
        print(f'  Markets backtested: {n_total}   Positive alpha: {n_pos}/{n_total}   '
              f'Mean α: {mean_a:+.1f}%   Mean SR: {mean_sr:.2f}   '
              f'Mean Calmar: {mean_cm:.2f}   Mean MaxDD: {mean_dd:.1f}%')
        print(f'  t-test sig (p<0.05): {n_t}/{n_total}   '
              f'Hansen SPA sig: {n_spa}/{n_total}   '
              f'DSR sig (>0.95): {n_dsr}/{n_total}')

    # By category
    cats = {}
    for tk, r in results.items():
        if not r: continue
        c = data[tk]['category']
        cats.setdefault(c, []).append(r['alpha'])
    print('\n  By category (mean alpha):')
    for c, alphas in sorted(cats.items(), key=lambda x: -np.mean(x[1])):
        print(f'    {c:<12} n={len(alphas):>3}   mean α = {np.mean(alphas)*100:+.1f}%   '
              f'pos={sum(1 for a in alphas if a > 0)}/{len(alphas)}')

    print(f'\n  CSV: {csv_path}')

    print('\n── Generating ALL-HIP-3 Premium Visualizations ──')
    png_summary(results, data)
    png_equity_curves(results, data)
    print(f'\n  Total time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
