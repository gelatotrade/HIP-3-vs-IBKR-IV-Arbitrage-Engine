#!/usr/bin/env python3
"""
HIP-3 vs IBKR Visualizations — matching Market-Making-Engine style.

Produces:
  1. hip3_iv_surface_comparison.png  — 3D IV surface: HIP-3 vs IBKR
  2. hip3_equity_curves.png          — Equity curves for all HIP-3 assets
  3. hip3_greeks_surface.png         — 3D Greeks surface (vanna/vomma/zomma)
  4. hip3_vol_spread_heatmap.png     — Vol spread heatmap across assets/time
  5. hip3_regime_dashboard.png       — Regime detection + MM parameters
  6. hip3_arbitrage_summary.png      — Alpha & strategy summary bars

Dark terminal aesthetic with neon colors, matching original repo.
"""
import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperliquid_hip3_client import generate_synthetic_hip3_data
from ibkr_options_client import IBKROptionsClient, bs_greeks
from iv_arbitrage_engine import IVArbitrageEngine
from greeks_strategies import GreeksStrategyEngine, classify_regime

# ── Style constants (matching original repo) ──
BG    = '#0a0a1a'
BG2   = '#0d0d22'
TEXT  = '#cccccc'
GREEN = '#00ff88'
RED   = '#ff3344'
YELLOW= '#ffaa00'
BLUE  = '#4488ff'
CYAN  = '#00ffcc'
WHITE = '#ffffff'
GRID_C= '#1a1a2e'
DIM   = '#556677'
ORANGE= '#ff8800'
PURPLE= '#cc66ff'

CMAP_BULL = LinearSegmentedColormap.from_list('bull', [
    '#0000aa','#0055cc','#22aa66','#00ff88','#eeffaa','#ffffdd'])
CMAP_NORMAL = LinearSegmentedColormap.from_list('normal', [
    '#001144','#003388','#2266aa','#44aacc','#88ddee','#ccffff'])
CMAP_CRISIS = LinearSegmentedColormap.from_list('crisis', [
    '#000044','#220066','#660088','#cc0022','#ff4444','#ffcc44'])
CMAP_VANNA = LinearSegmentedColormap.from_list('vanna', [
    '#000033','#003366','#006699','#3399cc','#66ccff','#ccffff'])
CMAP_VOMMA = LinearSegmentedColormap.from_list('vomma', [
    '#330033','#660066','#990099','#cc33cc','#ff66ff','#ffccff'])
CMAP_SPREAD = LinearSegmentedColormap.from_list('spread', [
    '#0000aa','#2244cc','#ffffff','#cc4422','#aa0000'])

OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'
DPI = 150


def style_ax(ax, title='', xlabel='', ylabel='', is_3d=False):
    ax.set_facecolor(BG2 if not is_3d else BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color=DIM, fontsize=9)
    ax.set_ylabel(ylabel, color=DIM, fontsize=9)
    ax.tick_params(colors=DIM, labelsize=7)
    if is_3d:
        ax.set_zlabel('', color=DIM, fontsize=9)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(GRID_C)
        ax.yaxis.pane.set_edgecolor(GRID_C)
        ax.zaxis.pane.set_edgecolor(GRID_C)
        ax.grid(True, color='#223344', alpha=0.2)
    else:
        ax.grid(True, color=GRID_C, alpha=0.4, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID_C)


# ════════════════════════════════════════════════════════════════
# 1. IV Surface Comparison: HIP-3 vs IBKR
# ════════════════════════════════════════════════════════════════
def plot_iv_surface_comparison(spot=42000, hip3_base_iv=0.70, ibkr_base_iv=0.62):
    print('  Generating hip3_iv_surface_comparison.png ...')
    ibkr = IBKROptionsClient()

    n = 50
    strikes = np.linspace(spot * 0.75, spot * 1.25, n)
    expiries = np.linspace(7, 180, n)
    X, Y = np.meshgrid(strikes, expiries)

    # IBKR IV surface: skew + smile + term structure
    moneyness = np.log(X / spot)
    ibkr_iv = ibkr_base_iv + (-0.15) * moneyness + 0.05 * moneyness**2 + 0.02 * (Y/365)
    ibkr_iv = np.clip(ibkr_iv, 0.05, 2.0)

    # HIP-3 IV surface: flatter, higher base, less skew
    hip3_iv = hip3_base_iv + (-0.03) * moneyness + 0.01 * moneyness**2 + 0.005 * (Y/365)
    hip3_iv = np.clip(hip3_iv, 0.10, 2.0)

    # Vol spread
    vol_spread = hip3_iv - ibkr_iv

    fig = plt.figure(figsize=(20, 7), facecolor=BG)
    fig.suptitle('Implied Volatility Surface: HIP-3 vs IBKR Options',
                 color=WHITE, fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: IBKR IV surface
    ax1 = fig.add_subplot(131, projection='3d')
    style_ax(ax1, 'IBKR Options IV Surface', 'Strike ($)', 'DTE (days)', is_3d=True)
    ax1.plot_surface(X/1000, Y, ibkr_iv*100, cmap=CMAP_NORMAL, alpha=0.9,
                     rstride=2, cstride=2, edgecolor='none', antialiased=True)
    ax1.set_zlabel('IV (%)', color=DIM, fontsize=8)
    ax1.view_init(elev=25, azim=220)

    # Panel 2: HIP-3 IV surface
    ax2 = fig.add_subplot(132, projection='3d')
    style_ax(ax2, 'HIP-3 Implied Vol Surface', 'Strike ($)', 'DTE (days)', is_3d=True)
    ax2.plot_surface(X/1000, Y, hip3_iv*100, cmap=CMAP_BULL, alpha=0.9,
                     rstride=2, cstride=2, edgecolor='none', antialiased=True)
    ax2.set_zlabel('IV (%)', color=DIM, fontsize=8)
    ax2.view_init(elev=25, azim=220)

    # Panel 3: Vol spread (arb opportunity)
    ax3 = fig.add_subplot(133, projection='3d')
    style_ax(ax3, 'Vol Spread (HIP3 - IBKR) = Arb Signal', 'Strike ($)', 'DTE (days)', is_3d=True)
    vmax = max(abs(vol_spread.min()), abs(vol_spread.max()))
    ax3.plot_surface(X/1000, Y, vol_spread*100, cmap=CMAP_SPREAD, alpha=0.9,
                     rstride=2, cstride=2, edgecolor='none', antialiased=True,
                     vmin=-vmax*100, vmax=vmax*100)
    ax3.set_zlabel('Spread (%)', color=DIM, fontsize=8)
    ax3.view_init(elev=25, azim=220)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / 'hip3_iv_surface_comparison.png', dpi=DPI, facecolor=BG)
    plt.close(fig)
    print('    Done.')


# ════════════════════════════════════════════════════════════════
# 2. Equity Curves for HIP-3 Assets
# ════════════════════════════════════════════════════════════════
def plot_equity_curves(results_df, curves):
    print('  Generating hip3_equity_curves.png ...')
    top = results_df.sort_values('alpha', ascending=False).head(6)
    n = len(top)
    if n == 0:
        return

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.25,
                  left=0.06, right=0.97, top=0.93, bottom=0.05)
    fig.suptitle('HIP-3 IV Arbitrage Backtest — Out-of-Sample Equity Curves',
                 fontsize=16, fontweight='bold', color=WHITE, y=0.97)

    colors_s = [GREEN, CYAN, ORANGE, YELLOW, PURPLE, '#66ffcc']
    color_b = '#555588'

    for i, (_, row) in enumerate(top.iterrows()):
        tk = row['asset']
        if tk not in curves:
            continue
        sr, br = curves[tk]
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_facecolor(BG2)

        cum_s = np.exp(np.cumsum(sr)) - 1
        cum_b = np.exp(np.cumsum(br)) - 1
        days = np.arange(len(sr)) / 365

        ax.plot(days, cum_b * 100, color=color_b, linewidth=1.2,
                alpha=0.7, label='Buy & Hold')
        ax.plot(days, cum_s * 100, color=colors_s[i], linewidth=1.5,
                label='Strategy')
        ax.fill_between(days, cum_b*100, cum_s*100,
                        where=cum_s > cum_b, alpha=0.15, color=colors_s[i])
        ax.fill_between(days, cum_b*100, cum_s*100,
                        where=cum_s <= cum_b, alpha=0.10, color=RED)

        ns = sum([row.get('sr_sig',False), row.get('boot_sig',False),
                  row.get('perm_sig',False), row.get('dsr_sig',False)])
        sig_str = f'{ns}/4 sig' if ns >= 2 else f'{ns}/4'
        ax.set_title(
            f'{tk}  |  Alpha={row["alpha"]*100:+.1f}%  Sharpe={row["sharpe"]:.2f}  '
            f'Calmar={row["calmar"]:.2f}  [{sig_str}]',
            fontsize=10, color=WHITE, fontweight='bold')
        ax.set_xlabel('Years (OOS)', fontsize=8, color=DIM)
        ax.set_ylabel('Cumulative Return (%)', fontsize=8, color=DIM)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.3,
                  facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)
        ax.tick_params(colors=DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID_C)
        ax.grid(True, alpha=0.15, color='#444466')

    fig.savefig(OUT_DIR / 'hip3_equity_curves.png', dpi=DPI, facecolor=BG)
    plt.close(fig)
    print('    Done.')


# ════════════════════════════════════════════════════════════════
# 3. Greeks 3D Surfaces (Vanna, Vomma, Zomma)
# ════════════════════════════════════════════════════════════════
def plot_greeks_surfaces(spot=42000, iv=0.65):
    print('  Generating hip3_greeks_surface.png ...')
    n = 50
    strikes = np.linspace(spot * 0.80, spot * 1.20, n)
    expiries = np.linspace(3, 90, n)

    vanna_surf = np.zeros((n, n))
    vomma_surf = np.zeros((n, n))
    zomma_surf = np.zeros((n, n))

    for i, dte in enumerate(expiries):
        for j, K in enumerate(strikes):
            T = dte / 365
            q = bs_greeks(spot, K, T, 0.05, iv, 'call')
            vanna_surf[i, j] = q.vanna
            vomma_surf[i, j] = q.vomma
            zomma_surf[i, j] = q.zomma

    X, Y = np.meshgrid(strikes / 1000, expiries)

    fig = plt.figure(figsize=(20, 7), facecolor=BG)
    fig.suptitle('Higher-Order Greeks — HIP-3 Arbitrage Edge',
                 color=WHITE, fontsize=14, fontweight='bold', y=0.98)

    titles = ['Vanna (dDelta/dVol)', 'Vomma (dVega/dVol)', 'Zomma (dGamma/dVol)']
    surfaces = [vanna_surf, vomma_surf, zomma_surf]
    cmaps = [CMAP_VANNA, CMAP_VOMMA, CMAP_CRISIS]

    for idx in range(3):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        style_ax(ax, titles[idx], 'Strike ($K)', 'DTE', is_3d=True)
        ax.plot_surface(X, Y, surfaces[idx], cmap=cmaps[idx], alpha=0.9,
                        rstride=2, cstride=2, edgecolor='none', antialiased=True)
        ax.plot_wireframe(X[::5,::5], Y[::5,::5], surfaces[idx][::5,::5],
                          color='white', alpha=0.05, linewidth=0.3)
        ax.view_init(elev=25, azim=225)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / 'hip3_greeks_surface.png', dpi=DPI, facecolor=BG)
    plt.close(fig)
    print('    Done.')


# ════════════════════════════════════════════════════════════════
# 4. Vol Spread Heatmap
# ════════════════════════════════════════════════════════════════
def plot_vol_spread_heatmap(data, arb_engine):
    print('  Generating hip3_vol_spread_heatmap.png ...')
    assets = list(data.keys())[:12]
    n_days = min(len(data[assets[0]]), 730)
    sample_days = np.linspace(30, n_days-1, 60, dtype=int)

    spread_matrix = np.zeros((len(assets), len(sample_days)))

    for i, asset in enumerate(assets):
        prices = data[asset]['close'].values
        hip3_iv = arb_engine.compute_hip3_implied_vol(prices)
        ibkr_iv = arb_engine.compute_ibkr_atm_iv(prices)
        for j, d in enumerate(sample_days):
            if d < len(hip3_iv) and d < len(ibkr_iv):
                spread_matrix[i, j] = (hip3_iv[d] - ibkr_iv[d]) * 100

    fig, ax = plt.subplots(figsize=(16, 8), facecolor=BG)
    style_ax(ax, 'IV Spread Heatmap: HIP-3 minus IBKR (% points)',
             'Time (day index)', '')

    vmax = max(abs(spread_matrix.min()), abs(spread_matrix.max()), 1)
    im = ax.imshow(spread_matrix, cmap=CMAP_SPREAD, aspect='auto',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_yticks(range(len(assets)))
    ax.set_yticklabels(assets, fontsize=9, color=TEXT)
    ax.set_xticks(np.linspace(0, len(sample_days)-1, 10, dtype=int))
    ax.set_xticklabels([str(sample_days[int(x)]) for x in
                        np.linspace(0, len(sample_days)-1, 10)],
                       fontsize=8, color=DIM)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=DIM, labelsize=8)
    cbar.set_label('Vol Spread (HIP3-IBKR) %', color=TEXT, fontsize=9)

    # Annotate significant spreads
    for i in range(spread_matrix.shape[0]):
        for j in range(spread_matrix.shape[1]):
            val = spread_matrix[i, j]
            if abs(val) > vmax * 0.6:
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        color=BG if abs(val) > vmax * 0.8 else TEXT,
                        fontsize=5, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'hip3_vol_spread_heatmap.png', dpi=DPI, facecolor=BG)
    plt.close(fig)
    print('    Done.')


# ════════════════════════════════════════════════════════════════
# 5. Regime Dashboard
# ════════════════════════════════════════════════════════════════
def plot_regime_dashboard(data, asset='BTC'):
    print('  Generating hip3_regime_dashboard.png ...')
    if asset not in data:
        asset = list(data.keys())[0]

    df = data[asset]
    prices = df['close'].values
    N = len(prices)
    returns = np.zeros(N)
    returns[1:] = np.diff(np.log(prices))

    regimes = []
    vols = []
    for i in range(N):
        regimes.append(classify_regime(returns, i))
        w = returns[max(0, i-20):i] if i >= 20 else returns[:max(i,1)]
        vols.append(np.std(w, ddof=1) * np.sqrt(365) * 100 if len(w) > 1 else 50)

    regime_colors_map = {
        'BULL': GREEN, 'NORMAL': BLUE, 'CAUTIOUS': YELLOW,
        'CRISIS': RED, 'RECOVERY': CYAN
    }
    regime_spread = {
        'BULL': 0.8, 'NORMAL': 1.0, 'CAUTIOUS': 2.0, 'CRISIS': 3.0, 'RECOVERY': 1.3
    }
    regime_base = {
        'BULL': 1.10, 'NORMAL': 1.00, 'CAUTIOUS': 0.70, 'CRISIS': 0.35, 'RECOVERY': 1.05
    }

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.30,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)
    fig.suptitle(f'HIP-3 Regime Dashboard — {asset}',
                 fontsize=16, fontweight='bold', color=WHITE, y=0.97)

    days = np.arange(N)

    # Panel 1: Price + regime coloring
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, f'{asset} Price with Regime Detection', 'Day', 'Price ($)')
    for i in range(1, N):
        color = regime_colors_map.get(regimes[i], BLUE)
        ax1.plot([days[i-1], days[i]], [prices[i-1], prices[i]],
                 color=color, linewidth=1.2, alpha=0.9)
    # Legend
    for regime, color in regime_colors_map.items():
        ax1.plot([], [], color=color, linewidth=3, label=regime)
    ax1.legend(fontsize=8, loc='upper left', framealpha=0.3,
               facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT, ncol=5)

    # Panel 2: Rolling volatility
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, '20d Rolling Volatility (annualized)', 'Day', 'Vol (%)')
    ax2.fill_between(days, 0, vols, alpha=0.2, color=YELLOW)
    ax2.plot(days, vols, color=YELLOW, linewidth=1.2)
    ax2.axhline(80, color=RED, linestyle='--', alpha=0.5, linewidth=0.8, label='Crisis')
    ax2.axhline(60, color=YELLOW, linestyle='--', alpha=0.5, linewidth=0.8, label='Cautious')
    ax2.legend(fontsize=7, facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)

    # Panel 3: Regime distribution
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, 'Regime Distribution', '', 'Count')
    regime_counts = {}
    for r in regimes:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    names = list(regime_counts.keys())
    counts = list(regime_counts.values())
    colors = [regime_colors_map.get(n, BLUE) for n in names]
    ax3.barh(names, counts, color=colors, alpha=0.85, height=0.6)
    for j, v in enumerate(counts):
        ax3.text(v + max(counts)*0.02, j, str(v), va='center', fontsize=9, color=TEXT)

    # Panel 4: MM parameters over time
    ax4 = fig.add_subplot(gs[2, 0])
    style_ax(ax4, 'Market-Making Spread Multiplier', 'Day', 'Multiplier')
    spreads = [regime_spread.get(r, 1.0) for r in regimes]
    ax4.fill_between(days, 0, spreads, alpha=0.15, color=CYAN)
    ax4.plot(days, spreads, color=CYAN, linewidth=1.2)
    ax4.set_ylim(0, 3.5)

    # Panel 5: Base position
    ax5 = fig.add_subplot(gs[2, 1])
    style_ax(ax5, 'Base Position Sizing', 'Day', 'Base (%)')
    bases = [regime_base.get(r, 1.0) * 100 for r in regimes]
    ax5.fill_between(days, 0, bases, alpha=0.15, color=GREEN)
    ax5.plot(days, bases, color=GREEN, linewidth=1.2)
    ax5.axhline(100, color=DIM, linestyle=':', alpha=0.5)

    fig.savefig(OUT_DIR / 'hip3_regime_dashboard.png', dpi=DPI, facecolor=BG)
    plt.close(fig)
    print('    Done.')


# ════════════════════════════════════════════════════════════════
# 6. Alpha & Strategy Summary
# ════════════════════════════════════════════════════════════════
def plot_arbitrage_summary(results_df, greeks_results):
    print('  Generating hip3_arbitrage_summary.png ...')
    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs = GridSpec(2, 2, hspace=0.40, wspace=0.30,
                  left=0.07, right=0.97, top=0.92, bottom=0.06)
    fig.suptitle('HIP-3 vs IBKR Arbitrage — Strategy Summary',
                 fontsize=16, fontweight='bold', color=WHITE, y=0.97)

    # Panel 1: Alpha by asset
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'Out-of-Sample Alpha by Asset (% p.a.)', '', 'Alpha (%)')
    assets = results_df.sort_values('alpha', ascending=True)['asset'].values
    alphas = results_df.sort_values('alpha', ascending=True)['alpha'].values * 100
    bar_colors = [GREEN if a > 0 else RED for a in alphas]
    ax1.barh(assets, alphas, color=bar_colors, height=0.6, alpha=0.85)
    ax1.axvline(0, color=WHITE, linewidth=0.8, alpha=0.5)
    for j, v in enumerate(alphas):
        ax1.text(v + (0.5 if v >= 0 else -0.5), j,
                 f'{v:+.1f}%', va='center', fontsize=7,
                 color=WHITE, ha='left' if v >= 0 else 'right')

    # Panel 2: Sharpe comparison
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'Sharpe Ratio: Strategy vs Benchmark', '', 'Sharpe')
    x = np.arange(len(results_df))
    w = 0.35
    sorted_df = results_df.sort_values('sharpe', ascending=True)
    ax2.barh(sorted_df['asset'].values, sorted_df['sharpe'].values,
             height=w, alpha=0.85, color=CYAN, label='Strategy')
    ax2.barh([a for a in sorted_df['asset'].values],
             sorted_df['sharpe_bench'].values,
             height=w, alpha=0.5, color='#666699', label='Benchmark',
             left=0)
    ax2.legend(fontsize=8, facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)

    # Panel 3: Greeks strategy Sharpe
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'Greeks Strategy Sharpe Ratios (Best per Asset)', '', 'Sharpe')
    if greeks_results:
        first_asset = next(iter(greeks_results))
        strat_names = list(greeks_results[first_asset].keys())
        avg_sharpes = []
        for sn in strat_names:
            sharpes = [greeks_results[a][sn].sharpe for a in greeks_results if sn in greeks_results[a]]
            avg_sharpes.append(np.mean(sharpes) if sharpes else 0)
        display_names = [greeks_results[first_asset][sn].name for sn in strat_names]
        sorted_idx = np.argsort(avg_sharpes)
        colors_g = [GREEN if s > 0 else RED for s in np.array(avg_sharpes)[sorted_idx]]
        ax3.barh(np.array(display_names)[sorted_idx],
                 np.array(avg_sharpes)[sorted_idx],
                 color=colors_g, height=0.6, alpha=0.85)
        ax3.axvline(0, color=WHITE, linewidth=0.8, alpha=0.5)
        for j, v in enumerate(np.array(avg_sharpes)[sorted_idx]):
            ax3.text(v + 0.02, j, f'{v:.2f}', va='center', fontsize=7, color=TEXT)

    # Panel 4: IV arb contribution
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'IV Arbitrage Annual Contribution (%)', '', 'IV Arb (%)')
    sorted_iv = results_df.sort_values('iv_arb_ann', ascending=True)
    iv_vals = sorted_iv['iv_arb_ann'].values * 100
    iv_colors = [GREEN if v > 0 else RED for v in iv_vals]
    ax4.barh(sorted_iv['asset'].values, iv_vals,
             color=iv_colors, height=0.6, alpha=0.85)
    ax4.axvline(0, color=WHITE, linewidth=0.8, alpha=0.5)
    for j, v in enumerate(iv_vals):
        ax4.text(v + 0.1, j, f'{v:+.1f}%', va='center', fontsize=7, color=TEXT)

    fig.savefig(OUT_DIR / 'hip3_arbitrage_summary.png', dpi=DPI, facecolor=BG)
    plt.close(fig)
    print('    Done.')


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def generate_all_visualizations(results_df=None, curves=None, greeks_results=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'\nOutput: {OUT_DIR}')

    # Generate data if not provided
    data = generate_synthetic_hip3_data(n_assets=15, n_days=730)
    arb_engine = IVArbitrageEngine()

    # 1. IV Surface Comparison
    plot_iv_surface_comparison()

    # 2. Greeks 3D surfaces
    plot_greeks_surfaces()

    # 3. Vol spread heatmap
    plot_vol_spread_heatmap(data, arb_engine)

    # 4. Regime dashboard
    plot_regime_dashboard(data, 'BTC')

    # 5. Equity curves (if results available)
    if results_df is not None and curves is not None:
        plot_equity_curves(results_df, curves)

    # 6. Arbitrage summary (if results available)
    if results_df is not None:
        plot_arbitrage_summary(results_df, greeks_results or {})

    print('\nAll visualizations generated.')


if __name__ == '__main__':
    # Run standalone — generate sample data + run backtest + visualize
    print('='*80)
    print('  HIP-3 vs IBKR Visualizations')
    print('='*80)

    # First run backtest to get data
    try:
        from hip3_backtest import main as run_backtest
        results_df, curves, greeks_results = run_backtest()
        generate_all_visualizations(results_df, curves, greeks_results)
    except Exception as e:
        print(f'  Backtest error: {e}, generating standalone visualizations ...')
        generate_all_visualizations()
