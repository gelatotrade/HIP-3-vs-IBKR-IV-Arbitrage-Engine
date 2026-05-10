#!/usr/bin/env python3
"""
Generate a 4-panel 3D arbitrage setup visualization showing how to identify
and trade four canonical HIP-3 perp vs IBKR options arbitrage setups using
synthetic data.

Each panel contains:
  1. A 3D surface depicting the relevant arbitrage signal
  2. Highlighted entry zones (BUY/SELL markers, threshold contours)
  3. A trade-construction annotation explaining how to execute

Output: docs/img/hip3_arbitrage_setups_3d.png
"""
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ibkr_options_client import bs_greeks

OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = '#080810'
PANEL_BG = '#0e0e1a'
TEXT = '#dddddd'
DIM = '#778899'
CYAN = '#00ffcc'
GREEN = '#22cc88'
RED = '#ff3344'
BLUE = '#3399ff'
GOLD = '#ffcc00'
ORANGE = '#ff9933'

CMAP_DIV = LinearSegmentedColormap.from_list('div', [
    '#0066ff', '#3399ff', '#1a1a2e', '#ff6633', '#ff2200'])
CMAP_SKEW = LinearSegmentedColormap.from_list('sk', [
    '#1a3300', '#446622', '#888844', '#ddaa33', '#ff6622'])
CMAP_TERM = LinearSegmentedColormap.from_list('tm', [
    '#000033', '#003366', '#0066aa', '#33aaee', '#aaccff'])
CMAP_GREEK = LinearSegmentedColormap.from_list('gr', [
    '#001833', '#003d66', '#00aa66', '#ccff44', '#ffff66'])


def style_3d(ax, title, xlabel, ylabel, zlabel):
    ax.set_facecolor(PANEL_BG)
    ax.xaxis.pane.set_facecolor(PANEL_BG)
    ax.yaxis.pane.set_facecolor(PANEL_BG)
    ax.zaxis.pane.set_facecolor(PANEL_BG)
    ax.xaxis.pane.set_edgecolor(DIM)
    ax.yaxis.pane.set_edgecolor(DIM)
    ax.zaxis.pane.set_edgecolor(DIM)
    ax.tick_params(axis='x', colors=DIM, labelsize=7)
    ax.tick_params(axis='y', colors=DIM, labelsize=7)
    ax.tick_params(axis='z', colors=DIM, labelsize=7)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=8, labelpad=2)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=8, labelpad=2)
    ax.set_zlabel(zlabel, color=TEXT, fontsize=8, labelpad=2)
    ax.set_title(title, color=CYAN, fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.15)


def annotation_box(fig, x, y, w, h, lines, header, header_color):
    """Add a trade-construction annotation box at figure-relative coords."""
    ax = fig.add_axes([x, y, w, h])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#101020')
    for spine in ax.spines.values():
        spine.set_edgecolor(header_color)
        spine.set_linewidth(1.2)
    ax.text(0.5, 0.92, header, transform=ax.transAxes, ha='center', va='top',
            color=header_color, fontsize=10, fontweight='bold',
            family='monospace')
    y0 = 0.78
    for line in lines:
        color = TEXT
        weight = 'normal'
        if line.startswith('SIGNAL:'):
            color = GOLD
            weight = 'bold'
        elif line.startswith('BUY ') or line.startswith('LONG '):
            color = GREEN
            weight = 'bold'
        elif line.startswith('SELL ') or line.startswith('SHORT '):
            color = RED
            weight = 'bold'
        elif line.startswith('HEDGE:'):
            color = BLUE
        elif line.startswith('EDGE:'):
            color = CYAN
        ax.text(0.04, y0, line, transform=ax.transAxes,
                ha='left', va='top', color=color, fontsize=8,
                family='monospace', fontweight=weight)
        y0 -= 0.13


def setup_1_vol_spread(ax):
    """HIP-3 implied vol vs IBKR options IV across moneyness × DTE.

    Synthetic regime: HIP-3 prices ~28% flat IV from funding+spread dynamics.
    IBKR has ~22% ATM with skew (puts richer) and term structure.
    The spread surface highlights where HIP-3 over/underprices vol.
    """
    moneyness = np.linspace(-0.20, 0.20, 35)
    dte = np.linspace(7, 90, 25)
    M, D = np.meshgrid(moneyness, dte)

    # IBKR IV: skew (puts richer) + term structure (long-dated higher)
    ibkr_iv = 0.22 + (-0.30) * M + 0.08 * M**2 + 0.0008 * D
    # HIP-3 IV: ~flat at 28% with mild noise
    hip3_iv = 0.28 + 0.005 * np.sin(D / 12) + 0.001 * M
    spread = (hip3_iv - ibkr_iv) * 100  # in vol points

    surf = ax.plot_surface(M, D, spread, cmap=CMAP_DIV, edgecolor='none',
                            alpha=0.92, vmin=-8, vmax=8, antialiased=True)

    # Threshold contour at +5 vol pts (SELL HIP-3 zone)
    ax.contour(M, D, spread, levels=[5], colors=[RED], linewidths=2,
               offset=spread.min())
    # Threshold contour at -3 vol pts (BUY HIP-3 zone)
    ax.contour(M, D, spread, levels=[-3], colors=[BLUE], linewidths=2,
               offset=spread.min())

    # Mark the SELL VOL hot spot (puts, short DTE, where HIP-3 most overprices)
    sell_idx = np.unravel_index(np.argmax(spread), spread.shape)
    ax.scatter([M[sell_idx]], [D[sell_idx]], [spread[sell_idx]],
               color=RED, s=120, edgecolor='white', linewidth=1.5, zorder=10)
    ax.text(M[sell_idx], D[sell_idx], spread[sell_idx] + 1.5,
            f'SELL\n{spread[sell_idx]:.1f}vp',
            color=RED, fontsize=8, fontweight='bold', ha='center')

    # Mark the BUY VOL hot spot (calls)
    buy_idx = np.unravel_index(np.argmin(spread), spread.shape)
    ax.scatter([M[buy_idx]], [D[buy_idx]], [spread[buy_idx]],
               color=BLUE, s=120, edgecolor='white', linewidth=1.5, zorder=10)
    ax.text(M[buy_idx], D[buy_idx], spread[buy_idx] - 1.5,
            f'BUY\n{spread[buy_idx]:.1f}vp',
            color=BLUE, fontsize=8, fontweight='bold', ha='center')

    style_3d(ax, '① Vol Spread Arb — HIP-3 IV vs IBKR IV',
             'log-moneyness', 'DTE (days)', 'Spread (vol pts)')
    ax.view_init(elev=24, azim=-58)
    return surf


def setup_2_term_structure(ax):
    """Calendar spread: IBKR has rich term structure, HIP-3 is flat.

    The signal: front-month vs back-month IV differential by moneyness.
    Steeper backwardation → buy back / sell front; contango → opposite.
    """
    moneyness = np.linspace(-0.15, 0.15, 30)
    dte_front = np.linspace(7, 30, 20)
    dte_back = np.linspace(60, 180, 20)
    M, D = np.meshgrid(moneyness, dte_back)

    # Term structure differential: long-dated vs short-dated IV at same K
    # IBKR: positive contango +0.06% per day → back > front
    ibkr_back_iv = 0.22 + 0.0008 * D - 0.30 * M + 0.08 * M**2
    ibkr_front_iv = 0.22 + 0.0008 * 14 - 0.30 * M + 0.08 * M**2  # ref 14d
    term_diff = (ibkr_back_iv - ibkr_front_iv) * 100  # vol pts

    # HIP-3 contributes nothing to term structure (single funding rate),
    # so the entire surface IS the arbitrage signal
    surf = ax.plot_surface(M, D, term_diff, cmap=CMAP_TERM, edgecolor='none',
                            alpha=0.92, antialiased=True)
    ax.contour(M, D, term_diff, levels=8, colors='white',
               alpha=0.15, linewidths=0.5, offset=term_diff.min())

    # Mark the steepest contango point (LONG BACK / SHORT FRONT)
    max_idx = np.unravel_index(np.argmax(term_diff), term_diff.shape)
    ax.scatter([M[max_idx]], [D[max_idx]], [term_diff[max_idx]],
               color=GOLD, s=150, edgecolor='white', linewidth=1.5,
               marker='^', zorder=10)
    ax.text(M[max_idx], D[max_idx], term_diff[max_idx] + 1.0,
            f'CALENDAR\n+{term_diff[max_idx]:.1f}vp',
            color=GOLD, fontsize=8, fontweight='bold', ha='center')

    # Add reference plane at HIP-3 (flat = 0)
    zz = np.zeros_like(term_diff)
    ax.plot_surface(M, D, zz, color=ORANGE, alpha=0.15, edgecolor='none')
    ax.text(0, D.mean(), 0.5, 'HIP-3 flat (0)',
            color=ORANGE, fontsize=7, fontweight='bold', ha='center')

    style_3d(ax, '② Calendar Spread — Term Structure',
             'log-moneyness', 'DTE back leg', 'Back − Front IV (vp)')
    ax.view_init(elev=22, azim=-50)
    return surf


def setup_3_skew_arb(ax):
    """Risk reversal: IBKR has put skew, HIP-3 is symmetric.

    The signal: 25-delta put IV - 25-delta call IV across spot × IV regime.
    Wide skew → sell put / buy call (risk reversal), delta-hedge with perp.
    """
    spot_pct = np.linspace(-0.10, 0.10, 30)  # spot move from anchor
    iv_regime = np.linspace(0.15, 0.50, 25)  # base IV regime
    S, V = np.meshgrid(spot_pct, iv_regime)

    # 25-delta skew: more negative when spot drops + IV high (crisis-style)
    # Skew magnitude grows with vol level (vol-of-vol effect)
    skew_iv = (-0.18 - 0.45 * V) * (1 + 0.5 * S * np.sign(-S)) * 100
    # Add some structure: skew worst when spot down + vol up
    skew_iv = skew_iv - 8 * np.maximum(0, -S) * V * 100 / 100

    surf = ax.plot_surface(S, V, skew_iv, cmap=CMAP_SKEW, edgecolor='none',
                            alpha=0.92, antialiased=True)
    # Top-decile contour
    threshold = np.percentile(skew_iv, 10)  # most negative = worst skew
    ax.contour(S, V, skew_iv, levels=[threshold], colors=[CYAN],
               linewidths=2.5, offset=skew_iv.min())

    # Mark widest skew zone (RISK REVERSAL trade)
    rr_idx = np.unravel_index(np.argmin(skew_iv), skew_iv.shape)
    ax.scatter([S[rr_idx]], [V[rr_idx]], [skew_iv[rr_idx]],
               color=ORANGE, s=140, edgecolor='white', linewidth=1.5,
               marker='D', zorder=10)
    ax.text(S[rr_idx], V[rr_idx], skew_iv[rr_idx] - 3,
            f'RISK REVERSAL\n{skew_iv[rr_idx]:.1f}vp',
            color=ORANGE, fontsize=8, fontweight='bold', ha='center')

    style_3d(ax, '③ Skew Arb — IBKR Put Skew vs HIP-3 Symmetric',
             'spot move (%)', 'IV regime', '25Δ Put − Call IV (vp)')
    ax.view_init(elev=24, azim=42)
    return surf


def setup_4_gamma_scalp(ax):
    """Gamma scalping: IBKR options have convex Greeks, HIP-3 is linear.

    The signal: gamma + |vanna| + |vomma| intensity across strike × IV.
    Long IBKR straddle + delta-hedge with HIP-3 perp captures the convexity.
    """
    spot = 100.0
    strikes = np.linspace(80, 120, 35)
    ivs = np.linspace(0.12, 0.50, 25)
    K, V = np.meshgrid(strikes, ivs)

    Z = np.zeros_like(K)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            q = bs_greeks(spot, K[i, j], 30 / 365.0, 0.05, V[i, j], 'call')
            # Composite convexity score: gamma + |vanna|/100 + |vomma|/1000
            Z[i, j] = q.gamma * 100 + abs(q.vanna) * 0.1 + abs(q.vomma) * 0.01

    surf = ax.plot_surface(K, V, Z, cmap=CMAP_GREEK, edgecolor='none',
                            alpha=0.92, antialiased=True)
    threshold = np.percentile(Z, 90)
    ax.contour(K, V, Z, levels=[threshold], colors=[CYAN],
               linewidths=2.5, offset=Z.min())

    # Mark the peak (ATM, low IV) — best gamma scalp
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    ax.scatter([K[max_idx]], [V[max_idx]], [Z[max_idx]],
               color=GREEN, s=150, edgecolor='white', linewidth=1.5,
               marker='*', zorder=10)
    ax.text(K[max_idx], V[max_idx], Z[max_idx] + 0.8,
            f'LONG STRADDLE\n{Z[max_idx]:.2f}',
            color=GREEN, fontsize=8, fontweight='bold', ha='center')

    style_3d(ax, '④ Gamma Scalp — Convexity Edge (HIP-3 Linear = 0)',
             'strike', 'IV level', 'Γ + |Vanna| + |Vomma|')
    ax.view_init(elev=26, azim=-40)
    return surf


def main():
    fig = plt.figure(figsize=(20, 14), facecolor=BG)

    # Title strip
    fig.text(0.5, 0.965, '4 Canonical Arbitrage Setups — HIP-3 Perp vs IBKR Options',
             ha='center', va='top', color=CYAN, fontsize=20, fontweight='bold',
             family='monospace')
    fig.text(0.5, 0.940,
             'Each 3D surface visualises one trade setup. Colored markers show entry signals; '
             'annotation boxes specify the trade construction.',
             ha='center', va='top', color=DIM, fontsize=11, family='monospace')

    # 4-panel grid: surface (left col) + annotation (right col) for each row
    gs = GridSpec(2, 2, figure=fig, left=0.04, right=0.62, top=0.91, bottom=0.06,
                  wspace=0.05, hspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')

    setup_1_vol_spread(ax1)
    setup_2_term_structure(ax2)
    setup_3_skew_arb(ax3)
    setup_4_gamma_scalp(ax4)

    # Annotation boxes (right side)
    annotation_box(fig, 0.64, 0.71, 0.34, 0.20,
        header='① VOL SPREAD ARB',
        header_color=RED,
        lines=[
            'SIGNAL: |HIP3_IV − IBKR_IV| > 5 vol pts',
            'SELL  HIP-3 perp (overpriced vol)',
            'BUY   IBKR ATM straddle (cheap vol)',
            'HEDGE: delta-neutral via perp leg',
            'EDGE: vol_spread × vega ≈ +$45/contract',
        ])
    annotation_box(fig, 0.64, 0.485, 0.34, 0.20,
        header='② CALENDAR SPREAD',
        header_color=BLUE,
        lines=[
            'SIGNAL: IBKR contango > 4 vp (back > front)',
            'BUY   IBKR back-month ATM (rich back IV)',
            'SELL  IBKR front-month ATM (cheap front)',
            'HEDGE: HIP-3 perp neutralises directional Δ',
            'EDGE: ride term-structure mean reversion',
        ])
    annotation_box(fig, 0.64, 0.26, 0.34, 0.20,
        header='③ RISK REVERSAL',
        header_color=ORANGE,
        lines=[
            'SIGNAL: 25Δ put−call IV skew < −12 vp',
            'SELL  IBKR 25Δ put (rich)',
            'BUY   IBKR 25Δ call (cheap)',
            'HEDGE: short HIP-3 perp covers downside Δ',
            'EDGE: skew normalisation on vol mean-revert',
        ])
    annotation_box(fig, 0.64, 0.035, 0.34, 0.20,
        header='④ GAMMA SCALP',
        header_color=GREEN,
        lines=[
            'SIGNAL: Γ-peak at ATM, short DTE, low IV',
            'BUY   IBKR ATM straddle (long Γ + vega)',
            'HEDGE: dynamically Δ-hedge via HIP-3 perp',
            'EDGE: realised vol > implied → scalp Γ',
            'NOTE: HIP-3 has Γ = 0 (linear payoff)',
        ])

    out_path = OUT_DIR / 'hip3_arbitrage_setups_3d.png'
    fig.savefig(out_path, dpi=130, facecolor=BG, edgecolor='none',
                bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f'Generated {out_path}: {out_path.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
