#!/usr/bin/env python3
"""
Generate animated 3D surfaces showing the arbitrage opportunity for each
of the 4 core strategies: Vol Spread, Term Structure, Skew, and Higher-
Order Greeks.

Fixed viewing angles (no rotation) — only the underlying surface changes
to show how arbitrage opportunities evolve across vol regimes. Threshold
contours and BUY/SELL zone annotations highlight when each opportunity
is most exploitable.

Output: docs/img/hip3_arbitrage_strategies_3d.gif
"""
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa
import io, sys
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ibkr_options_client import bs_greeks

BG = '#080810'; BG2 = '#0e0e1a'; TEXT = '#cccccc'; DIM = '#556677'
CYAN = '#00ffcc'; WHITE = '#ffffff'; HOT = '#ff3322'; COOL = '#22aaff'
OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 100; N_FRAMES = 60

# Diverging colormap (blue=BUY HIP-3, red=SELL HIP-3)
CMAP_DIV = LinearSegmentedColormap.from_list('div', [
    '#0066ff', '#3399ff', '#1a1a2e', '#ff6633', '#ff2200'])
CMAP_GREEK = LinearSegmentedColormap.from_list('gr', [
    '#001833', '#003d66', '#00aa66', '#ccff44', '#ffff66'])


def _img(fig, w=1800, h=1400):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    return Image.open(buf).convert('RGBA').resize((w, h), Image.LANCZOS)


def _gif(frames, path, dur=200):
    rgb = []
    for f in frames:
        bg = Image.new('RGB', f.size, (8, 8, 16))
        bg.paste(f, mask=f.split()[3])
        rgb.append(bg)
    rgb[0].save(path, save_all=True, append_images=rgb[1:],
                duration=dur, loop=0, optimize=True)
    print(f'  Saved {path.name} ({len(rgb)} fr, {path.stat().st_size/1024:.0f}KB)')


def _style_3d(ax):
    ax.set_facecolor(BG)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.set_edgecolor('#1a1a2e')
    ax.tick_params(colors=DIM, labelsize=5)
    ax.grid(False)


def _draw_diverging_surface(ax, X, Y, Z, threshold, xl, yl, zl, title,
                            buy_label='BUY VOL HIP-3',
                            sell_label='SELL VOL HIP-3'):
    """Draw a 3D surface with diverging colormap where the arbitrage
    opportunity is signaled by |Z| > threshold."""
    _style_3d(ax)

    vmax = max(abs(Z.min()), abs(Z.max()), threshold * 1.5)
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    colors = CMAP_DIV(norm(Z))

    mask = np.abs(Z) > threshold
    colors[..., 3] = np.where(mask, 1.0, 0.55)

    ax.plot_surface(X, Y, Z, facecolors=colors,
                    rstride=1, cstride=1, edgecolor='none',
                    shade=False, antialiased=True)

    z_floor = Z.min() - (Z.max() - Z.min()) * 0.15
    try:
        ax.contour(X, Y, Z, levels=[-threshold, 0, threshold],
                   offset=z_floor, colors=[COOL, '#444444', HOT],
                   linewidths=1.5, alpha=0.85)
    except Exception:
        pass

    ax.plot_surface(X, Y, np.zeros_like(Z), color='#222233',
                    alpha=0.12, rstride=12, cstride=12,
                    edgecolor='none', shade=False)

    ax.set_xlabel(xl, color=DIM, fontsize=7, labelpad=2)
    ax.set_ylabel(yl, color=DIM, fontsize=7, labelpad=2)
    ax.set_zlabel(zl, color=DIM, fontsize=6, labelpad=2)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=6)

    pct = mask.mean() * 100
    if Z.max() > threshold and Z.min() < -threshold:
        sig = f'{sell_label} ▲   ARB ZONE: {pct:4.0f}%   ▼ {buy_label}'
        sig_col = CYAN
    elif Z.max() > threshold:
        sig = f'{sell_label} ▲   ARB ZONE: {pct:4.0f}%'
        sig_col = HOT
    elif Z.min() < -threshold:
        sig = f'ARB ZONE: {pct:4.0f}%   ▼ {buy_label}'
        sig_col = COOL
    else:
        sig = f'no arb signal  (|spread| < {threshold:.0f} vol pts)'
        sig_col = DIM
    ax.text2D(0.5, -0.06, sig, transform=ax.transAxes,
              color=sig_col, fontsize=8, fontweight='bold',
              ha='center', fontfamily='monospace')


def _draw_greeks_surface(ax, X, Y, Z, xl, yl, zl, title):
    _style_3d(ax)
    zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-10)
    ax.plot_surface(X, Y, Z, facecolors=CMAP_GREEK(zn), alpha=0.95,
                    rstride=1, cstride=1, edgecolor='none', shade=True)
    peak_thr = np.percentile(Z, 90)
    peak_mask = Z > peak_thr
    z_floor = Z.min() - (Z.max() - Z.min()) * 0.15
    try:
        ax.contour(X, Y, Z, levels=[peak_thr],
                   offset=z_floor, colors=[CYAN], linewidths=2, alpha=0.9)
    except Exception:
        pass
    ax.set_xlabel(xl, color=DIM, fontsize=7, labelpad=2)
    ax.set_ylabel(yl, color=DIM, fontsize=7, labelpad=2)
    ax.set_zlabel(zl, color=DIM, fontsize=6, labelpad=2)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=6)
    pct = peak_mask.mean() * 100
    ax.text2D(0.5, -0.06,
              f'TRADE PEAK GREEKS ▲   top-decile zone: {pct:4.0f}%',
              transform=ax.transAxes, color=CYAN, fontsize=8,
              fontweight='bold', ha='center', fontfamily='monospace')


def generate():
    print('=' * 70)
    print('  Generating 4 Arbitrage Strategy 3D Surfaces (fixed view)')
    print('=' * 70)

    n = 50
    moneyness = np.linspace(0.75, 1.25, n)
    dte = np.linspace(7, 180, n)
    M, D = np.meshgrid(moneyness, dte)

    n_g = 40
    sr = np.linspace(0.80, 1.20, n_g)
    vr = np.linspace(0.15, 1.2, n_g)
    SG, VG = np.meshgrid(sr, vr)
    print('  Pre-computing Greeks surface...')
    Z_gamma = np.zeros((n_g, n_g))
    Z_vanna = np.zeros((n_g, n_g))
    Z_vomma = np.zeros((n_g, n_g))
    for i in range(n_g):
        for j in range(n_g):
            g = bs_greeks(100 * sr[j], 100, 30/365, 0.05,
                          max(vr[i], 0.05), 'call')
            Z_gamma[i, j] = g.gamma * 100
            Z_vanna[i, j] = g.vanna
            Z_vomma[i, j] = g.vomma
    Z_greeks_static = Z_gamma + np.abs(Z_vanna) * 2 + np.abs(Z_vomma) * 0.5
    Z_greeks_static = np.clip(Z_greeks_static, 0,
                              np.percentile(Z_greeks_static, 98))
    print('  Greeks surface ready.')

    THR_SPREAD = 5.0
    THR_TERM = 4.0
    THR_SKEW = 4.0

    # Fixed view angle (no rotation)
    ELEV = 22
    AZIM = 235

    frames = []
    for fi in range(N_FRAMES):
        t = fi / N_FRAMES
        phase = t * 2 * np.pi

        # Cycle through low → mid → high → mid → low IV regime so the
        # arbitrage opportunity grows and shrinks visibly without any
        # camera rotation.
        base_iv = 0.30 + 0.30 * np.sin(phase)
        regime_strength = 0.5 + 0.5 * np.sin(phase * 1.5)

        # Panel 1: Vol Spread
        ibkr_iv = base_iv + (-0.25) * (M - 1) + 0.35 * (M - 1)**2 + 0.0008 * D
        hip3_iv = base_iv * (1.10 + 0.12 * np.sin(phase)) + 0.0002 * D
        hip3_iv += (-0.04) * (M - 1) + 0.05 * (M - 1)**2
        Z_spread = (hip3_iv - ibkr_iv) * 100
        Z_spread *= 1 + 0.6 * regime_strength

        # Panel 2: Term Structure
        ibkr_term = base_iv * (0.85 + 0.003 * D - 0.000008 * D**2)
        ibkr_term += 0.04 * np.sin(phase * 0.7) * np.sqrt(D / 30)
        hip3_term = np.full_like(D, base_iv * (1.06 + 0.04 * np.sin(phase)))
        Z_term_diff = (hip3_term - ibkr_term) * 100
        Z_term_diff *= 1 + 0.4 * regime_strength

        # Panel 3: Skew
        skew_strength = 0.30 + 0.22 * np.sin(phase * 1.2)
        ibkr_skew = base_iv + (-skew_strength) * (M - 1)
        ibkr_skew += 0.50 * (M - 1)**2 + 0.0005 * D
        hip3_skew = base_iv * 1.06 + (-0.03) * (M - 1) + 0.03 * (M - 1)**2
        Z_skew_diff = (hip3_skew - ibkr_skew) * 100

        # Panel 4: Greeks (modulate amplitude only)
        Z_greeks = Z_greeks_static * (0.7 + 0.3 * regime_strength)

        fig = plt.figure(figsize=(18, 14), facecolor=BG)
        gs = GridSpec(2, 2, wspace=0.08, hspace=0.28,
                      left=0.03, right=0.97, top=0.92, bottom=0.05)

        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        _draw_diverging_surface(
            ax1, M, D, Z_spread, THR_SPREAD,
            'Moneyness (K/S)', 'DTE (days)', 'Spread (vol pts)',
            '1. Vol Spread  (HIP3 − IBKR)',
            buy_label='BUY VOL HIP-3', sell_label='SELL VOL HIP-3')
        ax1.view_init(elev=ELEV, azim=AZIM)

        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        _draw_diverging_surface(
            ax2, M, D, Z_term_diff, THR_TERM,
            'Moneyness (K/S)', 'DTE (days)', 'IV Diff (vol pts)',
            '2. Term Structure  (calendar spread)',
            buy_label='BUY BACK / SELL FRONT',
            sell_label='SELL BACK / BUY FRONT')
        ax2.view_init(elev=ELEV, azim=AZIM)

        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        _draw_diverging_surface(
            ax3, M, D, Z_skew_diff, THR_SKEW,
            'Moneyness (K/S)', 'DTE (days)', 'Skew Diff (vol pts)',
            '3. Skew Arbitrage  (risk reversal)',
            buy_label='BUY OTM PUT (HIP-3)',
            sell_label='SELL OTM PUT (HIP-3)')
        ax3.view_init(elev=ELEV, azim=AZIM)

        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        _draw_greeks_surface(
            ax4, SG, VG, Z_greeks,
            'Spot / Strike', 'Implied Vol', 'Greeks P&L',
            '4. Higher-Order Greeks  (Γ + |Vanna| + |Vomma|)')
        ax4.view_init(elev=ELEV, azim=AZIM)

        regime = ('HIGH-VOL' if base_iv > 0.45
                  else 'LOW-VOL' if base_iv < 0.25
                  else 'NORMAL')
        fig.suptitle(
            f'HIP-3 vs IBKR — 4 Arbitrage Strategies   |   '
            f'Base IV: {base_iv*100:3.0f}%   |   '
            f'Regime: {regime}',
            color=CYAN, fontsize=13, fontweight='bold', y=0.97,
            fontfamily='monospace')

        frames.append(_img(fig))
        plt.close(fig)
        if (fi + 1) % 10 == 0:
            print(f'    {fi + 1}/{N_FRAMES}')

    _gif(frames, OUT_DIR / 'hip3_arbitrage_strategies_3d.gif', dur=180)
    print(f'\nDone! Output: {OUT_DIR / "hip3_arbitrage_strategies_3d.gif"}')


if __name__ == '__main__':
    generate()
