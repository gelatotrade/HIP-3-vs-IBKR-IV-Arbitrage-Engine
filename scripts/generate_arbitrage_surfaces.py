#!/usr/bin/env python3
"""
Generate 4 animated 3D surfaces showing the arbitrage opportunity
for each of the 4 core strategies: Vol Spread, Term Structure, Skew,
and Higher-Order Greeks.

Output: docs/img/hip3_arbitrage_strategies_3d.gif (4 panels, 60 frames)
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
CYAN = '#00ffcc'; WHITE = '#ffffff'
OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 100; N_FRAMES = 60

CMAP_SPREAD = LinearSegmentedColormap.from_list('sp', [
    '#0044cc', '#2266dd', '#444444', '#dd4422', '#ff6600'])
CMAP_TERM = LinearSegmentedColormap.from_list('tm', [
    '#001155', '#003399', '#2266aa', '#44bbcc', '#aaffee'])
CMAP_SKEW = LinearSegmentedColormap.from_list('sk', [
    '#440066', '#8833aa', '#bb6600', '#ffaa00', '#ffffbb'])
CMAP_GREEK = LinearSegmentedColormap.from_list('gr', [
    '#002244', '#006688', '#00cc88', '#88ff44', '#ffff66'])


def _img(fig, w=1600, h=900):
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


def generate():
    print('=' * 70)
    print('  Generating 4 Arbitrage Strategy 3D Surfaces (animated)')
    print('=' * 70)

    n = 40
    moneyness = np.linspace(0.75, 1.25, n)
    dte = np.linspace(7, 180, n)
    M, D = np.meshgrid(moneyness, dte)

    # Pre-compute Greeks surface (spot/strike × vol)
    n_g = 35
    sr = np.linspace(0.80, 1.20, n_g)
    vr = np.linspace(0.15, 1.2, n_g)
    SG, VG = np.meshgrid(sr, vr)
    Z_gamma = np.zeros((n_g, n_g))
    Z_vanna = np.zeros((n_g, n_g))
    Z_vomma = np.zeros((n_g, n_g))
    for i in range(n_g):
        for j in range(n_g):
            g = bs_greeks(100 * sr[j], 100, 30/365, 0.05, max(vr[i], 0.05), 'call')
            Z_gamma[i, j] = g.gamma * 100
            Z_vanna[i, j] = g.vanna
            Z_vomma[i, j] = g.vomma
    Z_greeks = Z_gamma + np.abs(Z_vanna) * 2 + np.abs(Z_vomma) * 0.5
    Z_greeks = np.clip(Z_greeks, 0, np.percentile(Z_greeks, 98))
    print('  Greeks surface pre-computed.')

    frames = []
    for fi in range(N_FRAMES):
        t = fi / N_FRAMES
        phase = t * 2 * np.pi
        azim = 220 + fi * 3

        base_iv = 0.35 + 0.25 * np.sin(phase)

        # --- Panel 1: Vol Spread (HIP3 IV - IBKR IV) ---
        ibkr_iv = base_iv + (-0.25) * (M - 1) + 0.35 * (M - 1)**2 + 0.0008 * D
        hip3_iv = base_iv * (1.12 + 0.08 * np.sin(phase * 1.3)) + 0.0002 * D
        hip3_iv += (-0.04) * (M - 1) + 0.05 * (M - 1)**2
        Z_spread = (hip3_iv - ibkr_iv) * 100

        # --- Panel 2: Term Structure (IBKR curved vs HIP-3 flat) ---
        ibkr_term = base_iv * (0.85 + 0.003 * D - 0.000008 * D**2)
        ibkr_term += 0.03 * np.sin(phase * 0.7) * np.sqrt(D / 30)
        hip3_term = np.full_like(D, base_iv * 1.05)
        hip3_term += 0.02 * np.sin(phase * 1.1)
        Z_term_diff = (hip3_term - ibkr_term) * 100

        # --- Panel 3: Skew (IBKR negative skew vs HIP-3 flat) ---
        ibkr_skew = base_iv + (-0.30 - 0.10 * np.sin(phase)) * (M - 1)
        ibkr_skew += 0.50 * (M - 1)**2 + 0.0005 * D
        hip3_skew = base_iv * 1.08 + (-0.03) * (M - 1) + 0.03 * (M - 1)**2
        Z_skew_diff = (hip3_skew - ibkr_skew) * 100

        # --- Build 4-panel figure ---
        fig = plt.figure(figsize=(18, 14), facecolor=BG)
        gs = GridSpec(2, 2, wspace=0.05, hspace=0.18,
                      left=0.02, right=0.98, top=0.92, bottom=0.03)

        panels = [
            (gs[0, 0], M, D, Z_spread,
             '1. Vol Spread (HIP3 − IBKR)',
             'Moneyness (K/S)', 'DTE (days)', 'Spread (vol pts)',
             CMAP_SPREAD),
            (gs[0, 1], M, D, Z_term_diff,
             '2. Term Structure Arb',
             'Moneyness (K/S)', 'DTE (days)', 'IV Diff (vol pts)',
             CMAP_TERM),
            (gs[1, 0], M, D, Z_skew_diff,
             '3. Skew Arbitrage',
             'Moneyness (K/S)', 'DTE (days)', 'Skew Diff (vol pts)',
             CMAP_SKEW),
            (gs[1, 1], SG, VG, Z_greeks,
             '4. Higher-Order Greeks (Γ + |Vanna| + |Vomma|)',
             'Spot / Strike', 'Implied Vol', 'Greeks P&L',
             CMAP_GREEK),
        ]

        for spec, X, Y, Z, title, xl, yl, zl, cmap in panels:
            ax = fig.add_subplot(spec, projection='3d')
            _style_3d(ax)
            zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-10)
            ax.plot_surface(X, Y, Z, facecolors=cmap(zn), alpha=0.92,
                            rstride=2, cstride=2, edgecolor='none', shade=True)
            ax.set_xlabel(xl, color=DIM, fontsize=6, labelpad=1)
            ax.set_ylabel(yl, color=DIM, fontsize=6, labelpad=1)
            ax.set_zlabel(zl, color=DIM, fontsize=5, labelpad=1)
            ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=4)
            if spec == gs[1, 1]:
                ax.view_init(elev=28, azim=azim + 30)
            else:
                ax.view_init(elev=25, azim=azim)

        fig.suptitle(
            f'HIP-3 vs IBKR — 4 Arbitrage Strategies | Base IV: {base_iv*100:.0f}%',
            color=CYAN, fontsize=13, fontweight='bold', y=0.97,
            fontfamily='monospace')

        frames.append(_img(fig, 1800, 1400))
        plt.close(fig)
        if (fi + 1) % 10 == 0:
            print(f'    {fi + 1}/{N_FRAMES}')

    _gif(frames, OUT_DIR / 'hip3_arbitrage_strategies_3d.gif', dur=200)
    print(f'\nDone! Output: {OUT_DIR / "hip3_arbitrage_strategies_3d.gif"}')


if __name__ == '__main__':
    generate()
