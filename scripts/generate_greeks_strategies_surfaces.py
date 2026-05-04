#!/usr/bin/env python3
"""
Generate animated 3D surfaces for all 7 Higher-Order Greek Strategies.

For each Greek (Gamma, Vanna, Charm, Vomma, Speed, Color, Zomma) the
surface visualises where the arbitrage opportunity is largest — i.e.
where IBKR's convex payoff diverges most strongly from HIP-3's linear
payoff. Fixed viewing angle (no rotation); the surfaces evolve as the
underlying IV regime cycles.

Output: docs/img/hip3_greeks_strategies_3d.gif
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

BG = '#080810'; TEXT = '#cccccc'; DIM = '#556677'
CYAN = '#00ffcc'
OUT_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 100; N_FRAMES = 50

# 7 distinct colormaps so each strategy has its own visual identity
CMAPS = {
    'gamma':  LinearSegmentedColormap.from_list('cm', ['#001833', '#0066aa', '#00ddaa', '#ccff44']),
    'vanna':  LinearSegmentedColormap.from_list('cm', ['#330033', '#8833aa', '#dd66cc', '#ffcceb']),
    'charm':  LinearSegmentedColormap.from_list('cm', ['#332200', '#aa6600', '#ffaa22', '#ffee88']),
    'vomma':  LinearSegmentedColormap.from_list('cm', ['#003322', '#006688', '#22ccaa', '#aaffee']),
    'speed':  LinearSegmentedColormap.from_list('cm', ['#330011', '#aa3333', '#ff6655', '#ffccaa']),
    'color':  LinearSegmentedColormap.from_list('cm', ['#001144', '#3344aa', '#6688ee', '#aaccff']),
    'zomma':  LinearSegmentedColormap.from_list('cm', ['#222200', '#888822', '#ddcc44', '#ffff99']),
}


def _img(fig, w=2200, h=1300):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    return Image.open(buf).convert('RGBA').resize((w, h), Image.LANCZOS)


def _gif(frames, path, dur=180):
    rgb = []
    for f in frames:
        bg = Image.new('RGB', f.size, (8, 8, 16))
        bg.paste(f, mask=f.split()[3])
        rgb.append(bg)
    rgb[0].save(path, save_all=True, append_images=rgb[1:],
                duration=dur, loop=0, optimize=True)
    print(f'  Saved {path.name} ({len(rgb)} fr, {path.stat().st_size/1024:.0f}KB)')


def _style(ax):
    ax.set_facecolor(BG)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.set_edgecolor('#1a1a2e')
    ax.tick_params(colors=DIM, labelsize=4)
    ax.grid(False)


def _draw_greek_panel(ax, X, Y, Z, cmap, title, formula,
                      xl='Spot/Strike', yl='IV', zl=''):
    _style(ax)
    Z_abs = np.abs(Z)
    zn = (Z_abs - Z_abs.min()) / (Z_abs.max() - Z_abs.min() + 1e-12)
    ax.plot_surface(X, Y, Z_abs, facecolors=cmap(zn), alpha=0.95,
                    rstride=1, cstride=1, edgecolor='none', shade=True)

    # Highlight the top-decile arbitrage zone with a contour on the floor
    peak = np.percentile(Z_abs, 90)
    z_floor = -(Z_abs.max() - Z_abs.min()) * 0.12
    try:
        ax.contour(X, Y, Z_abs, levels=[peak],
                   offset=z_floor, colors=[CYAN], linewidths=1.8, alpha=0.95)
    except Exception:
        pass

    ax.set_xlabel(xl, color=DIM, fontsize=5, labelpad=1)
    ax.set_ylabel(yl, color=DIM, fontsize=5, labelpad=1)
    ax.set_zlabel(zl, color=DIM, fontsize=5, labelpad=1)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=2)
    ax.text2D(0.5, -0.04, formula, transform=ax.transAxes,
              color=CYAN, fontsize=7, ha='center',
              fontfamily='monospace', fontweight='bold')


def _compute_surfaces(spot_range, vol_range, T):
    """Compute full Greek surfaces over (spot/strike) × IV grid."""
    n_v = len(vol_range); n_s = len(spot_range)
    surfaces = {k: np.zeros((n_v, n_s)) for k in
                ['gamma', 'vanna', 'charm', 'vomma',
                 'speed', 'color', 'zomma']}
    for i, sigma in enumerate(vol_range):
        for j, s_ratio in enumerate(spot_range):
            g = bs_greeks(100 * s_ratio, 100, T, 0.05,
                          max(sigma, 0.05), 'call')
            surfaces['gamma'][i, j] = g.gamma * 100
            surfaces['vanna'][i, j] = g.vanna
            surfaces['charm'][i, j] = g.charm
            surfaces['vomma'][i, j] = g.vomma
            surfaces['speed'][i, j] = g.speed * 100
            surfaces['color'][i, j] = g.color
            surfaces['zomma'][i, j] = g.zomma
    return surfaces


def generate():
    print('=' * 70)
    print('  Generating 7 Greek Strategy 3D Surfaces (animated)')
    print('=' * 70)

    n = 35
    sr = np.linspace(0.80, 1.20, n)
    SG, VG = np.meshgrid(sr, np.linspace(0.15, 1.0, n))

    # Pre-compute surfaces at multiple maturities so animation cycles
    # through near-dated → far-dated time horizons (charm/color/speed
    # depend on T strongly).
    print('  Pre-computing Greek surfaces at 4 maturities...')
    maturity_grid = [10, 30, 60, 120]
    surfaces_by_T = {}
    for d in maturity_grid:
        surfaces_by_T[d] = _compute_surfaces(sr, np.linspace(0.15, 1.0, n),
                                             d / 365)
    print('  Surfaces ready.')

    # Strategy metadata: title + formula description shown beneath each panel
    strategies = [
        ('gamma', '1. Gamma Scalping',  'Γ = ∂²V/∂S²   ·   long straddle + Δ-hedge perp'),
        ('vanna', '2. Vanna Trade',     '∂δ/∂σ   ·   long OTM put + long perp'),
        ('charm', '3. Charm Trade',     '∂δ/∂t   ·   short near-expiry + Δ-hedge perp'),
        ('vomma', '4. Vomma Trade',     '∂ν/∂σ   ·   long OTM options + short vol perp'),
        ('speed', '5. Speed Trade',     '∂Γ/∂S   ·   butterfly + Δ-hedge perp'),
        ('color', '6. Color Trade',     '∂Γ/∂t   ·   short near-expiry straddle'),
        ('zomma', '7. Zomma Trade',     '∂Γ/∂σ   ·   long strangle + Δ-hedge perp'),
    ]

    frames = []
    for fi in range(N_FRAMES):
        t = fi / N_FRAMES
        phase = t * 2 * np.pi

        # Cycle DTE smoothly between maturities — interpolate two adjacent
        # pre-computed surfaces for cheap animation
        T_cycle = 65 + 55 * np.sin(phase)  # ranges ~10 → 120 days
        # Find bracket
        T_low = max([m for m in maturity_grid if m <= T_cycle], default=10)
        T_high = min([m for m in maturity_grid if m >= T_cycle], default=120)
        if T_high == T_low:
            alpha = 0.0
        else:
            alpha = (T_cycle - T_low) / (T_high - T_low)
        regime_mult = 0.7 + 0.3 * (0.5 + 0.5 * np.sin(phase * 1.4))

        # 4×2 grid: 7 strategy panels + 1 info/legend cell
        fig = plt.figure(figsize=(22, 13), facecolor=BG)
        gs = GridSpec(2, 4, wspace=0.10, hspace=0.32,
                      left=0.02, right=0.98, top=0.91, bottom=0.04)

        positions = [(0, 0), (0, 1), (0, 2), (0, 3),
                     (1, 0), (1, 1), (1, 2)]

        for (key, title, formula), pos in zip(strategies, positions):
            r, c = pos
            Z = (surfaces_by_T[T_low][key] * (1 - alpha)
                 + surfaces_by_T[T_high][key] * alpha) * regime_mult
            ax = fig.add_subplot(gs[r, c], projection='3d')
            _draw_greek_panel(ax, SG, VG, Z, CMAPS[key], title, formula)
            ax.view_init(elev=24, azim=235)

        # Info panel in the empty 8th cell
        ax_info = fig.add_subplot(gs[1, 3])
        ax_info.set_facecolor(BG)
        ax_info.axis('off')
        info_lines = [
            'HIP-3 perp payoff:  LINEAR  →  no Greeks priced',
            'IBKR option payoff: CONVEX  →  full Greek term structure',
            '',
            f'Current DTE:  {T_cycle:5.0f} days',
            f'Regime mult:  {regime_mult:5.2f}',
            '',
            'Cyan contour on each panel = top-decile',
            '    arbitrage zone (largest convexity gap).',
            '',
            'Trade direction: long IBKR convexity,',
            '    delta-hedge with HIP-3 perp.',
        ]
        for i, line in enumerate(info_lines):
            ax_info.text(0.05, 0.92 - i * 0.07, line,
                         transform=ax_info.transAxes,
                         color=CYAN if line.startswith('Trade') else TEXT,
                         fontsize=9, fontfamily='monospace',
                         fontweight='bold' if i in (0, 1, 9) else 'normal')

        fig.suptitle(
            f'7 Higher-Order Greek Strategies   |   '
            f'DTE: {T_cycle:3.0f}d   |   Frame {fi + 1:02d}/{N_FRAMES}',
            color=CYAN, fontsize=14, fontweight='bold', y=0.97,
            fontfamily='monospace')

        frames.append(_img(fig))
        plt.close(fig)
        if (fi + 1) % 10 == 0:
            print(f'    {fi + 1}/{N_FRAMES}')

    _gif(frames, OUT_DIR / 'hip3_greeks_strategies_3d.gif', dur=180)
    print(f'\nDone! Output: {OUT_DIR / "hip3_greeks_strategies_3d.gif"}')


if __name__ == '__main__':
    generate()
