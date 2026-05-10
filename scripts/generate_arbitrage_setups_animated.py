#!/usr/bin/env python3
"""
Generate four separate animated GIFs — one per canonical HIP-3 perp vs IBKR
options arbitrage setup. Each GIF shows a single 3D surface evolving as the
underlying regime cycles, with the trade-construction annotation panel
attached on the right.

Outputs:
  docs/img/hip3_arbitrage_setup1_vol_spread.gif
  docs/img/hip3_arbitrage_setup2_calendar_spread.gif
  docs/img/hip3_arbitrage_setup3_risk_reversal.gif
  docs/img/hip3_arbitrage_setup4_gamma_scalp.gif
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

N_FRAMES = 24
DPI = 80
GIF_W, GIF_H = 960, 640


def style_3d(ax, title, xlabel, ylabel, zlabel):
    ax.set_facecolor(PANEL_BG)
    ax.xaxis.pane.set_facecolor(PANEL_BG)
    ax.yaxis.pane.set_facecolor(PANEL_BG)
    ax.zaxis.pane.set_facecolor(PANEL_BG)
    ax.xaxis.pane.set_edgecolor(DIM)
    ax.yaxis.pane.set_edgecolor(DIM)
    ax.zaxis.pane.set_edgecolor(DIM)
    ax.tick_params(axis='x', colors=DIM, labelsize=8)
    ax.tick_params(axis='y', colors=DIM, labelsize=8)
    ax.tick_params(axis='z', colors=DIM, labelsize=8)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=9, labelpad=4)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9, labelpad=4)
    ax.set_zlabel(zlabel, color=TEXT, fontsize=9, labelpad=4)
    ax.set_title(title, color=CYAN, fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.15)


def annotation_box(fig, x, y, w, h, lines, header, header_color, status_line):
    """Add a trade-construction annotation box at figure-relative coords."""
    ax = fig.add_axes([x, y, w, h])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#101020')
    for spine in ax.spines.values():
        spine.set_edgecolor(header_color)
        spine.set_linewidth(1.5)
    ax.text(0.5, 0.94, header, transform=ax.transAxes, ha='center', va='top',
            color=header_color, fontsize=12, fontweight='bold',
            family='monospace')
    y0 = 0.83
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
        ax.text(0.05, y0, line, transform=ax.transAxes,
                ha='left', va='top', color=color, fontsize=10,
                family='monospace', fontweight=weight)
        y0 -= 0.085
    # Live status line at bottom
    ax.text(0.5, 0.08, status_line, transform=ax.transAxes,
            ha='center', va='bottom', color=CYAN, fontsize=9,
            family='monospace', fontweight='bold',
            bbox=dict(facecolor='#00332a', edgecolor=CYAN,
                      boxstyle='round,pad=0.3'))


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf).convert('RGBA').resize((GIF_W, GIF_H), Image.LANCZOS)
    plt.close(fig)
    return img


def save_gif(frames, path, duration=130):
    rgb = [f.convert('P', palette=Image.ADAPTIVE, colors=96) for f in frames]
    rgb[0].save(path, save_all=True, append_images=rgb[1:],
                duration=duration, loop=0, optimize=True, disposal=2)
    print(f'Saved {path.name}: {path.stat().st_size:,} bytes ({len(frames)} frames)')


# ── Setup 1: Vol Spread (HIP-3 IV regime cycles low → high → low) ──
def gen_setup1_vol_spread():
    frames = []
    moneyness = np.linspace(-0.20, 0.20, 30)
    dte = np.linspace(7, 90, 22)
    M, D = np.meshgrid(moneyness, dte)
    ibkr_iv = 0.22 + (-0.30) * M + 0.08 * M**2 + 0.0008 * D

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        # HIP-3 IV cycles 0.20 → 0.34 → 0.20
        hip3_base = 0.27 + 0.07 * np.sin(2 * np.pi * t)
        hip3_iv = hip3_base + 0.005 * np.sin(D / 12) + 0.001 * M
        spread = (hip3_iv - ibkr_iv) * 100

        fig = plt.figure(figsize=(13, 7), facecolor=BG)
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(M, D, spread, cmap=CMAP_DIV, edgecolor='none',
                        alpha=0.92, vmin=-12, vmax=12, antialiased=True)
        ax.set_zlim(-15, 15)

        # Threshold contours
        if spread.max() > 5:
            ax.contour(M, D, spread, levels=[5], colors=[RED], linewidths=2,
                       offset=-15)
        if spread.min() < -3:
            ax.contour(M, D, spread, levels=[-3], colors=[BLUE], linewidths=2,
                       offset=-15)

        # Mark hot spots
        if spread.max() > 1:
            sell_idx = np.unravel_index(np.argmax(spread), spread.shape)
            ax.scatter([M[sell_idx]], [D[sell_idx]], [spread[sell_idx]],
                       color=RED, s=140, edgecolor='white', linewidth=1.8,
                       zorder=10)
            ax.text(M[sell_idx], D[sell_idx], spread[sell_idx] + 1.8,
                    f'SELL\n{spread[sell_idx]:.1f}vp',
                    color=RED, fontsize=9, fontweight='bold', ha='center')
        if spread.min() < -1:
            buy_idx = np.unravel_index(np.argmin(spread), spread.shape)
            ax.scatter([M[buy_idx]], [D[buy_idx]], [spread[buy_idx]],
                       color=BLUE, s=140, edgecolor='white', linewidth=1.8,
                       zorder=10)
            ax.text(M[buy_idx], D[buy_idx], spread[buy_idx] - 1.8,
                    f'BUY\n{spread[buy_idx]:.1f}vp',
                    color=BLUE, fontsize=9, fontweight='bold', ha='center')

        style_3d(ax, '① Vol Spread Arb — HIP-3 IV vs IBKR IV',
                 'log-moneyness', 'DTE (days)', 'Spread (vol pts)')
        ax.view_init(elev=24, azim=-58)
        ax.set_position([0.02, 0.05, 0.55, 0.90])

        arb_zone = (np.abs(spread) > 5).mean() * 100
        annotation_box(fig, 0.60, 0.10, 0.38, 0.80,
            header='① VOL SPREAD ARB',
            header_color=RED,
            lines=[
                'SIGNAL: |HIP3_IV − IBKR_IV| > 5 vp',
                'SELL  HIP-3 perp (overpriced vol)',
                'BUY   IBKR ATM straddle',
                'HEDGE: delta-neutral via perp leg',
                'EDGE: spread × vega ≈ +$45/contract',
                '',
                'WHY: HIP-3 IV is funding-implied,',
                'IBKR IV is options-implied. Gap >',
                '5vp persists → fade it.',
            ],
            status_line=f'HIP3 IV: {hip3_base*100:.1f}%  |  ARB ZONE: {arb_zone:.0f}%')
        fig.text(0.5, 0.95,
                 f'Frame {i+1}/{N_FRAMES} — HIP-3 IV regime cycling',
                 ha='center', color=DIM, fontsize=10, family='monospace')
        frames.append(fig_to_pil(fig))

    save_gif(frames, OUT_DIR / 'hip3_arbitrage_setup1_vol_spread.gif')


# ── Setup 2: Calendar Spread (term-structure slope cycles) ──
def gen_setup2_calendar_spread():
    frames = []
    moneyness = np.linspace(-0.15, 0.15, 28)
    dte_back = np.linspace(60, 180, 22)
    M, D = np.meshgrid(moneyness, dte_back)

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        # Term slope: 0.0002 → 0.0014 → 0.0002 (flat → steep contango → flat)
        slope = 0.0008 + 0.0006 * np.sin(2 * np.pi * t)
        ibkr_back_iv = 0.22 + slope * D - 0.30 * M + 0.08 * M**2
        ibkr_front_iv = 0.22 + slope * 14 - 0.30 * M + 0.08 * M**2
        term_diff = (ibkr_back_iv - ibkr_front_iv) * 100

        fig = plt.figure(figsize=(13, 7), facecolor=BG)
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(M, D, term_diff, cmap=CMAP_TERM, edgecolor='none',
                        alpha=0.92, vmin=0, vmax=25, antialiased=True)
        ax.set_zlim(-2, 25)

        # Reference plane at HIP-3 (flat = 0)
        zz = np.zeros_like(term_diff)
        ax.plot_surface(M, D, zz, color=ORANGE, alpha=0.18, edgecolor='none')

        if term_diff.max() > 4:
            max_idx = np.unravel_index(np.argmax(term_diff), term_diff.shape)
            ax.scatter([M[max_idx]], [D[max_idx]], [term_diff[max_idx]],
                       color=GOLD, s=180, edgecolor='white', linewidth=1.8,
                       marker='^', zorder=10)
            ax.text(M[max_idx], D[max_idx], term_diff[max_idx] + 1.5,
                    f'CALENDAR\n+{term_diff[max_idx]:.1f}vp',
                    color=GOLD, fontsize=9, fontweight='bold', ha='center')

        ax.text(0.0, D.mean(), 0.5, 'HIP-3 flat (0)',
                color=ORANGE, fontsize=8, fontweight='bold', ha='center')

        style_3d(ax, '② Calendar Spread — Term Structure',
                 'log-moneyness', 'DTE back leg', 'Back − Front IV (vp)')
        ax.view_init(elev=22, azim=-50)
        ax.set_position([0.02, 0.05, 0.55, 0.90])

        regime = 'STEEP CONTANGO' if slope > 0.0010 else \
                 ('MILD CONTANGO' if slope > 0.0006 else 'FLAT (no edge)')
        annotation_box(fig, 0.60, 0.10, 0.38, 0.80,
            header='② CALENDAR SPREAD',
            header_color=BLUE,
            lines=[
                'SIGNAL: contango > 4 vp (back > front)',
                'BUY   IBKR back-month ATM',
                'SELL  IBKR front-month ATM',
                'HEDGE: HIP-3 perp neutralises Δ',
                'EDGE: term-structure mean reversion',
                '',
                'WHY: HIP-3 has ONE funding rate (flat',
                'term). IBKR options price each expiry',
                'separately → trade the slope.',
            ],
            status_line=f'TERM SLOPE: {slope*1000:.2f} vp/d  |  REGIME: {regime}')
        fig.text(0.5, 0.95,
                 f'Frame {i+1}/{N_FRAMES} — Contango slope cycling',
                 ha='center', color=DIM, fontsize=10, family='monospace')
        frames.append(fig_to_pil(fig))

    save_gif(frames, OUT_DIR / 'hip3_arbitrage_setup2_calendar_spread.gif')


# ── Setup 3: Risk Reversal (skew cycles with vol regime) ──
def gen_setup3_risk_reversal():
    frames = []
    spot_pct = np.linspace(-0.10, 0.10, 28)
    iv_regime = np.linspace(0.15, 0.50, 22)
    S, V = np.meshgrid(spot_pct, iv_regime)

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        # Skew intensity cycles: -0.10 → -0.30 → -0.10 (mild → crisis → mild)
        skew_mult = 0.6 + 0.8 * (1 - np.cos(2 * np.pi * t)) / 2
        skew_iv = (-0.18 - 0.45 * V) * skew_mult * (1 + 0.5 * S * np.sign(-S)) * 100
        skew_iv = skew_iv - 8 * skew_mult * np.maximum(0, -S) * V

        fig = plt.figure(figsize=(13, 7), facecolor=BG)
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(S, V, skew_iv, cmap=CMAP_SKEW, edgecolor='none',
                        alpha=0.92, vmin=-50, vmax=-5, antialiased=True)
        ax.set_zlim(-55, 0)

        threshold = np.percentile(skew_iv, 10)
        if threshold < -10:
            ax.contour(S, V, skew_iv, levels=[threshold], colors=[CYAN],
                       linewidths=2.5, offset=-55)

        rr_idx = np.unravel_index(np.argmin(skew_iv), skew_iv.shape)
        ax.scatter([S[rr_idx]], [V[rr_idx]], [skew_iv[rr_idx]],
                   color=ORANGE, s=160, edgecolor='white', linewidth=1.8,
                   marker='D', zorder=10)
        ax.text(S[rr_idx], V[rr_idx], skew_iv[rr_idx] - 4,
                f'RISK REV\n{skew_iv[rr_idx]:.1f}vp',
                color=ORANGE, fontsize=9, fontweight='bold', ha='center')

        style_3d(ax, '③ Skew Arb — IBKR Put Skew vs HIP-3 Symmetric',
                 'spot move (%)', 'IV regime', '25Δ Put − Call IV (vp)')
        ax.view_init(elev=24, azim=42)
        ax.set_position([0.02, 0.05, 0.55, 0.90])

        regime = 'CRISIS SKEW' if skew_mult > 1.2 else \
                 ('NORMAL SKEW' if skew_mult > 0.85 else 'CALM (low skew)')
        annotation_box(fig, 0.60, 0.10, 0.38, 0.80,
            header='③ RISK REVERSAL',
            header_color=ORANGE,
            lines=[
                'SIGNAL: 25Δ put−call skew < −12 vp',
                'SELL  IBKR 25Δ put (rich)',
                'BUY   IBKR 25Δ call (cheap)',
                'HEDGE: short HIP-3 perp (Δ-cover)',
                'EDGE: skew normalises on mean-revert',
                '',
                'WHY: HIP-3 prices puts and calls',
                'symmetrically. IBKR has crash-fear',
                'skew → fade it when extreme.',
            ],
            status_line=f'SKEW INTENSITY: {skew_mult:.2f}x  |  {regime}')
        fig.text(0.5, 0.95,
                 f'Frame {i+1}/{N_FRAMES} — Skew intensity cycling',
                 ha='center', color=DIM, fontsize=10, family='monospace')
        frames.append(fig_to_pil(fig))

    save_gif(frames, OUT_DIR / 'hip3_arbitrage_setup3_risk_reversal.gif')


# ── Setup 4: Gamma Scalp (DTE cycles 90 → 7 → 90) ──
def gen_setup4_gamma_scalp():
    frames = []
    spot = 100.0
    strikes = np.linspace(80, 120, 28)
    ivs = np.linspace(0.12, 0.50, 22)
    K, V = np.meshgrid(strikes, ivs)

    for i in range(N_FRAMES):
        t = i / (N_FRAMES - 1)
        # DTE cycles 90 → 7 → 90 days
        dte = 48.5 - 41.5 * np.cos(2 * np.pi * t)
        dte = max(7, dte)

        Z = np.zeros_like(K)
        for ii in range(K.shape[0]):
            for jj in range(K.shape[1]):
                q = bs_greeks(spot, K[ii, jj], dte / 365.0, 0.05, V[ii, jj], 'call')
                Z[ii, jj] = q.gamma * 100 + abs(q.vanna) * 0.1 + abs(q.vomma) * 0.01

        fig = plt.figure(figsize=(13, 7), facecolor=BG)
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(K, V, Z, cmap=CMAP_GREEK, edgecolor='none',
                        alpha=0.92, antialiased=True)
        ax.set_zlim(0, 12)

        threshold = np.percentile(Z, 90)
        ax.contour(K, V, Z, levels=[threshold], colors=[CYAN],
                   linewidths=2.5, offset=0)

        max_idx = np.unravel_index(np.argmax(Z), Z.shape)
        ax.scatter([K[max_idx]], [V[max_idx]], [Z[max_idx]],
                   color=GREEN, s=180, edgecolor='white', linewidth=1.8,
                   marker='*', zorder=10)
        ax.text(K[max_idx], V[max_idx], Z[max_idx] + 0.8,
                f'LONG STRADDLE\n{Z[max_idx]:.2f}',
                color=GREEN, fontsize=9, fontweight='bold', ha='center')

        style_3d(ax, '④ Gamma Scalp — Convexity Edge (HIP-3 Γ = 0)',
                 'strike', 'IV level', 'Γ + |Vanna| + |Vomma|')
        ax.view_init(elev=26, azim=-40)
        ax.set_position([0.02, 0.05, 0.55, 0.90])

        regime = 'GAMMA EXPLOSION' if dte < 15 else \
                 ('GROWING Γ' if dte < 45 else 'DORMANT Γ')
        annotation_box(fig, 0.60, 0.10, 0.38, 0.80,
            header='④ GAMMA SCALP',
            header_color=GREEN,
            lines=[
                'SIGNAL: Γ-peak ATM, short DTE, low IV',
                'BUY   IBKR ATM straddle (long Γ+vega)',
                'HEDGE: dynamically Δ-hedge HIP-3 perp',
                'EDGE: realised vol > implied → scalp Γ',
                'NOTE: HIP-3 has Γ = 0 (linear payoff)',
                '',
                'WHY: Convexity is free on HIP-3.',
                'IBKR options accumulate Γ near expiry',
                '→ buy them and hedge linearly.',
            ],
            status_line=f'DTE: {dte:.0f}d  |  PEAK Γ-SCORE: {Z.max():.2f}  |  {regime}')
        fig.text(0.5, 0.95,
                 f'Frame {i+1}/{N_FRAMES} — DTE cycling 90 → 7 → 90 days',
                 ha='center', color=DIM, fontsize=10, family='monospace')
        frames.append(fig_to_pil(fig))

    save_gif(frames, OUT_DIR / 'hip3_arbitrage_setup4_gamma_scalp.gif')


def main():
    print('Generating 4 separate animated arbitrage-setup GIFs...')
    gen_setup1_vol_spread()
    gen_setup2_calendar_spread()
    gen_setup3_risk_reversal()
    gen_setup4_gamma_scalp()
    print('Done.')


if __name__ == '__main__':
    main()
