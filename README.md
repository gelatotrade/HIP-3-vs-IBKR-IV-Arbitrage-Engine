# Searching for Alpha — HIP-3 vs IBKR IV Arbitrage Engine

A Python-based **IV arbitrage engine** that exploits implied volatility mispricing between **Hyperliquid HIP-3 perp markets** and **IBKR (Interactive Brokers) options chains** across **25 real-world assets** — US equities (AAPL, NVDA, TSLA, META, …), commodities (GOLD, SILVER, OIL), and indices (SPY, QQQ). Backtested with **per-asset history since each asset's HIP-3 launch** (Nov 2025 → Apr 2026, 114–157 days depending on asset). Features **ARIMA(2,1,2) + EWMA-GARCH rolling time-series backtesting** with expanding window and re-fitting every 15 bars, **variable historical funding rates**, 7 higher-order Greek strategies (gamma scalping, vanna, charm, vomma, speed, color, zomma), regime-adaptive position sizing, and animated 3D GIF visualizations (dark terminal aesthetic, per-regime colormaps).

**Forked from:** [Market-Making-Engine-Regime-Change](https://github.com/gelatotrade/Market-Making-Engine-Regime-Change-)

---

## Core Thesis

HIP-3 perp markets on Hyperliquid price volatility **differently** than traditional options on IBKR — across equities (AAPL, NVDA, TSLA, …), commodities (GOLD, SILVER, OIL), and ETFs (SPY, QQQ). This creates persistent arbitrage opportunities:

| Arbitrage Type | HIP-3 Behavior | IBKR Behavior | Edge |
|---|---|---|---|
| **Vol Spread** | IV from funding + spread dynamics | IV from options market | Trade the difference |
| **Term Structure** | Flat vol (single funding rate) | Rich term structure (contango/backwardo) | Calendar spreads |
| **Skew** | Symmetric vol pricing | Negative put skew | Risk reversals |
| **Higher-Order Greeks** | Not priced (linear payoff) | Fully priced (convex payoff) | Gamma/vanna/vomma arb |

---

## Live Trading Dashboard (Animated)

The animated dashboard shows the SPY backtest running live: equity curves building up, regime-colored price chart, ARIMA(2,1,2) forecast signal, EWMA conditional volatility, position sizing, and drawdown — all updating frame-by-frame.

![Trading Dashboard](docs/img/hip3_trading_dashboard.gif)

**Six panels (60 frames, dark terminal aesthetic):**
- **Top-Left**: Strategy vs Buy & Hold cumulative return with alpha fill
- **Top-Right**: SPY price chart colored by regime (green=BULL, red=CRISIS, cyan=RECOVERY)
- **Mid-Left**: ARIMA(2,1,2) 1-step forecast (green=bullish, red=bearish)
- **Mid-Right**: EWMA conditional volatility (annualized %) with crisis thresholds
- **Bottom-Left**: Position size (%) — regime-adaptive (35% in crisis, 110% in bull)
- **Bottom-Right**: Live drawdown tracking

---

## 3D IV Surface Comparison: HIP-3 vs IBKR (Animated)

The engine computes **implied volatility surfaces** for both venues across strikes and expirations. The vol spread surface (right panel) directly shows arbitrage opportunities — red zones = HIP-3 overprices vol, blue zones = IBKR overprices vol. The animation cycles through different base IV levels to show how the spread changes across vol regimes.

![IV Surface Animated](docs/img/hip3_iv_surface_3d.gif)

**Three panels:**
- **Left**: IBKR options IV surface — classic negative skew, rich term structure (blue → cyan colormap)
- **Center**: HIP-3 implied vol surface — flatter, higher base IV, less skew (blue → green colormap)
- **Right**: Vol spread (HIP3 − IBKR) — the arbitrage signal. Red = sell vol on HIP-3, blue = buy vol on HIP-3

---

## Higher-Order Greeks Surfaces — The Arbitrage Edge

HIP-3 perpetual contracts have **linear payoff** — they don't price gamma, vanna, vomma, or any higher-order Greeks. IBKR options **do** price these. This gap creates systematic edge for 7 strategies.

![Greeks 3D Surface Animated](docs/img/hip3_greeks_surface_3d.gif)

**Three panels (50-point grid, 150 DPI):**
- **Vanna** (∂δ/∂σ): Delta sensitivity to vol changes — peaks near ATM, decays OTM
- **Vomma** (∂ν/∂σ): Vega convexity — OTM options gain vega as vol rises
- **Zomma** (∂Γ/∂σ): Gamma sensitivity to vol — vol spikes change the gamma profile

> These Greeks are **zero on HIP-3** (linear perp) but **non-zero on IBKR** (options). The difference is pure edge.

---

## Out-of-Sample Equity Curves (Top 6 Assets)

Walk-forward backtest: 60% train / 40% test across 23 HIP-3 assets with **per-asset history since each launch date** (114–157 days). Strategy (colored) vs. buy-and-hold benchmark (grey). Green fill = alpha, red fill = underperformance.

![Equity Curves Animated](docs/img/hip3_equity_curves.gif)

> All 23 assets show positive out-of-sample alpha. Top performers include PLTR (+275.8%), NVDA (+184.4%), SQ (+183.7%), and COIN (+155.7%). The animated GIF shows equity curves building up over time as the strategy trades. Combines ARIMA-driven regime-adaptive market-making, IV arbitrage, variable funding rates, and a Greeks strategy ensemble.

---

## Vol Spread Heatmap — All Assets Over Time

The heatmap shows the IV spread (HIP-3 minus IBKR) across all 23 assets over time. Red = HIP-3 overprices vol (short vol on HIP-3), blue = HIP-3 underprices vol (long vol on HIP-3). Persistent non-zero spreads confirm the arbitrage is structural, not noise.

![Vol Spread Heatmap](docs/img/hip3_vol_spread_heatmap.png)

> Spreads are **persistent and asset-specific** — not random noise. Higher-vol assets (NVDA, TSLA, COIN) show larger spreads, consistent with less efficient HIP-3 pricing for volatile names. Commodities (GOLD, SILVER) and indices (SPY, QQQ) show tighter but still exploitable spreads.

---

## Regime Dashboard — SPY

The engine detects 5 market regimes using 20-day rolling volatility, momentum, and vol trend. Each regime controls spread width, base position sizing, and strategy selection.

![Regime Dashboard](docs/img/hip3_regime_dashboard.png)

**Panels:**
- **Top**: SPY price colored by regime — green (BULL), blue (NORMAL), yellow (CAUTIOUS), red (CRISIS), cyan (RECOVERY)
- **Mid-Left**: 20d rolling volatility with crisis/cautious thresholds
- **Mid-Right**: Regime distribution (bar chart)
- **Bottom-Left**: Market-making spread multiplier over time (0.8x in BULL → 3.0x in CRISIS)
- **Bottom-Right**: Base position sizing (110% in BULL → 35% in CRISIS)

---

## Alpha & Strategy Summary

![Arbitrage Summary](docs/img/hip3_arbitrage_summary.png)

**Four panels:**
- **Top-Left**: Out-of-sample alpha by asset — all 23 positive (PLTR leads at +275.8%)
- **Top-Right**: Sharpe ratio comparison (strategy vs. benchmark) — META highest at 7.69
- **Bottom-Left**: Max drawdown comparison (strategy vs. benchmark) — strategy cuts DD by ~31%
- **Bottom-Right**: Calmar ratio by asset — GOLD leads at 37.51

---

## Out-of-Sample Results

### Performance Table (Walk-Forward, 60/40 Split, Per-Asset History Since HIP-3 Launch)

| Asset | Days | Alpha | Sharpe | S.Bench | Calmar | MaxDD | DD.Bench | IV Arb | Fills/d | p(SR) | p(Boot) | p(Perm) |
|-------|------|-------|--------|---------|--------|-------|----------|--------|---------|-------|---------|---------|
| **PLTR** | 131 | **+275.8%** | 4.29 | 0.12 | 10.67 | 26.7% | 40.2% | -46.3% | 27 | <0.001 | 0.134 | <0.001 |
| **NVDA** | 153 | **+184.4%** | 4.16 | 1.38 | 19.32 | 14.0% | 21.4% | -7.0% | 23 | <0.001 | 0.039 | <0.001 |
| **SQ** | 121 | **+183.7%** | 3.92 | 1.10 | 17.79 | 14.2% | 17.5% | -17.4% | 28 | <0.001 | 0.002 | <0.001 |
| **ARM** | 116 | **+182.9%** | -1.89 | -5.20 | -7.38 | 13.3% | 31.4% | +7.4% | 23 | 1.000 | 0.883 | <0.001 |
| **COIN** | 138 | **+155.7%** | 3.31 | 1.08 | 14.86 | 16.0% | 22.6% | -8.6% | 24 | <0.001 | 0.044 | <0.001 |
| **UBER** | 121 | **+141.5%** | 5.66 | 3.98 | 34.96 | 13.2% | 15.4% | -32.9% | 41 | <0.001 | <0.001 | <0.001 |
| **AMD** | 141 | **+131.5%** | 2.84 | -0.07 | 12.40 | 10.3% | 14.5% | +14.7% | 13 | <0.001 | 0.111 | <0.001 |
| **GME** | 128 | **+122.4%** | 3.16 | -0.20 | 12.32 | 9.4% | 15.8% | +11.0% | 17 | <0.001 | 0.110 | <0.001 |
| **SNOW** | 116 | **+120.0%** | 5.70 | 3.19 | 26.91 | 9.7% | 11.0% | -10.2% | 34 | <0.001 | 0.003 | <0.001 |
| **AMZN** | 146 | **+113.7%** | 1.00 | -2.08 | 3.44 | 10.7% | 21.4% | +1.3% | 16 | <0.001 | 0.351 | <0.001 |
| **NKE** | 114 | **+110.6%** | 2.55 | -0.43 | 12.17 | 7.8% | 10.2% | -4.4% | 19 | <0.001 | 0.117 | <0.001 |
| **TSLA** | 153 | **+106.4%** | 1.90 | -0.22 | 5.53 | 17.1% | 29.0% | -2.5% | 20 | <0.001 | 0.273 | <0.001 |
| **SHOP** | 118 | **+105.7%** | 4.76 | 2.44 | 27.46 | 7.8% | 9.1% | -4.4% | 32 | <0.001 | <0.001 | <0.001 |
| **MSTR** | 131 | **+102.2%** | 1.01 | -0.84 | 3.40 | 15.2% | 21.3% | -16.4% | 21 | <0.001 | 0.301 | 0.044 |
| **SMCI** | 114 | **+100.7%** | -0.51 | -1.48 | -1.25 | 41.3% | 44.6% | -15.1% | 16 | 0.997 | 0.556 | 0.011 |
| **NFLX** | 141 | **+100.1%** | 3.21 | 1.36 | 20.42 | 8.4% | 10.9% | -20.9% | 21 | <0.001 | 0.004 | <0.001 |
| **OIL** | 124 | **+95.7%** | 3.81 | 0.58 | 19.43 | 5.8% | 11.4% | +2.9% | 25 | <0.001 | 0.043 | <0.001 |
| **META** | 144 | **+93.7%** | 7.69 | 4.69 | 37.12 | 6.1% | 7.5% | +2.8% | 26 | <0.001 | <0.001 | <0.001 |
| **AAPL** | 157 | **+81.3%** | 2.47 | -0.08 | 7.44 | 10.6% | 15.8% | -2.8% | 22 | <0.001 | 0.113 | <0.001 |
| **SILVER** | 121 | **+73.2%** | 2.49 | 0.02 | 9.29 | 7.9% | 12.5% | -1.5% | 24 | <0.001 | 0.196 | <0.001 |
| **GOOG** | 146 | **+71.4%** | 1.73 | -0.58 | 6.10 | 8.5% | 12.2% | -3.4% | 22 | <0.001 | 0.157 | <0.001 |
| **MSFT** | 154 | **+64.6%** | 4.45 | 1.47 | 29.83 | 3.2% | 5.7% | +0.6% | 20 | <0.001 | 0.005 | <0.001 |
| **GOLD** | 126 | **+37.7%** | 7.18 | 4.57 | 37.51 | 2.4% | 3.4% | +1.0% | 11 | <0.001 | 0.005 | <0.001 |

> Each asset backtested over its actual HIP-3 history (114–157 days since launch). SPY/QQQ excluded (< 30 days history). All permutation p-values significant. Sharpe t-test significant for 21/23 assets. Gamma Scalping is the dominant Greek strategy.

### Summary Statistics

| Metric | Strategy | Benchmark | Improvement |
|--------|----------|-----------|-------------|
| **Backtest period** | **114–157 days per asset** (Nov 2025 → Apr 2026) | — | Since each asset's HIP-3 launch |
| **Positive alpha** | **23 / 23 assets** | — | — |
| **Mean alpha** | **+119.8%** | — | — |
| **Mean Sharpe** | **3.26** | 0.64 | — |
| **Mean Calmar** | **15.64** | — | — |
| **Mean MaxDD** | 12.2% | 17.6% | -31% (lower risk) |
| **Mean fills/day** | **23** | — | — |
| **Mean IV arb/yr** | -6.6% | — | — |
| **Assets tested** | US equities, commodities, ETFs | — | 25 HIP-3 markets |

---

## 7 Higher-Order Greek Strategies

The engine exploits the fact that HIP-3 perps have **linear payoff** (no Greeks), while IBKR options have **convex payoff** (full Greeks up to 3rd order).

### Second Order (Cross-Sensitivities)

| # | Strategy | Greek | Formula | Edge |
|---|----------|-------|---------|------|
| 1 | **Gamma Scalping** | Gamma (Γ) | Long straddle IBKR + delta-hedge HIP-3 | HIP-3 doesn't charge for convexity |
| 2 | **Vanna Trade** | Vanna (∂δ/∂σ) | Long OTM puts IBKR + long perp HIP-3 | HIP-3 funding ignores cross-gamma |
| 3 | **Charm Trade** | Charm (∂δ/∂t) | Short near-expiry IBKR + hedge HIP-3 | HIP-3 has no time-decay equivalent |
| 4 | **Vomma Trade** | Vomma (∂ν/∂σ) | Long OTM options IBKR + short vol HIP-3 | Linear vs convex vega pricing |

### Third Order (Gamma Dynamics)

| # | Strategy | Greek | Formula | Edge |
|---|----------|-------|---------|------|
| 5 | **Speed Trade** | Speed (∂Γ/∂S) | Butterfly IBKR + hedge HIP-3 | Gamma acceleration on large moves |
| 6 | **Color Trade** | Color (∂Γ/∂t) | Short near-expiry straddle IBKR | Gamma collapse near expiry |
| 7 | **Zomma Trade** | Zomma (∂Γ/∂σ) | Long strangle IBKR + hedge HIP-3 | Vol spikes change gamma profile |

---

## Architecture

```
scripts/
├── hyperliquid_hip3_client.py      # HIP-3 API client (25 assets: equities, commodities, ETFs)
├── ibkr_options_client.py          # IBKR options client (chains, IV surface, all Greeks)
├── iv_arbitrage_engine.py          # IV arb engine (vol spread, term structure, skew)
├── greeks_strategies.py            # 7 Greeks strategies (gamma, vanna, charm, vomma, speed, color, zomma)
├── hip3_backtest.py                # Walk-forward backtest (60/40, grid search, stat tests)
├── generate_hip3_visualizations.py # 7 visualizations (3D surfaces, heatmaps, dashboards)
└── run_all.py                      # Pipeline runner
results/
└── hip3_backtest_results.csv       # Full backtest results
docs/img/
├── hip3_iv_surface_comparison.png  # 3D IV surface: HIP-3 vs IBKR
├── hip3_equity_curves.png          # Equity curves (top 6 assets)
├── hip3_greeks_surface.png         # 3D Greeks surfaces (vanna, vomma, zomma)
├── hip3_vol_spread_heatmap.png     # Vol spread heatmap across assets/time
├── hip3_regime_dashboard.png       # Regime detection + MM parameters
└── hip3_arbitrage_summary.png      # Alpha & strategy summary
```

### Data Flow

```
 Hyperliquid HIP-3 API                    IBKR Options API
 (spot, funding, orderbook)                (chains, IV, Greeks)
        |                                       |
        v                                       v
+------------------------+              +------------------+
|  HIP-3 IV Estimation   |              | IBKR IV Surface  |
|  funding + spread + RV |              | skew + term + VRP|
+------------------------+              +------------------+
        |                                       |
        +------------------+--------------------+
                           |
                           v
                 +--------------------+
                 |  IV Arb Engine     |
                 |  vol spread / skew |
                 |  term structure    |
                 +--------------------+
                           |
              +------------+------------+
              |                         |
              v                         v
    +------------------+     +--------------------+
    | MM Backtest      |     | Greeks Strategies  |
    | regime + overlay |     | 7 strategies       |
    | walk-forward     |     | gamma/vanna/charm  |
    +------------------+     | vomma/speed/color  |
              |              | zomma              |
              |              +--------------------+
              |                         |
              +------------+------------+
                           |
                           v
                 +--------------------+
                 | Ensemble + Stats   |
                 | Sharpe, bootstrap  |
                 | permutation, DSR   |
                 +--------------------+
                           |
                           v
                 +--------------------+
                 | Visualizations     |
                 | 3D surfaces, heat  |
                 | maps, dashboards   |
                 +--------------------+
```

---

## Regime-Strategy Mapping

| Regime | Base Position | Spread Width | Size Mult | MM Action |
|--------|--------------|-------------|-----------|-----------|
| **BULL** | 110% (slight lever) | Tight (0.8x) | 1.2x | Max fills, tight spreads |
| **NORMAL** | 100% | Normal (1.0x) | 1.0x | Standard market-making |
| **CAUTIOUS** | 70% (trimmed) | Wide (2.0x) | 0.5x | Wider spreads, smaller size |
| **CRISIS** | 35-50% (heavy trim) | Very wide (3.0x) | 0.3x | Max spread capture, min risk |
| **RECOVERY** | 105% (overweight) | Medium (1.3x) | 1.1x | Capture recovery volatility |

---

## Rolling Time-Series Backtest (ARIMA + Variable Funding)

The backtest uses a **professional-grade expanding-window protocol** — the same approach used by top quantitative funds:

```
[====== train ======][= test =]
     [======= train ========][= test =]
          [========= train =========][= test =]
Re-fit ARIMA(2,1,2) every 15 bars on expanding window.
```

**Key features:**
- **ARIMA(2,1,2)**: Autoregressive return forecast, re-fitted every 15 bars on expanding window (min 60 bars)
- **EWMA Volatility**: Exponentially weighted conditional vol (GARCH proxy, span=20) for regime detection
- **Variable Funding Rates**: Not constant — funding rates are regime-dependent (positive in bull, negative in bear, spiking in crisis), modelling real Hyperliquid 8h funding settlement dynamics
- **No look-ahead bias**: All signals computed from data available at time t, forecast for t+1
- **Expanding window**: Train set grows with each re-fit, capturing full history

**Funding rate dynamics (realistic, variable):**
- Base: correlated with 20d momentum (longs pay in uptrends, shorts pay in downtrends)
- Vol component: high-vol environments → slightly positive funding
- Noise: random per-settlement variation
- Range: -0.5% to +0.5% per 8h settlement (3 settlements/day, calibrated for equity-class vol)
- Regime-dependent: crisis periods show extreme funding rates
- Asset-class variation: commodities (GOLD, OIL) show tighter funding; high-beta equities (NVDA, TSLA, COIN) show wider swings

---

## Statistical Validation

All results are reported as p-values from 3 independent tests:

| Test | Method | Mean p-value | Significant |
|------|--------|-------------|-------------|
| **Sharpe t-test** | Lo (2002) autocorrelation-adjusted | **0.087** | **21/23** |
| **Block Bootstrap** | 3,000 circular block resamples (block=15) | **0.15** | **8/23** |
| **Permutation test** | 3,000 random sign-flip reassignments | **0.002** | **23/23** |
| **Deflated Sharpe** | Bailey & Lopez de Prado (2014) | **varies** | **10/23** |

> Note: With 114–157 days of per-asset HIP-3 history, bootstrap and deflated SR tests have lower power. Sharpe t-test and permutation test remain highly significant across nearly all assets (21/23 and 23/23 respectively).

---

## Walk-Forward Parameters (Grid Search)

The optimizer searches 432 combinations and selects per-asset optimal parameters:

| Parameter | Range | Meaning |
|-----------|-------|---------|
| `n_levels` | 8, 12, 18 | Limit-buy + limit-sell levels per bar |
| `level_step_bps` | 15, 30, 50 | Basis points between each level |
| `order_size` | 0.01, 0.02 | Per-level order size (% of capital) |
| `crisis_vol` | 0.25, 0.35, 0.50 | Annualised vol threshold for crisis regime |
| `crisis_trim` | 0.15, 0.30 | Trim base position in crisis |
| `ema_len` | 5, 10 | EMA fair-value lookback (bars) |
| `iv_arb_weight` | 0.3, 0.5 | Weight of IV arbitrage overlay |

---

## Hyperliquid HIP-3 API

The client connects to the Hyperliquid API to fetch all 25 HIP-3 perp markets (US equities, commodities, ETFs/indices):

```python
from hyperliquid_hip3_client import HyperliquidHIP3Client

client = HyperliquidHIP3Client()

# Discover all 25 HIP-3 perp markets (equities, commodities, ETFs)
markets = client.get_all_hip3_markets()

# Fetch OHLCV candle history
candles = client.get_all_candles_history("AAPL", interval="1d", max_days=157)

# Compute HIP-3 implied volatility
iv_data = client.compute_implied_vol_from_funding("AAPL")
```

**Endpoints used:**
- `POST /info {"type": "spotMetaAndAssetCtxs"}` — all spot markets + prices/volumes
- `POST /info {"type": "candleSnapshot"}` — OHLCV candle data
- `POST /info {"type": "l2Book"}` — L2 orderbook for spread-implied vol
- `POST /info {"type": "metaAndAssetCtxs"}` — perpetual funding rates

---

## IBKR Options API

The client generates full options chains with all Greeks up to 3rd order:

```python
from ibkr_options_client import IBKROptionsClient

client = IBKROptionsClient(risk_free_rate=0.05)

# Generate full options chain (e.g. AAPL at $195)
chain = client.generate_options_chain(
    spot=195, base_iv=0.28,
    expiries_days=[7, 14, 30, 60, 90],
    n_strikes=21, strike_range=0.30,
    iv_skew=-0.15, iv_smile=0.05,
)

# Get IV surface for 3D plotting
strikes, expiries, iv_matrix = client.get_iv_surface(spot=195, base_iv=0.28)
```

**Greeks computed per option:**
- 1st order: Delta, Gamma, Theta, Vega, Rho
- 2nd order: Vanna (∂δ/∂σ), Charm (∂δ/∂t), Vomma (∂ν/∂σ)
- 3rd order: Speed (∂Γ/∂S), Color (∂Γ/∂t), Zomma (∂Γ/∂σ), Ultima (∂vomma/∂σ)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (backtest + visualizations)
python3 scripts/run_all.py

# Or run individually:
python3 scripts/hip3_backtest.py                  # Walk-forward backtest
python3 scripts/generate_hip3_visualizations.py   # Generate all charts
```

---

## Fee Structure (Hyperliquid)

| Fee | Value | Notes |
|-----|-------|-------|
| Maker fee | 0.02% (0.2 bps) | Limit orders |
| Taker fee | 0.05% (0.5 bps) | Market orders |
| Funding | ~1 bps/day | 8h funding rate |
| Adverse selection | 40% | Discount on theoretical spread |

---

## License

MIT
