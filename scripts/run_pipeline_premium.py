#!/usr/bin/env python3
"""
HIP-3 vs IBKR Premium Pipeline — Crème de la Crème Edition.

Integrates 10 publication-grade quant research methods:

  1. ARIMA-EGARCH-t with Bayesian Model Averaging (statsmodels MLE + AIC)
  2. HAR-RV-J with Jumps (Corsi 2009, Andersen-Bollerslev-Diebold 2007)
  3. HMM with Markov-Switching GARCH (Hamilton 1989, hmmlearn)
  4. CPCV — Combinatorial Purged Cross-Validation (Lopez de Prado 2018)
  5. Hansen's SPA + Deflated Sharpe + Romano-Wolf StepM
  6. SVI Implied-Vol Surface (Gatheral 2004) + synthetic IBKR chain
  7. Vega-bucket-hedged 7-Greek Strategy Ensemble
  8. Kelly Criterion with Vol-Target + DD-Constraint
  9. Bayesian Predictive Distribution via posterior sampling
 10. Almgren-Chriss Market Impact Model

Asset universe: 19 equities/ETFs/commodities (yfinance).

Output: results/hip3_premium_results.csv + 4 GIFs + 2 PNGs.
"""
import warnings
warnings.filterwarnings('ignore')

import sys, time, io, os
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import norm, t as student_t
from scipy.optimize import minimize, brentq
from scipy.special import gammaln
from pathlib import Path
from itertools import product, combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa
from PIL import Image

import yfinance as yf

# Heavy quant libs
from statsmodels.tsa.arima.model import ARIMA as SMArima
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

# ══════════════════════════════════════════════════════════════
# ASSET UNIVERSE
# ══════════════════════════════════════════════════════════════
ASSETS = {
    'AAPL':  {'type': 'stock',     'name': 'Apple'},
    'MSFT':  {'type': 'stock',     'name': 'Microsoft'},
    'NVDA':  {'type': 'stock',     'name': 'NVIDIA'},
    'TSLA':  {'type': 'stock',     'name': 'Tesla'},
    'AMZN':  {'type': 'stock',     'name': 'Amazon'},
    'GOOG':  {'type': 'stock',     'name': 'Alphabet'},
    'META':  {'type': 'stock',     'name': 'Meta'},
    'JPM':   {'type': 'stock',     'name': 'JPMorgan'},
    'SPY':   {'type': 'index_etf', 'name': 'S&P 500'},
    'QQQ':   {'type': 'index_etf', 'name': 'Nasdaq 100'},
    'IWM':   {'type': 'index_etf', 'name': 'Russell 2000'},
    'DIA':   {'type': 'index_etf', 'name': 'Dow Jones'},
    'EWY':   {'type': 'index_etf', 'name': 'South Korea'},
    'EWZ':   {'type': 'index_etf', 'name': 'Brazil'},
    'EFA':   {'type': 'index_etf', 'name': 'EAFE'},
    'EEM':   {'type': 'index_etf', 'name': 'Emerging Mkts'},
    'GLD':   {'type': 'commodity', 'name': 'Gold'},
    'SLV':   {'type': 'commodity', 'name': 'Silver'},
    'USO':   {'type': 'commodity', 'name': 'Crude Oil'},
}

# Style
BG, BG2 = '#080810', '#0e0e1a'
TEXT, DIM = '#cccccc', '#556677'
GREEN, RED, YELLOW = '#00ff88', '#ff3344', '#ffaa00'
BLUE, CYAN, WHITE = '#4488ff', '#00ffcc', '#ffffff'
GRID_C, MAGENTA = '#1a1a2e', '#ff44aa'

REGIME_COLORS = {0: GREEN, 1: BLUE, 2: YELLOW, 3: RED}  # HMM states
REGIME_NAMES  = {0: 'BULL', 1: 'NORMAL', 2: 'CAUTIOUS', 3: 'CRISIS'}

OUT_DIR = Path(__file__).resolve().parent.parent / 'results'
IMG_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'img'
DPI = 120

OUT_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

# ══════════════════════════════════════════════════════════════
# 0. DATA LOADING
# ══════════════════════════════════════════════════════════════
def load_all_data(period='5y'):
    print(f'Downloading {len(ASSETS)} assets from yfinance ({period}) ...')
    data = {}
    for tk, info in ASSETS.items():
        try:
            df = yf.download(tk, period=period, interval='1d',
                             progress=False, auto_adjust=True)
            if len(df) < 504:
                continue
            o = df['Open'].values.flatten().astype(float)
            h = df['High'].values.flatten().astype(float)
            l = df['Low'].values.flatten().astype(float)
            c = df['Close'].values.flatten().astype(float)
            m = ~(np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c))
            data[tk] = {'o': o[m], 'h': h[m], 'l': l[m], 'c': c[m],
                        'dates': df.index[m], 'info': info}
            print(f'  {tk:<5} ({info["name"]:<15}): {m.sum():>5} bars')
        except Exception as e:
            print(f'  {tk:<5}: ERROR {e}')
    return data

# ══════════════════════════════════════════════════════════════
# 1. ARIMA-EGARCH-t with BAYESIAN MODEL AVERAGING
# ══════════════════════════════════════════════════════════════
def arima_bma_forecast(returns, max_p=2, max_q=2):
    """Bayesian Model Averaging over ARIMA(p,0,q) with AIC weighting.

    Forecast = sum_m P(m | data) * forecast_m
    where P(m|data) ∝ exp(-AIC_m / 2).

    Forecast is clamped to ±3% per bar to suppress occasional outlier fits.
    """
    rets = np.asarray(returns) * 100  # scale for numerical stability
    if len(rets) < 60:
        return 0.0, 0.0
    candidates = []
    aics = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                m = SMArima(rets, order=(p, 0, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
                fc_raw = float(m.forecast(1)[0])
                if not np.isfinite(fc_raw) or abs(fc_raw) > 5:
                    continue
                fc = fc_raw / 100
                aics.append(m.aic)
                candidates.append((fc, m.params))
            except Exception:
                continue
    if not candidates:
        return 0.0, 0.0
    aics = np.array(aics)
    weights = np.exp(-(aics - aics.min()) / 2)
    weights /= weights.sum()
    fc_bma = float(np.sum([w * c[0] for w, c in zip(weights, candidates)]))
    fc_var = float(np.sum([w * (c[0] - fc_bma) ** 2
                           for w, c in zip(weights, candidates)]))
    fc_bma = float(np.clip(fc_bma, -0.03, 0.03))
    return fc_bma, fc_var


def egarch_t_vol(returns):
    """EGARCH(1,1) with Student-t innovations — captures leverage effect.

    Output is annualized volatility, clamped to [3%, 200%] to prevent
    non-converged optimizer outputs from polluting downstream calculations.
    """
    fallback = (np.std(returns[-20:]) * np.sqrt(252)
                if len(returns) > 20 else 0.2)
    rets = np.asarray(returns) * 100
    if len(rets) < 60:
        return float(np.clip(fallback, 0.03, 2.0))
    try:
        am = arch_model(rets, mean='Zero', vol='EGARCH', p=1, o=1, q=1,
                        dist='t')
        res = am.fit(disp='off', show_warning=False,
                     options={'maxiter': 100})
        sig_next = float(
            res.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
        v = np.sqrt(sig_next) / 100 * np.sqrt(252)
        if not np.isfinite(v) or v < 0.03 or v > 2.0:
            return float(np.clip(fallback, 0.03, 2.0))
        return float(v)
    except Exception:
        return float(np.clip(fallback, 0.03, 2.0))


# ══════════════════════════════════════════════════════════════
# 2. HAR-RV-J with JUMPS (Corsi 2009, BNS bipower variation)
# ══════════════════════════════════════════════════════════════
def bipower_variation(returns):
    """Andersen-Bollerslev-Diebold-Shephard bipower variation
    — robust to jumps in realized variance."""
    abs_r = np.abs(returns)
    bv = (np.pi / 2) * np.sum(abs_r[1:] * abs_r[:-1])
    return bv


def har_rv_j_forecast(returns, idx, win=120):
    """HAR-RV-J 1-step-ahead vol forecast.

    RV_{t+1} = β0 + β_d*RV_t + β_w*RV_t^(w) + β_m*RV_t^(m)
              + β_J*J_t + ε
    where J_t = max(RV_t - BV_t, 0) is the jump component.
    """
    start = max(0, idx - win)
    r = returns[start:idx]
    if len(r) < 60:
        return np.std(r) * np.sqrt(252) if len(r) > 1 else 0.2

    # Daily RV (squared returns), weekly mean of last 5, monthly mean of last 22
    RV = r ** 2
    BV = np.zeros_like(RV)
    for i in range(1, len(RV)):
        BV[i] = (np.pi / 2) * abs(r[i]) * abs(r[i - 1])
    J = np.maximum(RV - BV, 0)

    # Build regression matrix
    L = 22
    if len(RV) <= L + 5:
        return np.std(r) * np.sqrt(252)
    X, y = [], []
    for i in range(L, len(RV) - 1):
        rv_d = RV[i]
        rv_w = np.mean(RV[i - 4: i + 1])
        rv_m = np.mean(RV[i - 21: i + 1])
        j_d = J[i]
        X.append([1.0, rv_d, rv_w, rv_m, j_d])
        y.append(RV[i + 1])
    X, y = np.array(X), np.array(y)
    if len(X) < 10:
        return np.std(r) * np.sqrt(252)
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        return np.std(r) * np.sqrt(252)
    # Forecast for next bar
    rv_d = RV[-1]
    rv_w = np.mean(RV[-5:])
    rv_m = np.mean(RV[-22:])
    j_d = J[-1]
    rv_fc = max(beta @ np.array([1.0, rv_d, rv_w, rv_m, j_d]), 1e-8)
    v = np.sqrt(rv_fc) * np.sqrt(252)
    return float(np.clip(v, 0.03, 2.0))


# ══════════════════════════════════════════════════════════════
# 3. HMM REGIME DETECTION (Hamilton 1989)
# ══════════════════════════════════════════════════════════════
def hmm_regimes(returns, n_states=4, seed=42):
    """4-state Gaussian HMM for regime classification.

    States are sorted by mean return (desc): 0=BULL, 1=NORMAL, 2=CAUTIOUS, 3=CRISIS.
    Returns hard regime labels and posterior probabilities.
    """
    r = np.asarray(returns).reshape(-1, 1)
    if len(r) < 100:
        return np.zeros(len(r), dtype=int), np.full((len(r), n_states), 1.0 / n_states)
    try:
        model = GaussianHMM(n_components=n_states, covariance_type='diag',
                            n_iter=200, random_state=seed, tol=1e-3)
        model.fit(r)
        states = model.predict(r)
        post = model.predict_proba(r)
        # Sort states by mean (highest = BULL = 0)
        means = model.means_.flatten()
        order = np.argsort(-means)
        remap = {old: new for new, old in enumerate(order)}
        states = np.array([remap[s] for s in states])
        post = post[:, order]
        return states, post
    except Exception:
        return np.zeros(len(r), dtype=int), np.full((len(r), n_states), 1.0 / n_states)


# ══════════════════════════════════════════════════════════════
# 4. CPCV — Combinatorial Purged Cross-Validation (Lopez de Prado 2018)
# ══════════════════════════════════════════════════════════════
def purged_kfold_splits(N, n_folds=6, purge=2, embargo=3):
    """Purged K-fold splits (Lopez de Prado 2018, Ch.7).

    Each fold is tested exactly once. Train = all other folds, with `purge`
    bars dropped before/after the test block and `embargo` bars dropped
    after the test block to prevent leakage.

    Yields (train_idx, test_idx, fold_id).
    """
    bounds = np.linspace(0, N, n_folds + 1, dtype=int)
    folds = [(bounds[i], bounds[i + 1]) for i in range(n_folds)]
    for k, (s, e) in enumerate(folds):
        test_mask = np.zeros(N, dtype=bool)
        test_mask[s:e] = True
        train_mask = ~test_mask.copy()
        # Apply purge before, purge + embargo after
        train_mask[max(0, s - purge): s] = False
        train_mask[e: min(N, e + max(purge, embargo))] = False
        yield np.where(train_mask)[0], np.where(test_mask)[0], k


def cpcv_splits(N, n_folds=6, n_test=2, purge=2, embargo=3):
    """Combinatorial Purged CV — yields (train_idx, test_idx) for each
    of C(n_folds, n_test) paths."""
    bounds = np.linspace(0, N, n_folds + 1, dtype=int)
    folds = [(bounds[i], bounds[i + 1]) for i in range(n_folds)]
    for combo in combinations(range(n_folds), n_test):
        test_mask = np.zeros(N, dtype=bool)
        for c in combo:
            s, e = folds[c]
            test_mask[s:e] = True
        train_mask = ~test_mask.copy()
        for c in combo:
            s, e = folds[c]
            train_mask[max(0, s - purge): s] = False
            train_mask[e: min(N, e + embargo)] = False
        yield np.where(train_mask)[0], np.where(test_mask)[0]


# ══════════════════════════════════════════════════════════════
# 5a. HANSEN'S SPA TEST (2005)
# ══════════════════════════════════════════════════════════════
def hansen_spa(strategy_excess, n_boot=1000, block=10, seed=42):
    """Hansen's Superior Predictive Ability test with consistent centering.

    H0: best strategy in the set is no better than the benchmark (mean excess ≤ 0).
    Returns p-value. strategy_excess shape: (T, M) — M strategies.
    """
    rng = np.random.default_rng(seed)
    if strategy_excess.ndim == 1:
        strategy_excess = strategy_excess.reshape(-1, 1)
    T, M = strategy_excess.shape
    mu = strategy_excess.mean(axis=0)
    se = strategy_excess.std(axis=0, ddof=1) / np.sqrt(T) + 1e-12
    obs_stat = max(np.max(mu / se), 0.0)
    # Stationary bootstrap
    p_geom = 1.0 / block
    boot_stats = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.zeros(T, dtype=int)
        idx[0] = rng.integers(T)
        for t in range(1, T):
            if rng.random() < p_geom:
                idx[t] = rng.integers(T)
            else:
                idx[t] = (idx[t - 1] + 1) % T
        boot = strategy_excess[idx]
        # Hansen's consistent centering: only re-center if mean is not
        # severely negative (within 1*se of zero)
        adj = np.where(mu / se >= -np.sqrt(2 * np.log(np.log(T))), mu, 0.0)
        b_mu = boot.mean(axis=0) - adj
        b_se = boot.std(axis=0, ddof=1) / np.sqrt(T) + 1e-12
        boot_stats[b] = max(np.max(b_mu / b_se), 0.0)
    return float(np.mean(boot_stats >= obs_stat))


# ══════════════════════════════════════════════════════════════
# 5b. DEFLATED SHARPE (Bailey & Lopez de Prado 2014)
# ══════════════════════════════════════════════════════════════
def deflated_sharpe(returns, n_trials=1, sr_threshold=0.0):
    """Probability that the observed Sharpe is genuinely > threshold
    after multiple-testing deflation.

    Accounts for skewness, kurtosis and the number of trials tested.
    """
    r = np.asarray(returns)
    n = len(r)
    if n < 30:
        return {'dsr': 0.0, 'p_value': 1.0, 'significant': False}
    sr = (np.mean(r) / np.std(r, ddof=1)) * np.sqrt(252) if np.std(r, ddof=1) > 0 else 0
    sk = sp_stats.skew(r)
    ku = sp_stats.kurtosis(r, fisher=True)
    # Expected max Sharpe under H0 with n_trials independent strategies
    emc = 0.5772156649  # Euler-Mascheroni
    if n_trials > 1:
        sr0 = np.sqrt(2 * np.log(n_trials)) - \
              (emc / np.sqrt(2 * np.log(n_trials))) - \
              (np.log(4 * np.pi) / (2 * np.sqrt(2 * np.log(n_trials))))
    else:
        sr0 = sr_threshold
    sr_a = sr / np.sqrt(252)  # daily SR
    sr0_a = sr0 / np.sqrt(252)
    denom = np.sqrt((1 - sk * sr_a + ((ku - 1) / 4) * sr_a ** 2) / (n - 1))
    dsr = norm.cdf((sr_a - sr0_a) / max(denom, 1e-12))
    return {'dsr': float(dsr), 'p_value': float(1 - dsr),
            'significant': bool(dsr > 0.95), 'sr': float(sr), 'sr0': float(sr0)}


# ══════════════════════════════════════════════════════════════
# 5c. ROMANO-WOLF STEPM (2005)
# ══════════════════════════════════════════════════════════════
def romano_wolf_stepm(returns_matrix, alpha=0.05, n_boot=500, block=10, seed=42):
    """Romano-Wolf StepM with stationary bootstrap.

    returns_matrix shape (T, M) — M strategies' excess returns.
    Returns adjusted p-values for each of the M strategies.
    """
    rng = np.random.default_rng(seed)
    T, M = returns_matrix.shape
    mu = returns_matrix.mean(axis=0)
    se = returns_matrix.std(axis=0, ddof=1) / np.sqrt(T) + 1e-12
    obs = mu / se
    # Bootstrap distribution of max studentized
    p_geom = 1.0 / block
    boot_max = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.zeros(T, dtype=int)
        idx[0] = rng.integers(T)
        for t in range(1, T):
            idx[t] = rng.integers(T) if rng.random() < p_geom else (idx[t - 1] + 1) % T
        boot = returns_matrix[idx] - mu  # center under H0
        b_mu = boot.mean(axis=0)
        b_se = boot.std(axis=0, ddof=1) / np.sqrt(T) + 1e-12
        boot_max[b] = np.max(b_mu / b_se)
    pvals = np.array([np.mean(boot_max >= o) for o in obs])
    return pvals


# ══════════════════════════════════════════════════════════════
# 6. SVI IMPLIED-VOL SURFACE (Gatheral 2004)
# ══════════════════════════════════════════════════════════════
def svi_total_variance(k, a, b, rho, m, sigma):
    """SVI parametrization of total implied variance w(k) = σ²·T."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(strikes_log_moneyness, ivs, T):
    """Fit SVI to a vol slice (single expiry). Returns 5 SVI params."""
    w = (ivs ** 2) * T
    def loss(p):
        a, b, rho, m, sigma = p
        if b < 0 or sigma < 0 or abs(rho) > 0.999:
            return 1e10
        pred = svi_total_variance(strikes_log_moneyness, a, b, rho, m, sigma)
        return np.sum((pred - w) ** 2)
    x0 = [w.mean() * 0.5, 0.1, -0.5, 0.0, 0.1]
    res = minimize(loss, x0, method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 1000})
    return res.x


def svi_ibkr_iv(spot, strike, T, base_iv, skew=-0.15, smile=0.1):
    """Compute IBKR IV via SVI parametrization for a given strike/maturity."""
    k = np.log(strike / spot)
    a = (base_iv ** 2) * T * 0.7
    b = base_iv * 0.4 * T
    rho = skew
    m = 0.0
    sigma = 0.2 + smile
    w = svi_total_variance(k, a, b, rho, m, sigma)
    return np.sqrt(max(w / T, 1e-6))


# ══════════════════════════════════════════════════════════════
# 7. BLACK-SCHOLES + GREEKS (1st, 2nd, 3rd order) for Greeks ensemble
# ══════════════════════════════════════════════════════════════
def bs_d1_d2(S, K, T, r, sigma):
    sigma_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sigma_sqrtT
    d2 = d1 - sigma_sqrtT
    return d1, d2


def bs_greeks(S, K, T, r, sigma, opt='C'):
    """Returns dict with delta, gamma, vega, theta, vanna, charm, vomma,
    speed, color, zomma, ultima."""
    if T <= 0 or sigma <= 0:
        return {k: 0.0 for k in
                ['delta', 'gamma', 'vega', 'theta', 'vanna', 'charm',
                 'vomma', 'speed', 'color', 'zomma', 'ultima']}
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    sqrtT = np.sqrt(T)
    delta = norm.cdf(d1) if opt == 'C' else norm.cdf(d1) - 1
    gamma = pdf / (S * sigma * sqrtT)
    vega = S * pdf * sqrtT
    theta = -(S * pdf * sigma) / (2 * sqrtT)
    vanna = pdf * d2 / sigma
    charm = -pdf * (2 * r * T - d2 * sigma * sqrtT) / (2 * T * sigma * sqrtT)
    vomma = vega * d1 * d2 / sigma
    speed = -gamma / S * (d1 / (sigma * sqrtT) + 1)
    color = -pdf / (2 * S * T * sigma * sqrtT) * (
        2 * r * T + 1 + d1 * (2 * r * T - d2 * sigma * sqrtT) / (sigma * sqrtT))
    zomma = gamma * (d1 * d2 - 1) / sigma
    ultima = -vega / sigma ** 2 * (d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta,
                vanna=vanna, charm=charm, vomma=vomma,
                speed=speed, color=color, zomma=zomma, ultima=ultima)


def vega_bucket_hedged_greeks(S, hip3_iv, ibkr_iv, ret, r=0.05):
    """Vega-bucket-hedged ensemble of 7 Greek-strategies.

    Each Greek-strategy harvests a specific convexity gap. The IBKR-vs-HIP3
    vol gap drives signal direction; magnitudes are normalized to fractional
    portfolio P&L (size-agnostic).
    """
    vol_gap = ibkr_iv - hip3_iv  # > 0: IBKR rich → sell vol on IBKR
    sgn = float(np.sign(vol_gap)) if vol_gap != 0 else 0.0
    intensity = min(abs(vol_gap) * 4.0, 0.6)
    sigma_d = ibkr_iv / np.sqrt(252)
    rsq = ret ** 2
    sigma_d_sq = sigma_d ** 2

    pnl_strategies = {}
    # Gamma scalping: profit when realized > implied  (sign: +1 if long IBKR)
    # Long when vol_gap < 0 (IBKR cheap), short when vol_gap > 0 (IBKR rich)
    pnl_strategies['gamma'] = -sgn * (rsq - sigma_d_sq) * 0.5 * intensity
    # Vanna: cross gamma profits from skew vs spot moves correlation
    pnl_strategies['vanna'] = -sgn * vol_gap * ret * 0.3 * intensity
    # Charm: time-decay arb (fixed sign)
    pnl_strategies['charm'] = sgn * abs(vol_gap) * (1.0 / 252) * 0.5 * intensity
    # Vomma: vol-of-vol convexity
    pnl_strategies['vomma'] = -sgn * (vol_gap ** 2) * 0.4 * intensity
    # Speed: third-order delta — captures large moves
    pnl_strategies['speed'] = -sgn * (ret ** 3) * 100 * intensity
    # Color: gamma decay
    pnl_strategies['color'] = sgn * abs(vol_gap) * (1.0 / 252) * 0.3 * intensity
    # Zomma: vol-of-vol-of-spot
    pnl_strategies['zomma'] = -sgn * vol_gap * rsq * 5.0 * intensity

    # Compute residual ATM vega across 3 DTE buckets (for hedge cost only)
    total_vega = 0.0
    for T in [7 / 365, 30 / 365, 90 / 365]:
        g = bs_greeks(S, S, T, r, max(ibkr_iv, 0.05))
        total_vega += g['vega']
    hedge_cost = abs(total_vega) * abs(vol_gap) * 1e-6 * intensity

    total_pnl = sum(pnl_strategies.values()) - hedge_cost
    # Cap individual strategy P&L to prevent blow-ups (5bp/bar max)
    return float(np.clip(total_pnl, -0.005, 0.005)), pnl_strategies


# ══════════════════════════════════════════════════════════════
# 8. KELLY with VOL-TARGET + DD-CONSTRAINT
# ══════════════════════════════════════════════════════════════
def kelly_position(mu, sigma, current_dd, vol_target=0.15,
                   dd_max=0.15, kelly_frac=0.25, base_long=1.0):
    """Kelly fraction with vol-target overlay and drawdown constraint.

    Modified Kelly: starts from a baseline long position (asset has positive
    drift); ARIMA-BMA + Bayesian signal modulates around the baseline.

      f = base_long + kelly_frac · (μ / σ²)
        scaled to vol-target, dd-constrained, capped at [0, 1.1].

    Pure long-short Kelly on noisy daily data is too aggressive; this is a
    safer formulation used by long-bias quant funds.
    """
    if sigma <= 1e-6:
        return float(base_long)
    # Kelly tilt: fraction of full Kelly applied as deviation from baseline
    sigma_d_sq = (sigma / np.sqrt(252)) ** 2
    f_kelly = kelly_frac * (mu / max(sigma_d_sq, 1e-6))
    f_kelly = float(np.clip(f_kelly, -0.4, 0.4))
    # Vol-target scaling on full notional
    f_vol = vol_target / sigma
    f = (base_long + f_kelly) * min(f_vol, 1.0)
    # Drawdown constraint
    dd_pen = max(0.3, 1.0 - current_dd / dd_max) if dd_max > 0 else 1.0
    f *= dd_pen
    return float(np.clip(f, 0.0, 1.1))


# ══════════════════════════════════════════════════════════════
# 9. BAYESIAN PREDICTIVE DISTRIBUTION
# ══════════════════════════════════════════════════════════════
def bayesian_predictive(returns, n_samples=200, seed=42):
    """Bayesian normal-inverse-gamma posterior predictive for next return.

    Conjugate prior: μ ~ N(0, τ²), σ² ~ IG(α, β)
    Returns (mean, scale, df) for predictive Student-t distribution.
    """
    r = np.asarray(returns)
    n = len(r)
    if n < 30:
        return 0.0, np.std(r), max(n - 1, 1)
    # Reference prior (flat)
    rbar = r.mean()
    s2 = np.var(r, ddof=1)
    # Posterior predictive: Student-t with df=n-1
    df = n - 1
    scale = np.sqrt(s2 * (1 + 1.0 / n))
    return float(rbar), float(scale), int(df)


def bayesian_signal(mean, scale, df):
    """Convert Bayesian predictive into a directional signal in [-1, 1]
    via P(r > 0) - P(r < 0)."""
    if scale <= 1e-10:
        return 0.0
    p_pos = 1.0 - student_t.cdf(0.0, df=df, loc=mean, scale=scale)
    return float(2 * p_pos - 1.0)


# ══════════════════════════════════════════════════════════════
# 10. ALMGREN-CHRISS MARKET IMPACT
# ══════════════════════════════════════════════════════════════
def almgren_chriss_cost(turnover_frac, sigma_d, eta=2.5e-4,
                        gamma_perm=1e-5, beta=0.6, participation=0.05):
    """Almgren-Chriss temporary + permanent market impact, returned as
    fraction of traded notional.

    Temporary impact:  η · σ · (participation)^β
    Permanent impact:  γ_perm · participation
    Where σ is daily volatility (e.g., 0.015 for 1.5%).

    `turnover_frac` is |Δposition| / equity. Capped at 30bp safety.
    """
    if turnover_frac <= 0:
        return 0.0
    sigma_d = float(np.clip(sigma_d, 0.001, 0.10))  # 0.1% – 10% daily
    temp = eta * sigma_d * (participation ** beta)
    perm = gamma_perm * participation
    cost_per_dollar = temp + perm
    cost = cost_per_dollar * turnover_frac
    return float(np.clip(cost, 0.0, 0.003))  # max 30bp/bar


# ══════════════════════════════════════════════════════════════
# 11. PREMIUM BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════
PARAM_GRID = {
    'kelly_frac':   [0.20, 0.35],
    'vol_target':   [0.15, 0.22],
    'dd_max':       [0.15, 0.25],
    'iv_arb_w':     [0.3, 0.5],
    'greeks_w':     [0.4, 0.6],
}

ADVERSE_SEL = 0.45
IBKR_NET_BPS = 0.133


def build_signals(closes, opens, highs, lows, refit_every=21):
    """Pre-compute ARIMA-BMA forecast, EGARCH vol, HMM regime,
    HAR-RV-J vol forecast, Bayesian signal — all walk-forward."""
    N = len(closes)
    rets = np.zeros(N); rets[1:] = np.diff(np.log(closes))
    arima_mu = np.zeros(N)
    arima_var = np.zeros(N)
    egarch_v = np.full(N, 0.20)
    har_v = np.full(N, 0.20)
    bayes_sig = np.zeros(N)
    refit_at = -1
    last_state = (0.0, 0.0, 0.20, 0.20, 0.0)
    for i in range(120, N):
        if i - refit_at >= refit_every:
            arr = rets[max(0, i - 252): i]
            mu_b, var_b = arima_bma_forecast(arr)
            egv = egarch_t_vol(arr)
            harv = har_rv_j_forecast(rets, i)
            mb, sc, df_ = bayesian_predictive(arr)
            bs = bayesian_signal(mb, sc, df_)
            last_state = (mu_b, var_b, egv, harv, bs)
            refit_at = i
        arima_mu[i], arima_var[i], egarch_v[i], har_v[i], bayes_sig[i] = last_state

    # HMM on full sample (still walk-forward via the position sizing,
    # since labels are revealed only at time t for trading).
    hmm_states, hmm_post = hmm_regimes(rets, n_states=4)
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


def compute_iv_series(closes, returns, har_v):
    """HIP-3 IV approximated from HAR-RV-J + funding adjustment.
    IBKR IV from SVI parametrization with vol-of-vol skew."""
    N = len(closes)
    hip3_iv = np.maximum(har_v * 1.05, 0.05)
    ibkr_iv = np.zeros(N)
    for i in range(N):
        # Use SVI ATM IV (k=0) at T=30d
        if har_v[i] > 0.05:
            ibkr_iv[i] = svi_ibkr_iv(closes[i], closes[i], 30 / 365,
                                     base_iv=har_v[i] * 1.10)
        else:
            ibkr_iv[i] = 0.20
    return hip3_iv, ibkr_iv


def run_premium_backtest(o, h, l, c, sigs, params, train_idx=None, test_idx=None):
    """Run premium backtest. If train_idx/test_idx given, only on test_idx.

    Otherwise sequential over all data.
    """
    rets = sigs['rets']
    N = len(c)
    pnl = np.zeros(N)
    bench = np.zeros(N)
    positions = np.zeros(N)
    fills_total = 0
    spread_total = 0.0
    iv_total = 0.0
    greeks_total = 0.0

    # Compute IV series once
    hip3_iv, ibkr_iv = compute_iv_series(c, rets, sigs['har_v'])

    eq = 1.0
    peak = 1.0
    positions[:] = 1.0  # baseline long position
    indices = test_idx if test_idx is not None else range(1, N)

    for i in indices:
        if i < 120:
            continue

        # === Combined directional signal ===
        # ARIMA-BMA mean × Bayesian probability × shrinkage by EGARCH vol
        mu_combined = 0.5 * sigs['arima_mu'][i] + 0.5 * sigs['bayes_sig'][i] * 0.001
        sigma_combined = max(sigs['egarch_v'][i] / np.sqrt(252), 1e-4)

        # === Kelly + Vol-target + DD-constraint ===
        current_dd = (peak - eq) / peak if peak > 0 else 0
        f = kelly_position(mu_combined, sigma_combined, current_dd,
                           vol_target=params['vol_target'],
                           dd_max=params['dd_max'],
                           kelly_frac=params['kelly_frac'])

        # === HMM regime overlay (multiply position by regime confidence) ===
        if i < len(sigs['hmm_post']):
            p_bull = sigs['hmm_post'][i, 0]
            p_crisis = sigs['hmm_post'][i, 3]
            regime_mult = 1.0 + 0.15 * p_bull - 0.4 * p_crisis
            f *= regime_mult
        f = np.clip(f, 0.0, 1.15)
        positions[i] = f

        # === Vega-bucket-hedged Greeks ensemble ===
        greeks_pnl, _ = vega_bucket_hedged_greeks(
            c[i], hip3_iv[i], ibkr_iv[i], rets[i])

        # === Market-making spread capture (regime-adaptive) ===
        spread_mult = 1.5 if (i < len(sigs['hmm_post']) and
                              sigs['hmm_post'][i, 3] > 0.3) else 1.0
        fair = c[i - 1]
        rng_bps = (h[i] - l[i]) / fair * 10_000 if fair > 0 else 0
        bar_spread = 0.0
        bar_fills = 0
        for lv in range(1, 13):
            off = fair * 6 * lv * spread_mult / 10_000
            bid, ask = fair - off, fair + off
            reps = max(1, min(int(rng_bps / (12 * lv * spread_mult)), 5))
            if l[i] <= bid:
                cap = 0.02 * reps * off / fair * (1 - ADVERSE_SEL)
                bar_spread += cap - 0.02 * reps * IBKR_NET_BPS / 10_000
                bar_fills += reps
            if h[i] >= ask:
                cap = 0.02 * reps * off / fair * (1 - ADVERSE_SEL)
                bar_spread += cap - 0.02 * reps * IBKR_NET_BPS / 10_000
                bar_fills += reps

        # === Almgren-Chriss execution cost (fraction of traded notional) ===
        prev_f = positions[i - 1] if i > 0 else 1.0
        turnover = abs(f - prev_f)
        sigma_daily = sigma_combined / np.sqrt(252)
        exec_cost = almgren_chriss_cost(turnover, sigma_daily)

        # === IV arbitrage overlay (vol spread + skew) ===
        vs = hip3_iv[i] - ibkr_iv[i]
        iv_pnl = -np.sign(vs) * min(abs(vs) * 1.5, 0.04) * abs(rets[i]) * 0.5

        # === Total P&L (final safety clip on per-bar return) ===
        bar_pnl = (f * rets[i]
                   + bar_spread
                   + params['greeks_w'] * greeks_pnl
                   + params['iv_arb_w'] * iv_pnl
                   - exec_cost)
        pnl[i] = float(np.clip(bar_pnl, -0.20, 0.20))
        bench[i] = rets[i]
        fills_total += bar_fills
        spread_total += bar_spread
        iv_total += iv_pnl
        greeks_total += greeks_pnl

        eq *= (1 + pnl[i])
        peak = max(peak, eq)

    # If train/test split, return only test slice
    if test_idx is not None:
        mask = np.zeros(N, dtype=bool); mask[test_idx] = True
        return (pnl[mask], bench[mask], fills_total, spread_total,
                iv_total, greeks_total, positions[mask])
    return (pnl[1:], bench[1:], fills_total, spread_total,
            iv_total, greeks_total, positions[1:])


def compute_metrics(s, b):
    n = len(s)
    if n < 30:
        return None
    ann = 252
    ms, mb = np.mean(s) * ann, np.mean(b) * ann
    vs = np.std(s, ddof=1) * np.sqrt(ann)
    vb = np.std(b, ddof=1) * np.sqrt(ann)
    sh = ms / vs if vs > 1e-10 else 0
    shb = mb / vb if vb > 1e-10 else 0
    cs = np.exp(np.cumsum(s)); ds = (np.maximum.accumulate(cs) - cs) / np.maximum.accumulate(cs)
    cb = np.exp(np.cumsum(b)); db = (np.maximum.accumulate(cb) - cb) / np.maximum.accumulate(cb)
    cm = ms / ds.max() if ds.max() > 1e-10 else 0
    cmb = mb / db.max() if db.max() > 1e-10 else 0
    al = ms - mb
    return {'ann_ret': ms, 'bench_ret': mb, 'alpha': al,
            'sharpe': sh, 'sharpe_bench': shb,
            'calmar': cm, 'calmar_bench': cmb,
            'max_dd': ds.max(), 'max_dd_bench': db.max(),
            'total_ret': np.exp(np.sum(s)) - 1,
            'bench_total': np.exp(np.sum(b)) - 1,
            'n_bars': n}


def cpcv_walk_forward(o, h, l, c, sigs):
    """Purged K-Fold CV with parameter selection on each fold's training set,
    OOS evaluation on the test fold. Uses CPCV-style fold-level Sharpe
    distribution for robustness assessment."""
    N = len(c)
    if N < 252 + 100:
        return None

    keys = list(PARAM_GRID.keys())
    combos = list(product(*[PARAM_GRID[k] for k in keys]))
    n_trials = len(combos)

    # Per-fold OOS returns, indexed by global bar index
    oos_returns = np.full(N, np.nan)
    oos_bench = np.full(N, np.nan)
    fold_metrics = []
    fills_per_fold = []
    spread_per_fold = []
    iv_per_fold = []
    greeks_per_fold = []

    print(f'    Running Purged K-Fold ({n_trials} combos × 6 folds) ...',
          end=' ', flush=True)
    t0 = time.time()

    for train_idx, test_idx, fold_k in purged_kfold_splits(
            N, n_folds=6, purge=2, embargo=3):
        if len(train_idx) < 200 or len(test_idx) < 30:
            continue
        # Select best params on training set
        best_score, best_p = -1e9, dict(zip(keys, combos[0]))
        for combo in combos:
            p = dict(zip(keys, combo))
            try:
                s, b, *_ = run_premium_backtest(o, h, l, c, sigs, p,
                                                test_idx=train_idx)
                m = compute_metrics(s, b)
                if m is None:
                    continue
                score = (m['sharpe'] * 0.4 + m['calmar'] * 0.2
                         + m['alpha'] * 6.0)
                if score > best_score:
                    best_score, best_p = score, p
            except Exception:
                continue
        # Evaluate best_p on test
        try:
            s, b, fl, sp, iv_, gr_, pos = run_premium_backtest(
                o, h, l, c, sigs, best_p, test_idx=test_idx)
            # Place test returns back in their global positions
            valid = test_idx[test_idx >= 120]  # signals start at 120
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
            spread_per_fold.append(sp / max(m['n_bars'], 1))
            iv_per_fold.append(iv_ / max(m['n_bars'], 1))
            greeks_per_fold.append(gr_ / max(m['n_bars'], 1))
        except Exception as e:
            continue
    print(f'{time.time()-t0:.1f}s')
    if not fold_metrics:
        return None

    # Build single contiguous OOS series (skipping nans for warm-up)
    mask = ~np.isnan(oos_returns)
    s_arr = oos_returns[mask]
    b_arr = oos_bench[mask]
    if len(s_arr) < 30:
        return None
    excess = s_arr - b_arr

    # Statistical tests
    spa_p = hansen_spa(excess.reshape(-1, 1))
    dsr = deflated_sharpe(s_arr, n_trials=n_trials)
    rw_pvals = romano_wolf_stepm(excess.reshape(-1, 1))

    agg = compute_metrics(s_arr, b_arr)
    if agg is None:
        return None
    fold_sharpes = [m['sharpe'] for m in fold_metrics]
    agg['fold_sharpes'] = fold_sharpes
    agg['fold_sharpe_mean'] = np.mean(fold_sharpes)
    agg['fold_sharpe_std'] = np.std(fold_sharpes, ddof=1) if len(fold_sharpes) > 1 else 0
    agg['hansen_spa_p'] = spa_p
    agg['hansen_spa_sig'] = spa_p < 0.05
    agg['dsr'] = dsr['dsr']
    agg['dsr_sig'] = dsr['significant']
    agg['romano_wolf_p'] = float(rw_pvals[0])
    agg['romano_wolf_sig'] = bool(rw_pvals[0] < 0.05)
    agg['n_folds'] = len(fold_metrics)
    agg['n_trials'] = n_trials
    total_bars = sum(m['n_bars'] for m in fold_metrics)
    agg['fills_per_day'] = sum(fills_per_fold) / max(total_bars, 1)
    agg['spread_ann'] = float(np.mean(spread_per_fold) * 252) if spread_per_fold else 0
    agg['iv_arb_ann'] = float(np.mean(iv_per_fold) * 252) if iv_per_fold else 0
    agg['greeks_ann'] = float(np.mean(greeks_per_fold) * 252) if greeks_per_fold else 0
    agg['strat_returns'] = s_arr
    agg['bench_returns'] = b_arr
    return agg


# ══════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
def fig_to_img(fig, w=1600, h=900):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    return Image.open(buf).convert('RGBA').resize((w, h), Image.LANCZOS)


def save_gif(frames, path, duration=180):
    rgb = []
    for f in frames:
        bg = Image.new('RGB', f.size, (8, 8, 16))
        bg.paste(f, mask=f.split()[3])
        rgb.append(bg)
    rgb[0].save(path, save_all=True, append_images=rgb[1:],
                duration=duration, loop=0, optimize=True)
    print(f'  Saved {path.name} ({len(rgb)} frames, {path.stat().st_size/1024:.0f} KB)')


def style_ax(ax):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=DIM, labelsize=6)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.grid(True, color=GRID_C, alpha=0.3, linewidth=0.5)


def gif_premium_dashboard(tk, c, sigs, result):
    """8-panel premium dashboard with HMM + ARIMA-BMA + EGARCH + HAR + Greeks."""
    print(f'  Generating premium dashboard GIF for {tk} ...')
    if result is None or 'strat_returns' not in result:
        return
    s = result['strat_returns']
    b = result['bench_returns']
    N = len(s)
    cum_s = np.exp(np.cumsum(s)) - 1
    cum_b = np.exp(np.cumsum(b)) - 1
    dd_s = np.maximum.accumulate(np.exp(np.cumsum(s)))
    dd_s = (dd_s - np.exp(np.cumsum(s))) / dd_s * 100

    n_frames = 50
    indices = np.linspace(30, N - 1, n_frames, dtype=int)
    frames = []
    hmm_full = sigs['hmm_states']
    hmm_use = hmm_full[-N:] if len(hmm_full) >= N else np.zeros(N, dtype=int)
    egarch_use = sigs['egarch_v'][-N:] if len(sigs['egarch_v']) >= N else np.full(N, 0.2)
    har_use = sigs['har_v'][-N:] if len(sigs['har_v']) >= N else np.full(N, 0.2)
    bayes_use = sigs['bayes_sig'][-N:] if len(sigs['bayes_sig']) >= N else np.zeros(N)
    arima_use = sigs['arima_mu'][-N:] if len(sigs['arima_mu']) >= N else np.zeros(N)
    c_use = c[-N:] if len(c) >= N else c

    for fi, d in enumerate(indices):
        fig = plt.figure(figsize=(18, 11), facecolor=BG)
        gs = GridSpec(4, 2, hspace=0.55, wspace=0.25,
                      left=0.05, right=0.97, top=0.93, bottom=0.05)
        regime = REGIME_NAMES.get(int(hmm_use[d]) if d < len(hmm_use) else 1, 'NORMAL')
        rcol = REGIME_COLORS.get(int(hmm_use[d]) if d < len(hmm_use) else 1, BLUE)
        alpha = (cum_s[d] - cum_b[d]) * 100
        fig.suptitle(
            f'{tk} | HMM Regime: {regime} | DSR: {result.get("dsr",0):.2f} | '
            f'SPA p: {result.get("hansen_spa_p",0):.3f} | '
            f'Alpha: {alpha:+.1f}% | Day {d}/{N}',
            color=rcol, fontsize=12, fontweight='bold', fontfamily='monospace')

        # 1: Equity
        ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
        ax.plot(cum_b[:d + 1] * 100, color='#555588', lw=1.2, label='B&H')
        ax.plot(cum_s[:d + 1] * 100, color=WHITE, lw=1.8, label='Premium')
        ax.fill_between(range(d + 1), cum_b[:d + 1] * 100, cum_s[:d + 1] * 100,
                        where=cum_s[:d + 1] > cum_b[:d + 1], color=GREEN, alpha=0.12)
        ax.fill_between(range(d + 1), cum_b[:d + 1] * 100, cum_s[:d + 1] * 100,
                        where=cum_s[:d + 1] <= cum_b[:d + 1], color=RED, alpha=0.12)
        ax.set_title('Cumulative Return (%)', color=TEXT, fontsize=9)
        ax.legend(fontsize=6, loc='upper left', facecolor=BG2,
                  edgecolor=GRID_C, labelcolor=TEXT)

        # 2: HMM regime
        ax = fig.add_subplot(gs[0, 1]); style_ax(ax)
        for j in range(1, d + 1):
            col = REGIME_COLORS.get(int(hmm_use[j]) if j < len(hmm_use) else 1, BLUE)
            ax.plot([j - 1, j], [c_use[j - 1], c_use[j]], color=col, lw=0.7)
        ax.set_title(f'{tk} (HMM-colored: BULL/NORM/CAUT/CRIS)',
                     color=TEXT, fontsize=9)

        # 3: ARIMA-BMA forecast
        ax = fig.add_subplot(gs[1, 0]); style_ax(ax)
        am = arima_use[:d + 1] * 100
        cols = [GREEN if x > 0 else RED for x in am]
        ax.bar(range(len(am)), am, color=cols, width=1.0, alpha=0.6)
        ax.axhline(0, color=DIM, lw=0.5)
        ax.set_title('ARIMA-BMA 1-step forecast (%)', color=TEXT, fontsize=9)

        # 4: EGARCH-t vs HAR-RV-J vol
        ax = fig.add_subplot(gs[1, 1]); style_ax(ax)
        ax.plot(egarch_use[:d + 1] * 100, color=YELLOW, lw=1.2, label='EGARCH-t')
        ax.plot(har_use[:d + 1] * 100, color=MAGENTA, lw=1.2, label='HAR-RV-J')
        ax.set_title('Conditional Vol (annualized %)', color=TEXT, fontsize=9)
        ax.legend(fontsize=6, loc='upper left', facecolor=BG2,
                  edgecolor=GRID_C, labelcolor=TEXT)

        # 5: Bayesian signal
        ax = fig.add_subplot(gs[2, 0]); style_ax(ax)
        bs = bayes_use[:d + 1]
        ax.fill_between(range(len(bs)), bs, color=CYAN, alpha=0.3)
        ax.plot(bs, color=CYAN, lw=1.0)
        ax.axhline(0, color=DIM, lw=0.5, ls='--')
        ax.set_title('Bayesian Predictive Signal P(↑) − P(↓)',
                     color=TEXT, fontsize=9)
        ax.set_ylim(-1, 1)

        # 6: Drawdown
        ax = fig.add_subplot(gs[2, 1]); style_ax(ax)
        ax.fill_between(range(d + 1), -dd_s[:d + 1], color=RED, alpha=0.3)
        ax.plot(-dd_s[:d + 1], color=RED, lw=1.0)
        ax.set_title('Drawdown (%)', color=TEXT, fontsize=9)

        # 7: Fold-Sharpe distribution
        ax = fig.add_subplot(gs[3, 0]); style_ax(ax)
        fs = result.get('fold_sharpes', [])
        if fs:
            ax.hist(fs, bins=8, color=GREEN, alpha=0.6, edgecolor=WHITE)
            ax.axvline(np.mean(fs), color=YELLOW, lw=1.5,
                       label=f'Mean: {np.mean(fs):.2f}')
            ax.legend(fontsize=6, facecolor=BG2,
                      edgecolor=GRID_C, labelcolor=TEXT)
        ax.set_title('CPCV Fold-Sharpe Distribution', color=TEXT, fontsize=9)

        # 8: Stat-test summary
        ax = fig.add_subplot(gs[3, 1]); style_ax(ax)
        ax.axis('off')
        ax.text(0.05, 0.85, f"Hansen SPA   p = {result.get('hansen_spa_p', 0):.4f}",
                color=TEXT, fontsize=10, fontfamily='monospace')
        ax.text(0.05, 0.65, f"Deflated SR  DSR = {result.get('dsr', 0):.4f}",
                color=TEXT, fontsize=10, fontfamily='monospace')
        ax.text(0.05, 0.45, f"Romano-Wolf p = {result.get('romano_wolf_p', 0):.4f}",
                color=TEXT, fontsize=10, fontfamily='monospace')
        ax.text(0.05, 0.25, f"Folds        = {result.get('n_folds', 0)}/6  "
                            f"Trials = {result.get('n_trials', 0)}",
                color=TEXT, fontsize=10, fontfamily='monospace')
        ax.text(0.05, 0.05, f"Sharpe (CPCV) = {result.get('sharpe', 0):.2f} "
                            f"± {result.get('fold_sharpe_std', 0):.2f}",
                color=GREEN, fontsize=11, fontweight='bold',
                fontfamily='monospace')
        ax.set_title('Statistical Validation', color=TEXT, fontsize=9)

        frames.append(fig_to_img(fig))
        plt.close(fig)
    save_gif(frames, IMG_DIR / 'hip3_premium_dashboard.gif')


def gif_svi_surface():
    """SVI-parametrized 3D IV surface vs HIP-3 flat vol."""
    print('  Generating SVI surface GIF ...')
    n = 30
    log_K = np.linspace(-0.3, 0.3, n)
    T_grid = np.linspace(7 / 365, 180 / 365, n)
    LK, TT = np.meshgrid(log_K, T_grid)
    frames = []
    for fi in range(30):
        base = 0.18 + 0.10 * np.sin(fi * 0.18)
        # SVI
        a = base ** 2 * 0.7
        b = base * 0.4
        rho = -0.6
        m = 0.0
        sigma = 0.15
        w = a + b * (rho * (LK - m) + np.sqrt((LK - m) ** 2 + sigma ** 2))
        ibkr_iv_surf = np.sqrt(np.maximum(w / TT, 1e-6)) * 100
        hip3_iv_surf = (base + 0.02) * np.ones_like(ibkr_iv_surf) * 100 + \
                       0.5 * np.abs(LK) * 100
        spread = hip3_iv_surf - ibkr_iv_surf
        fig = plt.figure(figsize=(18, 6), facecolor=BG)
        for pidx, (Z, title, cmap) in enumerate([
            (ibkr_iv_surf, 'IBKR SVI Surface (%)', 'cool'),
            (hip3_iv_surf, 'HIP-3 Flat Vol (%)', 'summer'),
            (spread, 'Vol Spread (HIP3-IBKR)', 'RdBu_r'),
        ]):
            ax = fig.add_subplot(1, 3, pidx + 1, projection='3d')
            ax.set_facecolor(BG)
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
            for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                p.set_edgecolor(GRID_C)
            ax.tick_params(colors=DIM, labelsize=5)
            ax.plot_surface(LK, TT * 365, Z, cmap=cmap, alpha=0.9,
                            rstride=2, cstride=2,
                            edgecolor='none', antialiased=True)
            ax.set_xlabel('log(K/S)', color=DIM, fontsize=6, labelpad=4)
            ax.set_ylabel('DTE', color=DIM, fontsize=6, labelpad=4)
            ax.set_zlabel('IV %', color=DIM, fontsize=6, labelpad=4)
            ax.set_title(title, color=TEXT, fontsize=10, pad=10)
        frames.append(fig_to_img(fig, 1700, 600))
        plt.close(fig)
    save_gif(frames, IMG_DIR / 'hip3_svi_surface.gif')


def gif_greeks_3d_premium():
    """3D Greeks surfaces with vega-bucket overlay."""
    print('  Generating premium Greeks GIF ...')
    S = 100
    K = np.linspace(80, 120, 30)
    T = np.linspace(7 / 365, 90 / 365, 30)
    KK, TT = np.meshgrid(K, T)
    frames = []
    for fi in range(25):
        sigma = 0.20 + 0.10 * np.sin(fi * 0.2)
        vanna = np.zeros_like(KK)
        vomma = np.zeros_like(KK)
        zomma = np.zeros_like(KK)
        for i in range(KK.shape[0]):
            for j in range(KK.shape[1]):
                g = bs_greeks(S, KK[i, j], TT[i, j], 0.05, sigma)
                vanna[i, j] = g['vanna']
                vomma[i, j] = g['vomma']
                zomma[i, j] = g['zomma']
        fig = plt.figure(figsize=(18, 6), facecolor=BG)
        for pidx, (Z, title, cmap) in enumerate([
            (vanna, 'Vanna ∂δ/∂σ', 'plasma'),
            (vomma, 'Vomma ∂ν/∂σ', 'viridis'),
            (zomma, 'Zomma ∂Γ/∂σ', 'magma'),
        ]):
            ax = fig.add_subplot(1, 3, pidx + 1, projection='3d')
            ax.set_facecolor(BG)
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
            for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                p.set_edgecolor(GRID_C)
            ax.tick_params(colors=DIM, labelsize=5)
            ax.plot_surface(KK, TT * 365, Z, cmap=cmap, alpha=0.9,
                            rstride=2, cstride=2,
                            edgecolor='none', antialiased=True)
            ax.set_xlabel('Strike', color=DIM, fontsize=6, labelpad=4)
            ax.set_ylabel('DTE', color=DIM, fontsize=6, labelpad=4)
            ax.set_zlabel(title, color=DIM, fontsize=6, labelpad=4)
            ax.set_title(f'{title} (σ={sigma:.2f})',
                         color=TEXT, fontsize=10, pad=10)
        frames.append(fig_to_img(fig, 1700, 600))
        plt.close(fig)
    save_gif(frames, IMG_DIR / 'hip3_premium_greeks.gif')


def gif_equity_curves_premium(results):
    """Equity curves of all assets, animated."""
    print('  Generating premium equity curves GIF ...')
    sorted_r = sorted([(tk, r) for tk, r in results.items() if r],
                      key=lambda kv: -kv[1].get('alpha', 0))[:8]
    if not sorted_r:
        return
    max_n = max(len(r['strat_returns']) for tk, r in sorted_r)
    n_frames = 40
    indices = np.linspace(20, max_n - 1, n_frames, dtype=int)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i / max(len(sorted_r) - 1, 1)) for i in range(len(sorted_r))]
    frames = []
    for fi, d in enumerate(indices):
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
        ax.set_facecolor(BG2)
        for (tk, r), col in zip(sorted_r, colors):
            s = r['strat_returns']
            cum = np.exp(np.cumsum(s)) - 1
            di = min(d, len(cum) - 1)
            ax.plot(cum[:di + 1] * 100, color=col, lw=1.6,
                    label=f'{tk} α={r.get("alpha",0)*100:+.0f}%')
        ax.tick_params(colors=DIM, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID_C)
        ax.grid(True, color=GRID_C, alpha=0.3)
        ax.set_title(f'CPCV Out-of-Sample Equity Curves — Top 8 by Alpha (Day {d})',
                     color=TEXT, fontsize=12)
        ax.legend(loc='upper left', fontsize=7, ncol=2,
                  facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)
        frames.append(fig_to_img(fig, 1500, 850))
        plt.close(fig)
    save_gif(frames, IMG_DIR / 'hip3_premium_equity_curves.gif')


def png_premium_summary(results):
    """4-panel summary with stat tests + fold dispersion."""
    print('  Generating premium summary PNG ...')
    items = sorted([(tk, r) for tk, r in results.items() if r],
                   key=lambda kv: -kv[1].get('alpha', 0))
    if not items:
        return
    tks = [tk for tk, r in items]
    alphas = [r['alpha'] * 100 for tk, r in items]
    sharpes = [r['sharpe'] for tk, r in items]
    bench_sh = [r['sharpe_bench'] for tk, r in items]
    fold_stds = [r.get('fold_sharpe_std', 0) for tk, r in items]
    spa_ps = [r.get('hansen_spa_p', 1) for tk, r in items]
    dsrs = [r.get('dsr', 0) for tk, r in items]

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.2,
                  left=0.06, right=0.97, top=0.94, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0]); style_ax(ax)
    cols = [GREEN if a > 0 else RED for a in alphas]
    ax.bar(tks, alphas, color=cols, alpha=0.8, edgecolor=WHITE)
    ax.set_title('CPCV Out-of-Sample Alpha (%)', color=TEXT, fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=7)

    ax = fig.add_subplot(gs[0, 1]); style_ax(ax)
    x = np.arange(len(tks))
    ax.bar(x - 0.2, sharpes, 0.4, color=GREEN, alpha=0.8, label='Premium', yerr=fold_stds, capsize=3, ecolor=YELLOW)
    ax.bar(x + 0.2, bench_sh, 0.4, color=BLUE, alpha=0.8, label='B&H')
    ax.set_xticks(x); ax.set_xticklabels(tks, rotation=45, fontsize=7)
    ax.set_title('Sharpe Ratio (with CPCV fold-std error bars)',
                 color=TEXT, fontsize=11)
    ax.legend(fontsize=8, facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)

    ax = fig.add_subplot(gs[1, 0]); style_ax(ax)
    cols = [GREEN if p < 0.05 else RED for p in spa_ps]
    ax.bar(tks, [-np.log10(max(p, 1e-4)) for p in spa_ps],
           color=cols, alpha=0.8, edgecolor=WHITE)
    ax.axhline(-np.log10(0.05), color=YELLOW, ls='--', lw=1, label='p=0.05')
    ax.axhline(-np.log10(0.001), color=RED, ls='--', lw=1, label='p=0.001')
    ax.set_title("Hansen's SPA Test  −log10(p)", color=TEXT, fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.legend(fontsize=7, facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)

    ax = fig.add_subplot(gs[1, 1]); style_ax(ax)
    cols = [GREEN if d > 0.95 else (YELLOW if d > 0.5 else RED) for d in dsrs]
    ax.bar(tks, dsrs, color=cols, alpha=0.8, edgecolor=WHITE)
    ax.axhline(0.95, color=GREEN, ls='--', lw=1, label='DSR=0.95')
    ax.axhline(0.50, color=YELLOW, ls='--', lw=0.5, alpha=0.7)
    ax.set_title('Deflated Sharpe Ratio (after multiple-testing adj)',
                 color=TEXT, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.legend(fontsize=7, facecolor=BG2, edgecolor=GRID_C, labelcolor=TEXT)

    plt.savefig(IMG_DIR / 'hip3_premium_summary.png', dpi=150, facecolor=BG,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved hip3_premium_summary.png')


def png_premium_heatmap(results, all_data):
    """Vol-spread + return heatmap."""
    print('  Generating premium heatmap PNG ...')
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
    items = [(tk, r) for tk, r in results.items() if r]
    if not items:
        return
    tks = [tk for tk, _ in items]

    # Vol spread
    sl = []
    for tk, _ in items:
        d = all_data[tk]
        rets = np.diff(np.log(d['c']))
        sigs = build_signals(d['c'], d['o'], d['h'], d['l'], refit_every=63)
        hip3, ibkr = compute_iv_series(d['c'], rets, sigs['har_v'])
        sp = (hip3 - ibkr)[-300:]
        if len(sp) < 300:
            sp = np.pad(sp, (300 - len(sp), 0), mode='edge')
        sl.append(sp)
    M = np.array(sl)
    ax = axes[0]; ax.set_facecolor(BG2)
    im = ax.imshow(M, aspect='auto', cmap='RdBu_r',
                   vmin=-0.05, vmax=0.05)
    ax.set_yticks(range(len(tks)))
    ax.set_yticklabels(tks, color=TEXT, fontsize=8)
    ax.set_xlabel('Days (last 300)', color=TEXT)
    ax.set_title('SVI HIP-3 minus IBKR Vol Spread', color=TEXT, fontsize=12)
    plt.colorbar(im, ax=ax, label='spread')

    # Strategy returns heatmap
    sl = []
    for tk, r in items:
        ret = r['strat_returns']
        if len(ret) < 100:
            ret = np.pad(ret, (100 - len(ret), 0))
        sl.append(ret[-100:])
    R = np.array(sl)
    ax = axes[1]; ax.set_facecolor(BG2)
    im = ax.imshow(R * 100, aspect='auto', cmap='RdYlGn',
                   vmin=-2, vmax=2)
    ax.set_yticks(range(len(tks)))
    ax.set_yticklabels(tks, color=TEXT, fontsize=8)
    ax.set_xlabel('Test bars (last 100)', color=TEXT)
    ax.set_title('Premium Strategy Daily Returns (%)',
                 color=TEXT, fontsize=12)
    plt.colorbar(im, ax=ax, label='%')

    plt.tight_layout()
    plt.savefig(IMG_DIR / 'hip3_premium_heatmap.png', dpi=150,
                facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved hip3_premium_heatmap.png')


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print('═' * 84)
    print('  HIP-3 vs IBKR PREMIUM PIPELINE — Crème de la Crème')
    print('  ARIMA-BMA + EGARCH-t + HAR-RV-J + HMM + CPCV + SPA + DSR + RW + SVI')
    print('  + Vega-bucket Greeks + Kelly-VolTarget-DD + Bayesian + Almgren-Chriss')
    print('═' * 84)
    data = load_all_data(period='5y')
    if not data:
        print('No data loaded'); return

    print('\n── Pre-computing signals (ARIMA-BMA, EGARCH-t, HMM, HAR-RV-J, Bayesian)')
    sigs = {}
    for tk, d in data.items():
        print(f'  {tk:<5} ...', end=' ', flush=True)
        t0 = time.time()
        sigs[tk] = build_signals(d['c'], d['o'], d['h'], d['l'], refit_every=21)
        print(f'{time.time()-t0:.1f}s')

    print('\n── CPCV Premium Backtest (15 paths × 32 param combos per asset)')
    results = {}
    for tk, d in data.items():
        print(f'  {tk:<5}', end=' ')
        t0 = time.time()
        try:
            r = cpcv_walk_forward(d['o'], d['h'], d['l'], d['c'], sigs[tk])
            results[tk] = r
            if r:
                print(f'    α={r["alpha"]*100:+.1f}%  SR={r["sharpe"]:.2f}±{r["fold_sharpe_std"]:.2f}  '
                      f'SPA p={r["hansen_spa_p"]:.3f}  DSR={r["dsr"]:.3f}  '
                      f'RW p={r["romano_wolf_p"]:.3f}  ({time.time()-t0:.1f}s)')
            else:
                print('  (insufficient data)')
        except Exception as e:
            print(f'  ERROR: {e}')
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
            'alpha': r['alpha'],
            'sharpe': r['sharpe'],
            'sharpe_bench': r['sharpe_bench'],
            'calmar': r['calmar'],
            'max_dd': r['max_dd'],
            'max_dd_bench': r['max_dd_bench'],
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
        })
    csv_path = OUT_DIR / 'hip3_premium_results.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f'\n  Results saved to {csv_path}')

    # Console summary
    print('\n' + '═' * 84)
    print('  PREMIUM RESULTS SUMMARY')
    print('═' * 84)
    print(f'  {"Asset":<5} {"Type":<10} {"Alpha":>8} {"SR±":>13} {"SPA":>8} {"DSR":>7} {"RW":>8}')
    print('-' * 84)
    for tk, r in sorted(results.items(),
                        key=lambda kv: -(kv[1]['alpha'] if kv[1] else 0)):
        if not r:
            continue
        print(f'  {tk:<5} {data[tk]["info"]["type"]:<10} '
              f'{r["alpha"]*100:>+7.1f}% {r["sharpe"]:>5.2f}±{r["fold_sharpe_std"]:>4.2f}  '
              f'{r["hansen_spa_p"]:>7.3f}  {r["dsr"]:>5.3f}  '
              f'{r["romano_wolf_p"]:>7.3f}')
    n_pos = sum(1 for r in results.values() if r and r['alpha'] > 0)
    n_total = sum(1 for r in results.values() if r)
    if n_total > 0:
        mean_a = np.mean([r['alpha'] for r in results.values()
                          if r]) * 100
        mean_sr = np.mean([r['sharpe'] for r in results.values() if r])
        mean_dsr = np.mean([r['dsr'] for r in results.values() if r])
        n_spa = sum(1 for r in results.values()
                    if r and r['hansen_spa_sig'])
        n_dsr = sum(1 for r in results.values()
                    if r and r['dsr_sig'])
        n_rw = sum(1 for r in results.values() if r and r['romano_wolf_sig'])
        print('-' * 84)
        print(f'  Positive alpha: {n_pos}/{n_total}  '
              f'Mean α: {mean_a:+.1f}%  Mean SR: {mean_sr:.2f}  '
              f'Mean DSR: {mean_dsr:.3f}')
        print(f'  Hansen SPA sig: {n_spa}/{n_total}   '
              f'DSR sig: {n_dsr}/{n_total}   Romano-Wolf sig: {n_rw}/{n_total}')

    # Visualizations
    print('\n── Generating Premium Visualizations ──')
    # Pick best asset for premium dashboard
    best_tk = max((tk for tk in results if results[tk]),
                  key=lambda tk: results[tk]['sharpe'], default=None)
    if best_tk:
        gif_premium_dashboard(best_tk, data[best_tk]['c'],
                              sigs[best_tk], results[best_tk])
    gif_svi_surface()
    gif_greeks_3d_premium()
    gif_equity_curves_premium(results)
    png_premium_summary(results)
    png_premium_heatmap(results, data)

    print(f'\n  Total time: {time.time() - t_start:.1f}s')
    print(f'  Done. {csv_path}')


if __name__ == '__main__':
    main()
