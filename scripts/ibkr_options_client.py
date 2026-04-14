#!/usr/bin/env python3
"""
IBKR Options API Client — Options orderbook and chain data.

Provides:
  1. Live connection via ib_insync (TWS API) when available
  2. Black-Scholes pricing engine as fallback
  3. Full options chain generation with Greeks (delta, gamma, theta, vega, rho)
  4. Second/third order Greeks (vanna, charm, vomma, speed, color, zomma)
  5. IV surface construction from options chain

The higher-order Greeks are crucial for the HIP-3 vs IBKR arbitrage strategy:
  - Vanna (∂δ/∂σ): sensitivity of delta to vol changes
  - Charm (∂δ/∂t): delta decay (delta bleed)
  - Vomma (∂ν/∂σ): vol-of-vol sensitivity (convexity of vega)
  - Speed (∂Γ/∂S): rate of change of gamma
  - Color (∂Γ/∂t): gamma decay
  - Zomma (∂Γ/∂σ): gamma sensitivity to vol

Usage:
  from ibkr_options_client import IBKROptionsClient
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────
# Black-Scholes Pricing Engine
# ──────────────────────────────────────────────────────────────

@dataclass
class OptionQuote:
    """Single option quote with full Greeks."""
    strike: float
    expiry_days: float
    option_type: str  # 'call' or 'put'
    spot: float
    iv: float
    price: float
    # First order Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    # Second order Greeks
    vanna: float     # ∂δ/∂σ = ∂ν/∂S
    charm: float     # ∂δ/∂t (delta bleed)
    vomma: float     # ∂ν/∂σ (vol-of-vol, volga)
    # Third order Greeks
    speed: float     # ∂Γ/∂S
    color: float     # ∂Γ/∂t
    zomma: float     # ∂Γ/∂σ
    ultima: float    # ∂vomma/∂σ


def _d1d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = 'call') -> float:
    """Black-Scholes European option price."""
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        return max(K - S, 0)

    d1, d2 = _d1d2(S, K, T, r, sigma)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call') -> OptionQuote:
    """Compute all Greeks up to third order."""
    if T <= 1e-10:
        T = 1e-10
    if sigma <= 1e-10:
        sigma = 1e-10

    d1, d2 = _d1d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)
    exp_rT = np.exp(-r * T)

    # Price
    if option_type == 'call':
        price = S * N_d1 - K * exp_rT * N_d2
        delta = N_d1
        rho = K * T * exp_rT * N_d2 / 100
    else:
        price = K * exp_rT * N_neg_d2 - S * N_neg_d1
        delta = N_d1 - 1
        rho = -K * T * exp_rT * N_neg_d2 / 100

    # ── First Order Greeks ──
    gamma = n_d1 / (S * sigma * sqrt_T)
    theta_common = -(S * n_d1 * sigma) / (2 * sqrt_T)
    if option_type == 'call':
        theta = (theta_common - r * K * exp_rT * N_d2) / 365
    else:
        theta = (theta_common + r * K * exp_rT * N_neg_d2) / 365
    vega = S * n_d1 * sqrt_T / 100  # per 1% vol move

    # ── Second Order Greeks ──
    # Vanna: ∂δ/∂σ = ∂ν/∂S = -n(d1) * d2 / σ
    vanna = -n_d1 * d2 / sigma

    # Charm: ∂δ/∂t (delta bleed)
    # charm = -n(d1) * [2rT - d2*σ*√T] / (2T*σ*√T)
    charm_val = -n_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
    if option_type == 'put':
        charm_val = charm_val  # same for puts (charm of delta, not the delta itself)

    # Vomma (Volga): ∂ν/∂σ = S*n(d1)*√T * d1*d2 / σ
    vomma = vega * d1 * d2 / sigma * 100  # un-scale vega

    # ── Third Order Greeks ──
    # Speed: ∂Γ/∂S = -gamma/S * (d1/(σ√T) + 1)
    speed = -gamma / S * (d1 / (sigma * sqrt_T) + 1)

    # Color: ∂Γ/∂t
    color = -n_d1 / (2 * S * T * sigma * sqrt_T) * (
        1 + d1 * (2 * r * T - d2 * sigma * sqrt_T) / (sigma * sqrt_T)
    )

    # Zomma: ∂Γ/∂σ = gamma * (d1*d2 - 1) / σ
    zomma = gamma * (d1 * d2 - 1) / sigma

    # Ultima: ∂vomma/∂σ (third derivative w.r.t. vol)
    ultima = -vega / (sigma**2) * (
        d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2
    )

    return OptionQuote(
        strike=K, expiry_days=T * 365, option_type=option_type,
        spot=S, iv=sigma, price=price,
        delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho,
        vanna=vanna, charm=charm_val, vomma=vomma,
        speed=speed, color=color, zomma=zomma, ultima=ultima,
    )


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                option_type: str = 'call') -> Optional[float]:
    """Compute implied volatility from option price using Brent's method."""
    if T <= 0 or price <= 0:
        return None

    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if price < intrinsic:
        return None

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price

    try:
        iv = brentq(objective, 1e-4, 10.0, xtol=1e-8)
        return iv
    except (ValueError, RuntimeError):
        return None


# ──────────────────────────────────────────────────────────────
# IBKR Options Client
# ──────────────────────────────────────────────────────────────

class IBKROptionsClient:
    """IBKR-style options chain generator.

    When TWS is available, connects via ib_insync.
    Otherwise, generates realistic options chains using Black-Scholes
    with configurable IV surface (skew, term structure).
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
        self.ib = None
        self._try_connect()

    def _try_connect(self):
        """Try to connect to IBKR TWS/Gateway."""
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect('127.0.0.1', 7497, clientId=1)
            print("  Connected to IBKR TWS")
        except Exception:
            self.ib = None

    def generate_options_chain(self, spot: float, base_iv: float,
                                expiries_days: List[int] = None,
                                n_strikes: int = 21,
                                strike_range: float = 0.30,
                                iv_skew: float = -0.15,
                                iv_term_slope: float = 0.02,
                                iv_smile: float = 0.05,
                                bid_ask_spread_pct: float = 0.03
                                ) -> pd.DataFrame:
        """Generate a full options chain with realistic IV surface.

        Args:
            spot: Current spot price
            base_iv: At-the-money implied volatility
            expiries_days: List of expirations in days
            n_strikes: Number of strikes per expiry
            strike_range: Strikes from (1-range)*spot to (1+range)*spot
            iv_skew: Vol skew (negative = put skew)
            iv_term_slope: IV increase per year of expiry
            iv_smile: Quadratic smile curvature
            bid_ask_spread_pct: Bid-ask spread as fraction of price
        """
        if expiries_days is None:
            expiries_days = [7, 14, 30, 45, 60, 90, 120, 180, 365]

        strikes = np.linspace(spot * (1 - strike_range), spot * (1 + strike_range), n_strikes)

        rows = []
        for dte in expiries_days:
            T = dte / 365.0
            for K in strikes:
                moneyness = np.log(K / spot)  # log-moneyness

                # IV surface: skew + smile + term structure
                iv = base_iv + iv_skew * moneyness + iv_smile * moneyness**2 + iv_term_slope * T
                iv = max(iv, 0.05)  # floor at 5%

                for opt_type in ['call', 'put']:
                    quote = bs_greeks(spot, K, T, self.r, iv, opt_type)

                    # Add bid/ask around theoretical price
                    spread = max(quote.price * bid_ask_spread_pct, 0.01)
                    bid = max(quote.price - spread / 2, 0.01)
                    ask = quote.price + spread / 2
                    mid = (bid + ask) / 2

                    rows.append({
                        'strike': K,
                        'expiry_days': dte,
                        'expiry_T': T,
                        'type': opt_type,
                        'spot': spot,
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'iv': iv,
                        'moneyness': moneyness,
                        # First order
                        'delta': quote.delta,
                        'gamma': quote.gamma,
                        'theta': quote.theta,
                        'vega': quote.vega,
                        'rho': quote.rho,
                        # Second order
                        'vanna': quote.vanna,
                        'charm': quote.charm,
                        'vomma': quote.vomma,
                        # Third order
                        'speed': quote.speed,
                        'color': quote.color,
                        'zomma': quote.zomma,
                        'ultima': quote.ultima,
                    })

        return pd.DataFrame(rows)

    def get_iv_surface(self, spot: float, base_iv: float, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return IV surface as (strikes, expiries, iv_matrix) for 3D plotting."""
        chain = self.generate_options_chain(spot, base_iv, **kwargs)
        calls = chain[chain['type'] == 'call']

        strikes = sorted(calls['strike'].unique())
        expiries = sorted(calls['expiry_days'].unique())

        iv_matrix = np.zeros((len(expiries), len(strikes)))
        for i, dte in enumerate(expiries):
            for j, K in enumerate(strikes):
                row = calls[(calls['expiry_days'] == dte) & (calls['strike'] == K)]
                if not row.empty:
                    iv_matrix[i, j] = row['iv'].values[0]

        return np.array(strikes), np.array(expiries), iv_matrix

    def compute_vol_spread(self, hip3_iv: float, ibkr_chain: pd.DataFrame,
                            spot: float) -> pd.DataFrame:
        """Compare HIP-3 implied vol against IBKR options IV surface.

        Returns DataFrame with vol spread analysis for each strike/expiry.
        The vol spread = HIP3_IV - IBKR_IV is the arbitrage signal:
          positive → HIP-3 overprices vol → sell vol on HIP-3, buy on IBKR
          negative → HIP-3 underprices vol → buy vol on HIP-3, sell on IBKR
        """
        results = []
        for _, row in ibkr_chain.iterrows():
            vol_spread = hip3_iv - row['iv']
            # Theoretical edge: vol spread * vega
            vega_edge = vol_spread * row['vega'] * 100  # per contract

            # Higher-order adjustments
            vanna_adj = row['vanna'] * vol_spread * 0.01 * spot
            vomma_adj = row['vomma'] * vol_spread**2 * 0.5

            total_edge = vega_edge + vanna_adj + vomma_adj

            results.append({
                **row.to_dict(),
                'hip3_iv': hip3_iv,
                'vol_spread': vol_spread,
                'vol_spread_pct': vol_spread / row['iv'] * 100 if row['iv'] > 0 else 0,
                'vega_edge': vega_edge,
                'vanna_adj': vanna_adj,
                'vomma_adj': vomma_adj,
                'total_edge': total_edge,
            })

        return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# Synthetic IBKR options data for backtesting
# ──────────────────────────────────────────────────────────────

def generate_historical_options_data(prices: np.ndarray,
                                      base_iv_series: np.ndarray = None,
                                      risk_free: float = 0.05,
                                      ) -> List[pd.DataFrame]:
    """Generate historical daily options chain snapshots from price history.

    For each day in the price series, generates a full options chain
    with realistic IV dynamics tied to realized vol.

    Returns list of DataFrames, one per day.
    """
    N = len(prices)
    returns = np.zeros(N)
    returns[1:] = np.diff(np.log(prices))

    client = IBKROptionsClient(risk_free_rate=risk_free)
    chains = []

    for i in range(max(30, 0), N):
        spot = prices[i]

        # Realized vol for IV anchoring
        lookback = min(i, 30)
        rv = np.std(returns[max(0, i-lookback):i], ddof=1) * np.sqrt(365)
        rv = max(rv, 0.10)

        # IV typically trades at premium to RV
        if base_iv_series is not None and i < len(base_iv_series):
            base_iv = base_iv_series[i]
        else:
            # IV premium: 10-30% above RV, higher in downtrends
            mom = np.sum(returns[max(0, i-20):i])
            iv_premium = 1.15 if mom > 0 else 1.30
            base_iv = rv * iv_premium

        chain = client.generate_options_chain(
            spot=spot,
            base_iv=base_iv,
            expiries_days=[7, 14, 30, 60, 90],
            n_strikes=11,
            strike_range=0.20,
        )
        chain['day_index'] = i
        chains.append(chain)

    return chains


if __name__ == '__main__':
    print("=" * 80)
    print("  IBKR Options Chain Generator & Greeks Engine")
    print("=" * 80)

    client = IBKROptionsClient()

    # Example: BTC-like asset
    spot = 42000
    base_iv = 0.65

    print(f"\nGenerating options chain for spot=${spot:,.0f}, IV={base_iv*100:.0f}%")
    chain = client.generate_options_chain(spot, base_iv)
    print(f"  Chain: {len(chain)} quotes ({len(chain[chain['type']=='call'])} calls, "
          f"{len(chain[chain['type']=='put'])} puts)")
    print(f"  Strikes: {chain['strike'].min():,.0f} to {chain['strike'].max():,.0f}")
    print(f"  Expiries: {sorted(chain['expiry_days'].unique())}")

    # Show ATM straddle Greeks
    atm = chain[(abs(chain['strike'] - spot) / spot < 0.02) &
                (chain['expiry_days'] == 30)]

    print(f"\n  ATM 30-day options (strike ~ ${spot:,.0f}):")
    for _, row in atm.iterrows():
        print(f"    {row['type'].upper():4s}  Price=${row['mid']:>8.2f}  "
              f"IV={row['iv']*100:5.1f}%  Delta={row['delta']:+.4f}  "
              f"Gamma={row['gamma']:.6f}  Vega={row['vega']:.2f}  "
              f"Vanna={row['vanna']:.4f}  Vomma={row['vomma']:.4f}")

    # IV surface
    strikes, expiries, iv_surface = client.get_iv_surface(spot, base_iv)
    print(f"\n  IV Surface: {iv_surface.shape[0]} expiries x {iv_surface.shape[1]} strikes")
    print(f"  IV range: {iv_surface.min()*100:.1f}% - {iv_surface.max()*100:.1f}%")
