#!/usr/bin/env python3
"""
IV Arbitrage Engine — Compare HIP-3 implied vol vs IBKR options IV.

Core thesis:
  HIP-3 markets on Hyperliquid price volatility differently than traditional
  options on IBKR. This creates persistent arbitrage opportunities:

  1. VOL SPREAD: HIP-3 IV (from funding + spread dynamics) vs IBKR option IV
     → When HIP3_IV > IBKR_IV: sell vol on HIP-3 (short perp, collect funding)
                                buy vol on IBKR (buy straddle/strangle)
     → When HIP3_IV < IBKR_IV: buy vol on HIP-3 (long perp, volatile asset)
                                sell vol on IBKR (sell options)

  2. TERM STRUCTURE ARB: HIP-3 funding rates imply a flat vol term structure,
     while IBKR options have a rich term structure with contango/backwardation.

  3. SKEW ARB: HIP-3 has symmetric vol pricing, while IBKR options exhibit
     put skew (negative skew). Trade the difference.

  4. HIGHER-ORDER GREEK EDGE: Exploit vanna, charm, vomma differences between
     HIP-3 (which doesn't price these) and IBKR options (which does).

Usage:
  from iv_arbitrage_engine import IVArbitrageEngine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ibkr_options_client import IBKROptionsClient, bs_greeks, bs_price, implied_vol


@dataclass
class ArbitrageSignal:
    """A single arbitrage opportunity."""
    asset: str
    signal_type: str       # 'vol_spread', 'term_structure', 'skew', 'greek_edge'
    direction: str         # 'long_vol', 'short_vol', 'long_skew', etc.
    hip3_iv: float
    ibkr_iv: float
    vol_spread: float      # HIP3 - IBKR
    expected_edge_bps: float
    confidence: float      # 0-1
    strategy: str          # e.g., 'sell_straddle_ibkr_long_hip3'
    greeks: Dict[str, float]
    details: str


class IVArbitrageEngine:
    """Engine for finding and sizing IV arbitrage between HIP-3 and IBKR."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
        self.ibkr = IBKROptionsClient(risk_free_rate=risk_free_rate)

    def compute_hip3_implied_vol(self, prices: np.ndarray,
                                   funding_rates: np.ndarray = None,
                                   spreads_bps: np.ndarray = None) -> np.ndarray:
        """Compute time series of HIP-3 implied volatility.

        HIP-3 IV = f(realized_vol, funding_rate, spread_dynamics)

        Args:
            prices: Daily close prices
            funding_rates: 8h funding rates (if available)
            spreads_bps: Bid-ask spreads in bps (if available)
        """
        N = len(prices)
        returns = np.zeros(N)
        returns[1:] = np.diff(np.log(prices))

        hip3_iv = np.zeros(N)

        for i in range(20, N):
            # Component 1: Realized vol (20d rolling)
            rv_20d = np.std(returns[i-20:i], ddof=1) * np.sqrt(365)

            # Component 2: Funding rate premium
            funding_premium = 0.0
            if funding_rates is not None and i < len(funding_rates):
                # Annualized funding → vol premium
                ann_funding = abs(funding_rates[i]) * 3 * 365
                funding_premium = np.sqrt(max(ann_funding, 0)) * 0.5

            # Component 3: Spread-implied vol
            spread_vol = 0.0
            if spreads_bps is not None and i < len(spreads_bps):
                spread_vol = spreads_bps[i] / 10000 * np.sqrt(365) * 0.5

            # Component 4: Vol-of-vol (GARCH-like clustering)
            if i >= 40:
                rv_prev = np.std(returns[i-40:i-20], ddof=1) * np.sqrt(365)
                vol_of_vol = abs(rv_20d - rv_prev) / max(rv_prev, 0.01)
            else:
                vol_of_vol = 0

            # HIP-3 IV: weighted blend with perp-market adjustments
            components = [rv_20d * 1.10]  # RV with base premium
            if funding_premium > 0:
                components.append(rv_20d + funding_premium)
            if spread_vol > 0:
                components.append(spread_vol)

            hip3_iv[i] = np.mean(components) * (1 + 0.25 * vol_of_vol)
            hip3_iv[i] = max(hip3_iv[i], 0.05)  # Floor at 5%

        # Fill first 20 bars
        if hip3_iv[20] > 0:
            hip3_iv[:20] = hip3_iv[20]

        return hip3_iv

    def compute_ibkr_atm_iv(self, prices: np.ndarray,
                              base_iv_premium: float = 1.10) -> np.ndarray:
        """Compute time series of IBKR ATM IV for comparison.

        IBKR options markets typically:
        - Trade at a premium to RV (variance risk premium)
        - Show mean-reversion in IV
        - Have richer term structure dynamics
        """
        N = len(prices)
        returns = np.zeros(N)
        returns[1:] = np.diff(np.log(prices))

        ibkr_iv = np.zeros(N)

        for i in range(20, N):
            # Realized vol
            rv_20d = np.std(returns[i-20:i], ddof=1) * np.sqrt(365)

            # IBKR IV: variance risk premium + mean reversion
            # VRP: options systematically overprice vol
            vrp_mult = base_iv_premium

            # Mean reversion: IV mean-reverts to long-term average
            if i >= 60:
                rv_60d = np.std(returns[i-60:i], ddof=1) * np.sqrt(365)
                # Blend short-term and long-term
                ibkr_iv[i] = (0.6 * rv_20d + 0.4 * rv_60d) * vrp_mult
            else:
                ibkr_iv[i] = rv_20d * vrp_mult

            # Add skew effect: IV higher when market falling
            mom_20d = np.sum(returns[max(0, i-20):i])
            if mom_20d < -0.05:
                ibkr_iv[i] *= 1.15  # Fear premium
            elif mom_20d > 0.10:
                ibkr_iv[i] *= 0.95  # Complacency discount

            ibkr_iv[i] = max(ibkr_iv[i], 0.08)

        if ibkr_iv[20] > 0:
            ibkr_iv[:20] = ibkr_iv[20]

        return ibkr_iv

    def find_vol_spread_opportunities(self, hip3_iv: np.ndarray,
                                        ibkr_iv: np.ndarray,
                                        prices: np.ndarray,
                                        threshold_pct: float = 5.0
                                        ) -> pd.DataFrame:
        """Find vol spread arbitrage opportunities over time.

        Args:
            hip3_iv: HIP-3 implied vol series
            ibkr_iv: IBKR ATM implied vol series
            prices: Spot price series
            threshold_pct: Min vol spread (%) to flag as opportunity
        """
        N = min(len(hip3_iv), len(ibkr_iv), len(prices))
        vol_spread = hip3_iv[:N] - ibkr_iv[:N]
        vol_spread_pct = np.where(ibkr_iv[:N] > 0,
                                   vol_spread / ibkr_iv[:N] * 100, 0)

        opportunities = []
        for i in range(30, N):
            spread_pct = vol_spread_pct[i]
            if abs(spread_pct) > threshold_pct:
                direction = 'short_hip3_vol' if spread_pct > 0 else 'long_hip3_vol'
                edge = abs(vol_spread[i])

                # Compute option Greeks at this point
                spot = prices[i]
                atm_quote = bs_greeks(spot, spot, 30/365, self.r, ibkr_iv[i], 'call')

                opportunities.append({
                    'day': i,
                    'spot': spot,
                    'hip3_iv': hip3_iv[i],
                    'ibkr_iv': ibkr_iv[i],
                    'vol_spread': vol_spread[i],
                    'vol_spread_pct': spread_pct,
                    'direction': direction,
                    'edge_vol_pts': edge,
                    'vega_edge': edge * atm_quote.vega * 100,
                    'vanna_edge': atm_quote.vanna * vol_spread[i] * 0.01 * spot,
                    'vomma_edge': atm_quote.vomma * vol_spread[i]**2 * 0.5,
                    'gamma': atm_quote.gamma,
                    'theta': atm_quote.theta,
                })

        return pd.DataFrame(opportunities)

    def compute_term_structure_arb(self, prices: np.ndarray,
                                     hip3_iv: np.ndarray) -> pd.DataFrame:
        """Find term structure arbitrage: flat HIP-3 vs rich IBKR term structure.

        HIP-3 implies a single (flat) vol level from funding rates.
        IBKR options have different IVs at different expirations.
        The spread between near-term and far-term IV on IBKR,
        compared to flat HIP-3 IV, creates calendar spread opportunities.
        """
        results = []
        for i in range(60, len(prices)):
            spot = prices[i]
            h_iv = hip3_iv[i]

            # Generate IBKR chain at this point
            chain = self.ibkr.generate_options_chain(
                spot=spot, base_iv=h_iv * 0.95,  # IBKR slightly below HIP-3
                expiries_days=[7, 30, 90, 180],
                n_strikes=5, strike_range=0.05,
            )

            # ATM calls only
            atm = chain[(abs(chain['strike'] - spot) / spot < 0.03) &
                         (chain['type'] == 'call')]

            if len(atm) >= 2:
                ivs = atm.groupby('expiry_days')['iv'].mean()
                if len(ivs) >= 2:
                    short_term = ivs.iloc[0]
                    long_term = ivs.iloc[-1]
                    term_spread = long_term - short_term
                    hip3_vs_short = h_iv - short_term
                    hip3_vs_long = h_iv - long_term

                    results.append({
                        'day': i,
                        'spot': spot,
                        'hip3_iv': h_iv,
                        'ibkr_short_iv': short_term,
                        'ibkr_long_iv': long_term,
                        'ibkr_term_spread': term_spread,
                        'hip3_vs_short': hip3_vs_short,
                        'hip3_vs_long': hip3_vs_long,
                        'calendar_edge': abs(term_spread) * 100,
                    })

        return pd.DataFrame(results)

    def compute_skew_arb(self, prices: np.ndarray,
                          hip3_iv: np.ndarray) -> pd.DataFrame:
        """Find skew arbitrage: symmetric HIP-3 vs skewed IBKR.

        HIP-3 prices vol symmetrically (same for longs and shorts).
        IBKR options have negative skew (puts more expensive than calls).
        This creates risk reversal opportunities.
        """
        results = []
        for i in range(60, len(prices), 5):  # Sample every 5 days
            spot = prices[i]
            h_iv = hip3_iv[i]

            chain = self.ibkr.generate_options_chain(
                spot=spot, base_iv=h_iv * 0.95,
                expiries_days=[30],
                n_strikes=11, strike_range=0.15,
                iv_skew=-0.15,  # Negative put skew
            )

            calls = chain[chain['type'] == 'call']
            puts = chain[chain['type'] == 'put']

            if len(calls) > 0 and len(puts) > 0:
                # 25-delta risk reversal
                otm_put = puts[puts['delta'].abs() < 0.30].sort_values('delta', ascending=False).head(1)
                otm_call = calls[calls['delta'] < 0.30].sort_values('delta', ascending=False).head(1)

                if len(otm_put) > 0 and len(otm_call) > 0:
                    put_iv = otm_put['iv'].values[0]
                    call_iv = otm_call['iv'].values[0]
                    skew = put_iv - call_iv  # Positive = put skew

                    # HIP-3 has no skew → symmetric
                    hip3_skew = 0  # HIP-3 doesn't differentiate up/down vol

                    skew_edge = skew - hip3_skew

                    results.append({
                        'day': i,
                        'spot': spot,
                        'hip3_iv': h_iv,
                        'ibkr_put_iv': put_iv,
                        'ibkr_call_iv': call_iv,
                        'ibkr_skew': skew,
                        'hip3_skew': hip3_skew,
                        'skew_edge': skew_edge,
                        'put_delta': otm_put['delta'].values[0],
                        'call_delta': otm_call['delta'].values[0],
                    })

        return pd.DataFrame(results)

    def full_arbitrage_scan(self, asset: str, prices: np.ndarray,
                             funding_rates: np.ndarray = None,
                             spreads_bps: np.ndarray = None
                             ) -> Dict[str, pd.DataFrame]:
        """Run complete arbitrage analysis for one asset.

        Returns dict with DataFrames for each arb type.
        """
        hip3_iv = self.compute_hip3_implied_vol(prices, funding_rates, spreads_bps)
        ibkr_iv = self.compute_ibkr_atm_iv(prices)

        return {
            'hip3_iv': hip3_iv,
            'ibkr_iv': ibkr_iv,
            'vol_spread': self.find_vol_spread_opportunities(hip3_iv, ibkr_iv, prices),
            'term_structure': self.compute_term_structure_arb(prices, hip3_iv),
            'skew': self.compute_skew_arb(prices, hip3_iv),
        }


if __name__ == '__main__':
    print("=" * 80)
    print("  IV Arbitrage Engine — HIP-3 vs IBKR")
    print("=" * 80)

    # Generate sample data
    from hyperliquid_hip3_client import generate_synthetic_hip3_data
    data = generate_synthetic_hip3_data(n_assets=3, n_days=365)

    engine = IVArbitrageEngine()

    for asset, df in data.items():
        print(f"\n{'─'*60}")
        print(f"  {asset}")
        print(f"{'─'*60}")

        prices = df['close'].values
        result = engine.full_arbitrage_scan(asset, prices)

        vol_opps = result['vol_spread']
        if not vol_opps.empty:
            print(f"  Vol spread opportunities: {len(vol_opps)}")
            print(f"  Avg vol spread: {vol_opps['vol_spread'].mean()*100:.1f}%")
            print(f"  Max edge (vega): {vol_opps['vega_edge'].max():.2f}")

        term = result['term_structure']
        if not term.empty:
            print(f"  Term structure arb signals: {len(term)}")

        skew = result['skew']
        if not skew.empty:
            print(f"  Skew arb signals: {len(skew)}")
            print(f"  Avg IBKR skew: {skew['ibkr_skew'].mean()*100:.1f}%")
