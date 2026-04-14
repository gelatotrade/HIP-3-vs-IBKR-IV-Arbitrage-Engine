#!/usr/bin/env python3
"""
Second & Third Order Greek Options Strategies.

Exploits the fact that HIP-3 markets do NOT price higher-order Greeks,
while IBKR options markets DO. This creates persistent edges in:

SECOND ORDER:
  1. Gamma Scalping — Long gamma on IBKR, hedge delta on HIP-3
     Edge: HIP-3 doesn't charge for gamma convexity
  2. Vanna Trade — Exploit ∂δ/∂σ: when vol moves, delta shifts
     Edge: HIP-3 funding doesn't adjust for cross-gamma
  3. Charm Trade — Exploit ∂δ/∂t: delta decay creates drift
     Edge: HIP-3 positions don't experience charm bleed
  4. Vomma Trade — Exploit vol-of-vol: vega convexity
     Edge: HIP-3 linear vol pricing vs IBKR's convex vega

THIRD ORDER:
  5. Speed Trade — Exploit ∂Γ/∂S: gamma acceleration
     Edge: Large spot moves → gamma changes → non-linear P&L
  6. Color Trade — Exploit ∂Γ/∂t: gamma decay near expiry
     Edge: Weekly options gamma collapse vs HIP-3 constant risk
  7. Zomma Trade — Exploit ∂Γ/∂σ: gamma-vol interaction
     Edge: Vol spikes change gamma profile → exploit on HIP-3

Usage:  from greeks_strategies import GreeksStrategyEngine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ibkr_options_client import IBKROptionsClient, bs_greeks, bs_price


@dataclass
class StrategyResult:
    """Result from a single strategy backtest."""
    name: str
    daily_pnl: np.ndarray
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_trade_pnl: float
    n_trades: int
    details: Dict


# ──────────────────────────────────────────────────────────────
# Regime detection (same as original repo)
# ──────────────────────────────────────────────────────────────
def classify_regime(returns, idx, crisis_vol=0.80):
    """Classify regime for crypto assets (higher vol thresholds)."""
    if idx < 40:
        return 'NORMAL'
    w = returns[max(0, idx-20):idx]
    vol = np.std(w, ddof=1) * np.sqrt(365)
    mom = np.sum(w)
    w_prev = returns[max(0, idx-40):max(0, idx-20)]
    vol_prev = np.std(w_prev, ddof=1) * np.sqrt(365) if len(w_prev) > 2 else vol

    crisis_thresh = crisis_vol + 0.30
    cautious_thresh = crisis_vol
    if vol > crisis_thresh and mom < -0.10:
        return 'CRISIS'
    if vol > cautious_thresh and mom < 0:
        return 'CAUTIOUS'
    if vol_prev > cautious_thresh + 0.15 and vol < vol_prev * 0.85 and mom > 0:
        return 'RECOVERY'
    if vol < 0.50 and mom > 0.03:
        return 'BULL'
    return 'NORMAL'


class GreeksStrategyEngine:
    """Engine for higher-order Greek arbitrage strategies."""

    def __init__(self, risk_free: float = 0.05):
        self.r = risk_free
        self.ibkr = IBKROptionsClient(risk_free_rate=risk_free)

    # ════════════════════════════════════════════════════════
    # STRATEGY 1: Gamma Scalping
    # ════════════════════════════════════════════════════════
    def gamma_scalping(self, prices: np.ndarray, hip3_iv: np.ndarray,
                        ibkr_iv: np.ndarray,
                        rebalance_threshold: float = 0.02,
                        option_dte: int = 30) -> StrategyResult:
        """
        Long gamma on IBKR (buy straddle), delta-hedge on HIP-3.

        Logic:
        - Buy ATM straddle on IBKR → long gamma
        - Delta-hedge on HIP-3 perp → neutralize delta
        - As spot moves, gamma creates P&L from rebalancing
        - Edge: theta cost on IBKR < gamma P&L from HIP-3 vol

        The edge exists because HIP-3 realizes vol at a different rate
        than IBKR options price it. If realized_vol > IBKR_IV, gamma
        scalping is profitable.
        """
        N = len(prices)
        returns = np.zeros(N)
        returns[1:] = np.diff(np.log(prices))

        daily_pnl = np.zeros(N)
        n_trades = 0
        total_gamma_pnl = 0.0
        total_theta_cost = 0.0

        # Position state
        has_position = False
        entry_day = 0
        straddle_cost = 0.0
        hedge_delta = 0.0
        current_strike = 0.0

        for i in range(max(30, 0), N):
            spot = prices[i]
            rv_20d = np.std(returns[max(0, i-20):i], ddof=1) * np.sqrt(365) if i >= 20 else 0.5
            regime = classify_regime(returns, i)

            # Entry: when HIP-3 vol > IBKR vol (realized vol likely > implied)
            if not has_position and i >= 40:
                vol_spread = hip3_iv[i] - ibkr_iv[i]
                if vol_spread > 0.03 and regime not in ('CRISIS',):
                    # Buy ATM straddle on IBKR
                    current_strike = spot
                    T = option_dte / 365.0
                    call = bs_greeks(spot, current_strike, T, self.r, ibkr_iv[i], 'call')
                    put = bs_greeks(spot, current_strike, T, self.r, ibkr_iv[i], 'put')

                    straddle_cost = (call.price + put.price) / spot
                    hedge_delta = -(call.delta + put.delta)  # hedge on HIP-3
                    has_position = True
                    entry_day = i
                    n_trades += 1

            elif has_position:
                days_held = i - entry_day
                T_remaining = max((option_dte - days_held) / 365.0, 1/365)

                # Current Greeks
                call = bs_greeks(spot, current_strike, T_remaining, self.r, ibkr_iv[i], 'call')
                put = bs_greeks(spot, current_strike, T_remaining, self.r, ibkr_iv[i], 'put')

                total_delta = call.delta + put.delta + hedge_delta
                total_gamma = call.gamma + put.gamma
                total_theta = call.theta + put.theta

                # Gamma P&L from spot movement
                spot_move = returns[i]
                gamma_pnl = 0.5 * total_gamma * (spot * spot_move)**2 / spot
                theta_cost = total_theta  # daily theta decay

                # Rebalance delta hedge
                if abs(total_delta) > rebalance_threshold:
                    hedge_delta = -(call.delta + put.delta)
                    rebal_cost = abs(total_delta) * 0.0005  # slippage
                else:
                    rebal_cost = 0

                daily_pnl[i] = gamma_pnl + theta_cost - rebal_cost
                total_gamma_pnl += gamma_pnl
                total_theta_cost += abs(theta_cost)

                # Exit: option near expiry or vol spread closed
                if days_held >= option_dte - 2 or regime == 'CRISIS':
                    has_position = False
                    hedge_delta = 0

        return StrategyResult(
            name='Gamma Scalping',
            daily_pnl=daily_pnl,
            total_return=np.sum(daily_pnl),
            sharpe=self._sharpe(daily_pnl),
            max_drawdown=self._max_dd(daily_pnl),
            win_rate=np.mean(daily_pnl[daily_pnl != 0] > 0) if np.any(daily_pnl != 0) else 0,
            avg_trade_pnl=np.sum(daily_pnl) / max(n_trades, 1),
            n_trades=n_trades,
            details={
                'total_gamma_pnl': total_gamma_pnl,
                'total_theta_cost': total_theta_cost,
                'gamma_theta_ratio': total_gamma_pnl / max(total_theta_cost, 1e-10),
            }
        )

    # ════════════════════════════════════════════════════════
    # STRATEGY 2: Vanna Trade
    # ════════════════════════════════════════════════════════
    def vanna_trade(self, prices: np.ndarray, hip3_iv: np.ndarray,
                     ibkr_iv: np.ndarray) -> StrategyResult:
        """
        Exploit vanna (∂δ/∂σ) — delta changes when vol changes.

        When vol rises: OTM puts gain more delta (vanna effect).
        HIP-3 doesn't price this cross-sensitivity, so:
        - Buy OTM puts on IBKR (positive vanna exposure)
        - Hedge with HIP-3 perp long
        - When vol spikes: put delta increases → gain from hedge adjustment

        Edge: HIP-3 perp funding doesn't reflect vanna risk.
        """
        N = len(prices)
        returns = np.zeros(N)
        returns[1:] = np.diff(np.log(prices))

        daily_pnl = np.zeros(N)
        n_trades = 0

        for i in range(60, N):
            spot = prices[i]
            regime = classify_regime(returns, i)

            # Vanna signal: expect vol to change
            if i >= 40:
                rv_recent = np.std(returns[i-10:i], ddof=1) * np.sqrt(365)
                rv_older = np.std(returns[i-30:i-10], ddof=1) * np.sqrt(365)
                vol_momentum = rv_recent - rv_older

                # Buy vanna when vol is rising (positive vol momentum)
                T = 30 / 365
                otm_put_strike = spot * 0.90  # 10% OTM put
                put = bs_greeks(spot, otm_put_strike, T, self.r, ibkr_iv[i], 'put')

                vanna_exposure = put.vanna
                vol_change = (hip3_iv[i] - hip3_iv[max(0, i-1)]) if i > 0 else 0

                # P&L = vanna * dVol * dSpot (cross-gamma effect)
                spot_change = returns[i]
                vanna_pnl = vanna_exposure * vol_change * spot_change * spot * 0.01

                # HIP-3 hedge: long perp at base position
                hip3_hedge_pnl = -put.delta * returns[i]  # delta hedge P&L

                # Funding cost on HIP-3
                funding_cost = abs(put.delta) * 0.0001  # ~1bps daily funding

                daily_pnl[i] = vanna_pnl + hip3_hedge_pnl - funding_cost

                if abs(vanna_pnl) > 0.0001:
                    n_trades += 1

        return StrategyResult(
            name='Vanna Trade',
            daily_pnl=daily_pnl,
            total_return=np.sum(daily_pnl),
            sharpe=self._sharpe(daily_pnl),
            max_drawdown=self._max_dd(daily_pnl),
            win_rate=np.mean(daily_pnl[daily_pnl != 0] > 0) if np.any(daily_pnl != 0) else 0,
            avg_trade_pnl=np.sum(daily_pnl) / max(n_trades, 1),
            n_trades=n_trades,
            details={'strategy': 'vanna_otm_put_hedge'}
        )

    # ════════════════════════════════════════════════════════
    # STRATEGY 3: Charm Trade (Delta Bleed)
    # ════════════════════════════════════════════════════════
    def charm_trade(self, prices: np.ndarray, hip3_iv: np.ndarray,
                     ibkr_iv: np.ndarray) -> StrategyResult:
        """
        Exploit charm (∂δ/∂t) — delta changes with time passage.

        As options approach expiry, delta drifts toward 0 or 1.
        This "delta bleed" is predictable and exploitable:
        - Sell near-expiry options on IBKR (short charm position)
        - Hedge delta on HIP-3
        - As charm pushes delta, adjust hedge → capture drift profit

        Edge: HIP-3 has no time-decay equivalent, so delta hedging
        on HIP-3 is cheaper than the charm benefit from IBKR options.
        """
        N = len(prices)
        returns = np.zeros(N)
        returns[1:] = np.diff(np.log(prices))

        daily_pnl = np.zeros(N)
        n_trades = 0

        for i in range(60, N):
            spot = prices[i]
            regime = classify_regime(returns, i)

            # Use 7-day options for maximum charm
            T = 7 / 365
            atm_call = bs_greeks(spot, spot, T, self.r, ibkr_iv[i], 'call')

            # Charm effect: predictable delta drift
            charm_value = atm_call.charm

            # Sell the option (short charm = profit from delta drift)
            # Hedge delta on HIP-3
            theta_income = -atm_call.theta  # collect theta
            charm_pnl = -charm_value * (1/365) * spot * 0.001  # charm drift P&L

            # Gamma risk (cost of being short gamma)
            gamma_cost = 0.5 * atm_call.gamma * (spot * returns[i])**2 / spot * 0.5

            # Net P&L: theta + charm - gamma_cost
            size_mult = 0.3 if regime in ('CAUTIOUS', 'CRISIS') else 1.0
            daily_pnl[i] = (theta_income + charm_pnl - gamma_cost) * size_mult

            if abs(daily_pnl[i]) > 1e-6:
                n_trades += 1

        return StrategyResult(
            name='Charm Trade',
            daily_pnl=daily_pnl,
            total_return=np.sum(daily_pnl),
            sharpe=self._sharpe(daily_pnl),
            max_drawdown=self._max_dd(daily_pnl),
            win_rate=np.mean(daily_pnl[daily_pnl != 0] > 0) if np.any(daily_pnl != 0) else 0,
            avg_trade_pnl=np.sum(daily_pnl) / max(n_trades, 1),
            n_trades=n_trades,
            details={'strategy': 'short_charm_weekly_options'}
        )

    # ════════════════════════════════════════════════════════
    # STRATEGY 4: Vomma Trade (Vol of Vol)
    # ════════════════════════════════════════════════════════
    def vomma_trade(self, prices: np.ndarray, hip3_iv: np.ndarray,
                     ibkr_iv: np.ndarray) -> StrategyResult:
        """
        Exploit vomma (∂ν/∂σ) — vega convexity.

        Vomma is positive for OTM options: vega increases as vol rises.
        HIP-3 pricing is linear in vol. IBKR options have convex vega.
        Strategy:
        - When vomma is high and vol is mean-reverting upward:
          Buy OTM options on IBKR (long vomma)
          Sell vol on HIP-3 (short vega, linear)
        - Net: long the convexity gap

        Edge: crypto vol tends to cluster → vomma captures
        the non-linear payoff from vol spikes.
        """
        N = len(prices)
        returns = np.zeros(N)
        returns[1:] = np.diff(np.log(prices))

        daily_pnl = np.zeros(N)
        n_trades = 0

        for i in range(60, N):
            spot = prices[i]
            regime = classify_regime(returns, i)

            T = 30 / 365
            otm_call_strike = spot * 1.10
            otm_call = bs_greeks(spot, otm_call_strike, T, self.r, ibkr_iv[i], 'call')

            # Vol change
            vol_change = hip3_iv[i] - hip3_iv[max(0, i-1)] if i > 0 else 0

            # Vomma P&L: vomma * (vol_change)^2 / 2
            vomma_pnl = 0.5 * otm_call.vomma * vol_change**2

            # Linear vega from HIP-3 short (sell vol on HIP-3)
            hip3_vega_pnl = -otm_call.vega * vol_change * 100

            # Net: convexity gain - linear cost
            # When vol moves a lot, vomma wins
            net_pnl = vomma_pnl + hip3_vega_pnl * 0.01

            # Theta cost of long options
            theta_cost = otm_call.theta

            size = 0.5 if regime == 'CRISIS' else 1.0
            daily_pnl[i] = (net_pnl + theta_cost) * size * 0.1

            if abs(daily_pnl[i]) > 1e-6:
                n_trades += 1

        return StrategyResult(
            name='Vomma Trade',
            daily_pnl=daily_pnl,
            total_return=np.sum(daily_pnl),
            sharpe=self._sharpe(daily_pnl),
            max_drawdown=self._max_dd(daily_pnl),
            win_rate=np.mean(daily_pnl[daily_pnl != 0] > 0) if np.any(daily_pnl != 0) else 0,
            avg_trade_pnl=np.sum(daily_pnl) / max(n_trades, 1),
            n_trades=n_trades,
            details={'strategy': 'long_vomma_short_linear_vega'}
        )

    # ════════════════════════════════════════════════════════
    # STRATEGY 5: Speed Trade (Third Order)
    # ════════════════════════════════════════════════════════
    def speed_trade(self, prices: np.ndarray, hip3_iv: np.ndarray,
                     ibkr_iv: np.ndarray) -> StrategyResult:
        """
        Exploit speed (∂Γ/∂S) — gamma acceleration on large moves.

        Speed measures how gamma changes with spot. On large moves:
        - Gamma increases for near-ATM options
        - Creates convex payoff that HIP-3 perps don't capture
        Strategy:
        - Buy butterfly spreads on IBKR (peak gamma at center)
        - Hedge on HIP-3
        - Profit from gamma changes on large spot moves
        """
        N = len(prices)
        returns = np.zeros(N)
        returns[1:] = np.diff(np.log(prices))

        daily_pnl = np.zeros(N)
        n_trades = 0

        for i in range(60, N):
            spot = prices[i]
            regime = classify_regime(returns, i)

            T = 14 / 365  # Short-dated for maximum speed
            # Butterfly: buy K-5%, sell 2x ATM, buy K+5%
            k_low = spot * 0.95
            k_mid = spot
            k_high = spot * 1.05

            call_low = bs_greeks(spot, k_low, T, self.r, ibkr_iv[i], 'call')
            call_mid = bs_greeks(spot, k_mid, T, self.r, ibkr_iv[i], 'call')
            call_high = bs_greeks(spot, k_high, T, self.r, ibkr_iv[i], 'call')

            # Butterfly Greeks
            fly_gamma = call_low.gamma - 2*call_mid.gamma + call_high.gamma
            fly_speed = call_low.speed - 2*call_mid.speed + call_high.speed
            fly_theta = call_low.theta - 2*call_mid.theta + call_high.theta

            # Speed P&L from large moves
            move = spot * returns[i]
            speed_pnl = fly_speed * move**3 / 6  # Third order Taylor
            gamma_pnl = 0.5 * fly_gamma * move**2
            theta = fly_theta

            size = 0.2 if regime == 'CRISIS' else 0.8
            daily_pnl[i] = (speed_pnl + gamma_pnl + theta) * size * 0.05

            if abs(daily_pnl[i]) > 1e-7:
                n_trades += 1

        return StrategyResult(
            name='Speed Trade',
            daily_pnl=daily_pnl,
            total_return=np.sum(daily_pnl),
            sharpe=self._sharpe(daily_pnl),
            max_drawdown=self._max_dd(daily_pnl),
            win_rate=np.mean(daily_pnl[daily_pnl != 0] > 0) if np.any(daily_pnl != 0) else 0,
            avg_trade_pnl=np.sum(daily_pnl) / max(n_trades, 1),
            n_trades=n_trades,
            details={'strategy': 'butterfly_speed_trade'}
        )

    # ════════════════════════════════════════════════════════
    # STRATEGY 6: Color Trade (Gamma Decay)
    # ════════════════════════════════════════════════════════
    def color_trade(self, prices: np.ndarray, hip3_iv: np.ndarray,
                     ibkr_iv: np.ndarray) -> StrategyResult:
        """
        Exploit color (∂Γ/∂t) — gamma decay near expiry.

        Near expiry, ATM gamma spikes then collapses (pin risk).
        HIP-3 perps have no equivalent decay dynamic.
        Strategy:
        - Sell gamma (short straddle) on IBKR near expiry (1-3 DTE)
        - The gamma is decaying → color is negative for ATM
        - Hedge delta on HIP-3 where there's no gamma decay cost
        """
        N = len(prices)
        returns = np.zeros(N)
        returns[1:] = np.diff(np.log(prices))

        daily_pnl = np.zeros(N)
        n_trades = 0

        for i in range(60, N):
            spot = prices[i]
            regime = classify_regime(returns, i)

            # Short near-expiry straddle (2 DTE)
            T = 2 / 365
            call = bs_greeks(spot, spot, T, self.r, ibkr_iv[i], 'call')
            put = bs_greeks(spot, spot, T, self.r, ibkr_iv[i], 'put')

            # Color: rate of gamma decay
            total_color = call.color + put.color
            total_gamma = call.gamma + put.gamma
            total_theta = call.theta + put.theta

            # Short straddle: collect theta, pay gamma, benefit from color
            theta_income = -(total_theta)  # collect (theta is negative for long)
            gamma_cost = 0.5 * total_gamma * (spot * returns[i])**2 / spot
            color_benefit = -total_color * (1/365) * 0.001  # gamma decays in our favor

            # Regime sizing
            size = 0.1 if regime in ('CRISIS', 'CAUTIOUS') else 0.5

            daily_pnl[i] = (theta_income - gamma_cost + color_benefit) * size

            if abs(daily_pnl[i]) > 1e-6:
                n_trades += 1

        return StrategyResult(
            name='Color Trade',
            daily_pnl=daily_pnl,
            total_return=np.sum(daily_pnl),
            sharpe=self._sharpe(daily_pnl),
            max_drawdown=self._max_dd(daily_pnl),
            win_rate=np.mean(daily_pnl[daily_pnl != 0] > 0) if np.any(daily_pnl != 0) else 0,
            avg_trade_pnl=np.sum(daily_pnl) / max(n_trades, 1),
            n_trades=n_trades,
            details={'strategy': 'short_straddle_color_decay'}
        )

    # ════════════════════════════════════════════════════════
    # STRATEGY 7: Zomma Trade (Gamma-Vol)
    # ════════════════════════════════════════════════════════
    def zomma_trade(self, prices: np.ndarray, hip3_iv: np.ndarray,
                     ibkr_iv: np.ndarray) -> StrategyResult:
        """
        Exploit zomma (∂Γ/∂σ) — gamma sensitivity to vol.

        When vol rises, gamma profile changes (peak shifts).
        HIP-3 perps don't have gamma-vol interaction.
        Strategy:
        - Buy OTM strangles on IBKR (long zomma)
        - When vol spikes: gamma increases → more convexity
        - Delta hedge on HIP-3 perp
        """
        N = len(prices)
        returns = np.zeros(N)
        returns[1:] = np.diff(np.log(prices))

        daily_pnl = np.zeros(N)
        n_trades = 0

        for i in range(60, N):
            spot = prices[i]
            regime = classify_regime(returns, i)

            T = 30 / 365
            otm_call = bs_greeks(spot, spot*1.08, T, self.r, ibkr_iv[i], 'call')
            otm_put = bs_greeks(spot, spot*0.92, T, self.r, ibkr_iv[i], 'put')

            # Strangle zomma
            total_zomma = otm_call.zomma + otm_put.zomma
            total_gamma = otm_call.gamma + otm_put.gamma
            total_theta = otm_call.theta + otm_put.theta

            # Vol change
            vol_change = hip3_iv[i] - hip3_iv[max(0, i-1)] if i > 0 else 0

            # Zomma P&L: gamma changes when vol moves
            zomma_pnl = total_zomma * vol_change * (spot * returns[i])**2 / (2 * spot)
            gamma_pnl = 0.5 * total_gamma * (spot * returns[i])**2 / spot
            theta = total_theta

            size = 0.3 if regime == 'CRISIS' else 0.8
            daily_pnl[i] = (zomma_pnl + gamma_pnl + theta) * size * 0.05

            if abs(daily_pnl[i]) > 1e-7:
                n_trades += 1

        return StrategyResult(
            name='Zomma Trade',
            daily_pnl=daily_pnl,
            total_return=np.sum(daily_pnl),
            sharpe=self._sharpe(daily_pnl),
            max_drawdown=self._max_dd(daily_pnl),
            win_rate=np.mean(daily_pnl[daily_pnl != 0] > 0) if np.any(daily_pnl != 0) else 0,
            avg_trade_pnl=np.sum(daily_pnl) / max(n_trades, 1),
            n_trades=n_trades,
            details={'strategy': 'long_strangle_zomma'}
        )

    # ════════════════════════════════════════════════════════
    # Combined strategy: Weighted ensemble
    # ════════════════════════════════════════════════════════
    def run_all_strategies(self, prices: np.ndarray,
                            hip3_iv: np.ndarray,
                            ibkr_iv: np.ndarray) -> Dict[str, StrategyResult]:
        """Run all 7 strategies and return results dict."""
        strategies = {
            'gamma_scalping': self.gamma_scalping(prices, hip3_iv, ibkr_iv),
            'vanna_trade': self.vanna_trade(prices, hip3_iv, ibkr_iv),
            'charm_trade': self.charm_trade(prices, hip3_iv, ibkr_iv),
            'vomma_trade': self.vomma_trade(prices, hip3_iv, ibkr_iv),
            'speed_trade': self.speed_trade(prices, hip3_iv, ibkr_iv),
            'color_trade': self.color_trade(prices, hip3_iv, ibkr_iv),
            'zomma_trade': self.zomma_trade(prices, hip3_iv, ibkr_iv),
        }
        return strategies

    def ensemble_strategy(self, strategies: Dict[str, StrategyResult],
                           weights: Dict[str, float] = None) -> np.ndarray:
        """Combine strategies into weighted ensemble."""
        if weights is None:
            # Weight by Sharpe ratio (risk-parity like)
            weights = {}
            total_sharpe = sum(max(s.sharpe, 0.01) for s in strategies.values())
            for name, s in strategies.items():
                weights[name] = max(s.sharpe, 0.01) / total_sharpe

        # Combine daily P&Ls
        first_key = next(iter(strategies))
        N = len(strategies[first_key].daily_pnl)
        ensemble_pnl = np.zeros(N)

        for name, s in strategies.items():
            w = weights.get(name, 1 / len(strategies))
            ensemble_pnl += w * s.daily_pnl

        return ensemble_pnl

    # ── Utilities ──
    @staticmethod
    def _sharpe(pnl: np.ndarray) -> float:
        active = pnl[pnl != 0]
        if len(active) < 30:
            return 0.0
        mu = np.mean(active) * 365
        sig = np.std(active, ddof=1) * np.sqrt(365)
        return mu / sig if sig > 1e-10 else 0.0

    @staticmethod
    def _max_dd(pnl: np.ndarray) -> float:
        cum = np.cumsum(pnl)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return dd.max() if len(dd) > 0 else 0.0


if __name__ == '__main__':
    print("=" * 80)
    print("  Higher-Order Greeks Strategies — HIP-3 vs IBKR Arbitrage")
    print("=" * 80)

    from hyperliquid_hip3_client import generate_synthetic_hip3_data
    from iv_arbitrage_engine import IVArbitrageEngine

    data = generate_synthetic_hip3_data(n_assets=3, n_days=500)
    arb_engine = IVArbitrageEngine()
    greeks_engine = GreeksStrategyEngine()

    for asset, df in data.items():
        print(f"\n{'═'*60}")
        print(f"  {asset}")
        print(f"{'═'*60}")

        prices = df['close'].values
        hip3_iv = arb_engine.compute_hip3_implied_vol(prices)
        ibkr_iv = arb_engine.compute_ibkr_atm_iv(prices)

        results = greeks_engine.run_all_strategies(prices, hip3_iv, ibkr_iv)

        print(f"\n  {'Strategy':<20s} {'Return':>8s} {'Sharpe':>8s} {'MaxDD':>8s} "
              f"{'WinRate':>8s} {'Trades':>7s}")
        print(f"  {'─'*60}")

        for name, r in results.items():
            print(f"  {r.name:<20s} {r.total_return*100:>+7.2f}% {r.sharpe:>8.3f} "
                  f"{r.max_drawdown*100:>7.2f}% {r.win_rate*100:>7.1f}% {r.n_trades:>7d}")

        # Ensemble
        ensemble_pnl = greeks_engine.ensemble_strategy(results)
        ens_ret = np.sum(ensemble_pnl)
        ens_sharpe = greeks_engine._sharpe(ensemble_pnl)
        print(f"\n  {'Ensemble':<20s} {ens_ret*100:>+7.2f}% {ens_sharpe:>8.3f}")
