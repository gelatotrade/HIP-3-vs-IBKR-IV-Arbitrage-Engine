"""Tests for Black-Scholes pricing and Greeks."""

import numpy as np
from ibkr_options_client import OptionQuote, bs_greeks, bs_price, implied_vol


class TestBSPrice:
    def test_call_atm(self):
        price = bs_price(100, 100, 1.0, 0.05, 0.20, "call")
        assert 8.0 < price < 12.0

    def test_put_atm(self):
        price = bs_price(100, 100, 1.0, 0.05, 0.20, "put")
        assert 3.0 < price < 8.0

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        c = bs_price(S, K, T, r, sigma, "call")
        p = bs_price(S, K, T, r, sigma, "put")
        assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-10

    def test_deep_itm_call(self):
        price = bs_price(200, 100, 0.5, 0.05, 0.20, "call")
        assert price > 95

    def test_deep_otm_call(self):
        price = bs_price(50, 100, 0.5, 0.05, 0.20, "call")
        assert price < 1.0

    def test_expired_call_itm(self):
        assert bs_price(110, 100, 0, 0.05, 0.20, "call") == 10.0

    def test_expired_put_otm(self):
        assert bs_price(110, 100, 0, 0.05, 0.20, "put") == 0.0

    def test_higher_vol_higher_price(self):
        low = bs_price(100, 100, 1.0, 0.05, 0.10, "call")
        high = bs_price(100, 100, 1.0, 0.05, 0.40, "call")
        assert high > low

    def test_longer_dte_higher_price(self):
        short = bs_price(100, 100, 0.1, 0.05, 0.20, "call")
        long = bs_price(100, 100, 2.0, 0.05, 0.20, "call")
        assert long > short


class TestBSGreeks:
    def test_returns_option_quote(self):
        q = bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert isinstance(q, OptionQuote)

    def test_call_delta_range(self):
        q = bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert 0.0 < q.delta < 1.0

    def test_put_delta_range(self):
        q = bs_greeks(100, 100, 1.0, 0.05, 0.20, "put")
        assert -1.0 < q.delta < 0.0

    def test_call_put_delta_relation(self):
        c = bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
        p = bs_greeks(100, 100, 1.0, 0.05, 0.20, "put")
        assert abs(c.delta - p.delta - 1.0) < 1e-10

    def test_gamma_positive(self):
        q = bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert q.gamma > 0

    def test_gamma_same_for_call_put(self):
        c = bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
        p = bs_greeks(100, 100, 1.0, 0.05, 0.20, "put")
        assert abs(c.gamma - p.gamma) < 1e-10

    def test_vega_positive(self):
        q = bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert q.vega > 0

    def test_theta_negative_for_long(self):
        q = bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert q.theta < 0

    def test_gamma_peaks_atm(self):
        itm = bs_greeks(100, 80, 0.25, 0.05, 0.20, "call")
        atm = bs_greeks(100, 100, 0.25, 0.05, 0.20, "call")
        otm = bs_greeks(100, 120, 0.25, 0.05, 0.20, "call")
        assert atm.gamma > itm.gamma
        assert atm.gamma > otm.gamma

    def test_higher_order_greeks_exist(self):
        q = bs_greeks(100, 100, 1.0, 0.05, 0.20, "call")
        assert hasattr(q, "vanna")
        assert hasattr(q, "charm")
        assert hasattr(q, "vomma")
        assert hasattr(q, "speed")
        assert hasattr(q, "color")
        assert hasattr(q, "zomma")

    def test_near_zero_time_no_crash(self):
        q = bs_greeks(100, 100, 1e-12, 0.05, 0.20, "call")
        assert np.isfinite(q.price)
        assert np.isfinite(q.delta)


class TestImpliedVol:
    def test_roundtrip_call(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        price = bs_price(S, K, T, r, sigma, "call")
        iv = implied_vol(price, S, K, T, r, "call")
        assert abs(iv - sigma) < 1e-6

    def test_roundtrip_put(self):
        S, K, T, r, sigma = 100, 95, 0.5, 0.05, 0.30
        price = bs_price(S, K, T, r, sigma, "put")
        iv = implied_vol(price, S, K, T, r, "put")
        assert abs(iv - sigma) < 1e-6

    def test_returns_none_for_invalid(self):
        iv = implied_vol(0.0001, 100, 100, 1.0, 0.05, "call")
        assert iv is None or iv < 0.01
