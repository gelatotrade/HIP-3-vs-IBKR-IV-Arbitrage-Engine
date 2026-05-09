"""Tests for IVArbitrageEngine."""

import numpy as np
import pandas as pd
import pytest
from iv_arbitrage_engine import IVArbitrageEngine


@pytest.fixture
def engine():
    return IVArbitrageEngine(risk_free_rate=0.05)


class TestHIP3ImpliedVol:
    def test_returns_array(self, engine, sample_prices):
        iv = engine.compute_hip3_implied_vol(sample_prices)
        assert isinstance(iv, np.ndarray)
        assert len(iv) == len(sample_prices)

    def test_iv_positive(self, engine, sample_prices):
        iv = engine.compute_hip3_implied_vol(sample_prices)
        assert (iv[~np.isnan(iv)] > 0).all()

    def test_iv_reasonable_range(self, engine, sample_prices):
        iv = engine.compute_hip3_implied_vol(sample_prices)
        valid = iv[~np.isnan(iv)]
        assert valid.mean() < 2.0
        assert valid.mean() > 0.01

    def test_with_funding_rates(self, engine, sample_prices, rng):
        funding = rng.normal(0.0001, 0.001, len(sample_prices))
        iv = engine.compute_hip3_implied_vol(sample_prices, funding_rates=funding)
        assert len(iv) == len(sample_prices)


class TestIBKRATMIV:
    def test_returns_array(self, engine, sample_prices):
        iv = engine.compute_ibkr_atm_iv(sample_prices)
        assert isinstance(iv, np.ndarray)
        assert len(iv) == len(sample_prices)

    def test_iv_positive(self, engine, sample_prices):
        iv = engine.compute_ibkr_atm_iv(sample_prices)
        valid = iv[~np.isnan(iv)]
        assert (valid > 0).all()


class TestVolSpreadOpportunities:
    def test_returns_dataframe(self, engine, sample_prices):
        hip3_iv = engine.compute_hip3_implied_vol(sample_prices)
        ibkr_iv = engine.compute_ibkr_atm_iv(sample_prices)
        signals = engine.find_vol_spread_opportunities(hip3_iv, ibkr_iv, sample_prices)
        assert isinstance(signals, pd.DataFrame)

    def test_has_vol_spread_column(self, engine, sample_prices):
        hip3_iv = engine.compute_hip3_implied_vol(sample_prices)
        ibkr_iv = engine.compute_ibkr_atm_iv(sample_prices)
        signals = engine.find_vol_spread_opportunities(hip3_iv, ibkr_iv, sample_prices)
        assert "vol_spread" in signals.columns or len(signals) == 0


class TestFullArbitrageScan:
    def test_returns_dict(self, engine, sample_prices):
        result = engine.full_arbitrage_scan("TEST", sample_prices)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, engine, sample_prices):
        result = engine.full_arbitrage_scan("TEST", sample_prices)
        assert "hip3_iv" in result
        assert "ibkr_iv" in result
        assert "vol_spread" in result
