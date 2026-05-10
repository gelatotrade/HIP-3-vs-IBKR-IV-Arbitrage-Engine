"""Tests for IBKROptionsClient chain generation and IV surface."""

import pandas as pd
import pytest
from ibkr_options_client import IBKROptionsClient


@pytest.fixture
def client():
    return IBKROptionsClient(risk_free_rate=0.05)


class TestOptionsChain:
    def test_chain_returns_dataframe(self, client):
        chain = client.generate_options_chain(spot=100, base_iv=0.25)
        assert isinstance(chain, pd.DataFrame)
        assert len(chain) > 0

    def test_chain_has_required_columns(self, client):
        chain = client.generate_options_chain(spot=100, base_iv=0.25)
        required = {"strike", "expiry_days", "type", "mid", "delta", "gamma", "vega"}
        assert required.issubset(set(chain.columns))

    def test_chain_call_put_ratio(self, client):
        chain = client.generate_options_chain(spot=100, base_iv=0.25)
        calls = chain[chain["type"] == "call"]
        puts = chain[chain["type"] == "put"]
        assert len(calls) == len(puts)

    def test_chain_custom_expiries(self, client):
        chain = client.generate_options_chain(spot=100, base_iv=0.25, expiries_days=[7, 30, 90])
        unique_expiries = chain["expiry_days"].nunique()
        assert unique_expiries == 3

    def test_chain_prices_positive(self, client):
        chain = client.generate_options_chain(spot=100, base_iv=0.25)
        assert (chain["mid"] >= 0).all()

    def test_chain_with_skew(self, client):
        chain_no_skew = client.generate_options_chain(spot=100, base_iv=0.25, iv_skew=0.0)
        chain_skew = client.generate_options_chain(spot=100, base_iv=0.25, iv_skew=-0.15)
        assert len(chain_skew) == len(chain_no_skew)


class TestIVSurface:
    def test_surface_returns_tuple(self, client):
        result = client.get_iv_surface(spot=100, base_iv=0.25)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_surface_shapes_consistent(self, client):
        strikes, expiries, iv_matrix = client.get_iv_surface(spot=100, base_iv=0.25)
        assert iv_matrix.shape[0] == len(expiries)
        assert iv_matrix.shape[1] == len(strikes)

    def test_surface_ivs_positive(self, client):
        _, _, iv_matrix = client.get_iv_surface(spot=100, base_iv=0.25)
        assert (iv_matrix > 0).all()


class TestVolSpread:
    def test_vol_spread_returns_dataframe(self, client):
        chain = client.generate_options_chain(spot=100, base_iv=0.25)
        result = client.compute_vol_spread(hip3_iv=0.30, ibkr_chain=chain, spot=100)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_vol_spread_has_spread_column(self, client):
        chain = client.generate_options_chain(spot=100, base_iv=0.20)
        result = client.compute_vol_spread(hip3_iv=0.35, ibkr_chain=chain, spot=100)
        assert "vol_spread" in result.columns
