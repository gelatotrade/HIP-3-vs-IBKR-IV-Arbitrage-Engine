"""Tests for HyperliquidHIP3Client synthetic data generation and helpers."""

import numpy as np
import pandas as pd
import pytest
from hyperliquid_hip3_client import (
    generate_synthetic_hip3_data,
    get_asset_n_days,
    HIP3_LAUNCH_DATES,
)


class TestLaunchDates:
    def test_launch_dates_dict(self):
        assert isinstance(HIP3_LAUNCH_DATES, dict)
        assert len(HIP3_LAUNCH_DATES) >= 16

    def test_known_assets(self):
        for asset in ["AAPL", "NVDA", "TSLA", "GOOGL", "GOLD", "OIL"]:
            assert asset in HIP3_LAUNCH_DATES


class TestGetAssetNDays:
    def test_returns_int(self):
        n = get_asset_n_days("AAPL")
        assert isinstance(n, int)
        assert n > 0

    def test_unknown_asset_returns_default(self):
        n = get_asset_n_days("NONEXISTENT_ASSET_XYZ")
        assert isinstance(n, int)
        assert n >= 0


class TestSyntheticData:
    def test_returns_dict(self):
        data = generate_synthetic_hip3_data(n_assets=3, seed=42)
        assert isinstance(data, dict)
        assert len(data) >= 1

    def test_dataframes_have_ohlcv(self):
        data = generate_synthetic_hip3_data(n_assets=3, seed=42)
        for asset, df in data.items():
            assert isinstance(df, pd.DataFrame)
            for col in ["open", "high", "low", "close", "volume"]:
                assert col in df.columns, f"{asset} missing column {col}"

    def test_prices_positive(self):
        data = generate_synthetic_hip3_data(n_assets=3, seed=42)
        for asset, df in data.items():
            assert (df["close"] > 0).all(), f"{asset} has non-positive close prices"
            assert (df["high"] >= df["low"]).all(), f"{asset} high < low"

    def test_seed_reproducible(self):
        d1 = generate_synthetic_hip3_data(n_assets=3, seed=123)
        d2 = generate_synthetic_hip3_data(n_assets=3, seed=123)
        for asset in d1:
            pd.testing.assert_frame_equal(d1[asset], d2[asset])

    def test_different_seeds_differ(self):
        d1 = generate_synthetic_hip3_data(n_assets=3, seed=1)
        d2 = generate_synthetic_hip3_data(n_assets=3, seed=2)
        common = set(d1.keys()) & set(d2.keys())
        assert len(common) > 0
        asset = list(common)[0]
        assert not np.allclose(d1[asset]["close"].values, d2[asset]["close"].values)
