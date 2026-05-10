import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_prices(rng):
    n = 100
    returns = rng.normal(0.0005, 0.02, n)
    prices = 100 * np.cumprod(1 + returns)
    return prices


@pytest.fixture
def sample_iv(rng):
    return 0.25 + 0.05 * rng.randn(100)
