"""Tests for GreeksStrategyEngine."""

import numpy as np
import pytest
from greeks_strategies import GreeksStrategyEngine, StrategyResult, classify_regime


@pytest.fixture
def engine():
    return GreeksStrategyEngine(risk_free=0.05)


@pytest.fixture
def strategy_inputs(rng):
    n = 100
    returns = rng.normal(0.0005, 0.02, n)
    prices = 100 * np.cumprod(1 + returns)
    hip3_iv = 0.25 + 0.05 * rng.randn(n)
    hip3_iv = np.clip(hip3_iv, 0.05, 1.0)
    ibkr_iv = hip3_iv + 0.02 * rng.randn(n)
    ibkr_iv = np.clip(ibkr_iv, 0.05, 1.0)
    return prices, hip3_iv, ibkr_iv


class TestClassifyRegime:
    def test_returns_string(self, sample_prices):
        returns = np.diff(np.log(sample_prices))
        regime = classify_regime(returns, 50)
        assert isinstance(regime, str)

    def test_valid_regime(self, sample_prices):
        returns = np.diff(np.log(sample_prices))
        regime = classify_regime(returns, 50)
        valid = {"BULL", "NORMAL", "CAUTIOUS", "CRISIS", "RECOVERY"}
        assert regime in valid


class TestGammaScalping:
    def test_returns_strategy_result(self, engine, strategy_inputs):
        prices, hip3_iv, ibkr_iv = strategy_inputs
        result = engine.gamma_scalping(prices, hip3_iv, ibkr_iv)
        assert isinstance(result, StrategyResult)

    def test_pnl_length(self, engine, strategy_inputs):
        prices, hip3_iv, ibkr_iv = strategy_inputs
        result = engine.gamma_scalping(prices, hip3_iv, ibkr_iv)
        assert len(result.daily_pnl) == len(prices)


class TestVannaTrade:
    def test_returns_strategy_result(self, engine, strategy_inputs):
        prices, hip3_iv, ibkr_iv = strategy_inputs
        result = engine.vanna_trade(prices, hip3_iv, ibkr_iv)
        assert isinstance(result, StrategyResult)


class TestCharmTrade:
    def test_returns_strategy_result(self, engine, strategy_inputs):
        prices, hip3_iv, ibkr_iv = strategy_inputs
        result = engine.charm_trade(prices, hip3_iv, ibkr_iv)
        assert isinstance(result, StrategyResult)


class TestVommaTrade:
    def test_returns_strategy_result(self, engine, strategy_inputs):
        prices, hip3_iv, ibkr_iv = strategy_inputs
        result = engine.vomma_trade(prices, hip3_iv, ibkr_iv)
        assert isinstance(result, StrategyResult)


class TestRunAllStrategies:
    def test_returns_dict(self, engine, strategy_inputs):
        prices, hip3_iv, ibkr_iv = strategy_inputs
        results = engine.run_all_strategies(prices, hip3_iv, ibkr_iv)
        assert isinstance(results, dict)
        assert len(results) >= 7

    def test_all_values_are_strategy_results(self, engine, strategy_inputs):
        prices, hip3_iv, ibkr_iv = strategy_inputs
        results = engine.run_all_strategies(prices, hip3_iv, ibkr_iv)
        for name, result in results.items():
            assert isinstance(result, StrategyResult), f"{name} is not StrategyResult"


class TestEnsembleStrategy:
    def test_returns_array(self, engine, strategy_inputs):
        prices, hip3_iv, ibkr_iv = strategy_inputs
        strategies = engine.run_all_strategies(prices, hip3_iv, ibkr_iv)
        ensemble = engine.ensemble_strategy(strategies)
        assert isinstance(ensemble, np.ndarray)
        assert len(ensemble) == len(prices)
