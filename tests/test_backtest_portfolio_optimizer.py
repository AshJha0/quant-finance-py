"""Pins for quantfinlib.backtest.portfolio.portfolio_optimizer.

Java sources: PortfolioOptimizerTest.java and FormulaPinsTest.java
(maxSharpeFindsTheAnalyticTangencyPortfolio). The optimizer is a
stochastic search, so every pin is an ANALYTIC solution the search must
find within the Java tolerances — the RNG stream itself is not shared
across ports.
"""

import numpy as np
import pytest

from quantfinlib.backtest.portfolio import PortfolioOptimizer


def test_max_sharpe_finds_the_analytic_tangency_portfolio():
    # Diagonal cov: tangency weights ~ (mu - rf)/sigma^2 = {4, 2} -> {2/3, 1/3}.
    # Ignoring rf lands at {0.706, 0.294} and fails the tolerance.
    opt = PortfolioOptimizer([0.06, 0.10], [[0.01, 0], [0, 0.04]], 42)
    tangency = opt.max_sharpe(0.02)
    assert tangency.weights[0] == pytest.approx(2.0 / 3, abs=0.03)
    assert tangency.weights[1] == pytest.approx(1.0 / 3, abs=0.03)


def test_min_volatility_matches_two_asset_analytic_solution():
    # Uncorrelated assets, vol 10% and 20%: w1* = s2^2 / (s1^2 + s2^2) = 0.8.
    a = PortfolioOptimizer([0.05, 0.10], [[0.01, 0.0], [0.0, 0.04]]).min_volatility()
    assert a.weights[0] == pytest.approx(0.8, abs=0.02)
    assert a.weights[1] == pytest.approx(0.2, abs=0.02)
    assert a.weights[0] + a.weights[1] == pytest.approx(1.0, abs=1e-9)
    # Analytic min vol = sqrt(0.8^2*0.01 + 0.2^2*0.04) ~ 0.0894.
    assert a.volatility == pytest.approx(0.0894, abs=0.005)


def test_max_sharpe_favors_dominant_asset():
    # Asset B dominates: same vol, higher return.
    a = PortfolioOptimizer([0.04, 0.12],
                           [[0.0225, 0.0], [0.0, 0.0225]]).max_sharpe(0.02)
    assert a.weights[1] > a.weights[0]
    assert a.sharpe > 0


def test_weights_always_on_simplex():
    mu = [0.06, 0.08, 0.05, 0.11]
    cov = [[0.04, 0.01, 0.00, 0.01],
           [0.01, 0.09, 0.02, 0.02],
           [0.00, 0.02, 0.02, 0.00],
           [0.01, 0.02, 0.00, 0.16]]
    for a in (PortfolioOptimizer(mu, cov).max_sharpe(0.02),
              PortfolioOptimizer(mu, cov).min_volatility()):
        assert np.all(a.weights >= -1e-9)
        assert float(np.sum(a.weights)) == pytest.approx(1.0, abs=1e-9)


def test_efficient_frontier_volatility_rises_with_return():
    frontier = PortfolioOptimizer(
        [0.04, 0.12], [[0.01, 0.002], [0.002, 0.05]]).efficient_frontier(5)
    assert len(frontier) == 5
    # Ends of the frontier: low-return end has lower vol than high-return end.
    assert frontier[0].volatility <= frontier[-1].volatility + 1e-6


def test_rebalance_deltas_sum_to_zero():
    deltas = PortfolioOptimizer.rebalance([0.6, 0.4], [0.5, 0.5])
    assert deltas[0] == pytest.approx(-0.1, abs=1e-12)
    assert deltas[1] == pytest.approx(0.1, abs=1e-12)


def test_dimension_mismatch_is_refused():
    with pytest.raises(ValueError):
        PortfolioOptimizer([0.05], [[0.01, 0.0], [0.0, 0.04]])
