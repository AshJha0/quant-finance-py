"""Pins for risk parity, Black-Litterman and the constrained optimizer.

Java source: PortfolioConstructionTest.java. Risk parity and
Black-Litterman are deterministic (no RNG) so the Java pins transfer at
the same tolerances; the constrained optimizer pins are feasibility and
dominance properties the stochastic search must satisfy.
"""

import numpy as np
import pytest

from quantfinlib.backtest.portfolio import (BlackLitterman,
                                            ConstrainedPortfolioOptimizer,
                                            RiskParityOptimizer)

MU = [0.06, 0.08, 0.05, 0.11]
COV = [[0.04, 0.01, 0.00, 0.01],
       [0.01, 0.09, 0.02, 0.02],
       [0.00, 0.02, 0.02, 0.00],
       [0.01, 0.02, 0.00, 0.16]]


# ------------------------------------------------------------- risk parity

def test_risk_parity_equalizes_contributions():
    erc = RiskParityOptimizer.equal_risk_contribution(MU, COV)
    rc = RiskParityOptimizer.risk_contributions(erc.weights, COV)
    for contribution in rc:
        assert contribution == pytest.approx(0.25, abs=1e-6)
    assert np.all(erc.weights > 0)
    assert float(np.sum(erc.weights)) == pytest.approx(1.0, abs=1e-9)


def test_risk_parity_matches_inverse_vol_for_uncorrelated_assets():
    # Uncorrelated 10%/20% vol: ERC weights are proportional to 1/sigma.
    erc = RiskParityOptimizer.equal_risk_contribution(
        [0.05, 0.05], [[0.01, 0], [0, 0.04]])
    assert erc.weights[0] == pytest.approx(2.0 / 3, abs=1e-6)
    assert erc.weights[1] == pytest.approx(1.0 / 3, abs=1e-6)


def test_risk_parity_refuses_zero_variance_asset():
    # A zero-variance asset has no risk to contribute: the ERC fixed
    # point does not exist — refuse rather than spin.
    with pytest.raises(ValueError):
        RiskParityOptimizer.equal_risk_contribution(
            [0.05, 0.05], [[0.01, 0], [0, 0.0]])
    with pytest.raises(ValueError):
        RiskParityOptimizer.equal_risk_contribution([0.05], COV)  # mismatch


# --------------------------------------------------------- Black-Litterman

def test_no_views_returns_equilibrium():
    market_weights = [0.3, 0.3, 0.2, 0.2]
    pi = BlackLitterman.implied_equilibrium_returns(2.5, COV, market_weights)
    posterior = BlackLitterman.posterior_returns(0.05, COV, pi, [], [], [])
    assert np.allclose(posterior, pi, atol=1e-12)
    # Riskier assets carry higher implied returns under reverse optimization.
    assert pi[3] > pi[2]


def test_confident_absolute_view_pulls_posterior_to_the_view():
    pi = BlackLitterman.implied_equilibrium_returns(
        2.5, COV, [0.25, 0.25, 0.25, 0.25])
    # View: asset 0 returns exactly 15%, with near-certainty.
    p = [[1, 0, 0, 0]]
    posterior = BlackLitterman.posterior_returns(0.05, COV, pi, p, [0.15], [1e-8])
    assert posterior[0] == pytest.approx(0.15, abs=1e-3)
    # Weak view barely moves the prior.
    weak = BlackLitterman.posterior_returns(0.05, COV, pi, p, [0.15], [10.0])
    assert weak[0] == pytest.approx(pi[0], abs=0.005)


def test_relative_view_moves_the_spread():
    pi = BlackLitterman.implied_equilibrium_returns(
        2.5, COV, [0.25, 0.25, 0.25, 0.25])
    prior_spread = pi[1] - pi[2]
    # View: asset 1 outperforms asset 2 by 10% (more than the prior spread).
    p = [[0, 1, -1, 0]]
    posterior = BlackLitterman.posterior_returns(0.05, COV, pi, p, [0.10], [1e-6])
    assert posterior[1] - posterior[2] == pytest.approx(0.10, abs=1e-3)
    assert posterior[1] - posterior[2] > prior_spread


def test_black_litterman_gates():
    pi = BlackLitterman.implied_equilibrium_returns(
        2.5, COV, [0.25, 0.25, 0.25, 0.25])
    with pytest.raises(ValueError):
        BlackLitterman.posterior_returns(
            0.05, COV, pi, [[1, 0, 0, 0]], [0.15, 0.2], [1e-8])  # q misaligned
    with pytest.raises(ValueError):
        BlackLitterman.posterior_returns(
            0.05, COV, pi, [[1, 0, 0, 0]], [0.15], [0.0])        # omega <= 0


# ------------------------------------------------------------- constrained

def test_position_caps_are_respected():
    capped = (ConstrainedPortfolioOptimizer(MU, COV)
              .with_bounds([0, 0, 0, 0], [0.30, 0.30, 0.30, 0.30])
              .max_sharpe(0.02))
    assert np.all(capped.weights <= 0.30 + 1e-9)
    assert np.all(capped.weights >= -1e-9)
    assert float(np.sum(capped.weights)) == pytest.approx(1.0, abs=1e-6)
    # Cap must bind somewhere: 4 assets at <= 30% forces near-full usage.
    assert capped.weights[2] > 0.05


def test_turnover_penalty_anchors_to_current_holdings():
    current = [0.25, 0.25, 0.25, 0.25]
    free = ConstrainedPortfolioOptimizer(MU, COV).max_sharpe(0.02)
    sticky = (ConstrainedPortfolioOptimizer(MU, COV)
              .with_turnover_penalty(current, 0.50)   # punitive cost
              .max_sharpe(0.02))
    free_turnover = float(np.sum(np.abs(free.weights - current)))
    sticky_turnover = float(np.sum(np.abs(sticky.weights - current)))
    assert sticky_turnover < 0.25 * free_turnover


def test_infeasible_bounds_are_rejected():
    with pytest.raises(ValueError):
        (ConstrainedPortfolioOptimizer(MU, COV)
         .with_bounds([0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2]))  # maxSum < 1
    with pytest.raises(ValueError):
        (ConstrainedPortfolioOptimizer(MU, COV)
         .with_bounds([0, -0.1, 0, 0], [1, 1, 1, 1]))       # negative floor
