"""Pins for quantfinlib.risk.var_engine, ported from Java MarketRiskTest.

Seeded Monte Carlo uses numpy's default_rng instead of java.util.Random
(different stream); every MC assertion below is an agreement/identity
pin with the same tolerances as the Java asserts, so the change of
stream cannot flip a verdict.
"""

import math

import numpy as np
import pytest

from quantfinlib.risk import var_engine as ve
from quantfinlib.util import math_utils as mu

EXPOSURES = [1_000_000.0, -500_000.0]
COV = [[4e-4, 1e-4], [1e-4, 2.25e-4]]   # 2%/1.5% daily, corr ~1/3


def test_portfolio_stdev_pin():
    # w'Sw = 1e12*4e-4 - 2*5e11*1e-4 + 2.5e11*2.25e-4
    #      = 4e8 - 1e8 + 0.5625e8 = 3.5625e8.
    assert ve.portfolio_stdev(EXPOSURES, COV) == pytest.approx(
        math.sqrt(3.5625e8), abs=1e-9)


def test_delta_normal_is_z_times_sigma():
    sigma = ve.portfolio_stdev(EXPOSURES, COV)
    assert ve.delta_normal_var(EXPOSURES, COV, 0.99) == pytest.approx(
        mu.norm_inv(0.99) * sigma, abs=1e-9), "delta-normal IS z times sigma"
    assert ve.delta_normal_es(EXPOSURES, COV, 0.99) > ve.delta_normal_var(
        EXPOSURES, COV, 0.99), "ES sits beyond VaR, always"


def test_monte_carlo_converges_to_delta_normal_on_linear_book():
    parametric = ve.delta_normal_var(EXPOSURES, COV, 0.99)
    mc = ve.monte_carlo_var(EXPOSURES, COV, 0.99, 200_000, 42)
    assert mc.var == pytest.approx(parametric, rel=0.03), "two routes, one linear book"
    assert mc.expected_shortfall > mc.var


def test_delta_gamma_tilts_the_quantile_by_gamma_sign():
    parametric = ve.delta_normal_var(EXPOSURES, COV, 0.99)
    short_gamma = [[-4_000_000.0, 0.0], [0.0, 0.0]]
    dg_var = ve.delta_gamma_var(EXPOSURES, short_gamma, COV, 0.99)
    assert dg_var > parametric, "short-gamma tails are worse than delta-normal admits"
    long_gamma = [[4_000_000.0, 0.0], [0.0, 0.0]]
    assert ve.delta_gamma_var(EXPOSURES, long_gamma, COV, 0.99) < parametric, \
        "long gamma cushions the same tail"


def test_delta_gamma_es_closed_form():
    # gamma = 0 reduces EXACTLY to delta-normal ES:
    # ES = -mu + sigma*phi(z)/(1-c)*(1 + z*s/6) with mu = 0, s = 0.
    assert ve.delta_gamma_es(EXPOSURES, np.zeros((2, 2)), COV, 0.99) == pytest.approx(
        ve.delta_normal_es(EXPOSURES, COV, 0.99), abs=1e-9)
    short_gamma = [[-4_000_000.0, 0.0], [0.0, 0.0]]
    dg_es = ve.delta_gamma_es(EXPOSURES, short_gamma, COV, 0.99)
    dg_var = ve.delta_gamma_var(EXPOSURES, short_gamma, COV, 0.99)
    assert dg_es > dg_var, "ES sits beyond VaR for the gamma book too"
    assert dg_es > ve.delta_normal_es(EXPOSURES, COV, 0.99), \
        "short gamma worsens the tail MEAN, not just the quantile"


def test_historical_var_pin():
    # 100 scenarios, factor 1 loses linearly more: loss_s = 10*(s+1).
    # 95% -> index ceil(0.95*100)-1 = 94 -> loss 950 (the 95th of 100).
    history = np.zeros((100, 2))
    history[:, 0] = -(np.arange(100) + 1) * 1e-5
    hist = ve.historical_var([1_000_000.0, 0.0], history, 0.95)
    assert hist.var == pytest.approx(950_000 * 1e-3, abs=1e-6), \
        "the 95th of 100 ordered losses"
    assert hist.expected_shortfall > hist.var


def test_full_revaluation_linear_pricer_is_historical():
    # 100 one-factor scenarios, monotonically worse: x_s = -(s+1)*5e-4.
    scenarios = -(np.arange(100).reshape(-1, 1) + 1) * 5e-4
    delta = 1_000_000.0
    linear = ve.full_revaluation_var(scenarios, lambda m: delta * m[0], 0.99)
    historical = ve.historical_var([delta], scenarios, 0.99)
    assert linear.var == pytest.approx(historical.var, abs=1e-9), \
        "a linear pricer IS historical simulation"
    assert linear.expected_shortfall == pytest.approx(
        historical.expected_shortfall, abs=1e-9)


def test_full_revaluation_sees_the_curvature():
    # 99% row is s = 98 (x = -0.0495): linear 49,500 + quadratic
    # 0.5*4e7*0.0495^2 = 49,005 -> 98,505.
    scenarios = -(np.arange(100).reshape(-1, 1) + 1) * 5e-4
    delta = 1_000_000.0
    gamma = -4e7
    quadratic = ve.full_revaluation_var(
        scenarios, lambda m: delta * m[0] + 0.5 * gamma * m[0] * m[0], 0.99)
    assert quadratic.var == pytest.approx(98_505, abs=1e-6), "hand-computed, to the dollar"
    linear = ve.full_revaluation_var(scenarios, lambda m: delta * m[0], 0.99)
    assert quadratic.var > linear.var, \
        "short gamma makes every down scenario worse than delta admits"
    assert quadratic.expected_shortfall > quadratic.var


def test_full_revaluation_gates():
    scenarios = -(np.arange(100).reshape(-1, 1) + 1) * 5e-4
    # A pricer that cannot price a scenario is a modelling problem.
    with pytest.raises(ValueError):
        ve.full_revaluation_var(scenarios, lambda m: math.nan, 0.99)
    with pytest.raises(ValueError):
        ve.full_revaluation_var(np.zeros((10, 1)), lambda m: 0.0, 0.99)


def test_tiny_unit_covariances_stay_positive_definite():
    # Java review regression: the Cholesky pivot floor is relative to the
    # diagonal scale, so two 0.5bp-vol rate factors are valid MC inputs.
    v = 2.5e-9
    tiny = [[v, 0.9 * v], [0.9 * v, v]]
    dn = ve.delta_normal_var(EXPOSURES, tiny, 0.99)
    mc = ve.monte_carlo_var(EXPOSURES, tiny, 0.99, 100_000, 7)
    assert mc.var == pytest.approx(dn, rel=0.04), \
        "the linear book agreement holds at any unit scale"


def test_input_gates():
    with pytest.raises(ValueError):
        ve.monte_carlo_var(EXPOSURES, COV, 0.99, 99, 1)      # too few scenarios
    with pytest.raises(ValueError):
        ve.historical_var(EXPOSURES, np.zeros((10, 2)), 0.99)  # too few rows
    with pytest.raises(ValueError):
        ve.tail([1.0] * 30, 0.5)                              # confidence gate
    with pytest.raises(ValueError):
        ve.tail([1.0] * 30, math.nan)                         # NaN-rejecting
    with pytest.raises(ValueError):
        ve.delta_normal_var([1.0, 2.0], [[1.0]], 0.99)        # shape mismatch
    with pytest.raises(ValueError):
        ve.delta_gamma_var([1.0], [[1.0], [2.0]], [[1.0]], 0.99)  # gamma mismatch


def test_tail_index_arithmetic_matches_java_ceil():
    # losses 1..1000 at 97.5%: index = ceil(975)-1 = 974 -> VaR 975,
    # ES = mean of 975..1000 (26 values) = 987.5 exactly.
    losses = np.arange(1, 1001, dtype=float)
    t = ve.tail(losses, 0.975)
    assert t.var == pytest.approx(975.0, abs=1e-12)
    assert t.expected_shortfall == pytest.approx(987.5, abs=1e-9)
