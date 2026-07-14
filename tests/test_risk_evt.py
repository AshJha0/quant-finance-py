"""Pins for quantfinlib.risk.extreme_value_theory, ported from Java
MarketRiskTest (evtRecoversAPlantedParetoTail / evtIgnoresTies).

The planted-Pareto draw uses numpy's default_rng instead of
java.util.Random; the assertions are the same statistical pins at the
same tolerances (shape 0.25 +/- 0.06 on 20k draws), so the stream
change cannot flip them.
"""

import math

import numpy as np
import pytest

from quantfinlib.risk import extreme_value_theory as evt


def test_recovers_a_planted_pareto_tail():
    # Pareto(alpha = 4) exceedances: GPD shape = 1/4 exactly.
    rng = np.random.default_rng(42)
    losses = rng.random(20_000) ** (-1.0 / 4)     # Pareto >= 1
    fit = evt.fit_pot(losses, 0.90)
    assert fit.shape == pytest.approx(0.25, abs=0.06), \
        f"the planted tail index: {fit.shape}"
    # The extrapolated 99.9% lands where the (large) sample says.
    var999 = fit.var(0.999)
    empirical = np.sort(losses)[int(0.999 * losses.shape[0])]
    assert var999 == pytest.approx(empirical, rel=0.25)
    assert fit.expected_shortfall(0.999) > var999


def test_refuses_infinite_means_and_inside_sample_quantiles():
    # A tail with no mean refuses to average itself (xi >= 1).
    infinite = evt.GpdFit(1, 1.2, 0.5, 100, 1000)
    with pytest.raises(RuntimeError):
        infinite.expected_shortfall(0.999)
    fit = evt.GpdFit(10, 0.3, 2, 100, 1000)
    with pytest.raises(ValueError):
        fit.var(0.5)      # inside-sample quantiles belong to historical VaR
    with pytest.raises(ValueError):
        fit.var(99.9)     # a confidence typo (99.9 for 0.999) throws, never NaN
    assert math.isfinite(fit.var(0.999))


def test_ignores_ties_at_the_threshold():
    # Discretized losses: ten observations sit EXACTLY at the threshold
    # value — counted as y = 0 exceedances they would deflate both PWMs
    # and bias the shape. Deterministic port of the Java test.
    losses = np.empty(100)
    losses[:45] = 1 + np.arange(45)                 # bulk below the threshold
    losses[45:55] = 50.0                            # tie block at the 0.5 quantile
    p = (np.arange(45) + 0.5) / 45                  # GPD(xi=0.25, beta=2) above
    losses[55:] = 50 + 2 / 0.25 * ((1 - p) ** -0.25 - 1)
    fit = evt.fit_pot(losses, 0.5)
    assert fit.exceedances == 45, "ties AT the threshold are not exceedances"
    assert fit.shape == pytest.approx(0.25, abs=0.15), \
        f"the planted tail survives ties: {fit.shape}"
    assert fit.var(0.99) > 50


def test_near_zero_shape_takes_the_exponential_branch():
    # |xi| < 1e-9: VaR = u - beta*ln((1-p)/tail_prob). u=10, beta=2,
    # tail 0.1, p=0.99: ratio = 0.1 -> VaR = 10 - 2*ln(0.1) = 10 + 2ln10.
    fit = evt.GpdFit(10, 0.0, 2, 100, 1000)
    assert fit.var(0.99) == pytest.approx(10 + 2 * math.log(10), abs=1e-12)


def test_gpd_var_closed_form_pin():
    # xi=0.25, beta=2, u=50, 45 exceedances of 100: tail_prob 0.45.
    # var(0.99): ratio = 0.01/0.45; VaR = 50 + 8*((0.01/0.45)^-0.25 - 1).
    fit = evt.GpdFit(50, 0.25, 2, 45, 100)
    expected = 50 + 2 / 0.25 * ((0.01 / 0.45) ** -0.25 - 1)
    assert fit.var(0.99) == pytest.approx(expected, abs=1e-12)
    # ES = (v + beta - xi*u)/(1 - xi).
    assert fit.expected_shortfall(0.99) == pytest.approx(
        (expected + 2 - 0.25 * 50) / 0.75, abs=1e-12)


def test_fit_gates():
    with pytest.raises(ValueError):
        evt.fit_pot(np.ones(49), 0.9)               # too few losses
    with pytest.raises(ValueError):
        evt.fit_pot(np.arange(100.0), 0.4)          # threshold quantile gate
    with pytest.raises(ValueError):
        evt.fit_pot(np.arange(100.0), math.nan)     # NaN-rejecting gate
    bad = np.arange(100.0)
    bad[3] = math.nan
    with pytest.raises(ValueError):
        evt.fit_pot(bad, 0.9)                       # NaN sorts into the tail
    with pytest.raises(ValueError):
        evt.fit_pot(np.arange(100.0), 0.95)         # only 5 exceedances (< 10)
    # All exceedances equal: PWM fit degenerates and says so.
    flat = np.concatenate([np.arange(50.0), np.full(50, 99.0)])
    with pytest.raises(ValueError):
        evt.fit_pot(flat, 0.5)
