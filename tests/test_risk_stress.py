"""Pins for quantfinlib.risk.stress_tester, ported from Java
MarketRiskTest.stressScenariosAndReverseStressCloseTheLoop (deterministic)."""

import math

import numpy as np
import pytest

from quantfinlib.risk import stress_tester as st
from quantfinlib.risk import var_engine as ve


def test_named_scenarios_and_delta_gamma():
    exposures = [2_000_000.0, -1_000_000.0, 500_000.0, 0.0, 300_000.0]
    pnl = st.scenario_pnl(exposures, st.lehman_2008())
    assert pnl < 0, "a long-equity short-rates book bleeds in Lehman week"
    # Delta-gamma: short gamma makes the same shock worse.
    gamma = np.zeros((5, 5))
    gamma[0, 0] = -1e7
    assert st.scenario_pnl(exposures, st.lehman_2008(), gamma) < pnl


def test_scenario_pnl_hand_pin():
    # d'Dx = 2e6*(-0.09) + (-1e6)*(-0.004) + 5e5*0.04 + 0 + 3e5*0.16
    #      = -180000 + 4000 + 20000 + 48000 = -108000.
    exposures = [2_000_000.0, -1_000_000.0, 500_000.0, 0.0, 300_000.0]
    assert st.scenario_pnl(exposures, st.lehman_2008()) == pytest.approx(
        -108_000, abs=1e-9)
    # Gamma term adds 0.5*(-1e7)*0.09^2 = -40500.
    gamma = np.zeros((5, 5))
    gamma[0, 0] = -1e7
    assert st.scenario_pnl(exposures, st.lehman_2008(), gamma) == pytest.approx(
        -148_500, abs=1e-9)


def test_sensitivity_ladder_is_linear_for_a_delta_book():
    exposures = [2_000_000.0, -1_000_000.0, 500_000.0, 0.0, 300_000.0]
    ladder = st.sensitivity_ladder(exposures, 0, 0.10, 4)
    assert ladder[0] == pytest.approx(-200_000, abs=1e-6), "-10% x 2M"
    assert ladder[2] == pytest.approx(0, abs=1e-6)
    assert ladder[4] == pytest.approx(200_000, abs=1e-6)


def test_delta_gamma_ladder_shows_the_curvature():
    # Short gamma costs 0.5 * 1e7 * 0.01 = 50k at BOTH +/-10% rungs.
    exposures = [2_000_000.0, -1_000_000.0, 500_000.0, 0.0, 300_000.0]
    gamma = np.zeros((5, 5))
    gamma[0, 0] = -1e7
    dg = st.sensitivity_ladder(exposures, 0, 0.10, 4, gamma)
    assert dg[0] == pytest.approx(-250_000, abs=1e-6), "the down rung is WORSE short gamma"
    assert dg[2] == pytest.approx(0, abs=1e-6)
    assert dg[4] == pytest.approx(150_000, abs=1e-6)


def test_reverse_stress_closes_the_loop():
    cov = [[4e-4, 1e-4], [1e-4, 2.25e-4]]
    expo = [1_000_000.0, -500_000.0]
    reverse = st.reverse_stress(expo, cov, 50_000)
    # The returned shock loses EXACTLY the target.
    assert st.scenario_pnl(expo, reverse.shocks) == pytest.approx(-50_000, abs=1e-6), \
        "the breaking scenario breaks by exactly the asked amount"
    assert reverse.mahalanobis_sigmas == pytest.approx(
        50_000 / ve.portfolio_stdev(expo, cov), abs=1e-9), \
        "and reports its own implausibility"


def test_gates():
    cov = [[4e-4, 1e-4], [1e-4, 2.25e-4]]
    with pytest.raises(ValueError):
        st.reverse_stress([0.0, 0.0], cov, 50_000)      # no factor risk
    with pytest.raises(ValueError):
        st.reverse_stress([1.0, 1.0], cov, -1.0)        # loss must be positive
    with pytest.raises(ValueError):
        st.reverse_stress([1.0, 1.0], cov, math.inf)
    # A NaN exposure fails at the stress gate, not as NaN per scenario.
    with pytest.raises(ValueError):
        st.scenario_pnl([1e6, math.nan, 0.0, 0.0, 0.0], st.lehman_2008())
    with pytest.raises(ValueError):
        st.scenario_pnl([1e6], [0.1, 0.2])              # misaligned
    with pytest.raises(ValueError):
        st.sensitivity_ladder([1e6], 0, 0.1, 1)         # steps < 2
    with pytest.raises(ValueError):
        st.sensitivity_ladder([1e6], 1, 0.1, 4)         # factor out of range
    with pytest.raises(ValueError):
        st.sensitivity_ladder([1e6], 0, math.nan, 4)    # NaN-rejecting range
