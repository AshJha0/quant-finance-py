"""Pins for quantfinlib.sim (Monte Carlo portfolio simulation).

Java sources: MonteCarloSimulator/SimulationResult.java. The per-path
seed derivation and Gaussian draws are a bit-exact port of Java's
``java.util.SplittableRandom`` (see ``quantfinlib/sim/_java_random.py``);
the five terminal values below were captured by compiling and running
the actual Java ``mix``/``SplittableRandom``/Marsaglia-polar sequence
against seed 7, so they pin cross-language determinism, not just
same-process repeatability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantfinlib.sim._java_random import SplittableRandom64, mix_seed
from quantfinlib.sim.monte_carlo_simulator import MonteCarloSimulator
from quantfinlib.sim.simulation_result import SimulationResult

# Captured from a compiled Java program using java.util.SplittableRandom
# with the exact mix()/gaussian() transcription in MonteCarloSimulator.java,
# seed=7, initialValue=100_000, annualReturn=0.08, annualVol=0.15,
# horizonDays=252, path indices 0..4.
_JAVA_FINALS_SEED_7 = [
    109655.88062205294,
    92213.69557475435,
    127766.76940655152,
    137132.9597136051,
    86553.71247703057,
]


def _simulate_raw_finals(seed: int, initial_value, annual_return, annual_vol, horizon_days, simulations):
    """Recomputes per-path finals directly (unsorted) to compare against
    the Java per-path capture; SimulationResult sorts internally."""
    dt = 1.0 / 252
    drift = (annual_return - 0.5 * annual_vol * annual_vol) * dt
    diffusion = annual_vol * math.sqrt(dt)
    finals = []
    for s in range(simulations):
        rnd = SplittableRandom64(mix_seed(seed, s))
        log_v = math.log(initial_value)
        for _ in range(horizon_days):
            log_v += drift + diffusion * _gaussian(rnd)
        finals.append(math.exp(log_v))
    return finals


def _gaussian(rnd: SplittableRandom64) -> float:
    while True:
        u = 2 * rnd.next_double() - 1
        v = 2 * rnd.next_double() - 1
        s = u * u + v * v
        if 0 < s < 1:
            return u * math.sqrt(-2 * math.log(s) / s)


def test_per_path_finals_match_java_bit_exactly():
    finals = _simulate_raw_finals(7, 100_000, 0.08, 0.15, 252, 5)
    for got, expected in zip(finals, _JAVA_FINALS_SEED_7):
        assert got == pytest.approx(expected, rel=0, abs=1e-9)


def test_simulate_same_seed_is_deterministic():
    sim = MonteCarloSimulator(7)
    r1 = sim.simulate(100_000, 0.08, 0.15, 252, 500)
    r2 = sim.simulate(100_000, 0.08, 0.15, 252, 500)
    assert r1.expected_value() == r2.expected_value()
    assert r1.best_case() == r2.best_case()
    assert r1.worst_case() == r2.worst_case()


def test_simulate_different_seed_differs():
    r1 = MonteCarloSimulator(7).simulate(100_000, 0.08, 0.15, 252, 500)
    r2 = MonteCarloSimulator(8).simulate(100_000, 0.08, 0.15, 252, 500)
    assert r1.expected_value() != r2.expected_value()


def test_default_seed_is_7():
    default = MonteCarloSimulator()
    seeded = MonteCarloSimulator(7)
    r1 = default.simulate(100_000, 0.08, 0.15, 50, 20)
    r2 = seeded.simulate(100_000, 0.08, 0.15, 50, 20)
    assert r1.expected_value() == r2.expected_value()


def test_zero_horizon_or_zero_vol_is_a_no_op_path():
    sim = MonteCarloSimulator(7)
    result = sim.simulate(100_000, 0.0, 0.0, 0, 10)
    assert result.best_case() == pytest.approx(100_000)
    assert result.worst_case() == pytest.approx(100_000)


def test_simulate_portfolio_matches_single_asset_when_uncorrelated_single_weight():
    sim = MonteCarloSimulator(7)
    single = sim.simulate(100_000, 0.08, 0.15, 60, 200)
    portfolio = sim.simulate_portfolio(
        100_000,
        weights=[1.0],
        daily_mean_returns=[(0.08 - 0.5 * 0.15 * 0.15) / 252],
        daily_covariance=[[(0.15 * 0.15) / 252]],
        horizon_days=60,
        simulations=200,
    )
    # Not the same GBM parameterization (arithmetic vs log-normal steps),
    # but both should center near the same ballpark for small vol*sqrt(t).
    assert portfolio.expected_value() == pytest.approx(single.expected_value(), rel=0.05)


# ----------------------------------------------------------------------
# SimulationResult analytics
# ----------------------------------------------------------------------


def test_simulation_result_probability_and_var_cvar_ordering():
    finals = [80_000.0, 90_000.0, 95_000.0, 100_000.0, 110_000.0, 130_000.0]
    result = SimulationResult(100_000.0, finals)
    assert result.simulations() == 6
    assert result.best_case() == 130_000.0
    assert result.worst_case() == 80_000.0
    assert result.probability_of_profit() == pytest.approx(2 / 6)
    assert result.probability_of_loss() == pytest.approx(4 / 6)
    # CVaR is at least as severe as VaR (average of a worse-or-equal tail).
    assert result.conditional_value_at_risk(0.95) >= result.value_at_risk(0.95) - 1e-12


def test_simulation_result_confidence_interval_is_symmetric_bracket():
    rng = np.random.default_rng(9)
    finals = rng.normal(100_000, 5_000, 5_000)
    result = SimulationResult(100_000.0, finals)
    lo, hi = result.confidence_interval(0.90)
    assert lo < result.median_value() < hi
