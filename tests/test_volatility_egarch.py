"""Pins for quantfinlib.volatility.egarch11.

Java source: EgarchTest — the planted-leverage recovery, the sign
semantics of gamma, and the exact hand-computed one-step recursion pins
(RNG-free, transferred verbatim). Simulations use numpy's Generator
with the Java model parameters and tolerances.
"""

import math

import numpy as np
import pytest

from quantfinlib.volatility import Egarch11

E_ABS_Z = math.sqrt(2 / math.pi)


def test_egarch_finds_planted_leverage_and_its_absence():
    # Simulate EGARCH with strong planted leverage (gamma = -0.12).
    rng = np.random.default_rng(42)
    n = 4_000
    beta, alpha, gamma = 0.9, 0.15, -0.12
    omega = (1 - beta) * math.log(1e-4)
    r = np.empty(n)
    ln_h = math.log(1e-4)
    zs = rng.standard_normal(n)
    for t in range(n):
        z = zs[t]
        r[t] = math.exp(ln_h / 2) * z
        ln_h = omega + beta * ln_h + alpha * (abs(z) - E_ABS_Z) + gamma * z
    fit = Egarch11.fit(r)
    assert fit.gamma < -0.05, f"the planted leverage is found AS A SIGN: {fit.gamma}"
    assert 0.8 < fit.beta < 0.98, f"log-variance persistence: {fit.beta}"
    assert fit.alpha > 0.03, f"the magnitude effect: {fit.alpha}"
    assert fit.unconditional_log_variance() == pytest.approx(
        fit.omega / (1 - fit.beta), abs=1e-12)

    # Symmetric GARCH data: the honest answer is gamma ~ 0.
    sym = np.empty(n)
    h = 2e-6 / (1 - 0.08 - 0.9)
    zs2 = rng.standard_normal(n)
    for t in range(n):
        sym[t] = math.sqrt(h) * zs2[t]
        h = 2e-6 + 0.08 * sym[t] * sym[t] + 0.9 * h
    assert abs(Egarch11.fit(sym).gamma) < 0.08, "no asymmetry to find"


def test_leverage_means_down_moves_raise_tomorrows_vol_more():
    rng = np.random.default_rng(7)
    base = 0.01 * rng.standard_normal(200)
    # Identical histories except the LAST return's sign.
    down = base.copy()
    up = base.copy()
    down[-1] = -0.03
    up[-1] = 0.03
    leveraged = Egarch11.Params(0.1 * math.log(1e-4), 0.15, -0.12, 0.9, 0)
    after_down = Egarch11.next_variance(down, leveraged)
    after_up = Egarch11.next_variance(up, leveraged)
    assert after_down > after_up, \
        f"gamma < 0: the down move frightens tomorrow more ({after_down} > {after_up})"

    # Every conditional variance is positive BY CONSTRUCTION — the log
    # form needs no parameter constraints to guarantee it.
    variances = Egarch11.conditional_variances(down, leveraged)
    assert np.all(variances > 0)
    with pytest.raises(ValueError):
        Egarch11.fit(np.zeros(50))
    with pytest.raises(ValueError):
        Egarch11.fit(np.zeros(200))  # zero variance is not a fittable series


def test_one_step_recursion_matches_hand_arithmetic_exactly():
    # Two returns {0.01, -0.02}: mean -0.005, sample var 4.5e-4.
    # z1 = 0.015/sqrt(4.5e-4) = 1/sqrt(2); the recursion is then fully
    # determined — computed by hand below and pinned.
    p = Egarch11.Params(0.1 * math.log(1e-4), 0.15, -0.12, 0.9, 0)
    r = [0.01, -0.02]
    ln_h0 = math.log(4.5e-4)
    z1 = 0.015 / math.sqrt(4.5e-4)
    ln_h1 = p.omega + 0.9 * ln_h0 + 0.15 * (z1 - E_ABS_Z) - 0.12 * z1
    cv = Egarch11.conditional_variances(r, p)
    assert cv[0] == pytest.approx(4.5e-4, abs=1e-18), "seeded at the sample variance"
    assert cv[1] == pytest.approx(math.exp(ln_h1), abs=1e-15), \
        "the exact one-step transition"

    z2 = -0.015 / math.sqrt(cv[1])
    ln_h2 = (p.omega + 0.9 * math.log(cv[1])
             + 0.15 * (abs(z2) - E_ABS_Z) - 0.12 * z2)
    assert Egarch11.next_variance(r, p) == pytest.approx(math.exp(ln_h2), abs=1e-15), \
        "next_variance IS the recursion's next step, nothing else"

    # The house gates: a constant series must throw, never return h = 0
    # (which would contradict 'positive by construction').
    with pytest.raises(ValueError):
        Egarch11.conditional_variances([0.01, 0.01, 0.01], p)
    with pytest.raises(ValueError):
        Egarch11.next_variance([0.01, math.nan], p)
    with pytest.raises(ValueError):
        Egarch11.conditional_variances([0.01], p)


def test_multi_step_forecast_is_refused():
    # Iterating the log recursion forecasts the MEDIAN variance, not the
    # mean — the Java reference omits the method entirely; the Python
    # port raises the refusal with the explanation.
    p = Egarch11.Params(0.1 * math.log(1e-4), 0.15, -0.12, 0.9, 0)
    with pytest.raises(RuntimeError):
        Egarch11.forecast_variance([0.01, -0.02], p, 5)
    with pytest.raises(RuntimeError):
        Egarch11.forecast_variance([0.01, -0.02], p, 1)
