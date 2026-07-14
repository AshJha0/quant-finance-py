"""Pins for quantfinlib.volatility.{garch11,gjr_garch11}.

Java sources: VolatilityModelsTest (GARCH recovery + forecast mean
reversion), FormulaPinsTest.garchForecastHorizonExponentAnchored (exact
horizon-exponent pins), MarketRiskTest.gjrFindsTheLeverageEffect... and
gjrGridIsNotAHiddenParameterCap. Simulations use numpy's Generator (not
Java's RNGs) with the same model parameters and the Java tolerances.
"""

import math

import numpy as np
import pytest

from quantfinlib.util import math_utils as mu
from quantfinlib.volatility import Garch11, GjrGarch11


def simulate_garch(omega, alpha, beta, n, seed):
    """GARCH(1,1) return series with known parameters (Java helper port)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    r = np.empty(n)
    h = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = math.sqrt(h) * z[t]
        h = omega + alpha * r[t] * r[t] + beta * h
    return r


def simulate_gjr(omega, alpha, gamma, beta, n, seed):
    """GJR return series: down moves feed variance at alpha + gamma."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    r = np.empty(n)
    h = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        r[t] = math.sqrt(h) * z[t]
        arch = alpha + gamma if r[t] < 0 else alpha
        h = omega + arch * r[t] * r[t] + beta * h
    return r


# ---------------------------------------------------------------- GARCH(1,1)

def test_garch_fit_recovers_simulated_parameters():
    # True: alpha=0.08, beta=0.90, daily vol 1% (uncond var 1e-4).
    uncond = 1e-4
    r = simulate_garch(uncond * 0.02, 0.08, 0.90, 4_000, 42)
    fit = Garch11.fit(r)
    assert 0.90 < fit.persistence() < 0.9995, f"persistence={fit.persistence()}"
    assert fit.alpha == pytest.approx(0.08, abs=0.06)
    assert fit.unconditional_variance() == pytest.approx(uncond, abs=uncond * 0.35)
    assert math.isfinite(fit.log_likelihood)


def test_garch_forecast_mean_reverts_to_unconditional():
    r = simulate_garch(2e-6, 0.08, 0.90, 3_000, 7)
    fit = Garch11.fit(r)
    short_horizon = Garch11.forecast_variance(r, fit, 1)
    long_horizon = Garch11.forecast_variance(r, fit, 5_000)
    assert long_horizon == pytest.approx(fit.unconditional_variance(),
                                         abs=fit.unconditional_variance() * 0.01)
    assert short_horizon > 0
    # Conditional variances track squared-return clusters.
    h = Garch11.conditional_variances(r, fit)
    assert h.shape[0] == r.shape[0]


def test_garch_forecast_horizon_exponent_anchored():
    # Ported from FormulaPinsTest: horizon 1 IS the one-step GARCH update
    # (persistence^0 = 1); horizon 2 applies ONE persistence factor.
    p = Garch11.Params(2e-6, 0.08, 0.9, 0)
    rng = np.random.default_rng(3)
    r = 0.01 * rng.standard_normal(300)
    mean = mu.mean(r)
    h = Garch11.conditional_variances(r, p)
    last_r = r[-1] - mean
    one_step = 2e-6 + 0.08 * last_r * last_r + 0.9 * h[-1]
    assert Garch11.forecast_variance(r, p, 1) == pytest.approx(one_step, abs=1e-18)
    uncond = p.unconditional_variance()
    assert Garch11.forecast_variance(r, p, 2) == pytest.approx(
        uncond + p.persistence() * (one_step - uncond), abs=1e-18)


def test_garch_gates():
    with pytest.raises(ValueError):
        Garch11.fit(np.zeros(99))
    with pytest.raises(ValueError):
        Garch11.forecast_variance(np.zeros(10), Garch11.Params(2e-6, 0.08, 0.9, 0), 0)


# ---------------------------------------------------------------- GJR-GARCH

def test_gjr_finds_the_leverage_effect_and_its_absence_honestly():
    # Planted leverage: omega=2e-6, alpha=0.03, gamma=0.12, beta=0.85.
    r = simulate_gjr(2e-6, 0.03, 0.12, 0.85, 4_000, 42)
    fit = GjrGarch11.fit(r)
    assert fit.gamma > 0.05, f"the planted asymmetry is found: gamma={fit.gamma}"
    assert fit.persistence() < 1, "stationary fit"
    assert fit.log_likelihood > Garch11.fit(r).log_likelihood, \
        "the model that generated the data likelihood-beats the symmetric one"

    # Symmetric GARCH data: the honest answer is gamma ~ 0.
    sym = simulate_garch(2e-6, 0.08, 0.9, 4_000, 43)
    assert GjrGarch11.fit(sym).gamma < 0.06, "no asymmetry to find"

    # Forecast mean-reverts toward the unconditional variance.
    uncond = fit.unconditional_variance()
    near = GjrGarch11.forecast_variance(r, fit, 1)
    far = GjrGarch11.forecast_variance(r, fit, 500)
    assert abs(far - uncond) < abs(near - uncond) + 1e-15, "long horizons forget today"
    with pytest.raises(ValueError):
        GjrGarch11.fit(np.zeros(50))


def test_gjr_grid_is_not_a_hidden_parameter_cap():
    # Low-persistence / high-ARCH data (short intraday windows and regime
    # breaks produce it): true alpha = 0.45, beta = 0.15 — a fit box
    # capped at alpha <= 0.30 can only creep to ~0.37 across the
    # refinement passes and pins at the edge silently. (Seed chosen so
    # the symmetric-data alpha/gamma split does not blur the alpha read:
    # this draw fits gamma = 0 and puts all the ARCH mass on alpha.)
    rng = np.random.default_rng(6)
    z = rng.standard_normal(4_000)
    hot = np.empty(4_000)
    hv = 1e-4
    for t in range(4_000):
        hot[t] = math.sqrt(hv) * z[t]
        hv = 4e-5 + 0.45 * hot[t] * hot[t] + 0.15 * hv
    fit = GjrGarch11.fit(hot)
    assert fit.alpha > 0.38, f"the MLE reaches the true high-ARCH region: {fit.alpha}"
    assert fit.beta < 0.35, f"and the true low beta: {fit.beta}"


def test_gjr_forecast_horizon_semantics_match_garch():
    # Same anchoring as the Garch11 pin, with the asymmetric first step:
    # last return's sign selects alpha or alpha + gamma.
    p = GjrGarch11.Params(2e-6, 0.05, 0.08, 0.85, 0)
    rng = np.random.default_rng(11)
    r = 0.01 * rng.standard_normal(300)
    mean = mu.mean(r)
    h = GjrGarch11.conditional_variances(r, p)
    last_r = r[-1] - mean
    arch = p.alpha + p.gamma if last_r < 0 else p.alpha
    one_step = p.omega + arch * last_r * last_r + p.beta * h[-1]
    assert GjrGarch11.forecast_variance(r, p, 1) == pytest.approx(one_step, abs=1e-18)
    uncond = p.unconditional_variance()
    assert GjrGarch11.forecast_variance(r, p, 2) == pytest.approx(
        uncond + p.persistence() * (one_step - uncond), abs=1e-18)
    # persistence = alpha + gamma/2 + beta = 0.05 + 0.04 + 0.85 = 0.94.
    assert p.persistence() == pytest.approx(0.94, abs=1e-15)
    # uncond = omega / (1 - persistence) = 2e-6 / 0.06.
    assert uncond == pytest.approx(2e-6 / 0.06, abs=1e-18)
