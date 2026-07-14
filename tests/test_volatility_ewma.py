"""Pins for quantfinlib.volatility.ewma_volatility.

Java sources: VolatilityModelsTest.ewmaReactsToVolatilityRegimeShift and
FormulaPinsTest.ewmaRecursionWeightsPinned (the exact hand-computed
recursion pins). Simulated series use numpy's Generator instead of
Java's SplittableRandom, so the behavioral asserts keep the Java
tolerances while the exact pins are RNG-free.
"""

import math

import numpy as np
import pytest

from quantfinlib.volatility import EwmaVolatility


def test_ewma_reacts_to_volatility_regime_shift():
    rng = np.random.default_rng(5)
    vol = np.where(np.arange(600) < 500, 0.005, 0.03)  # regime shift at t=500
    r = vol * rng.standard_normal(600)
    ewma = EwmaVolatility.risk_metrics()
    calm = ewma.latest_vol(r[:500])
    stressed = ewma.latest_vol(r)
    assert stressed > 2 * calm, f"stressed {stressed} vs calm {calm}"
    # Variance series aligns with returns and stays positive.
    h = ewma.variances(r)
    assert h.shape[0] == r.shape[0]
    assert np.all(h > 0)


def test_ewma_lambda_gates():
    for bad in (1.5, 1.0, 0.0, -0.1):
        with pytest.raises(ValueError):
            EwmaVolatility(bad)


def test_ewma_recursion_weights_pinned():
    # Ported from FormulaPinsTest: {0.01, -0.02}: seed = sample variance
    # 4.5e-4 (mean -0.005, devs +/-0.015, sum d^2 = 4.5e-4, n-1 = 1);
    # h[1] = 0.94*4.5e-4 + 0.06*0.01^2 = 4.29e-4 (a lambda swap gives
    # 1.21e-4); next = 0.94*4.29e-4 + 0.06*0.02^2 = 4.2726e-4 -> vol
    # 0.0206703.
    ewma = EwmaVolatility(0.94)
    h = ewma.variances([0.01, -0.02])
    assert h[0] == pytest.approx(4.5e-4, abs=1e-15)  # unconditional seed
    # lambda on the OLD variance, 1-lambda on r^2:
    assert h[1] == pytest.approx(4.29e-4, abs=1e-15)
    assert ewma.latest_vol([0.01, -0.02]) == pytest.approx(
        math.sqrt(4.2726e-4), abs=1e-12)


def test_ewma_annualized_vol_is_sqrt_scaled():
    # annualized = latest * sqrt(252), by definition.
    ewma = EwmaVolatility(0.94)
    r = [0.01, -0.02, 0.015]
    assert ewma.annualized_vol(r, 252) == pytest.approx(
        ewma.latest_vol(r) * math.sqrt(252), abs=1e-15)


def test_ewma_needs_two_returns():
    with pytest.raises(ValueError):
        EwmaVolatility(0.94).variances([0.01])
