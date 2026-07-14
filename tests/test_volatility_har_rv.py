"""Pins for quantfinlib.volatility.har_rv.

Java source: QuantSignalsTest.harRecoverssPlantedCoefficientsAndForecastsSanely.
The planted-HAR recovery uses numpy's Generator (uniform noise, same
amplitude) with the Java tolerances; the window-alignment forecast pin
and the zero floor transfer exactly (RNG-free).
"""

import numpy as np
import pytest

from quantfinlib.util import math_utils as mu
from quantfinlib.volatility import HarRv


def test_har_recovers_planted_coefficients_and_forecasts_sanely():
    # Plant the HAR process itself: c=0.02, bd=0.4, bw=0.3, bm=0.2
    # (persistence 0.9, stationary mean 0.2), small uniform noise.
    rng = np.random.default_rng(9)
    n = 1_500
    rv = np.empty(n)
    rv[:22] = 0.2
    noise = 0.005 * (2 * rng.random(n) - 1)
    for t in range(21, n - 1):
        w = mu.mean(rv, t - 4, t + 1)
        m = mu.mean(rv, t - 21, t + 1)
        rv[t + 1] = max(1e-6, 0.02 + 0.4 * rv[t] + 0.3 * w + 0.2 * m + noise[t])
    p = HarRv.fit(rv)
    assert p.beta_daily == pytest.approx(0.4, abs=0.08), f"daily: {p.beta_daily}"
    assert p.beta_weekly == pytest.approx(0.3, abs=0.20), f"weekly: {p.beta_weekly}"
    assert p.beta_monthly == pytest.approx(0.2, abs=0.20), f"monthly: {p.beta_monthly}"
    assert p.intercept == pytest.approx(0.02, abs=0.03)

    forecast = HarRv.forecast(rv, p)
    assert 0.1 < forecast < 0.3, \
        f"the forecast sits near the stationary level: {forecast}"

    # The floor: a pathological parameter set cannot forecast a negative
    # variance.
    assert HarRv.forecast(rv, HarRv.Params(-10, 0, 0, 0)) == 0.0


def test_har_window_alignment_pinned_exactly():
    # Window alignment pinned EXACTLY where the horizons disagree:
    # 21 flat days then a spiked last day. d = 0.5, w = (4*0.1+0.5)/5
    # = 0.18, m = (21*0.1+0.5)/22; a d/w swap or off-by-one window moves
    # this a lot.
    spiked = np.full(22, 0.1)
    spiked[21] = 0.5
    hand = HarRv.Params(0.01, 0.5, 0.3, 0.1)
    expected = 0.01 + 0.5 * 0.5 + 0.3 * 0.18 + 0.1 * (21 * 0.1 + 0.5) / 22
    assert HarRv.forecast(spiked, hand) == pytest.approx(expected, abs=1e-12), \
        "c + bd*d + bw*w + bm*m, by hand"


def test_har_gates():
    with pytest.raises(ValueError):
        HarRv.fit(np.zeros(59))
    rv = np.full(100, 0.2)
    bad = rv.copy()
    bad[50] = -0.1
    with pytest.raises(ValueError):
        HarRv.fit(bad)
    nan_bad = rv.copy()
    nan_bad[50] = np.nan  # the not (rv >= 0) gate is NaN-rejecting
    with pytest.raises(ValueError):
        HarRv.fit(nan_bad)
    with pytest.raises(ValueError):
        HarRv.forecast(np.full(21, 0.1), HarRv.Params(0, 1, 0, 0))
