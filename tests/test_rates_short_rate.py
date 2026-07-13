"""Short-rate model pins, ported from Java MarketRiskPricingTest.

Deterministic limits are exact, convexity has the Jensen sign, both
bonds fall in the short rate, the Feller ratio splits the two regimes,
and Hull-White reprices today's curve by construction.
"""

import math

import pytest

from quantfinlib.rates import ShortRateModels, YieldCurve


def test_vasicek_and_cir_bonds_behave_like_bonds():
    # sigma = 0, r = b: the rate never moves -> P = e^{-bT} exactly.
    assert ShortRateModels.vasicek_bond(0.03, 0.5, 0.03, 0, 5) == pytest.approx(
        math.exp(-0.03 * 5), abs=1e-12), "the deterministic limit is exact"
    # Convexity: with vol, the Gaussian rate makes the bond WORTH MORE
    # (Jensen on e^{-int r}).
    assert ShortRateModels.vasicek_bond(0.03, 0.5, 0.03, 0.02, 5) > math.exp(-0.03 * 5)
    # Both decrease in the short rate; both price below par for r > 0.
    assert (ShortRateModels.vasicek_bond(0.05, 0.5, 0.03, 0.01, 5)
            < ShortRateModels.vasicek_bond(0.02, 0.5, 0.03, 0.01, 5))
    assert (ShortRateModels.cir_bond(0.05, 0.5, 0.03, 0.1, 5)
            < ShortRateModels.cir_bond(0.02, 0.5, 0.03, 0.1, 5))
    assert ShortRateModels.cir_bond(0.03, 0.5, 0.03, 0.1, 5) < 1
    # Yields recover the log-price.
    assert ShortRateModels.vasicek_yield(0.03, 0.5, 0.03, 0, 5) == pytest.approx(
        0.03, abs=1e-12)
    # Feller: 2ab vs sigma^2.
    assert ShortRateModels.cir_feller(0.5, 0.03, 0.1) > 1
    assert ShortRateModels.cir_feller(0.1, 0.02, 0.2) < 1
    # Exact Vasicek step is mean-reverting in expectation (z = 0).
    assert ShortRateModels.vasicek_step(0.05, 0.5, 0.03, 0.01, 1.0, 0) == pytest.approx(
        0.03 + (0.05 - 0.03) * math.exp(-0.5), abs=1e-12)
    # CIR full truncation never sources vol from a negative rate.
    stepped = ShortRateModels.cir_step(-0.01, 0.5, 0.03, 0.1, 1.0 / 252, 3.0)
    assert math.isfinite(stepped), "no sqrt(negative)"
    with pytest.raises(ValueError):
        ShortRateModels.vasicek_bond(0.03, 0, 0.03, 0.01, 5)


def test_hull_white_reprices_todays_curve_by_construction():
    curve = YieldCurve.of_zero_rates([1, 2, 5, 10],
                                     [0.030, 0.030, 0.030, 0.030])
    # At t = 0 with r = f(0,0), P(0,T) must come back exactly.
    f0 = ShortRateModels.instantaneous_forward(curve, 0)
    assert f0 == pytest.approx(0.03, abs=1e-6), \
        "flat curve: instantaneous forward = the rate"
    assert ShortRateModels.hull_white_bond(curve, 0, 5, f0, 0.1, 0.01) == pytest.approx(
        curve.discount_factor(5), abs=1e-6), \
        "the fitted model disagrees with its own curve by nothing"
    # Away from the curve rate, higher r -> cheaper bond.
    assert (ShortRateModels.hull_white_bond(curve, 1, 5, 0.05, 0.1, 0.01)
            < ShortRateModels.hull_white_bond(curve, 1, 5, 0.02, 0.1, 0.01))

    # A steep short end: z(t) = 0.02 + 0.01t exactly (two pillars,
    # linear interpolation), so f(0,t) = 0.02 + 0.02t and g = -ln P is
    # quadratic — the short-end stencil is exact there, where a clamped
    # central difference would report f near (t+h)/2 instead.
    steep = YieldCurve.of_zero_rates([1e-4, 2], [0.020001, 0.04])
    assert ShortRateModels.instantaneous_forward(steep, 0.001) == pytest.approx(
        0.02002, abs=1e-7), "the forward AT t, not somewhere in a one-sided window"
    # Negative valuation time is rejected, not silently clamped to 0.
    with pytest.raises(ValueError):
        ShortRateModels.instantaneous_forward(steep, -5)
