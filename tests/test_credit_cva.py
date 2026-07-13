"""CVA pins, ported from Java CvaTest and CreditRoundTripTest.

Pinned by hand: flat exposure on a flat-hazard curve is a sum of
exponentials anyone can recompute, a riskless counterparty costs
(essentially) nothing, more spread always means more CVA, and the
formula is exactly linear in both exposure and LGD.
"""

import math

import pytest

from quantfinlib.credit import CreditCurve, CvaApproximator
from quantfinlib.rates import YieldCurve


def flat3() -> YieldCurve:
    return YieldCurve.of_zero_rates([1, 2, 3, 5, 7, 10], [0.03] * 6)


def quarterly_grid(years: float) -> list[float]:
    n = round(years * 4)
    return [(i + 1) * 0.25 for i in range(n)]


def test_flat_exposure_flat_hazard_matches_the_hand_sum():
    discount = flat3()
    # Single 10y pillar: the bootstrapped hazard is flat, so
    # Q(t) = e^{-h t} exactly and the CVA sum is pure arithmetic.
    curve = CreditCurve.bootstrap([10], [0.02], 0.40, discount)
    h = curve.hazard(1.0)
    lgd = 0.6
    grid = quarterly_grid(5)
    ee = [1_000_000.0] * len(grid)

    expected = 0.0
    prev_t = 0.0
    for t in grid:
        expected += 1_000_000 * (math.exp(-h * prev_t) - math.exp(-h * t)) \
            * math.exp(-0.03 * t)
        prev_t = t
    expected *= lgd
    assert CvaApproximator.cva(ee, grid, curve, discount, lgd) == pytest.approx(
        expected, abs=1e-6)
    # Order of magnitude sanity: LGD * EE * 5y default probability,
    # lightly discounted — tens of thousands on a million of EE.
    assert 50_000 < expected < 120_000, f"cva={expected}"


def test_riskless_counterparty_costs_nothing():
    discount = flat3()
    # The bootstrap gates demand a positive spread; a vanishing one is
    # the riskless limit and the CVA must vanish with it.
    curve = CreditCurve.bootstrap([5], [1e-8], 0.40, discount)
    grid = quarterly_grid(5)
    ee = [1.0] * len(grid)
    assert CvaApproximator.cva(ee, grid, curve, discount, 0.6) == pytest.approx(
        0, abs=1e-6)


def test_cva_increases_with_the_spread_level():
    discount = flat3()
    tight = CreditCurve.bootstrap([1, 3, 5], [0.01, 0.01, 0.01], 0.40, discount)
    wide = CreditCurve.bootstrap([1, 3, 5], [0.02, 0.02, 0.02], 0.40, discount)
    grid = quarterly_grid(5)
    ee = [1_000_000.0] * len(grid)
    cva_tight = CvaApproximator.cva(ee, grid, tight, discount, 0.6)
    cva_wide = CvaApproximator.cva(ee, grid, wide, discount, 0.6)
    assert cva_tight > 0
    assert cva_wide > cva_tight, \
        f"200bp CVA {cva_wide} must exceed 100bp CVA {cva_tight}"
    # Not quite 2x (survival decays faster on the wide curve), but close.
    assert cva_wide < 2 * cva_tight


def test_cva_is_linear_in_both_exposure_and_lgd():
    # CVA = LGD * sum EE_i * dPD_i * DF_i: doubling the exposure profile
    # or the LGD doubles the number exactly.
    discount = flat3()
    curve = CreditCurve.bootstrap([5], [0.02], 0.40, discount)
    t = [0.5, 1.0, 1.5, 2.0]
    ee = [100, 250, 175, 50]
    ee2 = [200, 500, 350, 100]
    base = CvaApproximator.cva(ee, t, curve, discount, 0.3)
    assert base > 0
    assert CvaApproximator.cva(ee2, t, curve, discount, 0.3) == pytest.approx(
        2 * base, abs=1e-12)
    assert CvaApproximator.cva(ee, t, curve, discount, 0.6) == pytest.approx(
        2 * base, abs=1e-12)


def test_two_period_cva_matches_the_hand_sum():
    # Two buckets, hand-computed: with flat hazard h and flat 3% cc
    # discounting,
    #   CVA = LGD * [ EE1 * (1 - e^{-h/2}) * e^{-0.015}
    #               + EE2 * (e^{-h/2} - e^{-h}) * e^{-0.03} ].
    discount = flat3()
    curve = CreditCurve.bootstrap([1], [0.02], 0.40, discount)
    h = curve.hazard(0.5)
    lgd = 0.6
    expected = lgd * (100 * (1 - math.exp(-h * 0.5)) * math.exp(-0.03 * 0.5)
                      + 200 * (math.exp(-h * 0.5) - math.exp(-h)) * math.exp(-0.03 * 1.0))
    assert CvaApproximator.cva([100, 200], [0.5, 1.0], curve, discount,
                               lgd) == pytest.approx(expected, abs=1e-12)


def test_gates_refuse_nonsense():
    discount = flat3()
    curve = CreditCurve.bootstrap([5], [0.01], 0.40, discount)
    ok_t = [0.5, 1.0]
    ok_e = [1.0, 1.0]
    with pytest.raises(ValueError):     # empty
        CvaApproximator.cva([], [], curve, discount, 0.6)
    with pytest.raises(ValueError):     # misaligned
        CvaApproximator.cva(ok_e, [1.0], curve, discount, 0.6)
    with pytest.raises(ValueError):     # non-ascending times
        CvaApproximator.cva(ok_e, [1.0, 0.5], curve, discount, 0.6)
    with pytest.raises(ValueError):     # t[0] = 0
        CvaApproximator.cva(ok_e, [0, 1.0], curve, discount, 0.6)
    with pytest.raises(ValueError):     # NaN time
        CvaApproximator.cva(ok_e, [0.5, float("nan")], curve, discount, 0.6)
    with pytest.raises(ValueError):     # negative exposure
        CvaApproximator.cva([-1, 1], ok_t, curve, discount, 0.6)
    with pytest.raises(ValueError):     # NaN exposure
        CvaApproximator.cva([float("nan"), 1], ok_t, curve, discount, 0.6)
    with pytest.raises(ValueError):     # lgd = 0
        CvaApproximator.cva(ok_e, ok_t, curve, discount, 0)
    with pytest.raises(ValueError):     # lgd > 1
        CvaApproximator.cva(ok_e, ok_t, curve, discount, 1.2)
