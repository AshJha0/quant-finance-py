"""BondPricer pins, ported from Java RatesTest (bond half).

Zero-coupon closed forms, the par-bond identity (coupon == yield ->
price == face), YTM round trip, and yield-vs-curve pricing agreement on
a flat curve.
"""

import pytest

import math

from quantfinlib.rates import BondPricer, YieldCurve


def test_zero_coupon_bond_analytics():
    # 5y zero-coupon at 4% annual yield.
    price = BondPricer.price_from_yield(100, 0, 1, 5, 0.04)
    assert price == pytest.approx(100 / 1.04 ** 5, abs=1e-10)
    # Zero's Macaulay duration equals its maturity.
    assert BondPricer.macaulay_duration(100, 0, 1, 5, 0.04) == pytest.approx(5.0, abs=1e-10)
    assert BondPricer.modified_duration(100, 0, 1, 5, 0.04) == pytest.approx(
        5.0 / 1.04, abs=1e-10)


def test_par_bond_prices_at_face_and_ytm_round_trips():
    # Coupon == yield -> price == face.
    assert BondPricer.price_from_yield(100, 0.06, 2, 10, 0.06) == pytest.approx(
        100, abs=1e-9)

    price = BondPricer.price_from_yield(100, 0.05, 2, 7, 0.043)
    assert BondPricer.yield_to_maturity(price, 100, 0.05, 2, 7) == pytest.approx(
        0.043, abs=1e-8)


def test_convexity_and_dv01_are_positive_and_consistent():
    convexity = BondPricer.convexity(100, 0.05, 2, 10, 0.05)
    assert convexity > 0
    dv01 = BondPricer.dv01(100, 0.05, 2, 10, 0.05)
    # First-order check: price change for 1bp ~ DV01.
    p0 = BondPricer.price_from_yield(100, 0.05, 2, 10, 0.05)
    p1 = BondPricer.price_from_yield(100, 0.05, 2, 10, 0.0501)
    assert p0 - p1 == pytest.approx(dv01, abs=dv01 * 0.01)


def test_curve_pricing_matches_yield_pricing_on_flat_curve():
    # Flat cc curve at z; equivalent semi-annual yield y = 2(e^{z/2}-1).
    z = 0.05
    flat = YieldCurve.of_zero_rates([1, 30], [z, z])
    y = 2 * (math.exp(z / 2) - 1)
    from_curve = BondPricer.price_from_curve(100, 0.06, 2, 10, flat)
    from_yield = BondPricer.price_from_yield(100, 0.06, 2, 10, y)
    assert from_curve == pytest.approx(from_yield, abs=1e-6)


def test_ytm_bracket_check_refuses_unattainable_prices():
    # A price no yield in [-90%, 1000%] can explain: bisection would
    # silently hand back an endpoint; the gate raises instead.
    with pytest.raises(ValueError):
        BondPricer.yield_to_maturity(1e9, 100, 0.05, 2, 7)
    with pytest.raises(ValueError):
        BondPricer.yield_to_maturity(1e-9, 100, 0.05, 2, 7)


def test_bond_gates_refuse_nonsense():
    with pytest.raises(ValueError):
        BondPricer.price_from_yield(100, 0.05, 0, 5, 0.05)    # frequency < 1
    with pytest.raises(ValueError):
        BondPricer.price_from_yield(100, 0.05, 2, 0, 0.05)    # zero maturity
