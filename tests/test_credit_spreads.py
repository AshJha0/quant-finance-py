"""Z-spread pins, ported from Java CreditTest and CreditRoundTripTest.

An on-curve bond has zero Z-spread, planted spreads round-trip through
the solver, price falls strictly in z, and impossible prices raise
instead of returning the bracket edge.
"""

import pytest

from quantfinlib.credit import CreditSpreads
from quantfinlib.rates import YieldCurve


def flat3() -> YieldCurve:
    return YieldCurve.of_zero_rates([1, 2, 3, 5, 7, 10], [0.03] * 6)


def test_curve_priced_bond_has_zero_z_spread_and_round_trips():
    curve = flat3()
    on_curve = CreditSpreads.price_with_z_spread(100, 0.05, 2, 5, curve, 0)
    assert CreditSpreads.z_spread(on_curve, 100, 0.05, 2, 5, curve) == pytest.approx(
        0.0, abs=1e-10)

    # A credit-risky price below the curve price implies positive z, and
    # the solved z reprices the bond exactly.
    risky = on_curve * 0.95
    z = CreditSpreads.z_spread(risky, 100, 0.05, 2, 5, curve)
    assert z > 0
    assert CreditSpreads.price_with_z_spread(100, 0.05, 2, 5, curve, z) == pytest.approx(
        risky, abs=1e-9)


def test_planted_z_spreads_round_trip_and_price_falls_in_z():
    curve = flat3()
    # Plant z, price, solve back: the solver must return the plant.
    for z in [0.005, 0.02, 0.08]:
        px = CreditSpreads.price_with_z_spread(100, 0.05, 2, 5, curve, z)
        assert CreditSpreads.z_spread(px, 100, 0.05, 2, 5, curve) == pytest.approx(
            z, abs=1e-10), f"z={z} must round-trip"
    # More z discounts every flow harder: price strictly decreasing.
    p0 = CreditSpreads.price_with_z_spread(100, 0.05, 2, 5, curve, 0.00)
    p1 = CreditSpreads.price_with_z_spread(100, 0.05, 2, 5, curve, 0.02)
    p2 = CreditSpreads.price_with_z_spread(100, 0.05, 2, 5, curve, 0.08)
    assert p0 > p1 > p2, "price must fall as z rises"


def test_impossible_prices_raise_instead_of_returning_the_bracket_edge():
    curve = flat3()
    # Far above the maximum attainable PV (z = -50%).
    with pytest.raises(ValueError):
        CreditSpreads.z_spread(1e7, 100, 0.05, 2, 5, curve)
    # Effectively free bond: below the z = 500% price.
    with pytest.raises(ValueError):
        CreditSpreads.z_spread(1e-9, 100, 0.05, 2, 5, curve)


def test_z_spread_input_gates():
    curve = flat3()
    with pytest.raises(ValueError):
        CreditSpreads.z_spread(-1, 100, 0.05, 2, 5, curve)      # negative price
    with pytest.raises(ValueError):
        CreditSpreads.z_spread(90, 0, 0.05, 2, 5, curve)        # zero face
    with pytest.raises(ValueError):
        CreditSpreads.z_spread(90, 100, -0.05, 2, 5, curve)     # negative coupon
    with pytest.raises(ValueError):
        CreditSpreads.z_spread(90, 100, 0.05, 0, 5, curve)      # frequency < 1
    with pytest.raises(ValueError):
        CreditSpreads.z_spread(90, 100, 0.05, 2, 0, curve)      # zero maturity
    with pytest.raises(ValueError):
        CreditSpreads.price_with_z_spread(100, 0.05, 2, 5, curve, float("nan"))
