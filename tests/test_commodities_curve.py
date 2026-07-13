"""Commodity curve pins, ported from Java AssetClassRoundTest and
AssetClassEdgeTest (commodities sections).

Contango charges the long, backwardation pays the roll, implied carry
is the exact storage-arbitrage inversion, and both wings refuse to
extrapolate.
"""

import math

import pytest

from quantfinlib.commodities import CommodityCurve


def test_contango_curve_charges_the_long_and_carry_is_exact():
    curve = CommodityCurve.of(100, [0.25, 0.5, 1.0], [101, 102, 104])
    assert curve.is_contango()
    assert not curve.is_backwardation()
    assert curve.price(0.75) == pytest.approx(103, abs=1e-12), "linear between pillars"
    assert curve.annualized_roll_yield(0.25, 1.0) < 0, \
        "rolling long in contango pays away"
    # F = S e^{(r+u-y)t}: implied u-y = ln(104/100)/1 - 3%.
    assert curve.implied_carry(1.0, 0.03) == pytest.approx(
        math.log(1.04) - 0.03, abs=1e-12)


def test_backwardation_pays_the_roll_and_gates_hold():
    curve = CommodityCurve.of(100, [0.25, 0.5, 1.0], [99, 97, 95])
    assert curve.is_backwardation()
    assert curve.annualized_roll_yield(0.25, 1.0) == pytest.approx(
        math.log(99.0 / 95.0) / 0.75, abs=1e-12)
    with pytest.raises(ValueError):
        curve.price(2.0)                                    # no extrapolation
    with pytest.raises(ValueError):
        CommodityCurve.of(100, [0.5, 0.25], [99, 98])       # descending
    with pytest.raises(ValueError):
        curve.annualized_roll_yield(1.0, 0.5)               # far <= near


def test_flat_commodity_curve_is_neither_shape_and_carries_minus_r():
    # All futures AT spot: both shape tests are strict, so both are
    # false; ln(F/F) = 0 makes the roll yield exactly zero and the
    # implied carry exactly -r (ln(F/S)/t = 0).
    flat = CommodityCurve.of(100, [0.25, 0.5, 1.0], [100, 100, 100])
    assert not flat.is_contango()
    assert not flat.is_backwardation()
    assert flat.annualized_roll_yield(0.25, 1.0) == 0.0
    assert flat.implied_carry(1.0, 0.03) == -0.03
    assert flat.spot() == 100


def test_pillar_prices_are_exact_and_both_wings_refuse_to_extrapolate():
    curve = CommodityCurve.of(100, [0.5, 1.0], [102, 104])
    assert curve.price(0.5) == 102, "exact at the pillar"
    assert curve.price(0.75) == pytest.approx(103, abs=1e-12), "linear midpoint"
    with pytest.raises(ValueError):
        curve.price(0.25)                                   # before first
    with pytest.raises(ValueError):
        curve.price(1.5)                                    # after last
    with pytest.raises(ValueError):
        curve.implied_carry(1.0, float("nan"))
    with pytest.raises(ValueError):
        CommodityCurve.of(0, [1], [100])                    # zero spot
    with pytest.raises(ValueError):
        CommodityCurve.of(100, [1], [-5])                   # negative px
