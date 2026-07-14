"""Swap-points curve, ported from Java SwapPointsCurveTest.

Pillar exactness, linear-in-days broken dates, slope extrapolation,
covered-interest-parity carry, and negative-points curves.
"""

import datetime as dt
import math

import pytest

from quantfinlib.fx import CurrencyPair, SwapPointsCurve

EURUSD = CurrencyPair.of("EURUSD")
TRADE = dt.date(2026, 1, 7)  # Wednesday, spot Fri 01-09


def _curve():
    return (SwapPointsCurve.builder(EURUSD, TRADE, 1.0850)
            .add("1M", 12.6)      # quoted in pips, out of order on purpose
            .add("1W", 3.1)
            .add("3M", 38.4)
            .build())


def test_pillars_reproduce_exactly():
    c = _curve()
    one_month = EURUSD.tenor_date(TRADE, "1M")
    assert c.outright(one_month) == pytest.approx(1.0850 + 12.6 * 0.0001, abs=1e-12)
    assert c.outright("1M") == pytest.approx(1.0850 + 12.6 * 0.0001, abs=1e-12)
    # Builder sorted the out-of-order quotes by date.
    assert c.pillar_tenors() == ["1W", "1M", "3M"]
    assert c.spot_date() == dt.date(2026, 1, 9)
    assert c.spot_rate() == 1.0850
    assert c.pair() == EURUSD


def test_broken_dates_interpolate_linearly_in_days():
    c = _curve()
    d1 = EURUSD.tenor_date(TRADE, "1M")
    d3 = EURUSD.tenor_date(TRADE, "3M")
    mid = d1 + dt.timedelta(days=(d3 - d1).days // 2)
    p1 = c.forward_points(d1)
    p3 = c.forward_points(d3)
    expected = p1 + (p3 - p1) * ((mid - d1).days / (d3 - d1).days)
    assert c.forward_points(mid) == pytest.approx(expected, abs=1e-12)
    # Before the first pillar: anchored at zero points on spot.
    assert c.forward_points(c.spot_date()) == pytest.approx(0, abs=1e-12)
    early = c.spot_date() + dt.timedelta(days=3)
    assert 0 < c.forward_points(early) < p1


def test_beyond_last_pillar_extends_the_final_slope():
    c = _curve()
    d1 = EURUSD.tenor_date(TRADE, "1M")
    d3 = EURUSD.tenor_date(TRADE, "3M")
    slope = (c.forward_points(d3) - c.forward_points(d1)) / (d3 - d1).days
    beyond = d3 + dt.timedelta(days=30)
    assert c.forward_points(beyond) == pytest.approx(
        c.forward_points(d3) + slope * 30, abs=1e-12)


def test_implied_carry_matches_covered_interest_parity():
    c = _curve()
    d3 = EURUSD.tenor_date(TRADE, "3M")
    tau = (d3 - c.spot_date()).days / 365.0
    carry = c.implied_carry(d3)
    assert carry == pytest.approx(math.log(c.outright(d3) / 1.0850) / tau, abs=1e-12)
    assert carry > 0


def test_negative_points_curve_is_supported():
    # Base yields more than quote: forwards below spot, negative carry.
    usdjpy = CurrencyPair.of("USDJPY")
    c = (SwapPointsCurve.builder(usdjpy, TRADE, 155.00)
         .add("1M", -22.0)
         .add("6M", -130.0)
         .build())
    d6 = usdjpy.tenor_date(TRADE, "6M")
    assert c.outright(d6) == pytest.approx(155.00 - 130.0 * 0.01, abs=1e-9)
    assert c.implied_carry(d6) < 0


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        SwapPointsCurve.builder(EURUSD, TRADE, 0).add("1M", 1).build()
    with pytest.raises(RuntimeError):
        SwapPointsCurve.builder(EURUSD, TRADE, 1.08).build()
    # Pre-spot legs belong to the roll, not the forward curve.
    with pytest.raises(ValueError):
        SwapPointsCurve.builder(EURUSD, TRADE, 1.08).add("ON", 0.1)
    with pytest.raises(ValueError):
        _curve().forward_points(TRADE)  # before spot
    with pytest.raises(ValueError):
        SwapPointsCurve.builder(EURUSD, TRADE, 1.08).add("1M", 1).add("1M", 2).build()
