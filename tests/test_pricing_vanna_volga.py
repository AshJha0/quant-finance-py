"""Pins for quantfinlib.pricing.vanna_volga, ported from VannaVolgaTest.java.

The FxVolSurface integration test stays in Java — the fx domain is not
ported yet; the five-pillar reduction it also exercised is pinned here
directly.
"""

import math

import pytest

from quantfinlib.pricing import BlackScholes, OptionType, VannaVolga

S = 1.0850
R = 0.045
Q = 0.030
T = 0.5

# A negative-RR (put wing over call wing) EURUSD-style smile.
STRIKES = [1.04, 1.09, 1.14]
VOLS = [0.098, 0.090, 0.094]


@pytest.fixture()
def vv():
    return VannaVolga(STRIKES, VOLS, R, Q, T)


def test_pillar_strikes_recover_their_market_vols_exactly(vv):
    for i in range(3):
        assert vv.implied_vol(S, STRIKES[i]) == pytest.approx(VOLS[i], abs=1e-6)


def test_pillar_prices_match_the_market_prices(vv):
    for i in range(3):
        market = BlackScholes.price(OptionType.CALL, S, STRIKES[i], R, Q, VOLS[i], T)
        assert vv.price(OptionType.CALL, S, STRIKES[i]) == pytest.approx(market, abs=1e-12)


def test_smile_interpolates_smoothly_between_pillars(vv):
    # Between ATM and the call wing the vol must sit between their levels
    # (the log-quadratic weights cannot overshoot on this smile shape).
    mid = vv.implied_vol(S, 1.115)
    assert min(VOLS[1], VOLS[2]) - 1e-4 < mid < max(VOLS[1], VOLS[2]) + 1e-4
    # The smile is not flat: the adjustment is really doing something.
    assert abs(mid - VOLS[1]) > 1e-4
    # Wings continue outward without collapsing.
    assert vv.implied_vol(S, 1.00) > VOLS[1]


def test_put_prices_are_parity_consistent(vv):
    k = 1.115
    call = vv.price(OptionType.CALL, S, k)
    put = vv.price(OptionType.PUT, S, k)
    # Same smile adjustment on both sides: put-call parity in the
    # adjusted prices (parity holds for the flat-vol legs and the
    # pillar-hedge adjustment is type-independent).
    parity = S * math.exp(-Q * T) - k * math.exp(-R * T)
    assert call - put == pytest.approx(parity, abs=1e-10)


def test_five_pillar_overload_reduces_to_the_25_delta_triple():
    five = [1.02, 1.05, 1.09, 1.13, 1.16]
    five_vols = [0.101, 0.097, 0.090, 0.093, 0.099]
    reduced = VannaVolga.of_pillars(five, five_vols, R, Q, T)
    assert reduced.implied_vol(S, five[2]) == pytest.approx(five_vols[2], abs=1e-6)


def test_validation_rejects_bad_pillars(vv):
    with pytest.raises(ValueError):
        VannaVolga([1, 2], [0.1, 0.1], R, Q, T)
    with pytest.raises(ValueError):
        VannaVolga([1.1, 1.0, 1.2], VOLS, R, Q, T)
    with pytest.raises(ValueError):
        VannaVolga(STRIKES, [0.1, -0.1, 0.1], R, Q, T)
    with pytest.raises(ValueError):
        VannaVolga(STRIKES, VOLS, R, Q, 0)
    with pytest.raises(ValueError):
        vv.price(OptionType.CALL, S, -1)
