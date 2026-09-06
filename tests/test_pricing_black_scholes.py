"""Pins for quantfinlib.pricing.black_scholes, ported from BlackScholesTest.java.

Textbook case (Hull): S=100, K=100, r=5%, q=0, sigma=20%, T=1 — the same
hand-checked values at the same tolerances as the Java asserts.
"""

import math

import pytest

from quantfinlib.pricing import BlackScholes, OptionType

S, K, R, Q, VOL, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0


def test_matches_textbook_values():
    assert BlackScholes.price(OptionType.CALL, S, K, R, Q, VOL, T) == pytest.approx(10.4506, abs=1e-3)
    assert BlackScholes.price(OptionType.PUT, S, K, R, Q, VOL, T) == pytest.approx(5.5735, abs=1e-3)
    assert BlackScholes.delta(OptionType.CALL, S, K, R, Q, VOL, T) == pytest.approx(0.6368, abs=1e-3)
    assert BlackScholes.gamma(S, K, R, Q, VOL, T) == pytest.approx(0.0188, abs=1e-3)
    assert BlackScholes.vega(S, K, R, Q, VOL, T) == pytest.approx(37.524, abs=1e-2)
    assert BlackScholes.theta(OptionType.CALL, S, K, R, Q, VOL, T) == pytest.approx(-6.414, abs=1e-2)
    assert BlackScholes.rho(OptionType.CALL, S, K, R, Q, VOL, T) == pytest.approx(53.232, abs=1e-2)


def test_put_call_parity_holds_with_carry():
    # C - P = S*e^(-qT) - K*e^(-rT), q = foreign rate (Garman-Kohlhagen).
    q = 0.03
    call = BlackScholes.price(OptionType.CALL, S, K, R, q, VOL, T)
    put = BlackScholes.price(OptionType.PUT, S, K, R, q, VOL, T)
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-R * T), abs=1e-9)


def test_greeks_record_consistent_with_individual_functions():
    g = BlackScholes.greeks(OptionType.PUT, S, K, R, Q, VOL, T)
    assert g.price == pytest.approx(BlackScholes.price(OptionType.PUT, S, K, R, Q, VOL, T), abs=1e-12)
    assert g.delta == pytest.approx(BlackScholes.delta(OptionType.PUT, S, K, R, Q, VOL, T), abs=1e-12)
    assert g.delta < 0 and g.gamma > 0 and g.vega > 0


def test_implied_vol_round_trips():
    price = BlackScholes.price(OptionType.CALL, S, 110, R, Q, 0.27, 0.5)
    iv = BlackScholes.implied_vol(OptionType.CALL, price, S, 110, R, Q, 0.5)
    assert iv == pytest.approx(0.27, abs=1e-6)


def test_implied_vol_unattainable_price_is_nan():
    # Below intrinsic: no vol can produce this price — NaN, not the bracket edge.
    assert math.isnan(BlackScholes.implied_vol(OptionType.CALL, 1e-12, S, 50, R, Q, 0.5))
    # Above the maximum attainable BS price.
    assert math.isnan(BlackScholes.implied_vol(OptionType.CALL, 2 * S, S, 100, R, Q, 0.5))


def test_expiry_collapses_to_intrinsic():
    assert BlackScholes.price(OptionType.CALL, 105, 100, R, Q, VOL, 0) == pytest.approx(5, abs=1e-12)
    assert BlackScholes.price(OptionType.PUT, 105, 100, R, Q, VOL, 0) == pytest.approx(0, abs=1e-12)
    assert BlackScholes.delta(OptionType.CALL, 105, 100, R, Q, VOL, 0) == pytest.approx(1, abs=1e-12)
    assert BlackScholes.gamma(105, 100, R, Q, VOL, 0) == pytest.approx(0, abs=1e-12)


def test_zero_vol_is_discounted_forward_intrinsic():
    # Deterministic world: max(0, S e^{-qT} - K e^{-rT}); ATM-forward must
    # not come back NaN (the 0/0 branch the Java comment documents).
    q = 0.02
    expected = max(0.0, S * math.exp(-q * T) - K * math.exp(-R * T))
    assert BlackScholes.price(OptionType.CALL, S, K, R, q, 0.0, T) == pytest.approx(expected, abs=1e-12)


# ----------------------------------------------------------------------
# Review M3/M4: implied_vol never returns a bracket endpoint; greeks at
# vol == 0 agree with price's deterministic branch instead of dividing
# by zero.
# ----------------------------------------------------------------------

def test_implied_vol_rejects_nan_price_and_expired_option():
    # Before the gate both returned 0.0001 — the lower search bound.
    with pytest.raises(ValueError):
        BlackScholes.implied_vol(OptionType.CALL, math.nan, 100, 100, 0.03, 0, 1)
    with pytest.raises(ValueError):
        BlackScholes.implied_vol(OptionType.CALL, 5.0, 100, 100, 0.03, 0, 0.0)
    with pytest.raises(ValueError):
        BlackScholes.implied_vol(OptionType.CALL, 5.0, 100, 100, 0.03, 0, -1.0)
    with pytest.raises(ValueError):
        BlackScholes.implied_vol(OptionType.CALL, -1.0, 100, 100, 0.03, 0, 1)
    with pytest.raises(ValueError):
        BlackScholes.implied_vol(OptionType.CALL, 5.0, math.nan, 100, 0.03, 0, 1)
    with pytest.raises(ValueError):
        BlackScholes.implied_vol(OptionType.CALL, 5.0, 100, -100, 0.03, 0, 1)


@pytest.mark.parametrize("vol", [0.05, 0.2, 1.0, 3.0])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_implied_vol_round_trips_across_the_bracket(vol, option_type):
    price = BlackScholes.price(option_type, 100, 100, 0.03, 0.01, vol, 1.0)
    iv = BlackScholes.implied_vol(option_type, price, 100, 100, 0.03, 0.01, 1.0)
    assert abs(iv - vol) < 1e-8


def test_unattainable_price_is_nan_not_a_bound():
    # Above the maximum BS price (S e^-qT) -> NaN, not 5.0.
    assert math.isnan(BlackScholes.implied_vol(OptionType.CALL, 150.0, 100, 100, 0.03, 0, 1))
    # Below intrinsic -> NaN, not 1e-4.
    assert math.isnan(BlackScholes.implied_vol(OptionType.CALL, 0.01, 120, 100, 0.03, 0, 1))


@pytest.mark.parametrize("spot", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_greeks_at_zero_vol_match_price_by_finite_difference(spot, option_type):
    K, r, q, T = 100.0, 0.03, 0.01, 1.0
    g = BlackScholes.greeks(option_type, spot, K, r, q, 0.0, T)      # used to raise
    assert all(math.isfinite(x) for x in (g.price, g.delta, g.gamma, g.vega, g.theta, g.rho))
    assert g.gamma == 0.0 and g.vega == 0.0
    h = 1e-5

    def price(**kw):
        args = dict(spot=spot, rate=r, vol=0.0, time_years=T)
        args.update(kw)
        return BlackScholes.price(option_type, args["spot"], K, args["rate"], q,
                                  args["vol"], args["time_years"])

    fd_delta = (price(spot=spot + h) - price(spot=spot - h)) / (2 * h)
    fd_rho = (price(rate=r + h) - price(rate=r - h)) / (2 * h)
    fd_theta = -(price(time_years=T + h) - price(time_years=T - h)) / (2 * h)
    assert g.delta == pytest.approx(fd_delta, abs=1e-6)
    assert g.rho == pytest.approx(fd_rho, abs=1e-5)
    assert g.theta == pytest.approx(fd_theta, abs=1e-5)


def test_zero_vol_greeks_atm_forward_take_the_call_side_of_the_step():
    # F == K exactly (r = q = 0 so F = S = K with no rounding): price is 0
    # for both, delta is the call step (1{F>K} = 0) and the put step
    # (-1{F<K} = 0) — no NaN, no exception. (Before the fix: 0/0 in d1.)
    for ot in (OptionType.CALL, OptionType.PUT):
        g = BlackScholes.greeks(ot, 100.0, 100.0, 0.0, 0.0, 0.0, 1.0)
        assert g.price == 0.0
        assert g.delta == 0.0 and g.theta == 0.0 and g.rho == 0.0


def test_nan_market_inputs_raise_instead_of_pricing_nan():
    for bad in (math.nan, math.inf, -1.0, 0.0):
        with pytest.raises(ValueError):
            BlackScholes.price(OptionType.CALL, bad, 100, 0.03, 0, 0.2, 1)
        with pytest.raises(ValueError):
            BlackScholes.greeks(OptionType.CALL, 100, bad, 0.03, 0, 0.2, 1)
    with pytest.raises(ValueError):
        BlackScholes.price(OptionType.CALL, 100, 100, math.nan, 0, 0.2, 1)
    with pytest.raises(ValueError):
        BlackScholes.price(OptionType.CALL, 100, 100, 0.03, 0, math.nan, 1)
    with pytest.raises(ValueError):
        BlackScholes.delta(OptionType.CALL, 100, 100, 0.03, 0, 0.2, math.nan)
