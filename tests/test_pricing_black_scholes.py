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
