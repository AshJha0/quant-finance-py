"""Pins for Black76, HigherOrderGreeks and Heston, ported from
MarketRiskPricingTest.java (the short-rate and key-rate-duration parts
stay in Java — the rates domain is not ported yet).

Port note: Heston.call_monte_carlo draws from numpy.random.default_rng
instead of java.util.Random; the agreement assertion is statistical
(tolerance 0.35 as in Java), so the RNG swap is immaterial.
"""

import math

import pytest

from quantfinlib.pricing import (Black76, BlackScholes, Heston, HestonParams,
                                 HigherOrderGreeks, OptionType)

# ------------------------------------------------------------------
# Black-76
# ------------------------------------------------------------------


def test_black76_is_black_scholes_with_zero_carry_on_the_forward():
    f = 102.0
    k = 100.0
    r = 0.03
    vol = 0.25
    t = 0.75
    # The identity in the house convention: BlackScholes' carry is the
    # YIELD q, so a driftless forward means q = r.
    assert Black76.price(OptionType.CALL, f, k, r, vol, t) == pytest.approx(
        BlackScholes.price(OptionType.CALL, f, k, r, r, vol, t), abs=1e-10)
    # Put-call parity on the forward: C - P = df*(F - K).
    c = Black76.price(OptionType.CALL, f, k, r, vol, t)
    p = Black76.price(OptionType.PUT, f, k, r, vol, t)
    assert c - p == pytest.approx(math.exp(-r * t) * (f - k), abs=1e-10)
    # Implied vol round-trips.
    assert Black76.implied_vol(OptionType.CALL, c, f, k, r, t) == pytest.approx(vol, abs=1e-6)
    # Delta bounds and vega positivity.
    assert Black76.delta(OptionType.CALL, f, k, r, vol, t) > 0.5  # ITM-ish call
    assert Black76.vega(f, k, r, vol, t) > 0
    # Zero vol = discounted intrinsic.
    assert Black76.price(OptionType.CALL, f, k, r, 0, t) == pytest.approx(
        math.exp(-r * t) * 2, abs=1e-12)


# ------------------------------------------------------------------
# Higher-order Greeks vs finite differences of the first-order ones
# ------------------------------------------------------------------


def test_vanna_and_volga_match_finite_differences_of_delta_and_vega():
    # Deliberately r != 2*carry etc. — parameters chosen so no
    # convention coincidence can fake agreement.
    s, k, r, carry, vol, t = 100.0, 110.0, 0.02, 0.035, 0.3, 0.5
    h = 1e-4

    vanna_fd = (BlackScholes.delta(OptionType.CALL, s, k, r, carry, vol + h, t)
                - BlackScholes.delta(OptionType.CALL, s, k, r, carry, vol - h, t)) / (2 * h)
    assert HigherOrderGreeks.vanna(s, k, r, carry, vol, t) == pytest.approx(vanna_fd, abs=1e-4)

    volga_fd = (BlackScholes.vega(s, k, r, carry, vol + h, t)
                - BlackScholes.vega(s, k, r, carry, vol - h, t)) / (2 * h)
    assert HigherOrderGreeks.volga(s, k, r, carry, vol, t) == pytest.approx(volga_fd, abs=1e-3)

    # An OTM call's volga is positive (long wings love vol of vol).
    assert HigherOrderGreeks.volga(s, k, r, carry, vol, t) > 0
    # Exchange-option cross-gamma is negative and shrinks as correlation rises.
    low_corr = HigherOrderGreeks.exchange_cross_gamma(100, 100, 0.3, 0.3, 0.2, 1)
    high_corr = HigherOrderGreeks.exchange_cross_gamma(100, 100, 0.3, 0.3, 0.8, 1)
    assert low_corr < 0 and high_corr < 0
    # Closer legs = sharper exchange-option kink.
    assert abs(high_corr) > abs(low_corr)


# ------------------------------------------------------------------
# Heston
# ------------------------------------------------------------------


def test_heston_collapses_to_black_scholes_when_vol_of_vol_vanishes():
    # Small (not degenerate) vol-of-vol with v0 = theta: variance is
    # pinned near v0, so the price sits within O(sigmaV^2) of
    # BS(sqrt(v0)) — sigmaV = 0.01 keeps the integrand well-conditioned.
    # This tolerance is deliberately tight: it caught a genuine bug in
    # the Java source — the naive complex sqrt lost the imaginary part
    # near u = 0 and biased every price ~0.5%; the stable form agrees to
    # 4+ decimals.
    p = HestonParams(5, 0.04, 0.01, 0.0, 0.04)
    heston = Heston.call(100, 100, 0.02, 0.0, 1.0, p)
    bs = BlackScholes.price(OptionType.CALL, 100, 100, 0.02, 0.0, 0.2, 1.0)
    assert heston == pytest.approx(bs, abs=5e-4)


def test_heston_survives_short_dated_low_vol_where_a_fixed_window_truncates():
    # 1 week at 4% vol: the integrand's decay scale ~1/(sigma*sqrt(T))
    # is ~180 here, so a FIXED u-window of 200 truncates real mass and
    # silently biases the price — the window must stretch with the
    # parameters (this pins that it does).
    p = HestonParams(5, 0.0016, 0.005, 0.0, 0.0016)
    t = 1.0 / 52
    heston = Heston.call(100, 100, 0.02, 0.0, t, p)
    bs = BlackScholes.price(OptionType.CALL, 100, 100, 0.02, 0.0, 0.04, t)
    assert heston == pytest.approx(bs, abs=1e-4)


def test_heston_semi_analytic_agrees_with_its_own_monte_carlo():
    # Realistic equity-skew parameters (Feller violated, as markets do).
    p = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    assert p.feller() < 1  # deliberately in the violated regime
    analytic = Heston.call(100, 100, 0.02, 0.0, 1.0, p)
    mc = Heston.call_monte_carlo(100, 100, 0.02, 0.0, 1.0, p, 200, 40_000, 42)
    # Two independent routes to one price.
    assert mc == pytest.approx(analytic, abs=0.35)
    # Put-call parity ties the put to the same integral.
    put = Heston.put(100, 100, 0.02, 0.0, 1.0, p)
    forward = 100 * math.exp(0.02)
    assert put == pytest.approx(analytic - math.exp(-0.02) * (forward - 100), abs=1e-9)
    # The skew: with rho < 0, low strikes carry MORE implied vol than high.
    low_k = Heston.call(100, 80, 0.02, 0.0, 1.0, p)
    high_k = Heston.call(100, 120, 0.02, 0.0, 1.0, p)
    iv_low = BlackScholes.implied_vol(OptionType.CALL, low_k, 100, 80, 0.02, 0.0, 1.0)
    iv_high = BlackScholes.implied_vol(OptionType.CALL, high_k, 100, 120, 0.02, 0.0, 1.0)
    assert iv_low > iv_high
    with pytest.raises(ValueError):
        HestonParams(0, 0.04, 0.5, -0.7, 0.04)
    with pytest.raises(ValueError):
        Heston.call(math.nan, 100, 0.02, 0, 1, p)
    # The MC cross-check gates its market inputs exactly like call().
    with pytest.raises(ValueError):
        Heston.call_monte_carlo(math.nan, 100, 0.02, 0, 1, p, 10, 100, 1)
    with pytest.raises(ValueError):
        Heston.call_monte_carlo(100, 100, 0.02, 0, -1, p, 10, 100, 1)
