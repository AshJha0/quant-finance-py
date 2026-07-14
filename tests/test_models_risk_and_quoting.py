"""Pins for JumpRobustVolatility, HawkesIntensity, EwmaCovariance,
AvellanedaStoikov and LiquidityMeasures.

Java sources: QuantModels5Test.java (JumpRobustVolatility),
QuantModels6Test.java (HawkesIntensity), QuantModels4Test.java
(EwmaCovariance, AvellanedaStoikov), QuantSignalsTest.java
(LiquidityMeasures).
"""

import math

import numpy as np
import pytest

from quantfinlib.microstructure.avellaneda_stoikov import AvellanedaStoikov
from quantfinlib.microstructure.ewma_covariance import EwmaCovariance
from quantfinlib.microstructure.hawkes_intensity import HawkesIntensity
from quantfinlib.microstructure.jump_robust_volatility import (
    JumpRobustVolatility)
from quantfinlib.microstructure.liquidity_measures import LiquidityMeasures

SEC = 1_000_000_000


# ------------------------------------------------------------------
# JumpRobustVolatility
# ------------------------------------------------------------------

def test_diffusion_agrees_but_a_jump_splits_the_estimators():
    vol = JumpRobustVolatility(10_000_000_000)
    rng = np.random.default_rng(7)
    dt = 1_000_000_000
    sigma = 1e-4
    for _ in range(5_000):
        vol.on_return(sigma * rng.standard_normal(), dt)
    # Pure diffusion: the two estimators agree (that is bipower's point).
    assert vol.raw_vol_per_sqrt_second() == pytest.approx(
        vol.vol_per_sqrt_second(), abs=0.25 * vol.raw_vol_per_sqrt_second())
    assert vol.jump_fraction() < 0.25

    # One 50-sigma headline print, then one normal return.
    vol.on_return(50 * sigma, dt)
    vol.on_return(sigma, dt)
    assert vol.raw_vol_per_sqrt_second() > 2 * vol.vol_per_sqrt_second()
    assert vol.jump_fraction() > 0.5


def test_irregular_sampling_is_not_misread_as_jumps():
    vol = JumpRobustVolatility(100_000_000_000)
    rng = np.random.default_rng(1)
    sigma = 1e-4
    for i in range(20_000):
        dt = 10_000_000_000 if i % 2 == 0 else 100_000_000
        dt_sec = dt * 1e-9
        vol.on_return(sigma * math.sqrt(dt_sec) * rng.standard_normal(), dt)
    assert vol.raw_vol_per_sqrt_second() == pytest.approx(
        vol.vol_per_sqrt_second(), abs=0.25 * vol.raw_vol_per_sqrt_second())
    assert vol.jump_fraction() < 0.25


def test_a_gap_breaks_the_pairing_so_neighbors_are_never_invented():
    vol = JumpRobustVolatility(10_000_000_000)
    vol.on_return(1e-4, 1_000_000_000)
    vol.on_return(1e-4, 1_000_000_000)      # bipower seeds here
    bipower_before = vol.vol_per_sqrt_second()
    vol.on_return(math.nan, 1_000_000_000)   # feed gap
    vol.on_return(5e-3, 1_000_000_000)       # big first return after the gap
    assert vol.vol_per_sqrt_second() == pytest.approx(bipower_before, abs=0.0)
    assert vol.raw_vol_per_sqrt_second() > bipower_before
    with pytest.raises(ValueError):
        JumpRobustVolatility(0)


# ------------------------------------------------------------------
# HawkesIntensity
# ------------------------------------------------------------------

def test_steady_flow_reads_as_baseline_not_as_a_burst():
    h = HawkesIntensity(2.0, 0.1, 2 * SEC)
    t = 0
    for _ in range(200):
        t += SEC // 2                        # exactly baseline pace
        h.on_event(t)
    assert h.burst_score(t) < 0.4
    assert h.events() == 200


def test_a_burst_spikes_the_score_then_decays_with_the_half_life():
    h = HawkesIntensity(2.0, 0.1, 2 * SEC)
    t = 0
    for _ in range(40):
        t += SEC // 1000                      # 1ms machine-gun burst
        h.on_event(t)
    at_burst = h.burst_score(t)
    assert at_burst == pytest.approx(1.0, abs=1e-9)
    later = h.burst_score(t + 4 * SEC)         # two half-lives on
    assert later < at_burst and later > 0
    assert h.intensity(t) > h.intensity(t + 60 * SEC)
    assert h.intensity(t + 600 * SEC) == pytest.approx(2.0, abs=1e-6)


def test_out_of_order_timestamps_are_dropped_and_explosiveness_is_rejected():
    h = HawkesIntensity(2.0, 0.1, 2 * SEC)
    assert h.intensity(0) == pytest.approx(2.0, abs=1e-12)
    h.on_event(SEC)
    h.on_event(SEC // 2)                       # feed-merge jitter
    assert h.events() == 1
    with pytest.raises(ValueError):
        HawkesIntensity(2.0, 0.5, 2 * SEC)
    with pytest.raises(ValueError):
        HawkesIntensity(0, 0.1, 2 * SEC)


# ------------------------------------------------------------------
# EwmaCovariance
# ------------------------------------------------------------------

def test_learns_the_planted_correlation_structure():
    cov = EwmaCovariance(3, 0.97)
    rng = np.random.default_rng(42)
    for _ in range(3_000):
        g = 1e-4 * rng.standard_normal()
        r = [g, g, 1e-4 * rng.standard_normal()]   # r[0]==r[1] perfectly correlated
        cov.on_returns(r)
    assert cov.correlation(0, 1) > 0.99
    assert abs(cov.correlation(0, 2)) < 0.2
    assert cov.variance(0) == pytest.approx(1e-8, abs=5e-9)
    assert cov.correlation(0, 1) == cov.correlation(1, 0)


def test_portfolio_arithmetic_is_exact_on_a_hand_built_matrix():
    cov = EwmaCovariance(2, 0.94)
    cov.on_returns([1, 1])
    assert cov.portfolio_variance([1, 1]) == pytest.approx(4, abs=1e-12)
    mrc = [0.0, 0.0]
    cov.marginal_contribution([1, 1], mrc)
    assert mrc[0] == pytest.approx(0.5, abs=1e-12)
    assert mrc[0] + mrc[1] == pytest.approx(1.0, abs=1e-12)
    assert cov.min_variance_hedge_ratio(0, 1) == pytest.approx(1.0, abs=1e-12)


def test_a_perfect_hedge_is_a_degenerate_risk_picture_not_a_crash():
    cov = EwmaCovariance(2, 0.94)
    cov.on_returns([1, -1])                     # perfectly anti-correlated
    assert cov.portfolio_variance([1, 1]) == pytest.approx(0, abs=1e-12)
    mrc = [99.0, 99.0]
    cov.marginal_contribution([1, 1], mrc)
    assert mrc[0] == 0.0
    assert cov.min_variance_hedge_ratio(0, 1) == pytest.approx(-1.0, abs=1e-12)


def test_a_bad_print_drops_the_whole_sample_never_half_of_it():
    cov = EwmaCovariance(2, 0.94)
    cov.on_returns([1e-4, 1e-4])
    before = cov.covariance(0, 1)
    cov.on_returns([math.nan, 5e-4])            # one bad symbol
    cov.on_returns([1e-4, math.inf])
    assert cov.samples() == 1
    assert cov.covariance(0, 1) == before


def test_covariance_validation():
    with pytest.raises(ValueError):
        EwmaCovariance(0, 0.94)
    with pytest.raises(ValueError):
        EwmaCovariance(2, 1.0)
    with pytest.raises(ValueError):
        EwmaCovariance(2, 0)
    cov = EwmaCovariance(3)
    with pytest.raises(ValueError):
        cov.on_returns(np.zeros(2))
    with pytest.raises(ValueError):
        cov.portfolio_variance(np.zeros(2))
    with pytest.raises(ValueError):
        EwmaCovariance(65_536)


# ------------------------------------------------------------------
# AvellanedaStoikov
# ------------------------------------------------------------------

def test_quotes_match_the_closed_form():
    gamma, kappa = 0.1, 1.5
    a_s = AvellanedaStoikov(gamma, kappa)
    var, tau, mid, inv = 1e-4, 60, 100, 100

    expected_r = mid - inv * gamma * var * tau
    assert a_s.reservation_price(mid, inv, var, tau) == pytest.approx(expected_r, abs=1e-12)

    expected_half = 0.5 * gamma * var * tau + math.log(1 + gamma / kappa) / gamma
    assert a_s.optimal_half_spread(var, tau) == pytest.approx(expected_half, abs=1e-12)

    assert a_s.bid_quote(mid, inv, var, tau) == pytest.approx(expected_r - expected_half, abs=1e-12)
    assert a_s.ask_quote(mid, inv, var, tau) == pytest.approx(expected_r + expected_half, abs=1e-12)


def test_long_inventory_shades_both_quotes_down():
    a_s = AvellanedaStoikov(0.1, 1.5)
    flat_bid = a_s.bid_quote(100, 0, 1e-4, 60)
    flat_ask = a_s.ask_quote(100, 0, 1e-4, 60)
    assert a_s.bid_quote(100, 500, 1e-4, 60) < flat_bid
    assert a_s.ask_quote(100, 500, 1e-4, 60) < flat_ask
    assert a_s.ask_quote(100, -500, 1e-4, 60) > flat_ask


def test_spread_widens_with_volatility_and_horizon():
    a_s = AvellanedaStoikov(0.1, 1.5)
    assert a_s.optimal_half_spread(4e-4, 60) > a_s.optimal_half_spread(1e-4, 60)
    assert a_s.optimal_half_spread(1e-4, 600) > a_s.optimal_half_spread(1e-4, 60)


def test_risk_neutral_limit_is_the_pure_liquidity_spread():
    near_neutral = AvellanedaStoikov(1e-9, 1.5)
    assert near_neutral.optimal_half_spread(0, 0) == pytest.approx(1 / 1.5, abs=1e-6)
    # Regression: plain log(1+x) rounds to 0 below x ~ 1e-16; log1p keeps
    # the floor exact.
    tiny = AvellanedaStoikov(1e-16, 1.5)
    assert tiny.optimal_half_spread(0, 0) == pytest.approx(1 / 1.5, abs=1e-9)


def test_garbage_variance_is_neutral_never_poisonous():
    a_s = AvellanedaStoikov(0.1, 1.5)
    floor = a_s.optimal_half_spread(0, 60)
    assert a_s.reservation_price(100, 500, math.nan, 60) == pytest.approx(100, abs=1e-12)
    assert a_s.optimal_half_spread(math.nan, 60) == pytest.approx(floor, abs=1e-12)
    assert a_s.optimal_half_spread(-1, 60) == pytest.approx(floor, abs=1e-12)
    assert a_s.optimal_half_spread(math.inf, 60) == pytest.approx(floor, abs=1e-12)
    with pytest.raises(ValueError):
        AvellanedaStoikov(0, 1)
    with pytest.raises(ValueError):
        AvellanedaStoikov(1, 0)


# ------------------------------------------------------------------
# LiquidityMeasures
# ------------------------------------------------------------------

def test_roll_spread_recovers_the_planted_bounce_and_refuses_trends():
    spread = 0.10
    rng = np.random.default_rng(11)
    prices = 100 + np.where(rng.random(4_000) < 0.5, 1, -1) * spread / 2
    assert LiquidityMeasures.roll_spread(prices) == pytest.approx(spread, abs=0.02)

    trend = 100 + np.arange(100)
    assert math.isnan(LiquidityMeasures.roll_spread(trend))
    with pytest.raises(ValueError):
        LiquidityMeasures.roll_spread([100, 101])


def test_corwin_schultz_and_amihud_match_hand_arithmetic():
    assert LiquidityMeasures.corwin_schultz_spread(101, 100, 101, 100) == \
        pytest.approx(0.00995, abs=2e-4)
    assert LiquidityMeasures.corwin_schultz_spread(100, 100, 100, 100) == 0.0
    # A gapping market (disjoint ranges) drives the estimator negative:
    # clamps to 0.
    assert LiquidityMeasures.corwin_schultz_spread(101, 100, 106, 105) == 0.0
    with pytest.raises(ValueError):
        LiquidityMeasures.corwin_schultz_spread(100, 101, 100, 99)

    assert LiquidityMeasures.amihud_illiquidity(
        [0.01, -0.02, 0.03], [1e6, 2e6, 3e6]) == pytest.approx(1e-8, abs=1e-20)
    with pytest.raises(ValueError):
        LiquidityMeasures.amihud_illiquidity([0.01], [0])
    liquid = LiquidityMeasures.amihud_illiquidity([0.01, 0.01], [5e7, 5e7])
    illiquid = LiquidityMeasures.amihud_illiquidity([0.03, 0.03], [2e6, 2e6])
    assert illiquid > 10 * liquid
