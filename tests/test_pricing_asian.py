"""Pins for quantfinlib.pricing.asian_option, ported from
AsianOptionTest.java: one fixing IS vanilla Black-Scholes (exact), zero
vol IS discounted intrinsic on the forward average (hand computation),
AM-GM orders arithmetic above geometric, and averaging cheapens the
call versus its vanilla cousin.
"""

import math

import pytest

from quantfinlib.pricing import AsianOption, BlackScholes, OptionType
from quantfinlib.util import math_utils as mu

CALL, PUT = OptionType.CALL, OptionType.PUT
S, R, Q, VOL, T = 100.0, 0.05, 0.01, 0.25, 0.75


def test_single_fixing_degenerates_to_vanilla_black_scholes_exactly():
    # n = 1: the "average" is the terminal price; both moments collapse
    # to the vanilla lognormal, so both pricers must equal BS to
    # machine precision — no tolerance games.
    for k in (85.0, 100.0, 115.0):
        call_bs = BlackScholes.price(CALL, S, k, R, Q, VOL, T)
        put_bs = BlackScholes.price(PUT, S, k, R, Q, VOL, T)
        assert AsianOption.geometric_price(CALL, S, k, R, Q, VOL, T, 1) == pytest.approx(
            call_bs, abs=1e-12)
        assert AsianOption.geometric_price(PUT, S, k, R, Q, VOL, T, 1) == pytest.approx(
            put_bs, abs=1e-12)
        assert AsianOption.arithmetic_price(CALL, S, k, R, Q, VOL, T, 1) == pytest.approx(
            call_bs, abs=1e-12)
        assert AsianOption.arithmetic_price(PUT, S, k, R, Q, VOL, T, 1) == pytest.approx(
            put_bs, abs=1e-12)


def test_averaging_cheapens_the_call_versus_vanilla():
    # The average has less variance AND (with r > q) a lower forward
    # than the terminal price: the ATM Asian call must be worth less.
    vanilla = BlackScholes.price(CALL, S, 100, R, Q, VOL, T)
    geo = AsianOption.geometric_price(CALL, S, 100, R, Q, VOL, T, 12)
    arith = AsianOption.arithmetic_price(CALL, S, 100, R, Q, VOL, T, 12)
    assert geo < vanilla
    assert arith < vanilla


def test_arithmetic_call_dominates_geometric_by_am_gm():
    # A >= G pathwise, so the arithmetic call is worth at least the
    # geometric call at every strike and fixing count.
    for n in (2, 4, 12, 52):
        for k in (85.0, 100.0, 115.0):
            geo = AsianOption.geometric_price(CALL, S, k, R, Q, VOL, T, n)
            arith = AsianOption.arithmetic_price(CALL, S, k, R, Q, VOL, T, n)
            assert arith >= geo - 1e-12, f"n={n} K={k}: arith {arith} < geo {geo}"
    # And strictly greater away from the n = 1 degenerate case.
    assert (AsianOption.arithmetic_price(CALL, S, 100, R, Q, VOL, T, 12)
            > AsianOption.geometric_price(CALL, S, 100, R, Q, VOL, T, 12))


def test_zero_vol_pays_discounted_intrinsic_on_the_forward_average():
    # Deterministic world, n = 4 quarterly fixings over one year:
    # the averages are known numbers, prices are exact hand sums.
    n = 4
    r, t, k = 0.05, 1.0, 90.0
    arith_avg = 0.0
    geo_log_avg = 0.0
    for i in range(1, n + 1):
        arith_avg += S * math.exp(r * i * t / n)
        geo_log_avg += math.log(S) + r * i * t / n
    arith_avg /= n
    geo_avg = math.exp(geo_log_avg / n)
    df = math.exp(-r * t)
    assert AsianOption.arithmetic_price(CALL, S, k, r, 0, 0, t, n) == pytest.approx(
        df * (arith_avg - k), abs=1e-10)
    assert AsianOption.geometric_price(CALL, S, k, r, 0, 0, t, n) == pytest.approx(
        df * (geo_avg - k), abs=1e-10)
    # Deep OTM call in a deterministic world is worth exactly zero.
    assert AsianOption.arithmetic_price(CALL, S, 200, r, 0, 0, t, n) == 0
    assert AsianOption.geometric_price(CALL, S, 200, r, 0, 0, t, n) == 0
    # Zero-vol put: discounted (K - average)+ on the same numbers.
    assert AsianOption.arithmetic_price(PUT, S, 200, r, 0, 0, t, n) == pytest.approx(
        df * (200 - arith_avg), abs=1e-10)


def test_more_fixings_cut_the_geometric_variance_toward_the_continuous_third():
    # Var[ln G] scales by (n+1)(2n+1)/(6n^2): 1 at n=1 down to 1/3 as
    # n grows — so the ATM price must fall monotonically in n.
    prev = math.inf
    for n in (1, 2, 4, 12, 52, 252):
        price = AsianOption.geometric_price(CALL, S, 100, 0, 0, VOL, T, n)
        assert price < prev, f"n={n} did not cheapen the call"
        prev = price
    # And the n = 252 price sits just above the continuous-averaging
    # limit (drift factor 1/2, variance factor 1/3), recomputed here
    # from scratch with the same lognormal-Black arithmetic.
    mean_log = math.log(S) - 0.5 * VOL * VOL * T * 0.5
    var_log = VOL * VOL * T / 3
    f = math.exp(mean_log + 0.5 * var_log)
    sd = math.sqrt(var_log)
    d1 = (math.log(f / 100) + 0.5 * var_log) / sd
    floor = f * mu.norm_cdf(d1) - 100 * mu.norm_cdf(d1 - sd)
    assert prev > floor
    assert prev < floor + 0.05  # n=252 should be within pennies of the limit


def test_gates_refuse_nonsense():
    with pytest.raises(ValueError):
        AsianOption.geometric_price(CALL, 0, 100, R, Q, VOL, T, 4)
    with pytest.raises(ValueError):
        AsianOption.geometric_price(CALL, S, math.nan, R, Q, VOL, T, 4)
    with pytest.raises(ValueError):
        AsianOption.geometric_price(CALL, S, 100, math.nan, Q, VOL, T, 4)
    with pytest.raises(ValueError):
        AsianOption.geometric_price(CALL, S, 100, R, Q, -0.1, T, 4)
    with pytest.raises(ValueError):
        AsianOption.geometric_price(CALL, S, 100, R, Q, VOL, 0, 4)
    with pytest.raises(ValueError):
        AsianOption.geometric_price(CALL, S, 100, R, Q, VOL, T, 0)
    with pytest.raises(ValueError):
        AsianOption.arithmetic_price(CALL, S, 100, R, Q, VOL, math.inf, 4)
    with pytest.raises(ValueError):
        AsianOption.arithmetic_price(CALL, -1, 100, R, Q, VOL, T, 4)
