"""Pins for quantfinlib.pricing.autocallable, ported from
AutocallableRfqTest.java (the RFQ auction/scorecard tests stay in Java —
the rfq domain is not ported).

Port note: the Java pricer draws from java.util.Random; this port draws
from numpy.random.default_rng. The Java suite pins only zero-vol EXACT
values (the MC collapses to arithmetic regardless of the RNG),
inequalities on 40k-path prices, and same-seed reproducibility — all of
which transfer unchanged.
"""

import math

import pytest

from quantfinlib.pricing import Autocallable

QUARTERS = [0.25, 0.5, 0.75, 1.0]

# ------------------------------------------------------------------
# Exact zero-vol cases (the MC collapses to arithmetic)
# ------------------------------------------------------------------


def test_zero_vol_flat_market_autocalls_at_the_first_observation():
    # vol=0, r=q=0: the path sits at spot forever; barrier 1.0 triggers
    # immediately -> notional + one coupon, undiscounted. Exact.
    note = Autocallable(1_000_000, QUARTERS, 1.0, 0.8, 0.6, 0.02, True)
    price = note.price(100, 100, 0, 0, 0, 10, 42)
    assert price == pytest.approx(1_020_000, abs=1e-6)  # notional + one 2% coupon


def test_zero_vol_below_autocall_collects_coupons_and_protection():
    # Barrier 1.2 never triggers; flat at 100 pays every coupon (>= 0.8
    # barrier) and redeems protected at maturity (>= 0.6 KI). Exact.
    note = Autocallable(1_000_000, QUARTERS, 1.2, 0.8, 0.6, 0.02, False)
    price = note.price(100, 100, 0, 0, 0, 10, 42)
    assert price == pytest.approx(1_080_000, abs=1e-6)  # four coupons + notional


def test_zero_vol_knocked_in_takes_the_equity_loss():
    # A heavy dividend drift sinks the path deterministically:
    # S_T = 100*e^{-0.6} = 54.88 < KI 60 -> redeem S_T/S0, no coupons.
    note = Autocallable(1_000_000, [1.0], 1.2, 0.8, 0.6, 0.02, False)
    price = note.price(100, 100, 0, 0, 0.6, 10, 42)
    assert price == pytest.approx(1_000_000 * math.exp(-0.6), abs=1e-3)


def test_memory_coupons_catch_up_at_the_next_paying_observation():
    # Compare memory ON vs OFF on a stochastic run — memory can only
    # ADD value.
    memory = Autocallable(1_000_000, QUARTERS, 1.05, 0.95, 0.6, 0.02, True)
    plain = Autocallable(1_000_000, QUARTERS, 1.05, 0.95, 0.6, 0.02, False)
    with_memory = memory.price(100, 100, 0.25, 0.02, 0, 40_000, 42)
    without = plain.price(100, 100, 0.25, 0.02, 0, 40_000, 42)
    assert with_memory > without


def test_monte_carlo_monotonicities_and_reproducibility():
    base = Autocallable(1_000_000, QUARTERS, 1.0, 0.8, 0.6, 0.02, True)
    p = base.price(100, 100, 0.20, 0.02, 0.01, 40_000, 42)
    # Fixed seed = bit-identical price.
    assert base.price(100, 100, 0.20, 0.02, 0.01, 40_000, 42) == p
    # Fatter coupon -> worth more.
    richer = Autocallable(1_000_000, QUARTERS, 1.0, 0.8, 0.6, 0.03, True)
    assert richer.price(100, 100, 0.20, 0.02, 0.01, 40_000, 42) > p
    # Higher knock-in barrier -> protection dies earlier -> worth less.
    fragile = Autocallable(1_000_000, QUARTERS, 1.0, 0.8, 0.8, 0.02, True)
    assert fragile.price(100, 100, 0.20, 0.02, 0.01, 40_000, 42) < p
    # More volatility -> the sold knock-in put dominates -> worth less.
    assert base.price(100, 100, 0.40, 0.02, 0.01, 40_000, 42) < p
    # Hard bound: never worth more than every cashflow undiscounted.
    assert p < 1_000_000 * (1 + 4 * 0.02)


def test_autocallable_validation():
    with pytest.raises(ValueError):
        Autocallable(1_000_000, QUARTERS, 1.0, 1.1, 0.6, 0.02, True)   # coupon > autocall
    with pytest.raises(ValueError):
        Autocallable(1_000_000, QUARTERS, 1.0, 0.8, 1.2, 0.02, True)   # KI above autocall
    with pytest.raises(ValueError):
        Autocallable(math.nan, QUARTERS, 1.0, 0.8, 0.6, 0.02, True)    # NaN fails HERE
    with pytest.raises(ValueError):
        Autocallable(1_000_000, QUARTERS, 1.0, 0.8, 0.6, math.nan, True)
    with pytest.raises(ValueError):
        Autocallable(1_000_000, [0.5, 0.5], 1.0, 0.8, 0.6, 0.02, True)
    note = Autocallable(1_000_000, QUARTERS, 1.0, 0.8, 0.6, 0.02, True)
    with pytest.raises(ValueError):
        note.price(math.nan, 100, 0.2, 0, 0, 100, 1)
    with pytest.raises(ValueError):
        note.price(100, 100, math.nan, 0, 0, 100, 1)
    assert note.notional() == 1_000_000
    assert note.observations() == 4
