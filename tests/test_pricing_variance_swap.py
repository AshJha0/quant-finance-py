"""Pins for quantfinlib.pricing.variance_swap, ported from
VarianceSwapTest.java.

The fair-variance chain-replication test stays in Java: it needs
``volatility.VolatilityIndex``, a domain not yet ported (and the Python
``VarianceSwap`` deliberately omits ``fair_variance`` until it is).
"""

import math

import pytest

from quantfinlib.pricing import VarianceSwap


def test_vega_to_variance_notional_bridge():
    # 100k vega at a 20-vol strike: 100000 / (2 * 0.20) = 250000.
    assert VarianceSwap.variance_notional(100_000, 0.20) == pytest.approx(250_000, abs=1e-9)
    with pytest.raises(ValueError):
        VarianceSwap.variance_notional(100_000, 0)
    with pytest.raises(ValueError):
        VarianceSwap.variance_notional(math.nan, 0.2)


def test_mark_to_market_blends_realized_and_remaining_additively():
    # Halfway: 0.5*0.09 + 0.5*0.05 - 0.04 = 0.03 per unit var notional.
    assert VarianceSwap.mark_to_market(0.04, 0.09, 0.05, 0.5, 1.0, 0.0) == pytest.approx(
        0.03, abs=1e-12)
    # Same with discounting at 2% for the remaining half year.
    assert VarianceSwap.mark_to_market(0.04, 0.09, 0.05, 0.5, 1.0, 0.02) == pytest.approx(
        0.03 * math.exp(-0.02 * 0.5), abs=1e-12)
    # At inception with the strike at fair: worth exactly zero.
    assert VarianceSwap.mark_to_market(0.04, 0.0, 0.04, 0.0, 1.0, 0.03) == pytest.approx(
        0.0, abs=1e-15)
    # At expiry: the settlement payoff, undiscounted.
    assert VarianceSwap.mark_to_market(0.04, 0.0625, 0.0, 1.0, 1.0, 0.05) == pytest.approx(
        0.0225, abs=1e-12)


def test_mark_to_market_gates():
    with pytest.raises(ValueError):
        VarianceSwap.mark_to_market(0, 0.04, 0.04, 0.5, 1, 0)
    with pytest.raises(ValueError):
        VarianceSwap.mark_to_market(0.04, -0.01, 0.04, 0.5, 1, 0)
    with pytest.raises(ValueError):
        VarianceSwap.mark_to_market(0.04, 0.04, 0.04, 1.5, 1, 0)  # t > T
    with pytest.raises(ValueError):
        VarianceSwap.mark_to_market(0.04, 0.04, 0.04, 0, 0, 0)    # T = 0
    with pytest.raises(ValueError):
        VarianceSwap.mark_to_market(0.04, 0.04, 0.04, 0.5, 1, math.nan)
