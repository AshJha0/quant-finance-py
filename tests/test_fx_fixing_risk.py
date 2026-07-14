"""Fixing-window analytics, ported from the FixingRisk half of Java
FixingRiskCrossRateTest (the streaming CrossRateEngine is bus-coupled
and is not ported; SyntheticCross/CrossOp cover its pure-math part).
"""

import math

import pytest

from quantfinlib.fx import CurrencyPair, FixingRisk


def test_window_averages_and_slippage():
    prices = [1.0850, 1.0852, 1.0854]
    sizes = [1, 1, 2]
    assert FixingRisk.window_twap(prices) == pytest.approx(1.0852, abs=1e-12)
    # VWAP tilts toward the heavy print.
    assert FixingRisk.window_vwap(prices, sizes) == pytest.approx(
        (1.0850 + 1.0852 + 2 * 1.0854) / 4, abs=1e-12)
    eurusd = CurrencyPair.of("EURUSD")
    assert FixingRisk.slippage_vs_fix(eurusd, 1.0852, 1.0850) == pytest.approx(2.0, abs=1e-9)


def test_tracking_error_follows_the_twap_variance_law():
    # sigma^2*T/3 law: doubling the window scales the std by sqrt(2).
    one_x = FixingRisk.tracking_error_std(0.0001, 5)
    two_x = FixingRisk.tracking_error_std(0.0001, 10)
    assert two_x / one_x == pytest.approx(math.sqrt(2), abs=1e-12)
    assert one_x == pytest.approx(0.0001 * math.sqrt(5.0 / 3), abs=1e-15)
    assert FixingRisk.participation_rate(25, 100) == pytest.approx(0.25, abs=1e-12)
    with pytest.raises(ValueError):
        FixingRisk.window_twap([])
    with pytest.raises(ValueError):
        FixingRisk.window_vwap([1], [0])
    with pytest.raises(ValueError):
        FixingRisk.tracking_error_std(0.1, 0)
    with pytest.raises(ValueError):
        FixingRisk.participation_rate(1, 0)
