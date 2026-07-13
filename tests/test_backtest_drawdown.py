"""Pins for quantfinlib.backtest.drawdown_analytics.

Java source: ValidationRobustnessTest.java (DrawdownAnalytics section) —
the hand-walked two-episode curve, the monotone curve, and the input
gates. The max depth must also agree exactly with the plain max-drawdown
estimator.
"""

import math

import pytest

from quantfinlib.backtest import DrawdownAnalytics
from quantfinlib.backtest._risk import max_drawdown


def test_drawdown_episodes_match_hand_walked_curve():
    # Peak 110 at i=1, trough 99 at i=2 (10% deep), recovered at i=4;
    # peak 121 at i=5, trough 100 at i=7, still open at series end.
    equity = [100, 110, 99, 104.5, 110, 121, 108.9, 100]
    r = DrawdownAnalytics.analyze(equity)

    assert len(r.episodes) == 2
    first = r.episodes[0]
    assert first.peak_index == 1
    assert first.trough_index == 2
    assert first.recovery_index == 4
    assert first.depth == pytest.approx(0.10, abs=1e-12)   # 1 - 99/110
    assert first.duration(len(equity)) == 3                # 4 - 1

    second = r.episodes[1]
    assert second.peak_index == 5
    assert second.trough_index == 7
    assert second.recovery_index == -1                     # honest: not recovered
    assert second.depth == pytest.approx(1 - 100.0 / 121.0, abs=1e-12)
    assert second.duration(len(equity)) == 2               # (8-1) - 5

    assert r.max_depth == pytest.approx(1 - 100.0 / 121.0, abs=1e-12)
    assert r.max_duration == 3
    assert r.time_under_water == pytest.approx(0.5, abs=1e-12)  # 4 of 8 periods
    # Must agree exactly with the plain max-drawdown estimator.
    assert r.max_depth == pytest.approx(max_drawdown(equity), abs=1e-12)


def test_monotonically_rising_equity_has_no_drawdowns():
    r = DrawdownAnalytics.analyze([100, 101, 105, 110])
    assert r.episodes == ()
    assert r.max_depth == 0.0
    assert r.max_duration == 0
    assert r.time_under_water == 0.0


def test_drawdown_analytics_refuses_non_positive_equity():
    with pytest.raises(ValueError):
        DrawdownAnalytics.analyze([100])                       # too short
    with pytest.raises(ValueError):
        DrawdownAnalytics.analyze([100, 0, 50])                # zero equity
    with pytest.raises(ValueError):
        DrawdownAnalytics.analyze([100, -5, 50])               # negative
    with pytest.raises(ValueError):
        DrawdownAnalytics.analyze([100, math.inf])             # infinite
    with pytest.raises(ValueError):
        DrawdownAnalytics.analyze([100, math.nan])             # NaN gate
