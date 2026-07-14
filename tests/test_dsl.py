"""Pins for quantfinlib.dsl (strategy DSL).

Java sources: Rule/Rules/StrategyBuilder.java. The core contract under
test is the NO-LOOKAHEAD regression: a Rule is a predicate over a bar
INDEX into precomputed arrays, so a cross rule may only combine values
at i and i-1, never i+1, and a series that OPENS on the far side of a
level must not count as a "cross" on bar 0 (the classic off-by-one
that fires a breakout signal at the start of every backtest). NaN
warm-up bars must never satisfy a rule.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantfinlib.data.bar_series import BarSeries
from quantfinlib.dsl import rules
from quantfinlib.dsl.rule import Rule
from quantfinlib.dsl.strategy_builder import StrategyBuilder
from quantfinlib.backtest.trade import Trade


# ----------------------------------------------------------------------
# No-lookahead contract
# ----------------------------------------------------------------------


def test_cross_above_does_not_fire_on_bar_zero_when_already_above():
    # a starts ABOVE b at index 0: since there is no i-1, this must
    # never be treated as a cross, no matter how far above a already is.
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 1.0, 1.0])
    rule = rules.cross_above(a, b)
    assert rule.is_satisfied(0) is False


def test_cross_above_requires_previous_bar_on_the_other_side():
    a = np.array([1.0, 1.0, 5.0, 6.0])
    b = np.array([2.0, 2.0, 2.0, 2.0])
    rule = rules.cross_above(a, b)
    assert rule.is_satisfied(1) is False  # a[1]=1 <= b[1]=2: no cross yet
    assert rule.is_satisfied(2) is True  # a[1]<=b[1] and a[2]>b[2]: genuine cross
    assert rule.is_satisfied(3) is False  # already above on both bars: not a fresh cross


def test_cross_below_symmetric_contract():
    a = np.array([5.0, 5.0, 1.0, 0.5])
    b = np.array([2.0, 2.0, 2.0, 2.0])
    rule = rules.cross_below(a, b)
    assert rule.is_satisfied(0) is False
    assert rule.is_satisfied(2) is True
    assert rule.is_satisfied(3) is False


def test_cross_rules_never_satisfied_on_nan_warmup():
    a = np.array([math.nan, math.nan, 1.0, 5.0])
    b = np.array([2.0, 2.0, 2.0, 2.0])
    rule = rules.cross_above(a, b)
    assert rule.is_satisfied(1) is False
    assert rule.is_satisfied(2) is False  # a[1] is NaN: previous bar unknown
    assert rule.is_satisfied(3) is True  # a[2]=1<=2, a[3]=5>2, both valid


def test_cross_above_value_and_below_value():
    a = np.array([50.0, 50.0, 71.0])
    above = rules.cross_above_value(a, 70.0)
    assert above.is_satisfied(1) is False
    assert above.is_satisfied(2) is True

    b = np.array([50.0, 50.0, 29.0])
    below = rules.cross_below_value(b, 30.0)
    assert below.is_satisfied(2) is True


def test_rising_and_falling_require_every_bar_in_window():
    a = np.array([1.0, 2.0, 3.0, 2.5, 4.0])
    assert rules.rising(a, 2).is_satisfied(2) is True  # 1->2->3 rising
    assert rules.rising(a, 3).is_satisfied(4) is False  # dip at index 3 breaks it
    assert rules.falling(a, 1).is_satisfied(3) is True  # 3.0 -> 2.5


def test_above_below_value_are_nan_safe():
    a = np.array([math.nan, 5.0])
    assert rules.above_value(a, 0.0).is_satisfied(0) is False
    assert rules.above_value(a, 0.0).is_satisfied(1) is True
    assert rules.below_value(a, 10.0).is_satisfied(0) is False


# ----------------------------------------------------------------------
# Rule combinators
# ----------------------------------------------------------------------


def test_rule_and_or_not():
    always_true = Rule(lambda i: True)
    always_false = Rule(lambda i: False)
    assert always_true.and_(always_false).is_satisfied(0) is False
    assert always_true.or_(always_false).is_satisfied(0) is True
    assert always_true.not_().is_satisfied(0) is False


# ----------------------------------------------------------------------
# StrategyBuilder: exact trade reproduction of the hand-designed
# SMA(1,2) cross series used across the backtest test suite.
# ----------------------------------------------------------------------


def test_strategy_builder_reproduces_sma_cross_trades():
    from quantfinlib.indicators.indicators import Indicators

    closes = [100.0, 100.0, 110.0, 120.0, 115.0, 100.0, 110.0]
    series = BarSeries.of("SMA", closes)
    fast = Indicators.sma(series.closes(), 1)
    slow = Indicators.sma(series.closes(), 2)

    strategy = (
        StrategyBuilder.named("dsl-sma-cross")
        .enter_when(rules.cross_above(fast, slow))
        .exit_when(rules.cross_below(fast, slow))
        .build()
    )
    result = strategy.backtest(series, 100_000)

    trades = result.trades()
    assert len(trades) == 2
    t1, t2 = trades
    assert (t1.entry_index, t1.exit_index) == (2, 4)
    assert t1.exit_reason == Trade.REASON_SIGNAL
    assert (t2.entry_index, t2.exit_index) == (6, 6)
    assert t2.exit_reason == Trade.REASON_END_OF_DATA


def test_strategy_builder_requires_entry_rule():
    with pytest.raises(ValueError):
        StrategyBuilder.named("no-entry").build()


def test_strategy_builder_default_exit_never_fires_on_its_own():
    closes = [100.0, 101.0, 102.0, 103.0]
    series = BarSeries.of("NOEXIT", closes)
    strategy = StrategyBuilder.named("buy-and-hold").enter_when(Rule(lambda i: i == 0)).build()
    result = strategy.backtest(series, 100_000)
    assert len(result.trades()) == 1
    assert result.trades()[0].exit_reason == Trade.REASON_END_OF_DATA


def test_strategy_builder_stop_loss_and_take_profit_wired_through():
    strategy = (
        StrategyBuilder.named("risk")
        .enter_when(Rule(lambda i: i == 0))
        .with_stop_loss(0.03)
        .with_take_profit(0.08)
        .build()
    )
    assert strategy.stop_loss_pct() == pytest.approx(0.03)
    assert strategy.take_profit_pct() == pytest.approx(0.08)
