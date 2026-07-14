"""Factory of common :class:`~quantfinlib.dsl.rule.Rule`\\ s over indicator
arrays (port of Java ``dsl.Rules``).

All rules are NaN-safe: a rule is never satisfied while its inputs are
in warm-up.

The design decision worth knowing: a :class:`Rule` is a predicate over
a BAR INDEX into precomputed indicator arrays, not over live values.
That keeps strategy definitions declarative ("RSI crossed under 30 AND
price above the 200-day") and -- more importantly -- makes look-ahead
bias structurally harder: the arrays are computed once by causal
indicator code, and a rule can only combine values at ``i`` and
``i-1``, never peek at ``i+1``. Cross rules require the previous bar
to be on the other side (with ``<=``/``>=``), so a series that OPENS
above the level does not count as a cross -- the classic off-by-one
that fires a "breakout" signal on bar 0 of every backtest. NaN
warm-up bars satisfy nothing, and combining rules with
:meth:`Rule.and_`/``or_``/``not_`` preserves that. Assembled into
strategies by :class:`~quantfinlib.dsl.strategy_builder.StrategyBuilder`.
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.dsl.rule import Rule


def _valid(*values: float) -> bool:
    return all(not math.isnan(v) for v in values)


def cross_above(a: np.ndarray, b: np.ndarray) -> Rule:
    """a crossed above b on this bar."""
    return Rule(
        lambda i: i > 0
        and _valid(a[i], b[i], a[i - 1], b[i - 1])
        and a[i - 1] <= b[i - 1]
        and a[i] > b[i]
    )


def cross_below(a: np.ndarray, b: np.ndarray) -> Rule:
    """a crossed below b on this bar."""
    return Rule(
        lambda i: i > 0
        and _valid(a[i], b[i], a[i - 1], b[i - 1])
        and a[i - 1] >= b[i - 1]
        and a[i] < b[i]
    )


def cross_above_value(a: np.ndarray, level: float) -> Rule:
    """a crossed above a constant level on this bar."""
    return Rule(lambda i: i > 0 and _valid(a[i], a[i - 1]) and a[i - 1] <= level and a[i] > level)


def cross_below_value(a: np.ndarray, level: float) -> Rule:
    """a crossed below a constant level on this bar."""
    return Rule(lambda i: i > 0 and _valid(a[i], a[i - 1]) and a[i - 1] >= level and a[i] < level)


def above(a: np.ndarray, b: np.ndarray) -> Rule:
    return Rule(lambda i: _valid(a[i], b[i]) and a[i] > b[i])


def below(a: np.ndarray, b: np.ndarray) -> Rule:
    return Rule(lambda i: _valid(a[i], b[i]) and a[i] < b[i])


def above_value(a: np.ndarray, level: float) -> Rule:
    return Rule(lambda i: _valid(a[i]) and a[i] > level)


def below_value(a: np.ndarray, level: float) -> Rule:
    return Rule(lambda i: _valid(a[i]) and a[i] < level)


def rising(a: np.ndarray, bars: int) -> Rule:
    """a has risen on each of the last ``bars`` bars."""

    def check(i: int) -> bool:
        if i < bars:
            return False
        for j in range(i - bars + 1, i + 1):
            if not _valid(a[j], a[j - 1]) or a[j] <= a[j - 1]:
                return False
        return True

    return Rule(check)


def falling(a: np.ndarray, bars: int) -> Rule:
    """a has fallen on each of the last ``bars`` bars."""

    def check(i: int) -> bool:
        if i < bars:
            return False
        for j in range(i - bars + 1, i + 1):
            if not _valid(a[j], a[j - 1]) or a[j] >= a[j - 1]:
                return False
        return True

    return Rule(check)
