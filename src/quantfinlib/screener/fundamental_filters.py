"""Fundamental screening filters (port of Java ``screener.FundamentalFilters``).

NaN fundamentals never match: every comparison below is a bare
``>``/``<``/``>=``/``<=`` against a possibly-NaN field, and IEEE 754
NaN comparisons are always false in Python exactly as in Java.
"""

from __future__ import annotations

from quantfinlib.screener.screen_filter import ScreenFilter


def market_cap_above(value: float) -> ScreenFilter:
    return ScreenFilter(lambda s: s.fundamentals.market_cap > value)


def pe_below(value: float) -> ScreenFilter:
    return ScreenFilter(lambda s: s.fundamentals.pe_ratio < value)


def pe_between(min_: float, max_: float) -> ScreenFilter:
    return ScreenFilter(
        lambda s: s.fundamentals.pe_ratio >= min_ and s.fundamentals.pe_ratio <= max_
    )


def pb_below(value: float) -> ScreenFilter:
    return ScreenFilter(lambda s: s.fundamentals.pb_ratio < value)


def eps_above(value: float) -> ScreenFilter:
    return ScreenFilter(lambda s: s.fundamentals.eps > value)


def roe_above(value: float) -> ScreenFilter:
    return ScreenFilter(lambda s: s.fundamentals.roe > value)


def dividend_yield_above(value: float) -> ScreenFilter:
    return ScreenFilter(lambda s: s.fundamentals.dividend_yield > value)


def debt_to_equity_below(value: float) -> ScreenFilter:
    return ScreenFilter(lambda s: s.fundamentals.debt_to_equity < value)
