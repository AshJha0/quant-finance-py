"""Technical screening filters (port of Java ``screener.TechnicalFilters``).

All filters are NaN-safe and return ``False`` when the series is too
short.

Design contract worth stating: a screen answers "does this stock look
like X TODAY", so every filter reads only the LAST valid indicator
value -- never a history of signals (that is a backtest's job, over in
``quantfinlib.backtest``). "Too short = false" rather than "too short
= throw" is deliberate: a screen runs across an entire universe, and
one recently-listed ticker with 30 bars must silently drop out of an
SMA(200) screen, not kill the run for the other 2,999 names. The cost
of that choice is that ``rsi_below(14, 70)`` and "not enough data" are
indistinguishable in the output -- compose an explicit
length/liquidity pre-filter first when the distinction matters.
Filters compose via :meth:`ScreenFilter.and_`/``or_``/``negate``; feed
survivors to :class:`~quantfinlib.screener.ranking_engine.RankingEngine`
to order them.

The Java methods take a ``BarSeries``; this port passes the raw
O/H/L/C/V arrays to :class:`~quantfinlib.indicators.indicators.Indicators`,
matching that module's array-based API.
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.indicators.indicators import Indicators
from quantfinlib.screener.screen_filter import ScreenFilter
from quantfinlib.screener.stock_snapshot import StockSnapshot


def _last_valid(v: np.ndarray) -> float:
    last = float(v[-1])
    return math.nan if math.isnan(last) else last


def _valid(*values: float) -> bool:
    return all(not math.isnan(v) for v in values)


def rsi_below(period: int, value: float) -> ScreenFilter:
    return ScreenFilter(
        lambda s: _last_valid(Indicators.rsi(s.series.closes(), period)) < value
    )


def rsi_above(period: int, value: float) -> ScreenFilter:
    return ScreenFilter(
        lambda s: _last_valid(Indicators.rsi(s.series.closes(), period)) > value
    )


def price_above_sma(period: int) -> ScreenFilter:
    return ScreenFilter(
        lambda s: s.last_close() > _last_valid(Indicators.sma(s.series.closes(), period))
    )


def price_below_sma(period: int) -> ScreenFilter:
    return ScreenFilter(
        lambda s: s.last_close() < _last_valid(Indicators.sma(s.series.closes(), period))
    )


def price_above_ema(period: int) -> ScreenFilter:
    return ScreenFilter(
        lambda s: s.last_close() > _last_valid(Indicators.ema(s.series.closes(), period))
    )


def macd_bullish() -> ScreenFilter:
    """MACD line above its signal line on the last bar."""

    def check(s: StockSnapshot) -> bool:
        m = Indicators.macd(s.series.closes(), 12, 26, 9)
        i = s.series.size() - 1
        return _valid(m.line[i], m.signal[i]) and m.line[i] > m.signal[i]

    return ScreenFilter(check)


def adx_above(period: int, value: float) -> ScreenFilter:
    def check(s: StockSnapshot) -> bool:
        series = s.series
        adx = Indicators.adx(series.highs(), series.lows(), series.closes(), period).adx
        return _last_valid(adx) > value

    return ScreenFilter(check)


def atr_percent_below(period: int, max_fraction: float) -> ScreenFilter:
    """ATR as a fraction of price below the threshold (low-volatility screen)."""

    def check(s: StockSnapshot) -> bool:
        series = s.series
        atr = _last_valid(
            Indicators.atr(series.highs(), series.lows(), series.closes(), period)
        )
        return not math.isnan(atr) and atr / s.last_close() < max_fraction

    return ScreenFilter(check)


def price_above_vwap() -> ScreenFilter:
    def check(s: StockSnapshot) -> bool:
        series = s.series
        vwap = Indicators.vwap(series.highs(), series.lows(), series.closes(), series.volumes())
        return s.last_close() > _last_valid(vwap)

    return ScreenFilter(check)


def super_trend_bullish(period: int, multiplier: float) -> ScreenFilter:
    def check(s: StockSnapshot) -> bool:
        series = s.series
        st = Indicators.super_trend(
            series.highs(), series.lows(), series.closes(), period, multiplier
        )
        return st.direction[series.size() - 1] == 1

    return ScreenFilter(check)


def bollinger_breakout(period: int, k: float) -> ScreenFilter:
    """Close above the upper Bollinger band (volatility breakout)."""

    def check(s: StockSnapshot) -> bool:
        upper = _last_valid(Indicators.bollinger(s.series.closes(), period, k).upper)
        return not math.isnan(upper) and s.last_close() > upper

    return ScreenFilter(check)


def above_ichimoku_cloud() -> ScreenFilter:
    """Close above both Ichimoku cloud spans on the last bar."""

    def check(s: StockSnapshot) -> bool:
        series = s.series
        ich = Indicators.ichimoku(series.highs(), series.lows(), series.closes(), 9, 26, 52)
        i = series.size() - 1
        a, b = ich.senkou_a[i], ich.senkou_b[i]
        return _valid(a, b) and s.last_close() > max(a, b)

    return ScreenFilter(check)


def breakout(lookback: int) -> ScreenFilter:
    """Close breaks above the highest high of the previous ``lookback`` bars."""

    def check(s: StockSnapshot) -> bool:
        series = s.series
        n = series.size()
        if n < lookback + 2:
            return False
        hh = -math.inf
        for i in range(n - 1 - lookback, n - 1):
            hh = max(hh, series.high(i))
        return series.close(n - 1) > hh

    return ScreenFilter(check)


def volume_spike(lookback: int, multiplier: float) -> ScreenFilter:
    """Last bar volume exceeds ``multiplier`` times the prior average volume."""

    def check(s: StockSnapshot) -> bool:
        from quantfinlib.util import mean

        series = s.series
        n = series.size()
        if n < lookback + 2:
            return False
        avg = mean(series.volumes(), n - 1 - lookback, n - 1)
        return avg > 0 and series.volume(n - 1) > multiplier * avg

    return ScreenFilter(check)


def gap_up(min_fraction: float) -> ScreenFilter:
    """Gap up at the last open of at least ``min_fraction`` versus the prior close."""

    def check(s: StockSnapshot) -> bool:
        series = s.series
        n = series.size()
        return n >= 2 and series.open(n - 1) >= series.close(n - 2) * (1 + min_fraction)

    return ScreenFilter(check)


def near_52_week_high(within_fraction: float) -> ScreenFilter:
    """Close within ``within_fraction`` of the 52-week (252-bar) high."""

    def check(s: StockSnapshot) -> bool:
        series = s.series
        n = series.size()
        lookback = min(252, n)
        hh = -math.inf
        for i in range(n - lookback, n):
            hh = max(hh, series.high(i))
        return series.close(n - 1) >= hh * (1 - within_fraction)

    return ScreenFilter(check)


def near_52_week_low(within_fraction: float) -> ScreenFilter:
    """Close within ``within_fraction`` of the 52-week (252-bar) low."""

    def check(s: StockSnapshot) -> bool:
        series = s.series
        n = series.size()
        lookback = min(252, n)
        ll = math.inf
        for i in range(n - lookback, n):
            ll = min(ll, series.low(i))
        return series.close(n - 1) <= ll * (1 + within_fraction)

    return ScreenFilter(check)
