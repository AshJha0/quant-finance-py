"""EMA crossover strategy (port of Java
``backtest.strategies.EmaCrossStrategy``)."""

from __future__ import annotations

import math

from quantfinlib.backtest.signal import Signal
from quantfinlib.backtest.trading_strategy import TradingStrategy
from quantfinlib.data.bar_series import BarSeries
from quantfinlib.indicators import Indicators


class EmaCrossStrategy(TradingStrategy):
    """Buy when the fast EMA crosses above the slow EMA, sell on the
    reverse cross."""

    def __init__(self, fast_period: int, slow_period: int) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._fast = None
        self._slow = None

    def name(self) -> str:
        return f"EMA_CROSS({self._fast_period},{self._slow_period})"

    def init(self, series: BarSeries) -> None:
        self._fast = Indicators.ema(series.closes(), self._fast_period)
        self._slow = Indicators.ema(series.closes(), self._slow_period)

    def on_bar(self, i: int) -> Signal:
        fast, slow = self._fast, self._slow
        if i < 1 or math.isnan(slow[i]) or math.isnan(slow[i - 1]):
            return Signal.HOLD
        if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            return Signal.BUY
        if fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            return Signal.SELL
        return Signal.HOLD
