"""Bollinger Band mean-reversion strategy (port of Java
``backtest.strategies.BollingerBandsStrategy``)."""

from __future__ import annotations

import math

from quantfinlib.backtest.signal import Signal
from quantfinlib.backtest.trading_strategy import TradingStrategy
from quantfinlib.data.bar_series import BarSeries
from quantfinlib.indicators import Indicators


class BollingerBandsStrategy(TradingStrategy):
    """Buy when the close dips below the lower band, sell when it
    recovers to the middle band or stretches above the upper band."""

    def __init__(self, period: int = 20, k: float = 2.0) -> None:
        self._period = period
        self._k = k
        self._upper = None
        self._middle = None
        self._lower = None
        self._close = None

    def name(self) -> str:
        return f"BOLLINGER({self._period},{self._k})"

    def init(self, series: BarSeries) -> None:
        self._close = series.closes()
        b = Indicators.bollinger(self._close, self._period, self._k)
        self._upper = b.upper
        self._middle = b.middle
        self._lower = b.lower

    def on_bar(self, i: int) -> Signal:
        close = self._close
        if math.isnan(self._middle[i]):
            return Signal.HOLD
        if close[i] < self._lower[i]:
            return Signal.BUY
        if ((close[i] >= self._middle[i] and close[i] > self._upper[i])
                or (close[i] >= self._middle[i] and i > 0
                    and close[i - 1] < self._middle[i - 1])):
            return Signal.SELL
        return Signal.HOLD
