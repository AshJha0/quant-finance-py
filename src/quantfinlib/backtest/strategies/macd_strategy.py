"""MACD signal-line crossover strategy (port of Java
``backtest.strategies.MacdStrategy``)."""

from __future__ import annotations

import math

from quantfinlib.backtest.signal import Signal
from quantfinlib.backtest.trading_strategy import TradingStrategy
from quantfinlib.data.bar_series import BarSeries
from quantfinlib.indicators import Indicators


class MacdStrategy(TradingStrategy):
    """Buy when MACD crosses above its signal line, sell on the reverse."""

    def __init__(self, fast: int = 12, slow: int = 26,
                 signal_period: int = 9) -> None:
        self._fast = fast
        self._slow = slow
        self._signal_period = signal_period
        self._line = None
        self._signal = None

    def name(self) -> str:
        return f"MACD({self._fast},{self._slow},{self._signal_period})"

    def init(self, series: BarSeries) -> None:
        m = Indicators.macd(series.closes(), self._fast, self._slow,
                            self._signal_period)
        self._line = m.line
        self._signal = m.signal

    def on_bar(self, i: int) -> Signal:
        line, signal = self._line, self._signal
        if i < 1 or math.isnan(signal[i]) or math.isnan(signal[i - 1]):
            return Signal.HOLD
        if line[i - 1] <= signal[i - 1] and line[i] > signal[i]:
            return Signal.BUY
        if line[i - 1] >= signal[i - 1] and line[i] < signal[i]:
            return Signal.SELL
        return Signal.HOLD
