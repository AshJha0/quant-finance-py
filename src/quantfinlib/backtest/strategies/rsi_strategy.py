"""RSI mean-reversion strategy (port of Java
``backtest.strategies.RsiStrategy``)."""

from __future__ import annotations

import math

from quantfinlib.backtest.signal import Signal
from quantfinlib.backtest.trading_strategy import TradingStrategy
from quantfinlib.data.bar_series import BarSeries
from quantfinlib.indicators import Indicators


class RsiStrategy(TradingStrategy):
    """Buy when RSI crosses up through the oversold level, sell when it
    crosses down through the overbought level."""

    def __init__(self, period: int, oversold: float,
                 overbought: float) -> None:
        self._period = period
        self._oversold = oversold
        self._overbought = overbought
        self._rsi = None

    def name(self) -> str:
        return f"RSI({self._period},{self._oversold},{self._overbought})"

    def init(self, series: BarSeries) -> None:
        self._rsi = Indicators.rsi(series.closes(), self._period)

    def on_bar(self, i: int) -> Signal:
        rsi = self._rsi
        if i < 1 or math.isnan(rsi[i]) or math.isnan(rsi[i - 1]):
            return Signal.HOLD
        if rsi[i - 1] <= self._oversold and rsi[i] > self._oversold:
            return Signal.BUY
        if rsi[i - 1] >= self._overbought and rsi[i] < self._overbought:
            return Signal.SELL
        return Signal.HOLD
