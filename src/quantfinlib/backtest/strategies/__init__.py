"""Reference trading strategies (port of Java
``com.quantfinlib.backtest.strategies``)."""

from quantfinlib.backtest.strategies.bollinger_bands_strategy import (
    BollingerBandsStrategy)
from quantfinlib.backtest.strategies.ema_cross_strategy import EmaCrossStrategy
from quantfinlib.backtest.strategies.macd_strategy import MacdStrategy
from quantfinlib.backtest.strategies.rsi_strategy import RsiStrategy
from quantfinlib.backtest.strategies.sma_cross_strategy import SmaCrossStrategy

__all__ = [
    "BollingerBandsStrategy",
    "EmaCrossStrategy",
    "MacdStrategy",
    "RsiStrategy",
    "SmaCrossStrategy",
]
