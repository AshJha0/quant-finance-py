"""Backtest analytics core (port of Java ``com.quantfinlib.backtest``).

This package carries the strategy-engine-free analytics layer: the
trade record, trade/equity-curve/drawdown/benchmark analytics, and the
``validation`` and ``portfolio`` subpackages. The Backtester engine,
strategies and execution models are a later phase.
"""

from quantfinlib.backtest.benchmark_comparison import BenchmarkComparison
from quantfinlib.backtest.drawdown_analytics import DrawdownAnalytics
from quantfinlib.backtest.performance_analytics import (PerformanceAnalytics,
                                                        PerformanceMetrics)
from quantfinlib.backtest.trade import Trade
from quantfinlib.backtest.trade_analytics import TradeAnalytics

__all__ = [
    "BenchmarkComparison",
    "DrawdownAnalytics",
    "PerformanceAnalytics",
    "PerformanceMetrics",
    "Trade",
    "TradeAnalytics",
]
