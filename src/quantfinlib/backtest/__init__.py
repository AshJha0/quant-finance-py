"""Backtest engine and analytics (port of Java ``com.quantfinlib.backtest``).

The strategy engine (Backtester, execution-aware backtester, execution
models), the trade record, trade/equity-curve/drawdown/benchmark
analytics, and the ``strategies``, ``validation`` and ``portfolio``
subpackages.
"""

from quantfinlib.backtest.backtest_config import BacktestConfig
from quantfinlib.backtest.backtester import Backtester, BacktestResult
from quantfinlib.backtest.benchmark_comparison import BenchmarkComparison
from quantfinlib.backtest.drawdown_analytics import DrawdownAnalytics
from quantfinlib.backtest.execution_aware_backtester import (
    ExecutionAwareBacktester, ExecutionAwareResult)
from quantfinlib.backtest.execution_models import (ExecutionModel,
                                                   IcebergExecution,
                                                   IcebergOrder,
                                                   InstantExecution,
                                                   LastLookExecution)
from quantfinlib.backtest.parent_order import ParentOrder
from quantfinlib.backtest.performance_analytics import (PerformanceAnalytics,
                                                        PerformanceMetrics)
from quantfinlib.backtest.signal import Signal
from quantfinlib.backtest.trade import Trade
from quantfinlib.backtest.trade_analytics import TradeAnalytics
from quantfinlib.backtest.trade_cost_model import TradeCostModel
from quantfinlib.backtest.trading_strategy import TradingStrategy

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "BenchmarkComparison",
    "DrawdownAnalytics",
    "ExecutionAwareBacktester",
    "ExecutionAwareResult",
    "ExecutionModel",
    "IcebergExecution",
    "IcebergOrder",
    "InstantExecution",
    "LastLookExecution",
    "ParentOrder",
    "PerformanceAnalytics",
    "PerformanceMetrics",
    "Signal",
    "Trade",
    "TradeAnalytics",
    "TradeCostModel",
    "TradingStrategy",
]
