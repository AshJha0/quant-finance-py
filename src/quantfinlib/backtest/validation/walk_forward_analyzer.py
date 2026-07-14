"""Walk-forward analysis (port of Java
``backtest.validation.WalkForwardAnalyzer``).

The standard defense against overfit backtests. The series is split
into rolling train/test windows; on each fold the parameter grid is
optimized *only on the train window* and the winner is evaluated on the
unseen test window. Out-of-sample test segments are stitched into one
continuous equity curve (capital carries across folds), giving honest
out-of-sample metrics and the walk-forward efficiency ratio (OOS / IS
objective — near 1 is robust, near 0 is curve-fitting).

Each test window is evaluated WARM: the backtest sees the preceding
train bars for indicator warm-up but only trades from the test boundary
(:meth:`Backtester.run` with ``trade_from``). Evaluating a bare test
slice would re-compute every indicator cold and force HOLD through each
fold's first lookback bars — systematically understating out-of-sample
activity.

The efficiency ratio is only meaningful when the in-sample objective
sum is positive; when it is zero or negative (the optimizer could not
find anything that even backtests well in-sample) efficiency is NaN — a
ratio of two losses saying "0.5" would read as robust when both sides
are failing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from quantfinlib.backtest.backtest_config import BacktestConfig
from quantfinlib.backtest.backtester import Backtester
from quantfinlib.backtest.performance_analytics import (PerformanceAnalytics,
                                                        PerformanceMetrics)
from quantfinlib.backtest.trade import Trade
from quantfinlib.backtest.validation.grid_search_optimizer import (
    GridSearchOptimizer, Objective, StrategyFactory)
from quantfinlib.backtest.validation.parameter_grid import ParameterGrid
from quantfinlib.data.bar_series import BarSeries


class WalkForwardAnalyzer:
    """Static walk-forward analysis; see the module docstring."""

    @dataclass(frozen=True)
    class Fold:
        """One train/test fold and its in/out-of-sample objectives."""

        train_from: int
        train_to: int
        test_from: int
        test_to: int
        best_params: Dict[str, float]
        in_sample_objective: float
        out_of_sample_objective: float

    @dataclass(frozen=True)
    class WalkForwardResult:
        """Stitched out-of-sample results across all folds."""

        folds: Tuple["WalkForwardAnalyzer.Fold", ...]
        out_of_sample_equity: np.ndarray
        out_of_sample_metrics: PerformanceMetrics
        out_of_sample_trades: Tuple[Trade, ...]
        efficiency: float

    @staticmethod
    def analyze(series: BarSeries, grid: ParameterGrid,
                factory: StrategyFactory, config: BacktestConfig,
                train_bars: int, test_bars: int, objective: Objective
                ) -> "WalkForwardAnalyzer.WalkForwardResult":
        """Runs the rolling walk-forward.

        Args:
            train_bars: Bars in each optimization window.
            test_bars: Bars in each out-of-sample window; the window
                rolls forward by this amount per fold.
        """
        n = series.size()
        if train_bars < 10 or test_bars < 2 or train_bars + test_bars > n:
            raise ValueError(f"invalid windows: train={train_bars} "
                             f"test={test_bars} bars={n}")
        folds: List[WalkForwardAnalyzer.Fold] = []
        oos_equity: List[float] = []
        oos_trades: List[Trade] = []
        carry_capital = config.initial_capital
        is_sum = 0.0
        oos_sum = 0.0

        start = 0
        while start + train_bars + test_bars <= n:
            train_to = start + train_bars
            test_to = train_to + test_bars

            train = series.slice(start, train_to)
            best = GridSearchOptimizer.best(grid, factory, train, config,
                                            objective)

            # Warm-up = the train window; trading starts at the test
            # boundary.
            warm_plus_test = series.slice(start, test_to)
            oos = Backtester.run(
                factory(best.params), warm_plus_test,
                config.with_initial_capital(carry_capital), train_bars)
            oos_score = objective(oos.metrics())

            folds.append(WalkForwardAnalyzer.Fold(
                start, train_to, train_to, test_to,
                best.params, best.objective, oos_score))
            oos_equity.extend(float(e) for e in oos.equity_curve())
            oos_trades.extend(oos.trades())
            carry_capital = oos.metrics().final_equity
            is_sum += best.objective
            oos_sum += oos_score
            start += test_bars

        if not folds:
            raise ValueError("series too short for even one fold")
        equity = np.asarray(oos_equity, dtype=float)
        metrics = PerformanceAnalytics.compute(equity, oos_trades,
                                               config.periods_per_year)
        efficiency = oos_sum / is_sum if is_sum > 0 else math.nan
        return WalkForwardAnalyzer.WalkForwardResult(
            tuple(folds), equity, metrics, tuple(oos_trades), efficiency)
