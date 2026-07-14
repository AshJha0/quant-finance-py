"""Exhaustive strategy parameter search (port of Java
``backtest.validation.GridSearchOptimizer``).

Backtests every grid combination and ranks by an objective (e.g.
``lambda m: m.sharpe_ratio``). Non-finite objectives rank last. Feed
the resulting in-sample winners into the walk-forward analyzer — never
trust them raw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List

from quantfinlib.backtest.backtest_config import BacktestConfig
from quantfinlib.backtest.backtester import Backtester
from quantfinlib.backtest.performance_analytics import PerformanceMetrics
from quantfinlib.backtest.trading_strategy import TradingStrategy
from quantfinlib.backtest.validation.parameter_grid import ParameterGrid
from quantfinlib.backtest.validation.sharpe_validation import SharpeValidation
from quantfinlib.data.bar_series import BarSeries
from quantfinlib.util import math_utils as mu

StrategyFactory = Callable[[Dict[str, float]], TradingStrategy]
Objective = Callable[[PerformanceMetrics], float]


class GridSearchOptimizer:
    """Static grid search; see the module docstring."""

    @dataclass(frozen=True)
    class Candidate:
        """One grid trial: its parameters, metrics and objective score."""

        params: Dict[str, float]
        metrics: PerformanceMetrics
        objective: float

    @staticmethod
    def search(grid: ParameterGrid, factory: StrategyFactory,
               series: BarSeries, config: BacktestConfig,
               objective: Objective
               ) -> List["GridSearchOptimizer.Candidate"]:
        """Every combination, backtested and ranked best-first."""
        if grid.size() == 0:
            raise ValueError("empty parameter grid: nothing to search")
        out: List[GridSearchOptimizer.Candidate] = []
        for params in grid.combinations():
            result = Backtester.run(factory(params), series, config)
            score = objective(result.metrics())
            out.append(GridSearchOptimizer.Candidate(
                params, result.metrics(),
                score if math.isfinite(score) else -math.inf))
        out.sort(key=lambda c: c.objective, reverse=True)
        return out

    @staticmethod
    def best(grid: ParameterGrid, factory: StrategyFactory,
             series: BarSeries, config: BacktestConfig,
             objective: Objective) -> "GridSearchOptimizer.Candidate":
        """The winning parameter set only."""
        return GridSearchOptimizer.search(grid, factory, series, config,
                                          objective)[0]

    @staticmethod
    def deflated_sharpe_of_winner(ranked, winner_returns,
                                  periods_per_year: int) -> float:
        """The MULTIPLE-TESTING HAIRCUT for the grid's winner: the
        probability that the top-ranked candidate's Sharpe beats what
        the best of ``len(ranked)`` zero-skill trials would have scored
        anyway (:meth:`SharpeValidation.deflated_sharpe`). A grid search
        computes a Sharpe for every trial and then quietly reports only
        the maximum — this is the one number that makes that selection
        honest. Values near 1 mean the winner survives its own search;
        below ~0.95 the "best" parameter set is indistinguishable from
        picking the luckiest of N random ones.

        Args:
            ranked: Result of :meth:`search` (uses every trial's Sharpe
                as the null distribution), >= 2 trials.
            winner_returns: The winner's per-period returns (derive from
                its equity curve), >= 4 observations.
            periods_per_year: The annualization used by the backtest
                metrics.
        """
        ranked = list(ranked)
        winner_returns = list(winner_returns)
        if len(ranked) < 2:
            raise ValueError(f"need >= 2 trials, got {len(ranked)}")
        if len(winner_returns) < 4:
            raise ValueError(
                f"need >= 4 winner returns, got {len(winner_returns)}")
        if periods_per_year <= 0:
            raise ValueError(
                f"periods_per_year must be > 0, got {periods_per_year}")
        # SharpeValidation works in per-period units; metrics store
        # annualized.
        per_period_scale = math.sqrt(periods_per_year)
        trial_sharpes = []
        for c in ranked:
            s = c.metrics.sharpe_ratio / per_period_scale
            trial_sharpes.append(s if math.isfinite(s) else 0.0)
        sd = mu.std_dev(winner_returns)
        observed = mu.mean(winner_returns) / sd if sd > 0 else 0.0
        return SharpeValidation.deflated_sharpe(
            observed, trial_sharpes, len(winner_returns),
            mu.skewness(winner_returns), mu.kurtosis(winner_returns))
