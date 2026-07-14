"""Multi-asset, long/short portfolio backtester (port of the classic
lane of Java ``backtest.portfolio.PortfolioBacktester``).

Rebalances positions (possibly fractional and negative) toward the
strategy's target weights at a configurable cadence, charging
commission on traded notional. This is where the optimizer lane meets
the backtester — feed optimizer weights, vol-target overlays, or
momentum rankings straight in.

The Java survivorship-aware overload (point-in-time universe, terminal
events, cash dividends) depends on ``data.PointInTimeUniverse`` and
``data.CorporateActions``, which are not in the Python port yet; only
the classic every-symbol-always-tradeable run is ported here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import numpy as np

from quantfinlib.backtest.performance_analytics import (PerformanceAnalytics,
                                                        PerformanceMetrics)
from quantfinlib.backtest.portfolio.portfolio_strategy import PortfolioStrategy
from quantfinlib.backtest.trade_cost_model import TradeCostModel
from quantfinlib.data.bar_series import BarSeries


class PortfolioBacktester:
    """Static engine; see the module docstring."""

    @dataclass(frozen=True)
    class Config:
        """Portfolio run parameters. ``cost_model``, when set, supersedes
        the flat ``commission_rate`` for every trade — the shared
        :class:`TradeCostModel` seam that makes a run execution-aware.
        ``None`` keeps the legacy flat commission behavior exactly."""

        initial_capital: float
        commission_rate: float
        rebalance_every_bars: int
        periods_per_year: int
        cost_model: Optional[TradeCostModel] = None

        @staticmethod
        def defaults() -> "PortfolioBacktester.Config":
            return PortfolioBacktester.Config(1_000_000, 0.001, 1, 252, None)

        def with_rebalance_every(self, bars: int
                                 ) -> "PortfolioBacktester.Config":
            return replace(self, rebalance_every_bars=bars)

        def with_cost_model(self, model: TradeCostModel
                            ) -> "PortfolioBacktester.Config":
            """Pluggable per-trade costs (e.g.
            ``TradeCostModel.institutional``)."""
            return replace(self, cost_model=model)

        def fee(self, series: BarSeries, index: int,
                notional: float) -> float:
            """One-way trade fee for ``notional`` of ``series`` at bar
            ``index``."""
            if self.cost_model is not None:
                return (notional
                        * self.cost_model.cost_bps(series, index, notional)
                        / 1e4)
            return notional * self.commission_rate

    @dataclass(frozen=True)
    class Result:
        """Equity curve, metrics, total costs/turnover and the final
        book."""

        equity_curve: np.ndarray
        metrics: PerformanceMetrics
        total_costs: float
        total_turnover_notional: float
        final_positions: Dict[str, float]

    @staticmethod
    def run(strategy: PortfolioStrategy, data: Dict[str, BarSeries],
            config: "PortfolioBacktester.Config"
            ) -> "PortfolioBacktester.Result":
        """Classic run: every supplied symbol is tradeable on every bar."""
        if not data:
            raise ValueError("no series supplied")
        # Sorted for determinism: with an arbitrary dict, insertion
        # order would otherwise decide within-bar processing order.
        symbols = sorted(data.keys())
        n = data[symbols[0]].size()
        for s in symbols:
            if data[s].size() != n:
                raise ValueError(f"series must be index-aligned: {s}")
        strategy.init(data)

        cash = config.initial_capital
        positions: Dict[str, float] = {}   # signed quantities
        equity = np.zeros(n)
        total_costs = 0.0
        total_turnover = 0.0

        for i in range(n):
            portfolio_value = cash + PortfolioBacktester._market_value(
                positions, data, i)

            if i % config.rebalance_every_bars == 0:
                weights = strategy.target_weights(i)
                for symbol in symbols:
                    weight = weights.get(symbol, 0.0)
                    close = data[symbol].close(i)
                    target_qty = weight * portfolio_value / close
                    current_qty = positions.get(symbol, 0.0)
                    delta = target_qty - current_qty
                    if delta == 0:
                        continue
                    notional = abs(delta) * close
                    fee = config.fee(data[symbol], i, notional)
                    cash -= delta * close + fee
                    total_costs += fee
                    total_turnover += notional
                    positions[symbol] = target_qty
            equity[i] = cash + PortfolioBacktester._market_value(
                positions, data, i)

        final_positions = {s: positions[s] for s in symbols
                           if positions.get(s, 0.0) != 0}
        metrics = PerformanceAnalytics.compute(equity, (),
                                               config.periods_per_year)
        return PortfolioBacktester.Result(equity, metrics, total_costs,
                                          total_turnover, final_positions)

    @staticmethod
    def _market_value(positions: Dict[str, float],
                      data: Dict[str, BarSeries], index: int) -> float:
        value = 0.0
        for symbol, qty in positions.items():
            value += qty * data[symbol].close(index)
        return value
