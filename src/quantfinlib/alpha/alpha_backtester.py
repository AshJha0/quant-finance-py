"""Execution-aware factor backtest (port of Java ``alpha.AlphaBacktester``):
runs a factor through a construction pipeline into weights, holds them
between rebalances, and charges the four costs that separate paper
alpha from real alpha:

* **Commission** -- flat bps on traded notional;
* **Bid-ask spread** -- half-spread bps paid on every trade (crossing
  the spread once per side);
* **Slippage** -- additional fixed bps of implementation noise
  (latency, partial fills, venue fees);
* **Market impact** -- the size-dependent cost, via the square-root
  law in :class:`~quantfinlib.microstructure.market_impact_model.MarketImpactModel`,
  with per-symbol ADV and daily vol estimated from the trailing
  window. This is the term that grows with capital: the same signal
  that nets 8% on $10m can net zero on $1b.

The simulation is *weight-based* (fractional book, returns compound
multiplicatively from 1.0): simpler and adequate for factor research.
For share-level accounting with lifecycle events, feed the constructed
weights into a portfolio-level backtester's survivorship-aware
overload instead.

Both gross and net curves are tracked, plus the cumulative drag of
each cost component -- "which cost kills this signal" is the
actionable output of an execution-aware backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from quantfinlib.alpha.alpha_context import AlphaContext
from quantfinlib.alpha.alpha_factor import AlphaFactor
from quantfinlib.alpha.portfolio_construction import PortfolioConstruction
from quantfinlib.backtest.performance_analytics import (PerformanceAnalytics,
                                                        PerformanceMetrics)
from quantfinlib.microstructure.market_impact_model import MarketImpactModel

#: Builds target weights from raw scores at a rebalance (the
#: construction hook): ``(ctx, scores, index) -> weights``.
WeightBuilder = Callable[[AlphaContext, np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class AlphaBacktestConfig:
    """Backtest configuration.

    Attributes:
        start_index: first tradeable bar (must cover factor warm-up
            and the impact estimation window).
        rebalance_every_bars: holding period between weight refreshes.
        commission_bps: commission per side on traded notional.
        half_spread_bps: half the quoted spread, paid per trade.
        slippage_bps: fixed implementation noise per trade.
        capital: book size in currency -- impact needs real size (0
            disables the impact model entirely).
        impact_window: trailing bars for ADV/vol estimation.
        periods_per_year: bar frequency for annualized metrics.
    """

    start_index: int
    rebalance_every_bars: int
    commission_bps: float
    half_spread_bps: float
    slippage_bps: float
    capital: float
    impact_window: int
    periods_per_year: int

    def __post_init__(self) -> None:
        if (self.start_index < 0 or self.rebalance_every_bars <= 0
                or self.commission_bps < 0 or self.half_spread_bps < 0
                or self.slippage_bps < 0 or self.capital < 0
                or self.impact_window < 2 or self.periods_per_year <= 0):
            raise ValueError("invalid backtest config")

    @staticmethod
    def defaults(start_index: int) -> "AlphaBacktestConfig":
        """Institutional-ish daily-bar defaults: 1bp commission, 2bp
        half-spread, 1bp slippage, $100m book."""
        return AlphaBacktestConfig(start_index, 21, 1.0, 2.0, 1.0,
                                   100_000_000, 20, 252)


@dataclass(frozen=True)
class AlphaBacktestResult:
    """Net/gross curves plus the cumulative fraction of equity each
    cost component consumed -- the cost autopsy."""

    net_equity: np.ndarray
    gross_equity: np.ndarray
    net_metrics: PerformanceMetrics
    gross_metrics: PerformanceMetrics
    commission_drag: float
    spread_drag: float
    slippage_drag: float
    impact_drag: float
    mean_turnover: float

    def total_cost_drag(self) -> float:
        """Total cost drag: gross minus net, decomposed."""
        return (self.commission_drag + self.spread_drag
                + self.slippage_drag + self.impact_drag)


def _default_builder(ctx: AlphaContext, scores: np.ndarray,
                     index: int) -> np.ndarray:
    return PortfolioConstruction.z_score_weights(scores, 1.0, 0.05)


class AlphaBacktester:
    """Static entry point; see the module docstring."""

    @staticmethod
    def run(ctx: AlphaContext, factor: AlphaFactor,
            config: AlphaBacktestConfig,
            builder: Optional[WeightBuilder] = None) -> AlphaBacktestResult:
        """Runs with ``builder`` (defaults to the standard z-score
        construction: gross 1.0, 5% name cap)."""
        if builder is None:
            builder = _default_builder
        n = ctx.bars()
        m = ctx.symbol_count()
        if config.start_index >= n - 1:
            raise ValueError("startIndex leaves no bars to trade")
        if config.capital > 0 and config.start_index < config.impact_window:
            # The impact model reads impact_window bars of ADV/vol
            # history before the first rebalance; enforcing it here
            # beats an uncontextualized index error deep in the loop.
            raise ValueError(
                f"startIndex {config.start_index} < impactWindow "
                f"{config.impact_window} -- impact estimation needs "
                "that history (or set capital to 0 to disable impact)")

        weights = np.zeros(m)          # current holdings (fractions of equity)
        net_equity = np.zeros(n - config.start_index)
        gross_equity = np.zeros(net_equity.shape[0])
        net_equity[0] = 1.0
        gross_equity[0] = 1.0
        commission_drag = 0.0
        spread_drag = 0.0
        slippage_drag = 0.0
        impact_drag = 0.0
        turnover_sum = 0.0
        rebalances = 0

        for t in range(config.start_index, n - 1):
            out_idx = t - config.start_index
            cost_factor = 1.0

            # Rebalance at the bar close, before earning the next bar's
            # return.
            if out_idx % config.rebalance_every_bars == 0:
                target = np.asarray(
                    builder(ctx, factor.scores(ctx, t), t), dtype=float)
                if target.shape[0] != m:
                    raise ValueError(
                        "weight builder returned misaligned array")
                turnover = 0.0
                commission = 0.0
                spread = 0.0
                slip = 0.0
                impact = 0.0
                for i in range(m):
                    traded = abs(target[i] - weights[i])
                    if traded == 0:
                        continue
                    turnover += traded
                    # Flat per-trade costs scale with traded fraction of
                    # equity.
                    commission += traded * config.commission_bps / 1e4
                    spread += traded * config.half_spread_bps / 1e4
                    slip += traded * config.slippage_bps / 1e4
                    # Impact needs real size: traded fraction * capital
                    # -> shares.
                    if config.capital > 0:
                        impact += (traded
                                  * AlphaBacktester._impact_bps(
                                      ctx, i, t, traded, config) / 1e4)
                cost = commission + spread + slip + impact
                if cost >= 1:
                    # The square-root law is unbounded: a book too big
                    # for its liquidity can cost more than the equity.
                    # Negative equity compounds into sign-inverted
                    # garbage, so fail loudly -- this IS the capacity
                    # answer.
                    raise ValueError(
                        f"execution cost {cost * 100:.1f}% of equity at "
                        f"bar {t} (impact {impact * 100:.1f}%) -- the "
                        "book is too large for this universe's "
                        "liquidity; reduce capital or the rebalance "
                        "size")
                cost_factor = 1 - cost
                commission_drag += commission
                spread_drag += spread
                slippage_drag += slip
                impact_drag += impact
                turnover_sum += turnover / 2  # buys and sells counted once
                rebalances += 1
                weights = target.copy()

            # Earn one bar of the held book: r_p = sum(w_i * r_i).
            portfolio_return = 0.0
            for i in range(m):
                if weights[i] != 0:
                    portfolio_return += weights[i] * ctx.return_over(i, t, t + 1)
            # Cost folds into the compounding step: each equity slot is
            # written exactly once and net_equity[0] stays 1.0 by
            # contract.
            net_equity[out_idx + 1] = (net_equity[out_idx] * cost_factor
                                       * (1 + portfolio_return))
            gross_equity[out_idx + 1] = (gross_equity[out_idx]
                                         * (1 + portfolio_return))

        return AlphaBacktestResult(
            net_equity, gross_equity,
            PerformanceAnalytics.compute(net_equity, (), config.periods_per_year),
            PerformanceAnalytics.compute(gross_equity, (), config.periods_per_year),
            commission_drag, spread_drag, slippage_drag, impact_drag,
            0.0 if rebalances == 0 else turnover_sum / rebalances)

    @staticmethod
    def _impact_bps(ctx: AlphaContext, symbol: int, index: int,
                    traded_fraction: float,
                    config: AlphaBacktestConfig) -> float:
        """Square-root-law impact for one name's trade, via the shared
        :meth:`MarketImpactModel.estimate` bridge (the same estimator
        the portfolio-level cost model uses, so the alpha and
        portfolio engines can never disagree on impact). Names without
        volume data contribute zero impact -- a data gap, not free
        liquidity; the flat costs still apply."""
        s = ctx.series(symbol)
        impact = MarketImpactModel.estimate(s, index, config.impact_window)
        if impact is None:
            return 0.0
        shares = traded_fraction * config.capital / s.close(index)
        return impact.square_root_impact_bps(shares)
