"""Pins for the execution-aware factor backtest.

Java source: AlphaBacktesterReportTest.java (AlphaBacktester half).
"""

import numpy as np
import pytest

from quantfinlib.alpha.alpha_backtester import (AlphaBacktestConfig,
                                                AlphaBacktester)
from quantfinlib.alpha.alpha_context import AlphaContext
from quantfinlib.alpha.factors import Factors
from quantfinlib.alpha.portfolio_construction import PortfolioConstruction
from quantfinlib.data.bar_series import BarSeries

BARS = 260
DRIFTS = (0.004, 0.002, 0.001, -0.001, -0.002, -0.004)


def _noisy_panel() -> AlphaContext:
    """Drift plus seeded per-bar noise: the impact model needs REAL
    return volatility (the square-root law scales with sigma -- a
    noiseless drift panel has sigma = 0 and correctly charges ~no
    impact)."""
    rng = np.random.default_rng(99)
    data = {}
    for s, drift in enumerate(DRIFTS):
        b = BarSeries.builder(f"S{s}")
        close = 100.0
        for i in range(BARS):
            open_ = close
            close = open_ * (1 + drift + 0.004 * (rng.random() - 0.5))
            b.add(i, open_, max(open_, close), min(open_, close), close, 500_000)
        data[f"S{s}"] = b.build()
    return AlphaContext.of(data)


def test_predictive_factor_profits_and_costs_bite():
    ctx = _noisy_panel()
    config = AlphaBacktestConfig(30, 21, 1.0, 2.0, 1.0, 100_000_000, 20, 252)
    r = AlphaBacktester.run(ctx, Factors.momentum(20, 0), config)

    # Long winners / short losers on a drift panel: gross must profit.
    gross_final = r.gross_equity[-1]
    net_final = r.net_equity[-1]
    assert gross_final > 1.0
    # Every cost component was charged, and net < gross by construction.
    assert net_final < gross_final
    assert r.commission_drag > 0
    assert r.spread_drag > 0
    assert r.slippage_drag > 0
    assert r.impact_drag > 0
    assert r.commission_drag + r.spread_drag + r.slippage_drag + r.impact_drag == \
        pytest.approx(r.total_cost_drag(), abs=1e-12)
    # Metrics computed on both curves by the shared engine.
    assert r.net_metrics.sharpe_ratio > 0
    assert r.gross_metrics.sharpe_ratio >= r.net_metrics.sharpe_ratio
    # Ranks are drift-anchored: rebalances shuffle the noisy middle but
    # never rebuild the book from scratch.
    assert r.mean_turnover < 0.5


def test_impact_scales_with_capital_and_disables_at_zero():
    ctx = _noisy_panel()
    small = AlphaBacktestConfig(30, 21, 1.0, 2.0, 1.0, 0, 20, 252)
    big = AlphaBacktestConfig(30, 21, 1.0, 2.0, 1.0, 1_000_000_000, 20, 252)
    no_impact = AlphaBacktester.run(ctx, Factors.momentum(20, 0), small)
    big_book = AlphaBacktester.run(ctx, Factors.momentum(20, 0), big)
    assert no_impact.impact_drag == 0.0
    assert big_book.impact_drag > 0
    # Same signal, same flat costs -- size alone degrades the net result:
    # the square-root law making "capacity" a number, not a slogan.
    assert big_book.net_equity[-1] < no_impact.net_equity[-1]
    with pytest.raises(ValueError):
        AlphaBacktestConfig(-1, 21, 1, 2, 1, 0, 20, 252)
    with pytest.raises(ValueError):
        AlphaBacktester.run(ctx, Factors.momentum(20, 0),
                           AlphaBacktestConfig(BARS - 1, 21, 1, 2, 1, 0, 20, 252))


def test_impact_precondition_and_oversized_books_fail_loud():
    ctx = _noisy_panel()
    # startIndex < impactWindow with capital > 0: the impact estimator
    # would read before bar 0 -- rejected with the constraint named.
    with pytest.raises(ValueError, match="impactWindow"):
        AlphaBacktester.run(ctx, Factors.momentum(5, 0),
                           AlphaBacktestConfig.defaults(10))
    # A book absurdly larger than the universe's liquidity: cost >= 100%
    # of equity at the first rebalance must throw the capacity error,
    # never compound a negative equity curve silently.
    with pytest.raises(ValueError, match="liquidity"):
        AlphaBacktester.run(ctx, Factors.momentum(20, 0),
                           AlphaBacktestConfig(30, 21, 1, 2, 1, 1e18, 20, 252))


def test_custom_construction_pipeline_plugs_in():
    ctx = _noisy_panel()

    def builder(c, scores, index):
        w = PortfolioConstruction.z_score_weights(scores, 1.0, 0.5)
        w = PortfolioConstruction.beta_neutralize(
            w, PortfolioConstruction.trailing_betas(c, index, 25))
        return PortfolioConstruction.inverse_vol_budget(
            w, PortfolioConstruction.trailing_vols(c, index, 25), 1.0)

    config = AlphaBacktestConfig(30, 21, 1, 2, 1, 0, 20, 252)
    r = AlphaBacktester.run(ctx, Factors.momentum(20, 0), config, builder)
    assert r.net_equity[-1] > 1.0
    # A misaligned builder is rejected, not silently truncated.
    with pytest.raises(ValueError):
        AlphaBacktester.run(ctx, Factors.momentum(20, 0), config,
                           lambda c, scores, index: np.zeros(1))
