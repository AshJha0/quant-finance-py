"""Portfolio-level scheduling: leg balance + capacity over per-symbol
executors. Ported from Java PortfolioExecutorTest.
"""

import math

import numpy as np
import pytest

from quantfinlib.execution.benchmark_executor import (Benchmark,
                                                       BenchmarkExecutor,
                                                       MarketState)
from quantfinlib.execution.portfolio_executor import Config, PortfolioExecutor
from quantfinlib.microstructure.execution import Side
from quantfinlib.microstructure.ewma_covariance import EwmaCovariance


def test_unconstrained_is_a_transparent_passthrough():
    pe = PortfolioExecutor(2, Config.unconstrained())
    buy = pe.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    sell = pe.add(BenchmarkExecutor.of(Side.SELL, 6_000, Benchmark.TWAP))

    states = [MarketState.neutral(100, 0.5), MarketState.neutral(50, 0.5)]
    due = [0, 0]
    pe.decide(0.5, states, due)

    assert due[buy] == 5_000
    assert due[sell] == 3_000


def test_leg_balance_stops_the_ahead_leg_when_the_other_is_stuck():
    config = Config(100_000, math.inf)
    pe = PortfolioExecutor(2, config)
    buy = pe.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    sell = pe.add(BenchmarkExecutor.of(Side.SELL, 10_000, Benchmark.TWAP))

    pe.on_fill(buy, 5_000, 100)
    assert pe.net_notional() == pytest.approx(500_000, abs=1e-9)

    due = [0, 0]
    buy_state = MarketState.neutral(100, 0.6)
    sell_state = MarketState(100, 0, 0, 0, 0.6, 0, 0)
    pe.decide(0.6, [buy_state, sell_state], due)

    assert due[sell] == 0
    assert due[buy] == 0, "the band stops the buy leg from running further ahead"

    free = PortfolioExecutor(2, Config.unconstrained())
    b2 = free.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    free.add(BenchmarkExecutor.of(Side.SELL, 10_000, Benchmark.TWAP))
    free.on_fill(b2, 5_000, 100)
    due2 = [0, 0]
    free.decide(0.6, [buy_state, sell_state], due2)
    assert due2[b2] == 1_000, "unconstrained, the buy child continues its schedule"


def test_leg_balance_throttles_partially_not_just_to_zero():
    config = Config(300_000, math.inf)
    pe = PortfolioExecutor(2, config)
    buy = pe.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    sell = pe.add(BenchmarkExecutor.of(Side.SELL, 10_000, Benchmark.TWAP))
    pe.on_fill(buy, 2_000, 100)

    due = [0, 0]
    buy_state = MarketState.neutral(100, 0.5)
    sell_state = MarketState(100, 0, 0, 0, 0.5, 0, 0)
    pe.decide(0.5, [buy_state, sell_state], due)

    assert due[buy] == 1_000, "scaled to the band, not zeroed"


def test_capacity_goes_to_the_riskiest_names_when_the_budget_binds():
    config = Config(math.inf, 500_000)
    pe = PortfolioExecutor(2, config)
    calm = pe.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    wild = pe.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))

    calm_state = MarketState.neutral(100, 0.5)
    wild_state = MarketState(100, 0, 1.0, math.inf, 0.5, 0, 0)
    due = [0, 0]
    pe.decide(0.5, [calm_state, wild_state], due)

    assert due[wild] == 2_500
    assert due[calm] == 2_500
    assert due[calm] * 100.0 + due[wild] * 100.0 <= 500_000


def test_capacity_allocation_cannot_reviolate_the_leg_band():
    config = Config(100_000, 1_000_000)
    pe = PortfolioExecutor(2, config)
    buy = pe.add(BenchmarkExecutor.of(Side.BUY, 20_000, Benchmark.TWAP))
    sell = pe.add(BenchmarkExecutor.of(Side.SELL, 20_000, Benchmark.TWAP))

    buy_state = MarketState.neutral(100, 0.5)
    sell_state = MarketState(100, 0, 1.0, math.inf, 0.5, 0, 0)
    due = [0, 0]
    pe.decide(0.5, [buy_state, sell_state], due)

    assert due[buy] == 3_750
    assert due[sell] == 4_750
    projected_net = due[buy] * 100.0 - due[sell] * 100.0
    assert abs(projected_net) <= 100_000
    assert due[buy] * 100.0 + due[sell] * 100.0 <= 1_000_000


def test_covariance_upgrades_capacity_from_diagonal_to_basket_risk():
    cov = EwmaCovariance(3, 0.97)
    rng = np.random.default_rng(7)
    for _ in range(2_000):
        g = 1e-4 * rng.standard_normal()
        r = [g, g, 1e-4 * rng.standard_normal()]
        cov.on_returns(r)

    pe = PortfolioExecutor(3, Config(math.inf, 750_000))
    a = pe.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    b = pe.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    c = pe.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    pe.use_risk_model(cov)

    s = MarketState.neutral(100, 0.5)
    due = [0, 0, 0]
    pe.decide(0.5, [s, s, s], due)

    assert due[a] == due[b], "symmetric legs get symmetric capacity"
    assert due[a] > due[c], "the correlated pair carries more basket risk"
    assert (due[a] + due[b] + due[c]) * 100.0 <= 750_000

    pe2 = PortfolioExecutor(1, Config(math.inf, 100_000))
    pe2.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    pe2.use_risk_model(EwmaCovariance(1))
    one = [0]
    pe2.decide(0.5, [MarketState.neutral(100, 0.5)], one)
    assert one[0] == 1_000, "unlearned model: plain budget cut still applies"

    with pytest.raises(ValueError):
        pe.use_risk_model(EwmaCovariance(2))


def test_overlays_only_ever_reduce_a_childs_own_due():
    config = Config(50_000, 100_000)
    pe = PortfolioExecutor(3, config)
    a = pe.add(BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP))
    b = pe.add(BenchmarkExecutor.of(Side.SELL, 4_000, Benchmark.ARRIVAL_PRICE))
    c = pe.add(BenchmarkExecutor.of(Side.BUY, 2_000, Benchmark.CLOSING_PRICE))

    states = [MarketState.neutral(100, 0.4), MarketState.neutral(25, 0.4),
             MarketState.neutral(400, 0.4)]
    portfolio = [0, 0, 0]
    pe.decide(0.4, states, portfolio)

    own = [
        BenchmarkExecutor.of(Side.BUY, 10_000, Benchmark.TWAP).due_quantity(0.4, states[0]),
        BenchmarkExecutor.of(Side.SELL, 4_000, Benchmark.ARRIVAL_PRICE).due_quantity(0.4, states[1]),
        BenchmarkExecutor.of(Side.BUY, 2_000, Benchmark.CLOSING_PRICE).due_quantity(0.4, states[2]),
    ]
    assert portfolio[a] <= own[0] and portfolio[b] <= own[1] and portfolio[c] <= own[2]


def test_fills_flow_through_to_children_and_the_ledger():
    pe = PortfolioExecutor(2, Config.unconstrained())
    buy = pe.add(BenchmarkExecutor.of(Side.BUY, 1_000, Benchmark.TWAP))
    sell = pe.add(BenchmarkExecutor.of(Side.SELL, 1_000, Benchmark.TWAP))

    pe.on_fill(buy, 1_000, 100)
    pe.on_fill(sell, 400, 50)
    assert pe.child(buy).executed() == 1_000
    assert pe.net_notional() == pytest.approx(100_000 - 20_000, abs=1e-9)
    assert pe.child(buy).done()
    assert not pe.done(), "sell leg still working"
    pe.on_fill(sell, 600, 50)
    assert pe.done()

    pe2 = PortfolioExecutor(1, Config.unconstrained())
    h = pe2.add(BenchmarkExecutor.of(Side.BUY, 100, Benchmark.TWAP))
    pe2.on_fill(h, 100, math.nan)
    assert pe2.child(h).executed() == 100
    assert pe2.net_notional() == pytest.approx(0, abs=1e-9)


def test_portfolio_executor_validation():
    with pytest.raises(ValueError):
        Config(0, 1)
    with pytest.raises(ValueError):
        Config(1, math.nan)
    with pytest.raises(ValueError):
        PortfolioExecutor(0, Config.unconstrained())

    pe = PortfolioExecutor(1, Config.unconstrained())
    pe.add(BenchmarkExecutor.of(Side.BUY, 100, Benchmark.TWAP))
    with pytest.raises(RuntimeError):
        pe.add(BenchmarkExecutor.of(Side.BUY, 100, Benchmark.TWAP))
    with pytest.raises(ValueError):
        pe.decide(0.5, [], [0])
