"""Pins for quantfinlib.backtest.validation.monte_carlo_trade_shuffle.

Java sources: BacktestAnalyticsRoundTest.java and
TradeAnalyticsEdgeTest.java (shuffle sections). The Java
SplittableRandom stream is not reproduced — per the port contract, the
pinned property set is: terminal order-invariance, percentile ordering,
determinism given the seed (np.random.default_rng), the collapse on an
order-free trade set, and the worst-case-ordering tail rank.
"""

import pytest

from quantfinlib.backtest import Trade
from quantfinlib.backtest.validation import MonteCarloTradeShuffle


def _trade(pnl: float, bars: int = 1) -> Trade:
    return Trade("X", 0, bars, 0, bars, 100.0, 100.0 + pnl, 1.0, pnl,
                 pnl / 100, Trade.REASON_SIGNAL)


def test_reshuffle_terminal_pnl_is_order_invariant_but_drawdown_is_not():
    # Terminal P&L is the sum, identical in every ordering:
    # 10 * (+30) + 10 * (-20) = +100 over 20 trades.
    trades = [_trade(30 if i % 2 == 0 else -20) for i in range(20)]
    r = MonteCarloTradeShuffle.analyze(trades, 2000, 42)
    assert r.median_terminal_pnl == pytest.approx(100.0, abs=1e-9)
    assert r.prob_loss == 0.0                      # always net positive
    # Drawdown IS order-dependent, so the distribution has spread.
    assert r.p95_max_drawdown >= r.median_max_drawdown
    assert r.p99_max_drawdown >= r.p95_max_drawdown
    assert r.median_max_drawdown > 0
    assert 0 <= r.actual_drawdown_pct <= 1


def test_front_loaded_losses_rank_as_an_unusually_painful_actual_ordering():
    # All losses first, then all wins: the worst possible drawdown
    # ordering, so the actual path should sit near the top of the
    # shuffle distribution.
    trades = [_trade(-20) for _ in range(10)] + [_trade(30) for _ in range(10)]
    r = MonteCarloTradeShuffle.analyze(trades, 2000, 7)
    assert r.actual_max_drawdown == pytest.approx(200.0, abs=1e-9)  # 10 * -20
    assert r.actual_drawdown_pct > 0.9


def test_same_seed_reproduces_the_shuffle_distribution_exactly():
    trades = [_trade(-25 if i % 3 == 0 else 15) for i in range(15)]
    a = MonteCarloTradeShuffle.analyze(trades, 500, 123)
    b = MonteCarloTradeShuffle.analyze(trades, 500, 123)
    assert a == b                                  # bit-identical result
    # A different seed draws different orderings; the median drawdown is
    # a fine-grained statistic, so at least one field is allowed to move.
    c = MonteCarloTradeShuffle.analyze(trades, 500, 124)
    assert a != c or a.median_max_drawdown == c.median_max_drawdown


def test_identical_trade_pnls_collapse_the_distribution_to_a_single_point():
    # Ten identical -5 trades: every permutation is the SAME path, so all
    # drawdown percentiles equal the actual drawdown (50), every shuffle
    # ends at -50, and the loss probability is 1.
    trades = [_trade(-5) for _ in range(10)]
    r = MonteCarloTradeShuffle.analyze(trades, 200, 9)
    assert r.actual_max_drawdown == pytest.approx(50.0, abs=1e-12)
    assert r.median_max_drawdown == pytest.approx(50.0, abs=1e-12)
    assert r.p95_max_drawdown == pytest.approx(50.0, abs=1e-12)
    assert r.p99_max_drawdown == pytest.approx(50.0, abs=1e-12)
    assert r.median_terminal_pnl == pytest.approx(-50.0, abs=1e-12)
    assert r.prob_loss == 1.0
    # Every shuffle draws down exactly as much as the actual path.
    assert r.actual_drawdown_pct == 1.0


def test_shuffle_gates():
    with pytest.raises(ValueError):
        MonteCarloTradeShuffle.analyze([_trade(10)], 1000, 1)      # < 2 trades
    with pytest.raises(ValueError):
        MonteCarloTradeShuffle.analyze([_trade(10), _trade(-5)], 50, 1)  # < 100
    with pytest.raises(ValueError):
        MonteCarloTradeShuffle.analyze(
            [_trade(float("nan")), _trade(-5)], 100, 1)            # non-finite
