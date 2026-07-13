"""Pins for quantfinlib.backtest.performance_analytics.

Java source: BacktestQualityRoundTest.java (PerformanceAnalytics pins) —
profit factor / win rate on hand trades, the zero-P&L trade boundary,
the all-zero flat curve, and the exact CAGR / annualized-return pins.
"""

import math

import pytest

from quantfinlib.backtest import PerformanceAnalytics, Trade


def _trade(pnl: float) -> Trade:
    return Trade("X", 0, 1, 0, 1, 100.0, 100.0 + pnl, 1.0, pnl,
                 pnl / 100, Trade.REASON_SIGNAL)


def test_profit_factor_and_win_rate_pinned_on_hand_trades():
    # Gross profit 150, gross loss 100 -> PF 1.5; 2 wins of 4 -> 0.5.
    trades = [_trade(100), _trade(-50), _trade(50), _trade(-50)]
    m = PerformanceAnalytics.compute([100, 110, 105, 120], trades, 252)
    assert m.profit_factor == pytest.approx(1.5, abs=1e-12)
    assert m.win_rate == pytest.approx(0.5, abs=1e-12)
    assert m.trade_count == 4


def test_zero_pnl_trade_is_neither_win_nor_loss():
    # The break-even trade adds nothing to gross loss, so PF is infinite;
    # but it is NOT counted as a win. One character (> vs >=) apart.
    m = PerformanceAnalytics.compute([100, 110], [_trade(0), _trade(100)], 252)
    assert m.profit_factor == math.inf
    assert m.win_rate == pytest.approx(0.5, abs=1e-12)


def test_flat_curve_scores_zero_everywhere_not_nan():
    m = PerformanceAnalytics.compute([100, 100, 100], [], 252)
    assert m.total_return == 0.0
    assert m.cagr == 0.0
    assert m.calmar_ratio == 0.0
    assert m.profit_factor == 0.0    # no profit either
    assert m.win_rate == 0.0


def test_cagr_and_annualized_return_pinned_exactly():
    # Equity 100 -> 110 -> 121: two +10% periods, 2 periods/year:
    # CAGR = 1.21^(2/2) - 1 = 21%; annualized mean return = 20%.
    m = PerformanceAnalytics.compute([100, 110, 121], [], 2)
    assert m.total_return == pytest.approx(0.21, abs=1e-12)
    assert m.cagr == pytest.approx(0.21, abs=1e-12)
    assert m.annualized_return == pytest.approx(0.20, abs=1e-12)


def test_max_drawdown_and_final_equity():
    # 100 -> 110 -> 99 -> 121: max drawdown = (110 - 99)/110 = 1/10.
    m = PerformanceAnalytics.compute([100, 110, 99, 121], [], 252)
    assert m.max_drawdown == pytest.approx(11.0 / 110.0, abs=1e-12)
    assert m.final_equity == pytest.approx(121.0, abs=1e-12)
    assert m.total_return == pytest.approx(0.21, abs=1e-12)
