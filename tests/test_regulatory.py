"""Pins for quantfinlib.regulatory (MiFID-style best-ex / market-quality).

Java sources: BestExecutionAnalyzer/FixAnalyzer/MarketQualityMetrics.java.
"""

from __future__ import annotations

import math

import pytest

from quantfinlib.microstructure.execution import Side
from quantfinlib.regulatory import market_quality_metrics as mqm
from quantfinlib.regulatory.best_execution_analyzer import (
    BestExecutionAnalyzer, OrderOutcome)
from quantfinlib.regulatory.fix_analyzer import analyze, calculate_fix


# ----------------------------------------------------------------------
# BestExecutionAnalyzer
# ----------------------------------------------------------------------


def test_best_execution_report_basic_aggregates():
    a = BestExecutionAnalyzer()
    a.add(OrderOutcome("1", "NYSE", Side.BUY, 100, 100.0, 100.05, 1_000_000, True))
    a.add(OrderOutcome("2", "NYSE", Side.SELL, 100, 100.0, 99.90, 2_000_000, True))
    a.add(OrderOutcome("3", "ARCA", Side.BUY, 100, 100.0, 100.0, 500_000, False))
    report = a.report()

    assert report.total_orders == 3
    assert report.fill_rate == pytest.approx(2 / 3)
    # slip1 = +1*(100.05-100)/100*1e4 = 5 bps; slip2 = -1*(99.90-100)/100*1e4 = 10 bps
    assert report.avg_slippage_bps == pytest.approx((5.0 + 10.0) / 2)
    assert report.median_latency_to_fill_millis == pytest.approx((1.0 + 2.0) / 2)
    # Both fills have slip > 0 (worse than arrival): 0% at-or-better.
    assert report.at_or_better_than_arrival_pct == pytest.approx(0.0)
    assert report.avg_slippage_bps_by_venue == {"NYSE": pytest.approx(7.5)}


def test_best_execution_report_empty_raises():
    with pytest.raises(RuntimeError):
        BestExecutionAnalyzer().report()


def test_best_execution_report_all_unfilled_gives_nan_slippage():
    a = BestExecutionAnalyzer()
    a.add(OrderOutcome("1", "NYSE", Side.BUY, 100, 100.0, 0.0, 0, False))
    report = a.report()
    assert report.fill_rate == 0.0
    assert math.isnan(report.avg_slippage_bps)
    assert math.isnan(report.at_or_better_than_arrival_pct)


def test_best_execution_at_or_better_than_arrival():
    a = BestExecutionAnalyzer()
    # Buy filled AT the arrival mid: slip == 0, counts as at-or-better.
    a.add(OrderOutcome("1", "NYSE", Side.BUY, 100, 100.0, 100.0, 0, True))
    report = a.report()
    assert report.at_or_better_than_arrival_pct == pytest.approx(1.0)


# ----------------------------------------------------------------------
# FixAnalyzer
# ----------------------------------------------------------------------


def test_calculate_fix_is_median_odd_and_even():
    assert calculate_fix([1.0, 3.0, 2.0]) == pytest.approx(2.0)
    assert calculate_fix([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_calculate_fix_empty_raises():
    with pytest.raises(ValueError):
        calculate_fix([])


def test_fix_analyzer_flags_banging_the_close():
    # Heavy one-sided buying (net_flow > 0), fix runs up from the
    # pre-window mid, and price reverts down after the window closes:
    # all three conditions for "banging the close" hold.
    mid_samples = [100.0, 100.5, 101.0, 101.5, 102.0]
    report = analyze(
        mid_samples_in_window=mid_samples,
        pre_window_mid=99.0,
        post_window_mid=100.0,
        participant_buy_qty=9000,
        participant_sell_qty=0,
        market_volume=10000,
        share_threshold=0.5,
    )
    assert report.fix_rate == pytest.approx(101.0)
    assert report.run_up_bps > 0
    assert report.reversion_bps < 0
    assert report.net_flow == 9000
    assert report.participation_share == pytest.approx(0.9)
    assert report.flagged is True


def test_fix_analyzer_does_not_flag_below_share_threshold():
    mid_samples = [100.0, 100.5, 101.0]
    report = analyze(
        mid_samples_in_window=mid_samples,
        pre_window_mid=99.0,
        post_window_mid=100.0,
        participant_buy_qty=100,
        participant_sell_qty=0,
        market_volume=10000,
        share_threshold=0.5,
    )
    assert report.participation_share < 0.5
    assert report.flagged is False


def test_fix_analyzer_zero_net_flow_never_flags():
    mid_samples = [100.0, 101.0, 102.0]
    report = analyze(
        mid_samples_in_window=mid_samples,
        pre_window_mid=99.0,
        post_window_mid=100.0,
        participant_buy_qty=500,
        participant_sell_qty=500,
        market_volume=1000,
        share_threshold=0.5,
    )
    assert report.net_flow == 0
    assert report.flagged is False


def test_fix_analyzer_zero_market_volume_gives_zero_share():
    report = analyze(
        mid_samples_in_window=[100.0],
        pre_window_mid=99.0,
        post_window_mid=100.0,
        participant_buy_qty=0,
        participant_sell_qty=0,
        market_volume=0,
        share_threshold=0.1,
    )
    assert report.participation_share == 0.0


# ----------------------------------------------------------------------
# MarketQualityMetrics
# ----------------------------------------------------------------------


def test_quoted_spread_bps():
    assert mqm.quoted_spread_bps(99.9, 100.1) == pytest.approx((100.1 - 99.9) / 100.0 * 1e4)


def test_quoted_spread_bps_zero_mid_is_nan():
    assert math.isnan(mqm.quoted_spread_bps(0.0, 0.0))


def test_effective_realized_and_price_impact_relationship():
    mid_exec = 100.0
    price = 100.05
    mid_after = 100.10
    eff = mqm.effective_spread_bps(Side.BUY, price, mid_exec)
    realized = mqm.realized_spread_bps(Side.BUY, price, mid_after)
    impact = mqm.price_impact_bps(Side.BUY, mid_exec, mid_after)
    # effective ~= realized + impact (Java doc's approximate identity --
    # exact only in the small-move limit, since realized/impact divide
    # by mid_after while effective divides by mid_exec).
    assert eff == pytest.approx(realized + impact, rel=1e-3)


def test_effective_spread_sign_follows_side():
    buy = mqm.effective_spread_bps(Side.BUY, 100.05, 100.0)
    sell = mqm.effective_spread_bps(Side.SELL, 99.95, 100.0)
    assert buy > 0  # taker paid above mid: cost
    assert sell > 0  # sold below mid: also a cost, same sign convention


def test_order_to_trade_ratio_zero_trades_is_infinite():
    assert mqm.order_to_trade_ratio(100, 0) == math.inf
    assert mqm.order_to_trade_ratio(100, 50) == pytest.approx(2.0)
