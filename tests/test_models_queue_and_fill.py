"""Pins for QueuePositionEstimator, HiddenLiquidityDetector and
FillProbabilityModel/QueueModel.

Java sources: QuantModelsTest.java (QueuePositionEstimator,
HiddenLiquidityDetector), QuantModels2Test.java (FillProbabilityModel).
"""

import math

import pytest

from quantfinlib.microstructure.fill_probability_model import (
    FillProbabilityModel)
from quantfinlib.microstructure.hidden_liquidity_detector import (
    HiddenLiquidityDetector)
from quantfinlib.microstructure.queue_model import QueueModel
from quantfinlib.microstructure.queue_position_estimator import (
    QueuePositionEstimator)


# ------------------------------------------------------------------
# QueuePositionEstimator
# ------------------------------------------------------------------

def test_queue_joins_at_the_back_and_trades_reduce_ahead():
    q = QueuePositionEstimator()
    q.join(5_000, 1_000)
    assert q.shares_ahead() == pytest.approx(5_000, abs=1e-9)
    q.on_trade(2_000)
    assert q.shares_ahead() == pytest.approx(3_000, abs=1e-9)
    q.on_trade(10_000)
    assert q.shares_ahead() == pytest.approx(0, abs=1e-9)


def test_queue_cancels_are_attributed_pro_rata():
    q = QueuePositionEstimator()
    q.join(8_000, 2_000)
    q.on_level_resize(6_000)
    assert q.shares_ahead() == pytest.approx(4_000, abs=1e-6)


def test_queue_adds_behind_do_not_change_ahead():
    q = QueuePositionEstimator()
    q.join(3_000, 1_000)
    q.on_level_resize(9_000)
    assert q.shares_ahead() == pytest.approx(3_000, abs=1e-9)


def test_queue_fill_probability_rises_as_we_advance():
    q = QueuePositionEstimator()
    q.join(50_000, 1_000)
    back = q.fill_probability(10_000)
    q.on_trade(49_000)
    front = q.fill_probability(10_000)
    assert front > back


def test_queue_progress_is_zero_at_join_and_one_at_the_front():
    q = QueuePositionEstimator()
    q.join(4_000, 1_000)
    assert q.queue_progress() == pytest.approx(0.0, abs=1e-9)
    q.on_trade(2_000)
    assert q.queue_progress() == pytest.approx(0.5, abs=1e-9)
    q.on_trade(2_000)
    assert q.queue_progress() == pytest.approx(1.0, abs=1e-9)


def test_queue_position_estimator_validation():
    with pytest.raises(ValueError):
        QueuePositionEstimator().join(-1, 100)


# ------------------------------------------------------------------
# HiddenLiquidityDetector
# ------------------------------------------------------------------

def test_detects_an_iceberg_when_a_single_print_exceeds_the_display():
    d = HiddenLiquidityDetector(4, 0.5)
    d.on_displayed(2, 1_000)
    d.on_execution(2, 3_000)
    assert d.is_iceberg(2)
    assert d.hidden_multiplier(2) == pytest.approx(3.0, abs=1e-9)
    assert d.estimated_true_depth(2) == pytest.approx(3_000, abs=1e-9)


def test_ordinary_fragmented_flow_is_not_false_flagged_as_an_iceberg():
    d = HiddenLiquidityDetector(4, 0.5)
    for _ in range(10):
        d.on_displayed(1, 5_000)
        d.on_execution(1, 3_000)          # each print < display
    assert not d.is_iceberg(1)
    assert d.hidden_multiplier(1) == pytest.approx(1.0, abs=1e-9)
    assert d.estimated_true_depth(1) == pytest.approx(5_000, abs=1e-9)


def test_learned_iceberg_memory_persists_across_a_clear():
    d = HiddenLiquidityDetector(2, 0.5)
    d.on_displayed(0, 1_000)
    d.on_execution(0, 1_500)
    assert d.is_iceberg(0)
    obs = d.refill_observations(0)
    d.on_level_cleared(0)
    d.on_displayed(0, 2_000)
    d.on_execution(0, 500)                # 500 < 2,000: no new event
    assert d.refill_observations(0) == obs
    assert d.hidden_multiplier(0) > 1.0


def test_hidden_liquidity_detector_validation():
    with pytest.raises(ValueError):
        HiddenLiquidityDetector(0)


# ------------------------------------------------------------------
# FillProbabilityModel / QueueModel
# ------------------------------------------------------------------

def test_touch_probability_behaves_like_a_barrier_hit():
    vol = 1e-4
    price = 100
    assert FillProbabilityModel.touch_probability(0, vol, 60, price) == \
        pytest.approx(1.0, abs=1e-12)
    assert FillProbabilityModel.touch_probability(-1, vol, 60, price) == \
        pytest.approx(1.0, abs=1e-12)
    near = FillProbabilityModel.touch_probability(0.01, vol, 60, price)
    far = FillProbabilityModel.touch_probability(0.05, vol, 60, price)
    far_longer = FillProbabilityModel.touch_probability(0.05, vol, 600, price)
    assert near > far
    assert far_longer > far
    assert 0 < near < 1
    one_sigma = price * vol * math.sqrt(60)
    assert FillProbabilityModel.touch_probability(one_sigma, vol, 60, price) == \
        pytest.approx(0.3173, abs=1e-3)
    assert FillProbabilityModel.touch_probability(0.01, 0, 60, price) == \
        pytest.approx(0, abs=1e-12)
    assert FillProbabilityModel.touch_probability(0.01, vol, 0, price) == \
        pytest.approx(0, abs=1e-12)
    assert FillProbabilityModel.touch_probability(0.01, math.nan, 60, price) == \
        pytest.approx(0, abs=1e-12)


def test_passive_fill_composes_touch_and_queue():
    vol = 1e-4
    price = 100
    best = FillProbabilityModel.passive_fill_probability(
        0, vol, 60, price, 0, 100, 1_000_000)
    assert best > 0.99
    queued = FillProbabilityModel.passive_fill_probability(
        0, vol, 60, price, 50_000, 100, 10_000)
    assert queued < 0.01
    far = FillProbabilityModel.passive_fill_probability(
        0.50, vol, 60, price, 0, 100, 1_000_000)
    assert far < 0.01
    both = FillProbabilityModel.passive_fill_probability(
        0.01, vol, 60, price, 5_000, 100, 10_000)
    assert both <= FillProbabilityModel.touch_probability(0.01, vol, 60, price)
    assert both <= QueueModel.fill_probability(5_000, 100, 10_000)
