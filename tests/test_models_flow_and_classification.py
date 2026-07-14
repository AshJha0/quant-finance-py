"""Pins for TradeClassifier (Lee-Ready), FlowSignals and Vpin.

Java sources: QuantModels2Test.java (TradeClassifier), FlowSignalsTest.java,
QuantSignalsTest.java (VPIN section).
"""

import math

import pytest

from quantfinlib.microstructure.trade_classifier import (BUY, SELL, UNKNOWN,
                                                          TradeClassifier)
from quantfinlib.microstructure.flow_signals import FlowSignals
from quantfinlib.microstructure.vpin import Vpin

T0 = 1_000_000_000


# ------------------------------------------------------------------
# TradeClassifier
# ------------------------------------------------------------------

def test_quote_rule_classifies_at_and_through_the_touch():
    tc = TradeClassifier()
    tc.on_quote(100.00, 100.02)
    assert tc.classify(100.02) == BUY          # at the ask = lifted
    assert tc.classify(100.05) == BUY          # through the ask
    assert tc.classify(100.00) == SELL         # at the bid = hit
    assert tc.classify(99.95) == SELL          # through the bid
    assert tc.classify(100.015) == BUY         # above mid
    assert tc.classify(100.005) == SELL        # below mid


def test_tick_test_resolves_midpoint_trades():
    tc = TradeClassifier()
    tc.on_quote(100.00, 100.02)
    assert tc.classify(100.005) == SELL        # below mid
    assert tc.classify(100.01) == BUY          # uptick from 100.005
    assert tc.classify(100.01) == BUY          # zero-tick: repeats last side


def test_unknown_without_quotes_or_history_and_on_garbage():
    tc = TradeClassifier()
    assert tc.classify(100.0) == UNKNOWN
    assert tc.classify(math.nan) == UNKNOWN
    assert tc.classify(-5) == UNKNOWN
    assert tc.classify(100.5) == BUY           # uptick from 100.0


def test_is_buy_aggressor_falls_back_to_the_last_known_side():
    tc = TradeClassifier()
    tc.on_quote(1.08500, 1.08502)
    assert tc.is_buy_aggressor(1.08502)
    tc.on_quote(math.nan, math.nan)
    assert tc.is_buy_aggressor(1.08502)         # unknown -> last known (buy)


# ------------------------------------------------------------------
# FlowSignals
# ------------------------------------------------------------------

def test_ofi_follows_the_best_level_formulation():
    f = FlowSignals(9_223_372_036_854_775_807 // 2)   # effectively no decay
    f.on_quote(100, 500, 101, 400, T0)
    assert f.ofi() == pytest.approx(0, abs=1e-12)      # first quote seeds only

    f.on_quote(100, 700, 101, 400, T0 + 1)              # bid size +200
    assert f.ofi() == pytest.approx(200, abs=1e-9)

    f.on_quote(101, 300, 102, 400, T0 + 2)              # bid price improves + ask up
    assert f.ofi() == pytest.approx(200 + 300 + 400, abs=1e-9)

    f.on_quote(101, 300, 101, 250, T0 + 3)              # ask drops
    assert f.ofi() == pytest.approx(900 - 250, abs=1e-9)

    f.on_quote(100, 100, 101, 250, T0 + 4)              # bid drops
    assert f.ofi() == pytest.approx(650 - 300, abs=1e-9)


def test_ofi_decays_toward_zero_with_the_configured_half_life():
    half_life = 1_000_000
    f = FlowSignals(half_life)
    f.on_quote(100, 500, 101, 400, T0)
    f.on_quote(100, 900, 101, 400, T0 + 1)              # OFI = +400
    assert f.ofi() == pytest.approx(400, abs=1e-6)
    assert f.ofi(T0 + 1 + half_life) == pytest.approx(200, abs=1e-6)
    assert f.ofi(T0 + 1 + 2 * half_life) == pytest.approx(100, abs=1e-6)
    f.on_quote(100, 1300, 101, 400, T0 + 1 + half_life)  # +400 on top of 200
    assert f.ofi() == pytest.approx(600, abs=1e-6)


def test_queue_imbalance_reads_the_inside():
    f = FlowSignals()
    assert f.queue_imbalance() == 0
    f.on_quote(100, 300, 101, 100, T0)
    assert f.queue_imbalance() == pytest.approx(0.5, abs=1e-12)
    f.on_quote(100, 100, 101, 300, T0 + 1)
    assert f.queue_imbalance() == pytest.approx(-0.5, abs=1e-12)


def test_trade_imbalance_is_signed_over_total_volume():
    f = FlowSignals(9_223_372_036_854_775_807 // 2)
    assert f.trade_imbalance() == 0
    f.on_trade(True, 300, T0)
    assert f.trade_imbalance() == pytest.approx(1.0, abs=1e-12)
    f.on_trade(False, 100, T0 + 1)
    assert f.trade_imbalance() == pytest.approx(0.5, abs=1e-12)
    f.on_trade(False, 400, T0 + 2)
    assert f.trade_imbalance() == pytest.approx(-0.25, abs=1e-12)
    assert f.trade_count() == 3


def test_old_trades_fade_from_the_imbalance():
    half_life = 1_000_000
    f = FlowSignals(half_life)
    f.on_trade(True, 1000, T0)
    f.on_trade(False, 1000, T0 + 10 * half_life)
    assert f.trade_imbalance() < -0.99


def test_one_sided_quotes_are_a_signal_gap_not_maximal_pressure():
    f = FlowSignals(9_223_372_036_854_775_807 // 2)
    f.on_quote(100, 500, 101, 400, T0)
    f.on_quote(100, 700, 101, 400, T0 + 1)               # OFI = +200
    assert f.ofi() == pytest.approx(200, abs=1e-9)

    # The last offering venue drops (NBBO sentinel): queue imbalance
    # must read 0, and OFI must NOT book a fake buy sweep.
    f.on_quote(100, 700, 2 ** 31 - 1, 0, T0 + 2)
    assert f.queue_imbalance() == pytest.approx(0, abs=1e-12)
    assert f.ofi() == pytest.approx(200, abs=1e-9)

    f.on_quote(100, 700, 101, 300, T0 + 3)               # book re-forms
    assert f.ofi() == pytest.approx(200, abs=1e-9)
    f.on_quote(100, 900, 101, 300, T0 + 4)
    assert f.ofi() == pytest.approx(400, abs=1e-9)
    f.on_quote(-(2 ** 31), 0, 101, 300, T0 + 5)
    assert f.queue_imbalance() == pytest.approx(0, abs=1e-12)


def test_zero_or_infinite_price_sentinels_are_gaps_even_with_sizes():
    f = FlowSignals(9_223_372_036_854_775_807 // 2)
    f.on_quote(100.0, 500, 101.0, 400, T0)
    f.on_quote(100.0, 700, 101.0, 400, T0 + 1)           # OFI = +200
    f.on_quote(0.0, 600, 0.0, 600, T0 + 2)                # zero-price sentinel
    assert f.ofi() == pytest.approx(200, abs=1e-9)
    assert f.queue_imbalance() == pytest.approx(0, abs=1e-12)
    f.on_quote(100.0, 900, 101.0, 300, T0 + 3)
    assert f.ofi() == pytest.approx(200, abs=1e-9)
    f.on_quote(100.0, 900, math.inf, 300, T0 + 4)
    assert f.queue_imbalance() == pytest.approx(0, abs=1e-12)
    assert f.ofi() == pytest.approx(200, abs=1e-9)


def test_flow_signals_validation():
    with pytest.raises(ValueError):
        FlowSignals(0)


# ------------------------------------------------------------------
# Vpin
# ------------------------------------------------------------------

def test_vpin_reads_one_sided_flow_as_toxic_and_balanced_flow_as_calm():
    toxic = Vpin(1_000, 5)
    assert math.isnan(toxic.vpin())
    for _ in range(5):
        toxic.on_trade(1_000, True)             # relentless one-way buying
    assert toxic.ready()
    assert toxic.vpin() == pytest.approx(1.0, abs=1e-12)

    calm = Vpin(1_000, 5)
    for i in range(10):
        calm.on_trade(500, i % 2 == 0)          # perfectly two-way
    assert calm.vpin() == pytest.approx(0.0, abs=1e-12)

    # A block trade SPLITS across buckets -- volume time, not clock time.
    split = Vpin(1_000, 5)
    split.on_trade(2_500, True)                 # 2 full buy buckets + 500
    split.on_trade(500, False)                   # completes bucket 3 balanced
    assert split.buckets_completed() == 3
    assert split.vpin() == pytest.approx((1.0 + 1.0 + 0.0) / 3, abs=1e-12)

    # Regime shift: toxicity RISES as informed flow displaces noise.
    regime = Vpin(1_000, 4)
    for i in range(8):
        regime.on_trade(500, i % 2 == 0)
    before = regime.vpin()
    for _ in range(4):
        regime.on_trade(1_000, True)
    assert regime.vpin() > before + 0.9

    with pytest.raises(ValueError):
        Vpin(0, 5)
    with pytest.raises(ValueError):
        calm.on_trade(0, True)


def test_vpin_evicts_exactly_the_oldest_bucket_and_survives_block_trades():
    vpin = Vpin(100, 3)
    vpin.on_trade(100, True)                     # bucket 1: imbalance 1
    for _ in range(2):
        vpin.on_trade(50, True)                  # buckets 2, 3: balanced
        vpin.on_trade(50, False)
    assert vpin.vpin() == pytest.approx(1.0 / 3, abs=1e-12)
    vpin.on_trade(50, True)
    vpin.on_trade(50, False)
    assert vpin.vpin() == pytest.approx(0.0, abs=1e-12)

    # A block trade of ANY size is O(window): returns instantly and
    # reads maximum toxicity.
    block = Vpin(1_000, 4)
    block.on_trade(9_223_372_036_854_775_806, True)
    assert block.ready()
    assert block.vpin() == pytest.approx(1.0, abs=1e-12)
