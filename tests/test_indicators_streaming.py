"""Pins for quantfinlib.indicators.streaming_indicators.

Java source: StreamingIndicatorsTest — the streaming indicators must
match the batch engine value-for-value (including NaN warm-up
positions) so live behavior equals backtested behavior. The GBM closes
come from the same generator as the batch indicator tests.
"""

import math

import numpy as np
import pytest

from quantfinlib.indicators import Indicators, StreamingIndicators

from test_indicators_batch import gbm_bars

_, _, _, CLOSES, _ = gbm_bars(500, 100, 0.08, 0.25, 99)


def assert_same(expected, actual, index, what):
    if math.isnan(expected):
        assert math.isnan(actual), \
            f"{what} expected NaN at {index} but was {actual}"
    else:
        assert actual == pytest.approx(expected, abs=1e-9), \
            f"{what} mismatch at {index}"


def test_sma_matches_batch():
    batch = Indicators.sma(CLOSES, 20)
    s = StreamingIndicators.Sma(20)
    for i, v in enumerate(CLOSES):
        assert_same(batch[i], s.update(v), i, "SMA")
    assert s.value() == pytest.approx(batch[-1], abs=1e-9)


def test_ema_matches_batch():
    batch = Indicators.ema(CLOSES, 20)
    e = StreamingIndicators.Ema(20)
    for i, v in enumerate(CLOSES):
        assert_same(batch[i], e.update(v), i, "EMA")


def test_rsi_matches_batch():
    batch = Indicators.rsi(CLOSES, 14)
    r = StreamingIndicators.Rsi(14)
    for i, v in enumerate(CLOSES):
        assert_same(batch[i], r.update(v), i, "RSI")


def test_macd_matches_batch_including_signal_and_histogram():
    batch = Indicators.macd(CLOSES, 12, 26, 9)
    m = StreamingIndicators.Macd(12, 26, 9)
    for i, v in enumerate(CLOSES):
        line = m.update(v)
        assert_same(batch.line[i], line, i, "MACD line")
        assert_same(batch.signal[i], m.signal(), i, "MACD signal")
        assert_same(batch.histogram[i], m.histogram(), i, "MACD histogram")


def test_vwap_accumulates():
    v = StreamingIndicators.Vwap()
    v.update(10, 100)
    res = v.update(20, 100)
    # (10*100 + 20*100) / 200 = 15.
    assert res == pytest.approx(15.0, abs=1e-12)
    assert v.value() == pytest.approx(15.0, abs=1e-12)
