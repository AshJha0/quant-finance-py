"""Pins for quantfinlib.data.Bar (port of the Java record com.quantfinlib.core.Bar).

The Java side has no dedicated BarTest — the record is exercised through
MarketDataTest fixtures like ``new Bar(1, 100, 102, 99, 101, 5_000)`` —
so these pins nail the record's own contract per house style: the
high >= low construction gate, the three derived quantities, and record
semantics (immutability, value equality).
"""

import dataclasses

import pytest

from quantfinlib.data import Bar


def test_fields_round_trip():
    # The MarketDataTest fixture bar.
    b = Bar(1, 100.0, 102.0, 99.0, 101.0, 5_000.0)
    assert b.timestamp == 1
    assert b.open == 100.0
    assert b.high == 102.0
    assert b.low == 99.0
    assert b.close == 101.0
    assert b.volume == 5_000.0


def test_typical_price_pin():
    # (high + low + close) / 3 = (102 + 99 + 101) / 3 = 302/3 = 100.666...
    b = Bar(1, 100.0, 102.0, 99.0, 101.0, 5_000.0)
    assert b.typical_price() == pytest.approx(302.0 / 3.0, abs=1e-12)


def test_range_pin():
    # 102 - 99 = 3.
    b = Bar(1, 100.0, 102.0, 99.0, 101.0, 5_000.0)
    assert b.range() == pytest.approx(3.0, abs=1e-15)


def test_is_bullish_is_strict():
    assert Bar(1, 100.0, 102.0, 99.0, 101.0, 0.0).is_bullish()       # close > open
    assert not Bar(1, 100.0, 102.0, 99.0, 99.5, 0.0).is_bullish()    # close < open
    assert not Bar(1, 100.0, 102.0, 99.0, 100.0, 0.0).is_bullish()   # doji: strict >


def test_high_below_low_fails_loudly():
    with pytest.raises(ValueError):
        Bar(1, 100.0, 99.0, 102.0, 101.0, 0.0)


def test_high_equal_low_is_allowed():
    # A locked-limit bar trades at one price all interval: range 0 is legal.
    b = Bar(1, 5.0, 5.0, 5.0, 5.0, 0.0)
    assert b.range() == 0.0


def test_bar_is_frozen():
    b = Bar(1, 100.0, 102.0, 99.0, 101.0, 5_000.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.close = 200.0


def test_value_equality_like_java_record():
    assert Bar(1, 100.0, 102.0, 99.0, 101.0, 5_000.0) == \
        Bar(1, 100.0, 102.0, 99.0, 101.0, 5_000.0)
    assert Bar(1, 100.0, 102.0, 99.0, 101.0, 5_000.0) != \
        Bar(2, 100.0, 102.0, 99.0, 101.0, 5_000.0)


# ----------------------------------------------------------------------
# Review M7: BarSeries timestamps must be strictly increasing.
# ----------------------------------------------------------------------

def test_bar_series_rejects_out_of_order_timestamps():
    from quantfinlib.data.bar_series import BarSeries
    b = BarSeries.builder("X")
    for ts in (3, 1, 2):
        b.add(ts, 1.0, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        b.build()


def test_bar_series_rejects_duplicate_timestamps():
    from quantfinlib.data.bar_series import BarSeries
    b = BarSeries.builder("X")
    for ts in (1, 2, 2):
        b.add(ts, 1.0, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        b.build()
    with pytest.raises(ValueError):
        BarSeries("X", [5, 5], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1])


def test_bar_series_accepts_sorted_timestamps_and_single_bar():
    from quantfinlib.data.bar_series import BarSeries
    s = BarSeries("X", [1, 2, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1])
    assert s.size() == 3
    assert BarSeries("X", [7], [1], [1], [1], [1], [1]).size() == 1
