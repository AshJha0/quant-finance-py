"""Pins for quantfinlib.screener.

Java sources: Fundamentals/StockSnapshot/ScreenFilter/FundamentalFilters/
TechnicalFilters/RankingEngine/StockScreener.java. Covers: NaN-never-matches
fundamental filters, the "too short = false" technical-filter contract,
three-sleeve composition (fundamental AND technical filters feeding
RankingEngine), min-max normalization edge cases (single valid value ->
0.5, negative weight inversion), and the point-in-time membership hook.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantfinlib.data.bar_series import BarSeries
from quantfinlib.data.point_in_time_universe import PointInTimeUniverse
from quantfinlib.screener import fundamental_filters as ff
from quantfinlib.screener import technical_filters as tf
from quantfinlib.screener.fundamentals import Fundamentals
from quantfinlib.screener.ranking_engine import RankingEngine
from quantfinlib.screener.screen_filter import ScreenFilter
from quantfinlib.screener.stock_screener import StockScreener
from quantfinlib.screener.stock_snapshot import StockSnapshot


def _snapshot(symbol: str, closes, fundamentals: Fundamentals) -> StockSnapshot:
    return StockSnapshot(symbol, BarSeries.of(symbol, closes), fundamentals)


# ----------------------------------------------------------------------
# Fundamentals / FundamentalFilters: NaN never matches
# ----------------------------------------------------------------------


def test_unknown_fundamentals_are_all_nan():
    u = Fundamentals.unknown()
    assert math.isnan(u.market_cap)
    assert math.isnan(u.pe_ratio)
    assert math.isnan(u.debt_to_equity)


def test_fundamental_filters_never_match_nan():
    unknown = _snapshot("X", [1.0, 2.0, 3.0], Fundamentals.unknown())
    assert ff.market_cap_above(0).matches(unknown) is False
    assert ff.pe_below(1e9).matches(unknown) is False
    assert ff.pe_between(-1e9, 1e9).matches(unknown) is False
    assert ff.pb_below(1e9).matches(unknown) is False
    assert ff.eps_above(-1e9).matches(unknown) is False
    assert ff.roe_above(-1e9).matches(unknown) is False
    assert ff.dividend_yield_above(-1e9).matches(unknown) is False
    assert ff.debt_to_equity_below(1e9).matches(unknown) is False


def test_fundamental_filters_basic_thresholds():
    known = _snapshot("X", [1.0], Fundamentals(1e11, 15, 3, 5, 0.2, 0.02, 0.5))
    assert ff.market_cap_above(1e9).matches(known)
    assert ff.pe_below(20).matches(known)
    assert ff.pe_between(10, 20).matches(known)
    assert not ff.pe_between(20, 30).matches(known)
    assert ff.roe_above(0.1).matches(known)
    assert not ff.roe_above(0.5).matches(known)


# ----------------------------------------------------------------------
# ScreenFilter combinators
# ----------------------------------------------------------------------


def test_screen_filter_and_or_negate():
    always_true = ScreenFilter(lambda s: True)
    always_false = ScreenFilter(lambda s: False)
    snap = _snapshot("X", [1.0], Fundamentals.unknown())

    assert always_true.and_(always_false).matches(snap) is False
    assert always_true.or_(always_false).matches(snap) is True
    assert always_true.negate().matches(snap) is False
    assert always_false.negate().matches(snap) is True


# ----------------------------------------------------------------------
# TechnicalFilters: "too short = false", never throw
# ----------------------------------------------------------------------


def test_technical_filter_too_short_is_false_not_throw():
    tiny = _snapshot("X", [100.0, 101.0, 99.0], Fundamentals.unknown())
    # SMA(200) on a 3-bar series: not enough data, must be False (not raise).
    assert tf.price_above_sma(200).matches(tiny) is False
    assert tf.breakout(252).matches(tiny) is False
    assert tf.volume_spike(252, 2.0).matches(tiny) is False


def test_technical_filter_monotone_up_series():
    closes = list(np.linspace(100.0, 200.0, 300))
    snap = _snapshot("UP", closes, Fundamentals.unknown())
    # A strictly increasing series: RSI saturates near 100, price stays
    # above its own SMA/EMA, and the last close is the 52-week high.
    assert tf.rsi_above(14, 90).matches(snap)
    assert not tf.rsi_below(14, 50).matches(snap)
    assert tf.price_above_sma(20).matches(snap)
    assert tf.price_above_ema(20).matches(snap)
    assert tf.near_52_week_high(0.001).matches(snap)
    assert not tf.near_52_week_low(0.5).matches(snap)


def test_gap_up_and_volume_spike():
    series = BarSeries.builder("G")
    for i in range(10):
        series.add(i, 100.0, 100.0, 100.0, 100.0, 100.0)
    # Gap up 5% on the last bar's open, plus a volume spike.
    series.add(10, 106.0, 108.0, 105.0, 107.0, 10_000.0)
    snap = StockSnapshot("G", series.build(), Fundamentals.unknown())
    assert tf.gap_up(0.03).matches(snap)
    assert not tf.gap_up(0.10).matches(snap)
    assert tf.volume_spike(9, 5.0).matches(snap)


# ----------------------------------------------------------------------
# RankingEngine: min-max normalization edge cases
# ----------------------------------------------------------------------


def test_ranking_engine_requires_criteria():
    engine = RankingEngine()
    with pytest.raises(RuntimeError):
        engine.rank([_snapshot("X", [1.0], Fundamentals.unknown())])


def test_ranking_engine_single_valid_value_scores_half():
    known = _snapshot("A", [1.0], Fundamentals(0, 0, 0, 0, 0.2, 0, 0))
    unknown = _snapshot("B", [1.0], Fundamentals.unknown())
    engine = RankingEngine().add_criterion("roe", 1.0, lambda s: s.fundamentals.roe)
    ranked = engine.rank([known, unknown])
    scores = {r.stock.symbol: r.score for r in ranked}
    # Only one valid value among the candidates: everyone is "average".
    assert scores["A"] == pytest.approx(0.5)
    assert scores["B"] == pytest.approx(0.5)


def test_ranking_engine_negative_weight_inverts_and_sorts_best_first():
    low_pe = _snapshot("LOW", [1.0], Fundamentals(0, 10, 0, 0, 0, 0, 0))
    high_pe = _snapshot("HIGH", [1.0], Fundamentals(0, 30, 0, 0, 0, 0, 0))
    mid_pe = _snapshot("MID", [1.0], Fundamentals(0, 20, 0, 0, 0, 0, 0))
    engine = RankingEngine().add_criterion("pe", -1.0, lambda s: s.fundamentals.pe_ratio)
    ranked = engine.rank([high_pe, mid_pe, low_pe])
    assert [r.stock.symbol for r in ranked] == ["LOW", "MID", "HIGH"]
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[-1].score == pytest.approx(0.0)


def test_ranking_engine_weighted_blend_normalizes_by_total_abs_weight():
    a = _snapshot("A", [1.0], Fundamentals(100, 0, 0, 0, 0, 0, 0))
    b = _snapshot("B", [1.0], Fundamentals(200, 0, 0, 0, 0, 0, 0))
    engine = RankingEngine().add_criterion("cap", 1.0, lambda s: s.fundamentals.market_cap)
    engine.add_criterion("cap2", 1.0, lambda s: s.fundamentals.market_cap)
    ranked = engine.rank([a, b])
    scores = {r.stock.symbol: r.score for r in ranked}
    # Same criterion twice at equal weight: identical to running it once.
    assert scores["A"] == pytest.approx(0.0)
    assert scores["B"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# StockScreener: three-sleeve composition (fundamental AND technical),
# screen_and_rank, and the point-in-time survivorship hook.
# ----------------------------------------------------------------------


def test_stock_screener_three_sleeve_composition():
    up_closes = list(np.linspace(50.0, 100.0, 300))
    cheap_quality = _snapshot("GOOD", up_closes, Fundamentals(5e9, 12, 2, 5, 0.25, 0.02, 0.3))
    expensive_quality = _snapshot("PRICEY", up_closes, Fundamentals(5e9, 45, 2, 5, 0.25, 0.02, 0.3))
    down_closes = list(np.linspace(100.0, 50.0, 300))
    cheap_but_falling = _snapshot("FALLING", down_closes, Fundamentals(5e9, 12, 2, 5, 0.25, 0.02, 0.3))

    screener = StockScreener([cheap_quality, expensive_quality, cheap_but_falling])
    survivors = screener.screen(
        ff.pe_below(20),
        ff.roe_above(0.1),
        tf.price_above_sma(20),
    )
    assert [s.symbol for s in survivors] == ["GOOD"]


def test_screen_and_rank_orders_survivors_best_first():
    up_closes = list(np.linspace(50.0, 100.0, 300))
    stock_a = _snapshot("A", up_closes, Fundamentals(5e9, 10, 2, 5, 0.30, 0.02, 0.3))
    stock_b = _snapshot("B", up_closes, Fundamentals(5e9, 18, 2, 5, 0.12, 0.02, 0.3))
    screener = StockScreener([stock_a, stock_b])
    ranking = RankingEngine().add_criterion("roe", 1.0, lambda s: s.fundamentals.roe)
    ranked = screener.screen_and_rank(ranking, ff.pe_below(20))
    assert [r.stock.symbol for r in ranked] == ["A", "B"]


def test_members_as_of_survivorship_hook():
    universe = PointInTimeUniverse()
    universe.add_membership("AAPL", 0)
    universe.add_membership("DEAD", 0, 500)  # left the index at t=500

    snaps = [
        _snapshot("AAPL", [1.0], Fundamentals.unknown()),
        _snapshot("DEAD", [1.0], Fundamentals.unknown()),
    ]
    early = StockScreener.members_as_of(snaps, universe, 100)
    late = StockScreener.members_as_of(snaps, universe, 1000)
    assert {s.symbol for s in early} == {"AAPL", "DEAD"}
    assert {s.symbol for s in late} == {"AAPL"}


def test_export_csv_writes_header_and_rows(tmp_path):
    a = _snapshot("A", [1.0, 2.0], Fundamentals(1e11, 15, 3, 5, 0.2, 0.01, 0.5))
    engine = RankingEngine().add_criterion("roe", 1.0, lambda s: s.fundamentals.roe)
    ranked = engine.rank([a])
    out = tmp_path / "screen.csv"
    StockScreener.export_csv(str(out), ranked)
    content = out.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    assert lines[0] == "symbol,score,lastClose,marketCap,peRatio,pbRatio,eps,roe,dividendYield,debtToEquity"
    assert lines[1].startswith("A,")
