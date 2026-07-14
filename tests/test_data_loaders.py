"""Pins for the extended quantfinlib.data loaders.

Java sources: CsvBarLoader/SeriesAligner/CorporateActions/
UniverseCsvLoader/PointInTimeUniverse.java.

Covers the classic CsvBarLoader traps from Java project history: a
semicolon-delimited (European decimal) file, and the DEAD-CODE
regression where a comma file has a quoted extra column containing a
literal ';' that must NOT flip the whole file to European decimal
parsing. Also covers the epoch-seconds-vs-millis heuristic shared with
UniverseCsvLoader, SeriesAligner's intersect/union-forward-fill pins,
CorporateActions ordering/back-adjustment pins, and PointInTimeUniverse
delisting/merger/one-terminal-event pins.
"""

from __future__ import annotations

import math

import pytest

from quantfinlib.data import csv_bar_loader as cbl
from quantfinlib.data import series_aligner as sa
from quantfinlib.data import universe_csv_loader as ucl
from quantfinlib.data.bar_series import BarSeries
from quantfinlib.data.corporate_actions import ActionType, CorporateAction, adjust
from quantfinlib.data.point_in_time_universe import PointInTimeUniverse

# ----------------------------------------------------------------------
# CsvBarLoader
# ----------------------------------------------------------------------


def test_csv_bar_loader_us_comma_thousands_separator():
    lines = [
        "timestamp,open,high,low,close,volume",
        '2020-01-01,"1,234.50",1235.00,1230.00,1234.00,1000',
    ]
    series = cbl.parse(lines, "US")
    assert series.open(0) == pytest.approx(1234.50)
    assert series.close(0) == pytest.approx(1234.00)


def test_csv_bar_loader_european_semicolon_decimal_comma():
    # Semicolon-delimited: the comma is now the DECIMAL separator, not a
    # thousands separator. Stripping commas here (the US-file logic)
    # would silently read "100,25" as 10025.
    lines = [
        "timestamp;open;high;low;close;volume",
        "2020-01-01;100,25;101,50;99,75;100,50;1000",
    ]
    series = cbl.parse(lines, "EU")
    assert series.open(0) == pytest.approx(100.25)
    assert series.high(0) == pytest.approx(101.50)
    assert series.close(0) == pytest.approx(100.50)


def test_csv_bar_loader_quoted_semicolon_in_header_does_not_flip_to_european():
    # DEAD-CODE regression: a comma-delimited file with a quoted extra
    # column name containing a literal ';' must still be parsed as a US
    # (comma-decimal) file, not flipped to European decimals.
    lines = [
        'timestamp,open,high,low,close,volume,"note;extra"',
        "2020-01-01,100.25,101.50,99.75,100.50,1000,foo",
    ]
    series = cbl.parse(lines, "TRAP")
    assert series.open(0) == pytest.approx(100.25)
    assert series.close(0) == pytest.approx(100.50)


def test_contains_unquoted_ignores_semicolons_inside_quotes():
    assert cbl.contains_unquoted('a,"b;c",d', ";") is False
    assert cbl.contains_unquoted("a;b,c", ";") is True


def test_parse_number_us_vs_european():
    assert cbl.parse_number("1,234.5", european_decimals=False) == pytest.approx(1234.5)
    assert cbl.parse_number("1.234,56", european_decimals=True) == pytest.approx(1234.56)


def test_csv_bar_loader_epoch_seconds_heuristic_scales_to_millis():
    lines = [
        "timestamp,open,high,low,close,volume",
        "1577836800,100,101,99,100.5,1000",
        "1577923200,100.5,102,100,101.5,1100",
    ]
    series = cbl.parse(lines, "SECS")
    assert series.timestamp(0) == 1577836800000
    assert series.timestamp(1) == 1577923200000


def test_csv_bar_loader_epoch_millis_not_rescaled():
    lines = [
        "timestamp,open,high,low,close,volume",
        "1577836800000,100,101,99,100.5,1000",
    ]
    series = cbl.parse(lines, "MILLIS")
    assert series.timestamp(0) == 1577836800000


def test_csv_bar_loader_iso_date_only_is_midnight_utc():
    lines = ["date,open,high,low,close", "2020-01-01,1,1,1,1"]
    series = cbl.parse(lines, "D")
    assert series.timestamp(0) == 1577836800000  # 2020-01-01T00:00:00Z


def test_csv_bar_loader_iso_datetime_with_offset():
    lines = ["timestamp,open,high,low,close", "2020-01-01T02:00:00+02:00,1,1,1,1"]
    series = cbl.parse(lines, "OFF")
    assert series.timestamp(0) == 1577836800000  # same instant as UTC midnight


def test_csv_bar_loader_rows_sorted_by_time_even_if_unordered_in_file():
    lines = [
        "timestamp,open,high,low,close",
        "3,3,3,3,3",
        "1,1,1,1,1",
        "2,2,2,2,2",
    ]
    series = cbl.parse(lines, "UNORDERED")
    assert list(series.timestamps()) == [1, 2, 3]


def test_csv_bar_loader_missing_required_column_raises():
    with pytest.raises(ValueError):
        cbl.parse(["timestamp,open,high,low", "1,1,1,1"], "BAD")


def test_csv_bar_loader_empty_raises():
    with pytest.raises(ValueError):
        cbl.parse([], "EMPTY")


def test_csv_bar_loader_header_only_raises():
    with pytest.raises(ValueError):
        cbl.parse(["timestamp,open,high,low,close"], "NOROWS")


def test_csv_bar_loader_save_and_load_round_trip(tmp_path):
    b = BarSeries.builder("RT")
    b.add(1000, 10.0, 11.0, 9.0, 10.5, 100.0)
    b.add(2000, 10.5, 12.0, 10.0, 11.5, 200.0)
    series = b.build()
    path = tmp_path / "rt.csv"
    cbl.save(series, str(path))
    reloaded = cbl.load(str(path), "RT")
    assert list(reloaded.timestamps()) == list(series.timestamps())
    assert list(reloaded.closes()) == pytest.approx(list(series.closes()))


# ----------------------------------------------------------------------
# SeriesAligner
# ----------------------------------------------------------------------


def _series(symbol, timestamps, base):
    b = BarSeries.builder(symbol)
    for i, ts in enumerate(timestamps):
        px = base + i
        b.add(ts, px, px, px, px, 100.0)
    return b.build()


def test_series_aligner_intersect_keeps_only_common_timestamps():
    a = _series("A", [1, 2, 3, 4], 10.0)
    b = _series("B", [2, 3, 4], 20.0)
    out = sa.intersect({"A": a, "B": b})
    assert list(out["A"].timestamps()) == [2, 3, 4]
    assert list(out["A"].closes()) == pytest.approx([11.0, 12.0, 13.0])
    assert list(out["B"].closes()) == pytest.approx([20.0, 21.0, 22.0])


def test_series_aligner_intersect_no_common_timestamps_raises():
    a = _series("A", [1, 2], 10.0)
    b = _series("B", [3, 4], 20.0)
    with pytest.raises(ValueError):
        sa.intersect({"A": a, "B": b})


def test_series_aligner_intersect_empty_input_raises():
    with pytest.raises(ValueError):
        sa.intersect({})


def test_series_aligner_union_forward_fill_carries_previous_close():
    a = _series("A", [1, 2, 3, 4], 10.0)  # closes 10,11,12,13
    b = _series("B", [2, 4], 20.0)  # closes 20,21 (missing ts=1,3)
    out = sa.union_forward_fill({"A": a, "B": b})
    # Timeline starts at max(first ts) = 2 (B's first bar).
    assert list(out["A"].timestamps()) == [2, 3, 4]
    # B is missing ts=3: forward-filled at its previous close (20.0),
    # zero volume.
    assert list(out["B"].closes()) == pytest.approx([20.0, 20.0, 21.0])
    assert out["B"].volume(1) == 0.0


def test_series_aligner_union_forward_fill_empty_input_raises():
    with pytest.raises(ValueError):
        sa.union_forward_fill({})


# ----------------------------------------------------------------------
# CorporateActions
# ----------------------------------------------------------------------


def test_corporate_actions_split_halves_price_doubles_volume_before_ex_date():
    b = BarSeries.builder("S")
    b.add(1, 100.0, 100.0, 100.0, 100.0, 1000.0)
    b.add(2, 100.0, 100.0, 100.0, 100.0, 1000.0)
    b.add(3, 50.0, 50.0, 50.0, 50.0, 1000.0)  # ex-date: already split-adjusted by vendor
    series = b.build()
    adjusted = adjust(series, [CorporateAction(3, ActionType.SPLIT, 2.0)])
    assert list(adjusted.closes()) == pytest.approx([50.0, 50.0, 50.0])
    assert list(adjusted.volumes()) == pytest.approx([2000.0, 2000.0, 1000.0])


def test_corporate_actions_cash_dividend_back_adjustment():
    b = BarSeries.builder("D")
    b.add(1, 100.0, 100.0, 100.0, 100.0, 1000.0)
    b.add(2, 98.0, 98.0, 98.0, 98.0, 1000.0)  # prev close 100, $2 dividend -> ex drop
    series = b.build()
    adjusted = adjust(series, [CorporateAction(2, ActionType.CASH_DIVIDEND, 2.0)])
    # factor = (100 - 2) / 100 = 0.98, applied only to bar 0 (before ex-date).
    assert adjusted.close(0) == pytest.approx(98.0)
    assert adjusted.close(1) == pytest.approx(98.0)  # unchanged: on/after ex-date


def test_corporate_actions_multiple_actions_compound_in_time_order():
    # A 2019 split, then a 2020 dividend: applying them out of input order
    # must still compound correctly because CorporateActions sorts by
    # ex_timestamp internally (Java: Comparator.comparingLong).
    b = BarSeries.builder("M")
    b.add(1, 200.0, 200.0, 200.0, 200.0, 1000.0)  # before both actions
    b.add(2, 100.0, 100.0, 100.0, 100.0, 1000.0)  # after split, before dividend
    b.add(3, 98.0, 98.0, 98.0, 98.0, 1000.0)  # after both
    series = b.build()
    dividend_first = [
        CorporateAction(3, ActionType.CASH_DIVIDEND, 2.0),
        CorporateAction(2, ActionType.SPLIT, 2.0),
    ]
    adjusted = adjust(series, dividend_first)
    # Bar 0: split factor (1/2) then dividend factor (0.98) both apply.
    assert adjusted.close(0) == pytest.approx(200.0 * 0.5 * 0.98)
    # Bar 1: only the dividend factor applies (already past the split ex-date).
    assert adjusted.close(1) == pytest.approx(100.0 * 0.98)
    assert adjusted.close(2) == pytest.approx(98.0)


def test_corporate_actions_dividend_exceeding_prior_close_raises():
    b = BarSeries.builder("BAD")
    b.add(1, 1.0, 1.0, 1.0, 1.0, 1.0)
    b.add(2, 1.0, 1.0, 1.0, 1.0, 1.0)
    series = b.build()
    with pytest.raises(ValueError):
        adjust(series, [CorporateAction(2, ActionType.CASH_DIVIDEND, 5.0)])


def test_corporate_action_nonpositive_value_rejected():
    with pytest.raises(ValueError):
        CorporateAction(1, ActionType.SPLIT, 0.0)
    with pytest.raises(ValueError):
        CorporateAction(1, ActionType.SPLIT, -1.0)


def test_corporate_actions_no_bars_before_ex_date_is_a_no_op():
    b = BarSeries.builder("NOOP")
    b.add(5, 100.0, 100.0, 100.0, 100.0, 100.0)
    series = b.build()
    adjusted = adjust(series, [CorporateAction(1, ActionType.SPLIT, 2.0)])
    assert adjusted.close(0) == pytest.approx(100.0)


# ----------------------------------------------------------------------
# PointInTimeUniverse
# ----------------------------------------------------------------------


def test_point_in_time_universe_open_ended_membership():
    u = PointInTimeUniverse()
    u.add_membership("AAPL", 100)
    assert u.is_member("AAPL", 100) is True
    assert u.is_member("AAPL", 10_000_000) is True
    assert u.is_member("AAPL", 99) is False


def test_point_in_time_universe_delisting_default_shumway_return():
    u = PointInTimeUniverse()
    u.add_membership("WCOM", 0)
    u.record_delisting("WCOM", 500, PointInTimeUniverse.DEFAULT_INVOLUNTARY_DELISTING_RETURN)
    assert u.is_member("WCOM", 400) is True
    assert u.is_member("WCOM", 500) is False  # dead at and after the event
    assert u.terminal_event("WCOM").delisting_return == pytest.approx(-0.30)


def test_point_in_time_universe_merger_requires_acquirer_when_stock_component():
    u = PointInTimeUniverse()
    with pytest.raises(ValueError):
        u.record_merger("TWX", 100, 48.53, 0.5471, None)


def test_point_in_time_universe_all_cash_merger_allows_no_acquirer():
    u = PointInTimeUniverse()
    u.record_merger("CASH_DEAL", 100, 50.0, 0.0, None)
    event = u.terminal_event("CASH_DEAL")
    assert event.cash_per_share == pytest.approx(50.0)
    assert event.acquirer is None


def test_point_in_time_universe_one_terminal_event_rule():
    u = PointInTimeUniverse()
    u.record_delisting("X", 100, -0.5)
    with pytest.raises(ValueError):
        u.record_merger("X", 200, 10.0, 0.0, None)


def test_point_in_time_universe_members_as_of_sorted():
    u = PointInTimeUniverse()
    u.add_membership("ZZZ", 0)
    u.add_membership("AAA", 0)
    u.add_membership("MMM", 0)
    assert u.members_as_of(50) == ["AAA", "MMM", "ZZZ"]


def test_point_in_time_universe_delisting_return_below_negative_one_rejected():
    u = PointInTimeUniverse()
    with pytest.raises(ValueError):
        u.record_delisting("X", 100, -1.5)


def test_point_in_time_universe_membership_end_before_start_rejected():
    u = PointInTimeUniverse()
    with pytest.raises(ValueError):
        u.add_membership("X", 200, 100)


# ----------------------------------------------------------------------
# UniverseCsvLoader
# ----------------------------------------------------------------------


def test_universe_csv_loader_full_lifecycle_pin():
    lines = [
        "symbol,event,date,end_date,value,acquirer_shares,acquirer",
        "AAPL,MEMBER,2010-01-01,,,,",
        "LEH,MEMBER,2000-01-03,2008-09-15,,,",
        "LEH,DELIST,2008-09-15,,-1.0,,",
        "WCOM,MEMBER,2000-01-01,,,,",
        "WCOM,DELIST,2002-07-01,,,,",
        "TWX,MEMBER,2000-01-01,,,,",
        "TWX,MERGER,2018-06-15,,48.53,0.5471,T",
    ]
    u = ucl.parse(lines)
    d2015 = cbl.parse_timestamp("2015-01-01")
    assert u.is_member("AAPL", d2015) is True
    assert u.is_member("LEH", d2015) is False  # dead since 2008
    leh_event = u.terminal_event("LEH")
    assert leh_event.delisting_return == pytest.approx(-1.0)
    wcom_event = u.terminal_event("WCOM")
    assert wcom_event.delisting_return == pytest.approx(
        PointInTimeUniverse.DEFAULT_INVOLUNTARY_DELISTING_RETURN
    )
    twx_event = u.terminal_event("TWX")
    assert twx_event.cash_per_share == pytest.approx(48.53)
    assert twx_event.acquirer_shares_per_share == pytest.approx(0.5471)
    assert twx_event.acquirer == "T"
    assert u.members_as_of(d2015) == ["AAPL", "TWX"]


def test_universe_csv_loader_epoch_seconds_scaled_like_bar_files():
    lines = [
        "symbol,event,date,end_date,value,acquirer_shares,acquirer",
        "AAPL,MEMBER,1577836800,,,,",  # 2020-01-01 in epoch SECONDS
    ]
    u = ucl.parse(lines)
    expected_millis = cbl.parse_timestamp("2020-01-01")
    assert u.is_member("AAPL", expected_millis) is True
    assert u.is_member("AAPL", expected_millis - 1) is False


def test_universe_csv_loader_comments_and_blank_lines_ignored():
    lines = [
        "# a comment",
        "",
        "symbol,event,date,end_date,value,acquirer_shares,acquirer",
        "",
        "# another comment",
        "AAPL,MEMBER,2010-01-01,,,,",
    ]
    u = ucl.parse(lines)
    assert u.all_symbols() == ["AAPL"]


def test_universe_csv_loader_wrong_header_raises():
    with pytest.raises(ValueError):
        ucl.parse(["wrong,header", "AAPL,MEMBER,2010-01-01,,,,"])


def test_universe_csv_loader_wrong_column_count_raises_with_line_number():
    lines = [
        "symbol,event,date,end_date,value,acquirer_shares,acquirer",
        "AAPL,MEMBER,2010-01-01,,,",  # only 6 columns
    ]
    with pytest.raises(ValueError, match="line 2"):
        ucl.parse(lines)


def test_universe_csv_loader_unknown_event_raises_with_line_number():
    lines = [
        "symbol,event,date,end_date,value,acquirer_shares,acquirer",
        "AAPL,BOGUS,2010-01-01,,,,",
    ]
    with pytest.raises(ValueError, match="line 2"):
        ucl.parse(lines)


def test_universe_csv_loader_empty_symbol_raises():
    lines = [
        "symbol,event,date,end_date,value,acquirer_shares,acquirer",
        ",MEMBER,2010-01-01,,,,",
    ]
    with pytest.raises(ValueError):
        ucl.parse(lines)


def test_universe_csv_loader_empty_file_raises():
    with pytest.raises(ValueError):
        ucl.parse([])
