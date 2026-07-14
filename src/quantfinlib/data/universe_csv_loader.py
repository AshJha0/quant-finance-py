"""Loads a :class:`~quantfinlib.data.point_in_time_universe.PointInTimeUniverse`
from a user-supplied CSV file (port of Java ``data.UniverseCsvLoader``) --
the defined interchange format for the membership/lifecycle data the
engine cannot invent.

File format
-----------
Header (required, column order fixed)::

    symbol,event,date,end_date,value,acquirer_shares,acquirer

One row per membership interval or lifecycle event::

    # comments and blank lines are ignored
    AAPL,MEMBER,2010-01-01,,,,                  <- member since date, still in
    YHOO,MEMBER,2000-01-03,2017-06-13,,,        <- left the index on end_date
    LEH,MEMBER,2000-01-03,2008-09-15,,,
    LEH,DELIST,2008-09-15,,-1.0,,               <- shareholders wiped out
    WCOM,DELIST,2002-07-01,,,,                  <- value empty: Shumway -30% default
    TWX,MERGER,2018-06-15,,48.53,0.5471,T       <- $48.53 + 0.5471 T shares per share

* ``event`` -- ``MEMBER``, ``DELIST`` or ``MERGER`` (case-insensitive)
* ``date`` / ``end_date`` -- ISO dates (``2018-06-15``) or epoch
  numbers, converted exactly like :mod:`~quantfinlib.data.csv_bar_loader`
  bar timestamps (epoch millis internally), so universe dates and bar
  timestamps line up by construction. Empty ``end_date`` = open-ended
  membership.
* ``value`` -- DELIST: the delisting return (final-day return on the
  last close, -1 = worthless; empty defaults to
  ``PointInTimeUniverse.DEFAULT_INVOLUNTARY_DELISTING_RETURN``);
  MERGER: cash per share (empty = 0).
* ``acquirer_shares`` / ``acquirer`` -- MERGER stock component (empty
  = all-cash deal).
"""

from __future__ import annotations

from typing import List, Sequence

from quantfinlib.data.csv_bar_loader import is_epoch_seconds_file
from quantfinlib.data.csv_bar_loader import parse_timestamp as _bar_parse_timestamp
from quantfinlib.data.point_in_time_universe import PointInTimeUniverse

_EXPECTED_HEADER = "symbol,event,date,end_date,value,acquirer_shares,acquirer"


def load(path: str) -> PointInTimeUniverse:
    """Loads a universe file (see the module doc for the format)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    return parse(lines)


def parse(lines: Sequence[str]) -> PointInTimeUniverse:
    """Parses in-memory lines -- same format, no file required (tests, HTTP)."""
    # Two passes: rows are validated and collected first so the same
    # whole-file epoch-seconds detection CsvBarLoader applies to bar
    # files can apply here -- otherwise a seconds-stamped universe file
    # would silently land in January 1970 next to millis-stamped bars.
    rows: List[List[str]] = []
    line_numbers: List[int] = []
    header_seen = False
    all_numeric = True
    min_ts = 2**63 - 1
    max_ts = -(2**63)

    for line_no, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not header_seen:
            if _normalize(line) != _EXPECTED_HEADER:
                raise ValueError(f"expected header '{_EXPECTED_HEADER}' but found: {line}")
            header_seen = True
            continue
        # -1 keeps trailing empty fields, so short rows are still 7 columns.
        f = line.split(",")
        if len(f) != 7:
            raise ValueError(f"line {line_no + 1}: expected 7 columns, found {len(f)}")
        for raw in (f[2].strip(), f[3].strip()):
            if not raw:
                continue
            if raw.isdigit():
                try:
                    v = int(raw)
                except ValueError as e:
                    # >19-digit corruption: line-tag it here too -- pass 1
                    # fires before pass 2's wrapping would.
                    raise ValueError(f"line {line_no + 1}: unparseable timestamp '{raw}'") from e
                min_ts = min(min_ts, v)
                max_ts = max(max_ts, v)
            else:
                all_numeric = False  # ISO dates present: values are millis
        rows.append(f)
        line_numbers.append(line_no + 1)

    if not header_seen:
        raise ValueError("empty universe file (no header)")

    # THE seconds-vs-millis heuristic, owned by CsvBarLoader so bar and
    # universe files can never diverge on timestamp scale.
    scale = 1000 if is_epoch_seconds_file(all_numeric, min_ts, max_ts) else 1

    universe = PointInTimeUniverse()
    for f, line_no in zip(rows, line_numbers):
        symbol = f[0].strip()
        event = f[1].strip().upper()
        if not symbol:
            raise ValueError(f"line {line_no}: empty symbol")
        try:
            if event == "MEMBER":
                from_ts = _timestamp(f[2].strip(), scale)
                if not f[3].strip():
                    universe.add_membership(symbol, from_ts)
                else:
                    universe.add_membership(symbol, from_ts, _timestamp(f[3].strip(), scale))
            elif event == "DELIST":
                ret = (
                    PointInTimeUniverse.DEFAULT_INVOLUNTARY_DELISTING_RETURN
                    if not f[4].strip()
                    else float(f[4].strip())
                )
                universe.record_delisting(symbol, _timestamp(f[2].strip(), scale), ret)
            elif event == "MERGER":
                cash = 0.0 if not f[4].strip() else float(f[4].strip())
                shares = 0.0 if not f[5].strip() else float(f[5].strip())
                acquirer = f[6].strip() or None
                universe.record_merger(symbol, _timestamp(f[2].strip(), scale), cash, shares, acquirer)
            else:
                raise ValueError(f"unknown event '{f[1]}'")
        except Exception as e:
            # Re-wrap with the line number: universe files are hand-curated,
            # and "which row is wrong" is the whole error message battle.
            raise ValueError(f"line {line_no} ({symbol}): {e}") from e
    return universe


def _timestamp(raw: str, scale: int) -> int:
    """ISO dates parse to millis directly; numeric values scale per the
    file heuristic."""
    if raw.isdigit():
        return int(raw) * scale
    return _bar_parse_timestamp(raw)


def _normalize(header: str) -> str:
    return header.lower().replace(" ", "")
