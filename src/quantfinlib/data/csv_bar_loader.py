"""CSV market data I/O (port of Java ``data.CsvBarLoader``).

Loads real historical OHLCV bars into a :class:`~quantfinlib.data.bar_series.BarSeries`
and saves series back out -- the interchange format for the whole
library.

The loader is tolerant of real-world files: header names are matched
case-insensitively (``date``/``time``/``timestamp``/``datetime``,
``open``, ``high``, ``low``, ``close``, optional ``volume``),
timestamps may be epoch millis, epoch seconds, ``yyyy-MM-dd``, ISO
local or offset date-times, rows may be unordered (they are sorted by
time), and blank lines are skipped.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import List, Sequence

from quantfinlib.data.bar import Bar
from quantfinlib.data.bar_series import BarSeries

_OFFSET_RE = re.compile(r"(Z|[+-]\d{2}:\d{2})$")
_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def load(path: str, symbol: str) -> BarSeries:
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    return parse(lines, symbol)


def parse(lines: Sequence[str], symbol: str) -> BarSeries:
    if not lines:
        raise ValueError("empty CSV")
    header_line = lines[0].replace("﻿", "")  # strip BOM
    # Delimiter is a FILE-level fact: a semicolon-delimited file is the
    # European convention, where the comma is the DECIMAL separator --
    # exactly the character the US-convention number parser strips.
    # Detect once from the header, counting only semicolons OUTSIDE
    # quotes: a quoted extra-column name containing ';' in a comma file
    # must not flip the whole file to European decimals.
    semicolon = contains_unquoted(header_line, ";")
    delimiter = ";" if semicolon else ","
    headers = split_csv(header_line, delimiter)
    ts_col = o_col = h_col = l_col = c_col = v_col = -1
    for i, raw_h in enumerate(headers):
        h = raw_h.strip().lower()
        if h in ("timestamp", "time", "date", "datetime"):
            ts_col = i if ts_col < 0 else ts_col
        elif h == "open":
            o_col = i
        elif h == "high":
            h_col = i
        elif h == "low":
            l_col = i
        elif h in ("close", "adj close", "adj_close"):
            c_col = i if c_col < 0 else c_col
        elif h in ("volume", "vol"):
            v_col = i
        # else: ignore extra columns

    if ts_col < 0 or o_col < 0 or h_col < 0 or l_col < 0 or c_col < 0:
        raise ValueError(
            f"CSV must have timestamp/date, open, high, low, close columns; got: {header_line}"
        )

    bars: List[Bar] = []
    all_numeric = True
    min_ts = 2**63 - 1
    max_ts = -(2**63)
    for r in range(1, len(lines)):
        line = lines[r].strip()
        if not line:
            continue
        cells = split_csv(line, delimiter)
        raw_ts = cells[ts_col].strip()
        all_numeric = all_numeric and raw_ts.isdigit()
        ts = parse_timestamp(raw_ts)
        min_ts = min(min_ts, ts)
        max_ts = max(max_ts, ts)
        bars.append(
            Bar(
                ts,
                parse_number(cells[o_col], semicolon),
                parse_number(cells[h_col], semicolon),
                parse_number(cells[l_col], semicolon),
                parse_number(cells[c_col], semicolon),
                parse_number(cells[v_col], semicolon) if 0 <= v_col < len(cells) else 0.0,
            )
        )
    if not bars:
        raise ValueError("CSV has a header but no data rows")

    # Whole-file epoch-seconds detection: if every numeric timestamp sits in the
    # plausible seconds range (years 1973..5138), the file is in seconds.
    if is_epoch_seconds_file(all_numeric, min_ts, max_ts):
        bars = [Bar(b.timestamp * 1000, b.open, b.high, b.low, b.close, b.volume) for b in bars]

    bars.sort(key=lambda b: b.timestamp)
    return BarSeries.from_bars(symbol, bars)


def save(series: BarSeries, path: str) -> None:
    """Writes the series as ``timestamp,open,high,low,close,volume`` with
    epoch-millis timestamps."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as w:
        w.write("timestamp,open,high,low,close,volume\n")
        for i in range(series.size()):
            w.write(
                "%d,%s,%s,%s,%s,%s\n"
                % (
                    series.timestamp(i),
                    series.open(i),
                    series.high(i),
                    series.low(i),
                    series.close(i),
                    series.volume(i),
                )
            )


def split_csv(line: str, delimiter: str = ",") -> List[str]:
    """RFC-4180-style field split on ONE delimiter. Splitting a semicolon
    file on commas too would shred its decimal commas ("100,25" is one
    number, not two cells) -- the delimiter is a file-level fact decided
    once from the header, never per line."""
    out: List[str] = []
    current: List[str] = []
    quoted = False
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if quoted:
            if c == '"':
                if i + 1 < n and line[i + 1] == '"':
                    current.append('"')
                    i += 1
                else:
                    quoted = False
            else:
                current.append(c)
        elif c == '"':
            quoted = True
        elif c == delimiter:
            out.append("".join(current))
            current = []
        else:
            current.append(c)
        i += 1
    out.append("".join(current))
    return out


def contains_unquoted(line: str, c: str) -> bool:
    """Whether ``c`` occurs outside RFC-4180 double-quoted regions."""
    quoted = False
    for ch in line:
        if ch == '"':
            quoted = not quoted  # "" inside quotes toggles twice: net unchanged
        elif ch == c and not quoted:
            return True
    return False


def parse_number(value: str, european_decimals: bool = False) -> float:
    """Locale-aware numeric parse. US convention (comma files): the comma
    is a thousands separator and is stripped ("1,234.5"). European
    convention (semicolon files): the DOT is the thousands separator and
    the COMMA is the decimal point ("1.234,56") -- stripping commas
    there would silently read 1,6 as 16, corrupting every value by
    10x."""
    v = value.strip()
    if european_decimals:
        return float(v.replace(".", "").replace(",", "."))
    return float(v.replace(",", ""))


def is_epoch_seconds_file(all_numeric: bool, min_ts: int, max_ts: int) -> bool:
    """The whole-file epoch-seconds heuristic, shared with
    :mod:`~quantfinlib.data.universe_csv_loader` so bar files and universe
    files can never disagree on the scale: all-numeric timestamps sitting
    entirely in the plausible seconds range (years 1973..5138) are
    seconds, x1000."""
    return all_numeric and min_ts >= 100_000_000 and max_ts < 100_000_000_000


def parse_timestamp(value: str) -> int:
    """Epoch millis from a numeric timestamp (seconds-vs-millis is
    resolved at file level in :func:`parse`), ``yyyy-MM-dd``, or ISO
    date-times (UTC assumed when unzoned)."""
    if value.isdigit():
        return int(value)
    if _DATE_ONLY_RE.fullmatch(value):
        dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return _to_epoch_millis(dt)
    v = value.replace(" ", "T")
    if not _OFFSET_RE.search(v):
        # LocalDateTime: no zone/offset in the text, UTC assumed.
        dt = datetime.fromisoformat(v)
        dt = dt.replace(tzinfo=timezone.utc)
        return _to_epoch_millis(dt)
    # OffsetDateTime: explicit zone/offset present.
    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    return _to_epoch_millis(dt)


def _to_epoch_millis(dt: datetime) -> int:
    # Java's Instant.toEpochMilli() is epochSecond * 1000 +
    # nanoOfSecond / 1_000_000 using integer division, and nanoOfSecond
    # is always in [0, 999_999_999) -- i.e. it TRUNCATES (floors) any
    # sub-millisecond remainder, it never rounds to the nearest
    # millisecond. round() would round a .5-or-later microsecond
    # fraction up to the next millisecond where Java keeps it down;
    # floor division on the timedelta reproduces the exact Java
    # truncation, including for pre-1970 (negative) instants.
    delta = dt - _EPOCH
    return delta // timedelta(milliseconds=1)
