"""Aligns multi-asset bar series onto one shared timeline (port of Java
``data.SeriesAligner``).

The bridge from raw vendor files (:mod:`~quantfinlib.data.csv_bar_loader`)
to the index-aligned input a portfolio backtester requires. Real data
never lines up: different holidays, listing dates, and gaps.

* :func:`intersect` -- keep only timestamps present in EVERY series
  (strictest, no synthetic bars).
* :func:`union_forward_fill` -- union of timestamps from each series'
  first bar onward; gaps carry the previous close forward as a flat
  zero-volume bar.
"""

from __future__ import annotations

import math
from typing import Dict

from quantfinlib.data.bar_series import BarSeries


def intersect(input_: Dict[str, BarSeries]) -> Dict[str, BarSeries]:
    """Timestamps common to all series, in order; input dict order preserved."""
    if not input_:
        raise ValueError("no series supplied")
    common = None
    for s in input_.values():
        timestamps = {int(s.timestamp(i)) for i in range(s.size())}
        common = timestamps if common is None else (common & timestamps)
    if not common:
        raise ValueError("series share no common timestamps")
    ordered = sorted(common)

    out: Dict[str, BarSeries] = {}
    for key, s in input_.items():
        index_by_ts = {int(s.timestamp(i)): i for i in range(s.size())}
        b = BarSeries.builder(s.symbol())
        for ts in ordered:
            bar = s.bar(index_by_ts[ts])
            b.add_bar(bar)
        out[key] = b.build()
    return out


def union_forward_fill(input_: Dict[str, BarSeries]) -> Dict[str, BarSeries]:
    """Union of timestamps from the latest series start onward; missing
    bars are forward-filled as flat bars at the previous close with
    zero volume."""
    if not input_:
        raise ValueError("no series supplied")
    start = -(2**63)
    union = set()
    for s in input_.values():
        start = max(start, int(s.timestamp(0)))
        union.update(int(s.timestamp(i)) for i in range(s.size()))

    # Only from the point where every series has traded at least once.
    timeline = sorted(ts for ts in union if ts >= start)
    if not timeline:
        raise ValueError("series share no overlapping period")

    out: Dict[str, BarSeries] = {}
    for key, s in input_.items():
        b = BarSeries.builder(s.symbol())
        cursor = 0
        last_close = math.nan
        n = s.size()
        for ts in timeline:
            while cursor < n and s.timestamp(cursor) < ts:
                last_close = s.close(cursor)
                cursor += 1
            if cursor < n and s.timestamp(cursor) == ts:
                b.add_bar(s.bar(cursor))
                last_close = s.close(cursor)
                cursor += 1
            else:
                # Gap: carry the previous close as a flat zero-volume bar.
                b.add(ts, last_close, last_close, last_close, last_close, 0.0)
        out[key] = b.build()
    return out
