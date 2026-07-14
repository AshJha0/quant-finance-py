"""Benchmark-fixing execution schedule, WMR-style (port of Java
``execution.WmrFixingScheduler``).

Orders benchmarked to a fixing (the WM/Refinitiv 4pm London fix and its
cousins) are executed by spreading the parent evenly across the
fixing's calculation window, so realized cost tracks the benchmark
rather than betting against it -- the window is 5 minutes for major
pairs, and the benchmark is computed from observations inside it, so
TWAP-in-window IS the neutral replication.

Deliberately excluded, with reasons: executing ahead of the window to
"get done early" is pre-hedging risk against the client's benchmark
(the conduct the 2013-15 FX fix scandals were about), and skewing
inside the window is a bet, not benchmark replication. Anyone who wants
a bet can use :mod:`quantfinlib.execution.implementation_shortfall_scheduler`
and own it explicitly.
"""

from __future__ import annotations

from typing import List

from quantfinlib.execution import twap_scheduler
from quantfinlib.execution.slice import Slice

#: The standard WMR calculation window for major pairs.
WINDOW_MILLIS = 5 * 60 * 1_000

_MAX_WINDOW_MILLIS = (1 << 63) - 1


def schedule(total_qty: int, num_slices: int,
            window_millis: int = WINDOW_MILLIS) -> List[Slice]:
    """Even slices across the fixing window. Offsets are relative to
    the schedule's own start; align slice 0 with the window open (fix
    time minus half the window).

    Port note: the Java source overloads ``schedule(qty, window,
    slices)`` and ``schedule(qty, slices)`` (standard window); Python
    has no overloading, so ``window_millis`` is a keyword default here
    -- call ``schedule(qty, slices)`` for the standard 5-minute window
    or ``schedule(qty, slices, window_millis=...)`` to override it.

    Args:
        total_qty: parent quantity (sliced exactly, largest-remainder).
        num_slices: child count; more slices = closer benchmark
            tracking, more tickets.
        window_millis: calculation-window length (:data:`WINDOW_MILLIS`
            for majors).
    """
    if total_qty <= 0 or window_millis <= 0 or num_slices < 1:
        raise ValueError("need totalQty > 0, windowMillis > 0, numSlices >= 1")
    if num_slices > total_qty:
        # Even slicing would emit zero-quantity children -- venue
        # rejects in the middle of the fixing window. Fail at schedule
        # time.
        raise ValueError(
            f"numSlices ({num_slices}) exceeds totalQty ({total_qty}): "
            "zero-quantity child orders")
    if window_millis > _MAX_WINDOW_MILLIS // num_slices:
        # Offset arithmetic (windowMillis * i) would overflow and place
        # slices BEFORE the window -- the pre-window execution this
        # class exists to refuse.
        raise ValueError(f"windowMillis too large: {window_millis}")
    # The schedule IS a TWAP over the window -- delegate so the claim
    # "TWAP-in-window is neutral replication" is true by construction
    # and cannot drift from the TWAP implementation.
    return twap_scheduler.schedule(total_qty, window_millis, num_slices)
