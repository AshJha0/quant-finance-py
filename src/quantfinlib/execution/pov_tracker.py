"""POV (percentage-of-volume) execution tracker (port of Java
``execution.PovTracker``) -- the streaming counterpart of the
precomputed TWAP/VWAP schedules.

POV cannot be prescheduled -- it chases realized market volume -- so
this class maintains the participation ledger live: feed it market
prints and own fills, and it answers "how many shares am I allowed to
send right now" (:meth:`due_quantity`).

The target is ``executed ~= participation * market_volume``, where
``market_volume`` excludes our own fills (participation is measured
against OTHER people's trading, else the algo chases itself: p/(1+p)
instead of p). Child sizes are clamped to ``[min_slice, max_slice]`` --
the minimum suppresses dribble orders, the maximum caps signaling risk.
"""

from __future__ import annotations

import math


class PovTracker:
    """Streaming POV participation tracker; see the module docstring."""

    __slots__ = ("_parent_qty", "_participation", "_min_slice", "_max_slice",
                "_market_volume", "_executed")

    def __init__(self, parent_qty: int, participation: float,
                min_slice: int, max_slice: int) -> None:
        """
        Args:
            parent_qty: total shares to execute.
            participation: target participation rate in (0, 1], e.g. 0.1 = 10%.
            min_slice: smallest child worth sending (0 = no minimum).
            max_slice: largest child ever sent (caps information leakage).
        """
        if parent_qty <= 0 or participation <= 0 or participation > 1:
            raise ValueError("need parentQty > 0, participation in (0,1]")
        if min_slice < 0 or max_slice < max(1, min_slice):
            raise ValueError("need 0 <= minSlice <= maxSlice")
        self._parent_qty = parent_qty
        self._participation = participation
        self._min_slice = min_slice
        self._max_slice = max_slice
        self._market_volume = 0
        self._executed = 0

    def on_market_volume(self, qty: int) -> None:
        """A market trade print that was NOT our fill."""
        if qty > 0:
            self._market_volume += qty

    def on_executed(self, qty: int) -> None:
        """Our own child fill (do not also feed it to
        :meth:`on_market_volume`)."""
        if qty > 0:
            self._executed += qty

    def due_quantity(self) -> int:
        """Shares to send now to restore the target participation: the
        behind-schedule quantity, clamped to the slice bounds and the
        parent remainder. Returns 0 while within ``min_slice`` of
        schedule (or when done) -- poll it on every print, act when
        it's positive."""
        remaining = self._parent_qty - self._executed
        if remaining <= 0:
            return 0
        target = int(self._participation * self._market_volume)
        behind = target - self._executed
        if behind < max(1, self._min_slice):
            return 0
        return min(min(behind, self._max_slice), remaining)

    def realized_participation(self) -> float:
        """Realized participation so far vs other-flow volume (NaN
        before any print)."""
        return (math.nan if self._market_volume == 0
               else self._executed / self._market_volume)

    def executed(self) -> int:
        return self._executed

    def remaining(self) -> int:
        return self._parent_qty - self._executed

    def market_volume(self) -> int:
        return self._market_volume

    def done(self) -> bool:
        return self._executed >= self._parent_qty
