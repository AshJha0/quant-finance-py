"""Midpoint-cross dark pool model (port of Java
``execution.DarkPoolSimulator``).

Hidden resting orders match at the current lit-market midpoint,
honoring minimum-execution-quantity constraints (a standard anti-gaming
feature). No pre-trade transparency: resting interest is only
observable through fills.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from quantfinlib.microstructure.execution import Side


@dataclass(frozen=True, slots=True)
class Fill:
    buy_order_id: int
    sell_order_id: int
    price: float
    quantity: int


class _Resting:
    __slots__ = ("id", "min_qty", "qty")

    def __init__(self, order_id: int, qty: int, min_qty: int) -> None:
        self.id = order_id
        self.qty = qty
        self.min_qty = min_qty


class DarkPoolSimulator:
    """Midpoint-cross dark pool with aggregate-MEQ dry-run matching;
    see the module docstring."""

    def __init__(self) -> None:
        self._buys: Deque[_Resting] = deque()
        self._sells: Deque[_Resting] = deque()
        self._mid = math.nan
        self._next_id = 1

    def on_quote(self, bid: float, ask: float) -> None:
        """Update the lit reference mid. A LOCKED or CROSSED reference
        (bid >= ask) or a non-positive/non-finite side invalidates the
        mid: a real midpoint pool is prohibited from executing during a
        locked/crossed NBBO, so crossing pauses until a valid two-sided
        market returns -- resting interest stays resting."""
        self._mid = ((bid + ask) / 2
                    if (bid > 0 and ask > 0 and ask != math.inf and bid < ask)
                    else math.nan)

    def submit(self, side: Side, quantity: int, min_execution_qty: int) -> List[Fill]:
        """Submits an order: crosses immediately against resting contra
        interest at the current mid (time priority), then rests the
        remainder. Returns the fills generated (empty if it fully
        rested).

        Minimum-execution-quantity is honored AGGREGATE-first, the
        common pool semantics: the incoming order's MEQ is checked
        against the total crossable contra quantity (an order wanting
        100 fills against two resting 60s), while each resting order's
        own MEQ still gates its individual slice.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if min_execution_qty < 0:
            raise ValueError(
                f"minExecutionQty must be >= 0, got {min_execution_qty}")
        order_id = self._next_id
        self._next_id += 1
        fills: List[Fill] = []
        remaining = quantity
        if not math.isnan(self._mid):
            contra = self._sells if side == Side.BUY else self._buys
            # Dry-run the exact time-priority consumption below to
            # learn the aggregate quantity that would actually execute.
            # A static scan against the ORIGINAL remaining overcounts: a
            # later resting order's MEQ can become unsatisfiable once
            # earlier fills shrink the remainder, and crossing on the
            # inflated total would violate the INCOMING order's
            # aggregate MEQ.
            crossable = 0
            dry_remaining = remaining
            for r in contra:
                if dry_remaining == 0:
                    break
                fill_qty = min(dry_remaining, r.qty)
                if fill_qty < r.min_qty:
                    continue
                crossable += fill_qty
                dry_remaining -= fill_qty
            if crossable >= min_execution_qty:
                to_remove = []
                for r in contra:
                    if remaining <= 0:
                        break
                    fill_qty = min(remaining, r.qty)
                    if fill_qty < r.min_qty:
                        continue           # the RESTING order's own constraint
                    r.qty -= fill_qty
                    remaining -= fill_qty
                    fills.append(
                        Fill(order_id, r.id, self._mid, fill_qty) if side == Side.BUY
                        else Fill(r.id, order_id, self._mid, fill_qty))
                    if r.qty == 0:
                        to_remove.append(r)
                for r in to_remove:
                    contra.remove(r)
        if remaining > 0:
            (self._buys if side == Side.BUY else self._sells).append(
                _Resting(order_id, remaining, min_execution_qty))
        return fills

    def resting_qty(self, side: Side) -> int:
        """Total hidden resting quantity on a side (for simulation
        introspection only)."""
        book = self._buys if side == Side.BUY else self._sells
        return sum(r.qty for r in book)
