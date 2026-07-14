"""Hot-lane smart order router (port of Java ``execution.HftSor``).

The zero-allocation sibling of a research-lane smart order router, for
when the routing decision sits on the tick-to-order path. Venue state
lives in parallel primitive arrays (updated in place from per-venue
feeds -- plug in NBBO-style venue quotes directly); a route decision is
a greedy sweep by all-in price with no sorting, no lists, no boxing.

Fees are expressed in ticks (venue taker fee converted at the
instrument's tick size, once, at configuration time), so the comparison
stays in integer-friendly arithmetic. Negative fee ticks model rebates.
With the handful of venues real equities routing faces, the O(V^2)
selection sweep is faster than any heap.

Single-writer; reuse one router per routing thread. Python note: the
Java class avoids array allocation per route() call; this port keeps
that discipline (``out_qty`` is provided by the caller and mutated in
place) even though CPython's per-call cost profile differs from the
JVM's.
"""

from __future__ import annotations

import numpy as np

from quantfinlib.microstructure.execution import Side

_NO_BID = -(2 ** 31)
_NO_ASK = 2 ** 31 - 1


class HftSor:
    """Zero-allocation greedy multi-venue router; see the module
    docstring."""

    def __init__(self, venue_count: int) -> None:
        if venue_count < 1:
            raise ValueError("venueCount must be >= 1")
        self._venues = venue_count
        self._bid_tick = np.full(venue_count, _NO_BID, dtype=np.int64)
        self._bid_size = np.zeros(venue_count, dtype=np.int64)
        self._ask_tick = np.full(venue_count, _NO_ASK, dtype=np.int64)
        self._ask_size = np.zeros(venue_count, dtype=np.int64)
        self._fee_ticks = np.zeros(venue_count)
        self._route_count = 0

    def fee(self, venue: int, taker_fee_ticks: float) -> None:
        """Per-venue taker fee in ticks (negative = rebate). Configure once."""
        self._fee_ticks[venue] = taker_fee_ticks

    def venue_quote(self, venue: int, bid: int, bid_sz: int, ask: int, ask_sz: int) -> None:
        """One venue's displayed top of book (zero size = side unavailable)."""
        self._bid_tick[venue] = bid if bid_sz > 0 else _NO_BID
        self._bid_size[venue] = bid_sz if bid_sz > 0 else 0
        self._ask_tick[venue] = ask if ask_sz > 0 else _NO_ASK
        self._ask_size[venue] = ask_sz if ask_sz > 0 else 0

    def venue_down(self, venue: int) -> None:
        """Removes a venue from routing (feed loss / venue halt) -- the
        symmetric call to an NBBO's venue-down handling, so a dead
        venue's stale quote can never keep receiving child orders."""
        self.venue_quote(venue, _NO_BID, 0, _NO_ASK, 0)

    def route(self, side: Side, quantity: int, limit_tick: int,
             out_qty: np.ndarray) -> int:
        """Routes a marketable order across venues by best all-in price
        (quote +/- fee), splitting at displayed size. Child quantities
        are written into ``out_qty[venue]``: the array must be at
        least :meth:`venue_count` long and indices ``[0,
        venue_count)`` are fully overwritten (entries beyond that are
        untouched). The return value is the total routed quantity --
        anything short of ``quantity`` found no displayed liquidity.

        Args:
            limit_tick: worst acceptable raw price in ticks (before
                fees); pass a very large/very negative sentinel for a
                pure market order.
        """
        out_qty[:self._venues] = 0
        self._route_count += 1
        remaining = quantity
        buy = side == Side.BUY
        while remaining > 0:
            best = -1
            best_all_in = 0.0
            for v in range(self._venues):
                if out_qty[v] != 0:
                    continue                # already swept this venue
                px = self._ask_tick[v] if buy else self._bid_tick[v]
                sz = self._ask_size[v] if buy else self._bid_size[v]
                if sz <= 0 or px == (_NO_ASK if buy else _NO_BID) \
                        or (px > limit_tick if buy else px < limit_tick):
                    continue
                all_in = px + self._fee_ticks[v] if buy else px - self._fee_ticks[v]
                if best == -1 or (all_in < best_all_in if buy else all_in > best_all_in):
                    best = v
                    best_all_in = all_in
            if best == -1:
                break
            # A selected venue always takes > 0, so out_qty doubles as
            # the "already used" marker -- no separate scratch array
            # needed.
            take = min(remaining, self._ask_size[best] if buy else self._bid_size[best])
            out_qty[best] = take
            remaining -= take
        return quantity - remaining

    def venue_count(self) -> int:
        return self._venues

    def route_count(self) -> int:
        return self._route_count
