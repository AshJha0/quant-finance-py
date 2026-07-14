"""National Best Bid and Offer (port of Java ``marketdata.Nbbo``):
aggregates per-venue top-of-book quotes for one symbol into the
consolidated best bid/ask, the size available at those prices, and a
bitmask of which venues are at the inside -- the three inputs a smart
order router actually consumes. The equities counterpart of
``fx.AggregatedBook``, in integer ticks.

Single-writer (one consolidated-feed thread). Recomputation is a
linear scan over venues -- with the <= 64 venues of a real consolidated
tape, a scan over four parallel arrays beats any incremental structure.

An optional listener fires only when the NBBO actually changes (price
or size at the inside), so downstream logic is naturally conflated to
inside-quote updates.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional

NO_BID = -(1 << 31)
NO_ASK = (1 << 31) - 1

#: fired after the NBBO (price or inside size) changes:
#: listener(bid_tick, bid_size, ask_tick, ask_size, timestamp_nanos)
NbboListener = Callable[[int, int, int, int, int], None]


class Nbbo:
    """Consolidated best bid/offer across venues for one symbol."""

    def __init__(self, venue_count: int) -> None:
        """
        Args:
            venue_count: number of venues (<= 64, so venue sets fit
                one bitmask).
        """
        if venue_count < 1 or venue_count > 64:
            raise ValueError("venueCount must be 1..64")
        self._venues = venue_count
        self._bid_tick: List[int] = [NO_BID] * venue_count
        self._bid_size: List[int] = [0] * venue_count
        self._ask_tick: List[int] = [NO_ASK] * venue_count
        self._ask_size: List[int] = [0] * venue_count

        self._nbb = NO_BID
        self._nbb_size = 0
        self._nbb_venue_bits = 0
        self._nbo = NO_ASK
        self._nbo_size = 0
        self._nbo_venue_bits = 0

        self._listener: Optional[NbboListener] = None
        self._update_count = 0
        self._change_count = 0

    def listener(self, l: Optional[NbboListener]) -> None:
        """Installs the (single) inside-change callback."""
        self._listener = l

    def on_venue_quote(self, venue: int, bid: int, bid_sz: int, ask: int,
                       ask_sz: int, timestamp_nanos: int) -> bool:
        """One venue's new top of book. Pass ``NO_BID``/``NO_ASK`` (or
        zero sizes) for an empty side; use :meth:`on_venue_down` when
        the venue drops entirely. Returns True when the NBBO changed."""
        new_bid = bid if bid_sz > 0 else NO_BID
        new_ask = ask if ask_sz > 0 else NO_ASK
        self._bid_tick[venue] = new_bid
        self._bid_size[venue] = bid_sz if bid_sz > 0 else 0
        self._ask_tick[venue] = new_ask
        self._ask_size[venue] = ask_sz if ask_sz > 0 else 0
        self._update_count += 1
        # Fast path: a venue that was not at either inside and stays
        # strictly off it cannot move the NBBO -- most consolidated-
        # tape updates are exactly this off-inside flicker, so skip the
        # scan for them. Any comparison against an absent inside
        # (NO_BID/NO_ASK sentinel) is False, correctly forcing the scan.
        bit = 1 << venue
        if ((self._nbb_venue_bits & bit) == 0 and (self._nbo_venue_bits & bit) == 0
                and (new_bid == NO_BID or new_bid < self._nbb)
                and (new_ask == NO_ASK or new_ask > self._nbo)):
            return False
        return self._recompute(timestamp_nanos)

    def on_venue_down(self, venue: int, timestamp_nanos: int) -> bool:
        """Removes a venue's quotes (feed loss / venue halt). Returns
        True on NBBO change."""
        return self.on_venue_quote(venue, NO_BID, 0, NO_ASK, 0, timestamp_nanos)

    def _recompute(self, ts: int) -> bool:
        bb = NO_BID
        bb_sz = 0
        bb_bits = 0
        bo = NO_ASK
        bo_sz = 0
        bo_bits = 0
        for v in range(self._venues):
            b = self._bid_tick[v]
            if b != NO_BID:
                if b > bb:
                    bb = b
                    bb_sz = self._bid_size[v]
                    bb_bits = 1 << v
                elif b == bb:
                    bb_sz += self._bid_size[v]
                    bb_bits |= 1 << v
            a = self._ask_tick[v]
            if a != NO_ASK:
                if a < bo:
                    bo = a
                    bo_sz = self._ask_size[v]
                    bo_bits = 1 << v
                elif a == bo:
                    bo_sz += self._ask_size[v]
                    bo_bits |= 1 << v
        changed = (bb != self._nbb or bb_sz != self._nbb_size
                   or bo != self._nbo or bo_sz != self._nbo_size)
        self._nbb = bb
        self._nbb_size = bb_sz
        self._nbb_venue_bits = bb_bits
        self._nbo = bo
        self._nbo_size = bo_sz
        self._nbo_venue_bits = bo_bits
        if changed:
            self._change_count += 1
            if self._listener is not None:
                self._listener(bb, bb_sz, bo, bo_sz, ts)
        return changed

    def bid_tick(self) -> int:
        """National best bid in ticks; ``NO_BID`` when no venue bids."""
        return self._nbb

    def bid_size(self) -> int:
        """Total displayed size at the national best bid, across venues."""
        return self._nbb_size

    def ask_tick(self) -> int:
        """National best offer in ticks; ``NO_ASK`` when no venue offers."""
        return self._nbo

    def ask_size(self) -> int:
        """Total displayed size at the national best offer, across venues."""
        return self._nbo_size

    def bid_venues(self) -> int:
        """Bitmask of venues quoting at the national best bid (bit v = venue v)."""
        return self._nbb_venue_bits

    def ask_venues(self) -> int:
        """Bitmask of venues quoting at the national best offer."""
        return self._nbo_venue_bits

    def crossed(self) -> bool:
        """Crossed market flag (NBB above NBO)."""
        return self._nbb != NO_BID and self._nbo != NO_ASK and self._nbb > self._nbo

    def locked(self) -> bool:
        """Locked market flag (NBB equal to NBO)."""
        return self._nbb != NO_BID and self._nbb == self._nbo

    def mid_tick(self) -> float:
        """Mid in tick units, NaN when either side is absent."""
        if self._nbb == NO_BID or self._nbo == NO_ASK:
            return math.nan
        return (self._nbb + self._nbo) / 2.0

    def update_count(self) -> int:
        return self._update_count

    def change_count(self) -> int:
        """Updates that moved the inside (price or size) -- the
        conflation ratio."""
        return self._change_count
