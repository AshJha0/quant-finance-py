"""Multi-venue aggregated top-of-book (port of Java
``com.quantfinlib.fx.AggregatedBook``).

Each liquidity provider / ECN streams its own two-sided quote; the
aggregator maintains the composite best bid/offer with venue
attribution. A venue with no quote (or a cleared one) holds NaN and
never wins the scan. Crossed composites (bid >= ask across venues) are
real in e-FX — latency between feeds — and are reported via
``is_crossed`` rather than "fixed" silently.
"""

from __future__ import annotations

import math

import numpy as np


class AggregatedBook:

    def __init__(self, venue_count: int):
        if venue_count <= 0:
            raise ValueError("venue_count must be > 0")
        self._venue_count = venue_count
        self._bids = np.full(venue_count, math.nan)
        self._bid_sizes = np.zeros(venue_count)
        self._asks = np.full(venue_count, math.nan)
        self._ask_sizes = np.zeros(venue_count)
        self._best_bid = math.nan
        self._best_ask = math.nan
        self._best_bid_size = 0.0
        self._best_ask_size = 0.0
        self._best_bid_venue = -1
        self._best_ask_venue = -1
        self._update_count = 0

    def on_quote(self, venue: int, bid: float, bid_size: float,
                 ask: float, ask_size: float) -> None:
        """A venue's fresh two-sided quote (NaN on a side pulls that
        side); refreshes the composite in the same call."""
        self._bids[venue] = bid
        self._bid_sizes[venue] = bid_size
        self._asks[venue] = ask
        self._ask_sizes[venue] = ask_size
        self._rescan()
        self._update_count += 1

    def clear(self, venue: int) -> None:
        """Pulls a venue entirely (disconnect, last-look withdrawal)."""
        self.on_quote(venue, math.nan, 0, math.nan, 0)

    def _rescan(self) -> None:
        """Recomputes the composite: highest bid, lowest ask, with
        attribution."""
        bb = math.nan
        ba = math.nan
        bb_v = -1
        ba_v = -1
        for v in range(self._venue_count):
            b = self._bids[v]
            # NaN comparisons are False, so empty venues lose automatically.
            if not math.isnan(b) and (math.isnan(bb) or b > bb):
                bb = b
                bb_v = v
            a = self._asks[v]
            if not math.isnan(a) and (math.isnan(ba) or a < ba):
                ba = a
                ba_v = v
        self._best_bid = bb
        self._best_ask = ba
        self._best_bid_venue = bb_v
        self._best_ask_venue = ba_v
        self._best_bid_size = self._bid_sizes[bb_v] if bb_v >= 0 else 0.0
        self._best_ask_size = self._ask_sizes[ba_v] if ba_v >= 0 else 0.0

    # ------------------------------------------------------------------
    # Composite queries
    # ------------------------------------------------------------------

    def best_bid(self) -> float:
        return self._best_bid

    def best_ask(self) -> float:
        return self._best_ask

    def best_bid_size(self) -> float:
        """Size shown by the single venue owning the best bid."""
        return self._best_bid_size

    def best_ask_size(self) -> float:
        return self._best_ask_size

    def best_bid_venue(self) -> int:
        """Venue index owning the best bid, -1 when no venue bids."""
        return self._best_bid_venue

    def best_ask_venue(self) -> int:
        return self._best_ask_venue

    def mid(self) -> float:
        """Composite mid; NaN until both sides are quoted."""
        return 0.5 * (self._best_bid + self._best_ask)

    def spread(self) -> float:
        """Composite spread; NaN until both sides are quoted."""
        return self._best_ask - self._best_bid

    def total_bid_size_at_best(self, tolerance: float) -> float:
        """Total size quoted within ``tolerance`` of the best bid across
        all venues — the sweepable size at the composite level."""
        if math.isnan(self._best_bid):
            return 0.0
        total = 0.0
        for v in range(self._venue_count):
            if (not math.isnan(self._bids[v])
                    and self._best_bid - self._bids[v] <= tolerance):
                total += self._bid_sizes[v]
        return total

    def total_ask_size_at_best(self, tolerance: float) -> float:
        """Mirror of :meth:`total_bid_size_at_best` for the offer side."""
        if math.isnan(self._best_ask):
            return 0.0
        total = 0.0
        for v in range(self._venue_count):
            if (not math.isnan(self._asks[v])
                    and self._asks[v] - self._best_ask <= tolerance):
                total += self._ask_sizes[v]
        return total

    def is_crossed(self) -> bool:
        """Whether the composite is crossed or locked (best bid >= best
        ask): common transiently in aggregated e-FX, and exactly what
        arbitrage/SOR logic wants to see, not have hidden."""
        return self._best_bid >= self._best_ask  # NaN either side -> False

    def venue_count(self) -> int:
        return self._venue_count

    def update_count(self) -> int:
        return self._update_count
