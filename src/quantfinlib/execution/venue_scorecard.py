"""Streaming per-venue execution quality (port of Java
``execution.VenueScorecard``).

Displayed prices tell you where a venue CLAIMS you'll trade; the
scorecard tells you what actually happens when you send there.
Everything :class:`~quantfinlib.execution.adaptive_sor.AdaptiveSor`
needs beyond the quote:

* **Fill rate** -- EWMA of {1 fill, 0 miss} per marketable child: the
  venue's reliability.
* **Response latency** -- EWMA of send->ack/fill time as YOU measure it.
* **Hidden liquidity** -- for dark venues, an EWMA of the shares each
  probe actually found.
* **Post-fill markout (adverse selection)** -- what the mid does one
  horizon after your fill, signed in your trading direction: positive =
  the price kept going your way (a clean fill), negative = it reverted
  (you paid the spread to trade against informed or stale flow).

All statistics are exponentially weighted per event so the card tracks
current behavior (each EWMA seeds from the prior / first observation,
so a venue is never scored below "never tried" for its first fill).
Before any data, :meth:`fill_rate` returns a configurable optimistic
prior (a new venue deserves flow until it proves otherwise).
"""

from __future__ import annotations

import numpy as np

from quantfinlib.persist import Checkpoint

#: Pending fill-markout slots per venue (bursts deeper overwrite oldest).
PENDING_RING = 4


class VenueScorecard:
    """Streaming per-venue execution-quality card; see the module
    docstring."""

    def __init__(self, venue_count: int, alpha: float = 0.05,
                fill_rate_prior: float = 0.95,
                markout_horizon_nanos: int = 100_000_000) -> None:
        """
        Args:
            venue_count: number of venues (dense indices).
            alpha: EWMA weight per event, e.g. 0.05.
            fill_rate_prior: fill rate assumed before any data, e.g. 0.95.
            markout_horizon_nanos: how long after a fill the markout is
                read, e.g. 100ms = 100_000_000.
        """
        if (venue_count < 1 or alpha <= 0 or alpha > 1
                or fill_rate_prior <= 0 or fill_rate_prior > 1
                or markout_horizon_nanos <= 0):
            raise ValueError(
                "need venueCount >= 1, alpha in (0,1], prior in (0,1], "
                "horizon > 0")
        self._venue_count = venue_count
        self._alpha = alpha
        self._fill_rate_prior = fill_rate_prior
        self._markout_horizon_nanos = markout_horizon_nanos

        self._sent = np.zeros(venue_count, dtype=np.int64)
        self._filled = np.zeros(venue_count, dtype=np.int64)
        self._fill_rate_ewma = np.zeros(venue_count)
        self._latency_nanos_ewma = np.zeros(venue_count)
        self._hidden_fill_ewma = np.zeros(venue_count)
        self._probes = np.zeros(venue_count, dtype=np.int64)

        self._markout_ewma = np.zeros(venue_count)
        self._markout_count = np.zeros(venue_count, dtype=np.int64)
        n = venue_count * PENDING_RING
        self._pending_side = np.zeros(n, dtype=np.int8)   # 0 empty, +1 buy, -1 sell
        self._pending_mid = np.zeros(n)
        self._pending_ts = np.zeros(n, dtype=np.int64)
        self._pending_cursor = np.zeros(venue_count, dtype=np.int8)
        self._pending_count = 0
        self._matured_fill_markouts = 0

    # ------------------------------------------------------------------
    # Event feed
    # ------------------------------------------------------------------

    def on_fill(self, venue: int, response_nanos: float, buy: bool | None = None,
               mid_at_fill: float | None = None,
               timestamp_nanos: int | None = None) -> None:
        """A marketable child filled (fully or partially counts as a
        fill).

        The extended form (pass ``buy``, ``mid_at_fill``,
        ``timestamp_nanos``) also arms this fill's markout: feed mids
        via :meth:`on_mid` and the move one horizon later is attributed
        to this venue. A non-finite mid still counts the fill but can
        never start a markout (maturing against a non-finite value
        would poison the EWMA and silently distort routing).
        """
        # EWMAs seed from the prior / first observation -- ramping from
        # 0 would record a venue's FIRST successful fill as fillRate
        # 0.05 and get it vetoed by the router: success must never
        # score below "never tried".
        self._seed_on_first_event(venue, response_nanos)
        self._sent[venue] += 1
        self._filled[venue] += 1
        self._fill_rate_ewma[venue] += self._alpha * (1 - self._fill_rate_ewma[venue])
        self._latency_nanos_ewma[venue] += self._alpha * (
            response_nanos - self._latency_nanos_ewma[venue])

        if buy is None:
            return
        if not np.isfinite(mid_at_fill):
            return                          # an Inf sentinel must not arm either
        slot = venue * PENDING_RING + self._pending_cursor[venue]
        if self._pending_side[slot] == 0:
            self._pending_count += 1        # fresh slot; overwrite keeps count
        self._pending_side[slot] = 1 if buy else -1
        self._pending_mid[slot] = mid_at_fill
        self._pending_ts[slot] = timestamp_nanos
        self._pending_cursor[venue] = (self._pending_cursor[venue] + 1) % PENDING_RING

    def on_miss(self, venue: int, response_nanos: float) -> None:
        """A marketable child that came back unfilled (faded,
        rejected, expired)."""
        self._seed_on_first_event(venue, response_nanos)
        self._sent[venue] += 1
        self._fill_rate_ewma[venue] += self._alpha * (0 - self._fill_rate_ewma[venue])
        self._latency_nanos_ewma[venue] += self._alpha * (
            response_nanos - self._latency_nanos_ewma[venue])

    def _seed_on_first_event(self, venue: int, response_nanos: float) -> None:
        if self._sent[venue] == 0:
            self._fill_rate_ewma[venue] = self._fill_rate_prior
            self._latency_nanos_ewma[venue] = response_nanos

    def on_mid(self, mid: float, timestamp_nanos: int) -> None:
        """Mid update FOR THE CARD'S ONE SYMBOL (see the class doc):
        matures every pending fill markout whose horizon has elapsed.
        Non-finite mids are ignored -- one bad sentinel maturing a slot
        would seed the EWMA at an extreme and the next blend would
        poison it forever, silently disabling the router's
        adverse-selection term. The common no-pending case is a single
        compare.
        """
        if self._pending_count == 0 or not np.isfinite(mid):
            return
        n = self._venue_count * PENDING_RING
        for slot in range(n):
            if (self._pending_side[slot] != 0
                    and timestamp_nanos - self._pending_ts[slot]
                    >= self._markout_horizon_nanos):
                venue = slot // PENDING_RING
                move = self._pending_side[slot] * (mid - self._pending_mid[slot])
                # Seed from the first matured markout -- ramping from 0
                # would read a venue's first adverse fills at ~5% strength.
                if self._markout_count[venue] == 0:
                    self._markout_ewma[venue] = move
                else:
                    self._markout_ewma[venue] += self._alpha * (
                        move - self._markout_ewma[venue])
                self._markout_count[venue] += 1
                self._pending_side[slot] = 0
                self._pending_count -= 1
                self._matured_fill_markouts += 1

    def on_dark_probe(self, venue: int, shares_filled: int) -> None:
        """A dark probe's outcome: how many shares it actually found (0
        is a real observation -- an empty pool teaches as much as a
        full one). The estimate seeds from the first probe rather than
        ramping from 0, so one good probe doesn't collapse subsequent
        probe sizes."""
        if self._probes[venue] == 0:
            self._hidden_fill_ewma[venue] = shares_filled
        else:
            self._hidden_fill_ewma[venue] += self._alpha * (
                shares_filled - self._hidden_fill_ewma[venue])
        self._probes[venue] += 1

    # ------------------------------------------------------------------
    # The card
    # ------------------------------------------------------------------

    def fill_rate(self, venue: int) -> float:
        """EWMA fill probability; the optimistic prior before any data."""
        return (self._fill_rate_prior if self._sent[venue] == 0
               else float(self._fill_rate_ewma[venue]))

    def fill_rate_prior(self) -> float:
        """The before-any-data prior (also what unregistered venues
        score as)."""
        return self._fill_rate_prior

    def measured_latency_nanos(self, venue: int) -> float:
        """EWMA measured response latency in nanos (0 before any data)."""
        return float(self._latency_nanos_ewma[venue])

    def expected_hidden_shares(self, venue: int) -> float:
        """EWMA shares found per dark probe (0 before any probe)."""
        return float(self._hidden_fill_ewma[venue])

    def post_fill_markout(self, venue: int) -> float:
        """EWMA post-fill markout in price units -- positive means the
        mid kept moving your way after fills at this venue; negative
        means it reverted. 0 before any matured markout."""
        return float(self._markout_ewma[venue])

    def matured_fill_markouts(self) -> int:
        """Fill markouts matured across all venues -- the wiring
        canary: zero while fills accrue means :meth:`on_mid` is not
        being fed and the router's adverse-selection term is silently
        disabled."""
        return self._matured_fill_markouts

    def sent(self, venue: int) -> int:
        return int(self._sent[venue])

    def filled(self, venue: int) -> int:
        return int(self._filled[venue])

    def probes(self, venue: int) -> int:
        return int(self._probes[venue])

    def venue_count(self) -> int:
        return self._venue_count

    # ------------------------------------------------------------------
    # Persistence (persist.Checkpoint)
    # ------------------------------------------------------------------

    def write_state(self, out) -> None:
        """Persists the learned venue quality -- fill rates, measured
        latencies, dark-probe estimates and fill markouts are exactly
        what a router should not have to relearn every morning. Format
        version 2 (version 1, from before the markout existed, is still
        readable)."""
        out.write_byte(2)
        Checkpoint.write_longs(out, self._sent)
        Checkpoint.write_longs(out, self._filled)
        Checkpoint.write_longs(out, self._probes)
        Checkpoint.write_doubles(out, self._fill_rate_ewma)
        Checkpoint.write_doubles(out, self._latency_nanos_ewma)
        Checkpoint.write_doubles(out, self._hidden_fill_ewma)
        Checkpoint.write_doubles(out, self._markout_ewma)
        Checkpoint.write_longs(out, self._markout_count)
        out.write_long(self._matured_fill_markouts)

    def read_state(self, inp) -> None:
        """Restores the card; pending fill markouts (intraday) reset.
        Reads both format versions: a v1 checkpoint restores everything
        it has and leaves the markout state cold. Raises on a
        venue-count mismatch or an unknown version."""
        v = inp.read_byte()
        if v not in (1, 2):
            raise ValueError(
                f"VenueScorecard state version {v} not supported "
                "(this build reads versions 1-2)")
        Checkpoint.read_longs_into(inp, self._sent)
        Checkpoint.read_longs_into(inp, self._filled)
        Checkpoint.read_longs_into(inp, self._probes)
        Checkpoint.read_doubles_into(inp, self._fill_rate_ewma)
        Checkpoint.read_doubles_into(inp, self._latency_nanos_ewma)
        Checkpoint.read_doubles_into(inp, self._hidden_fill_ewma)
        if v >= 2:
            Checkpoint.read_doubles_into(inp, self._markout_ewma)
            Checkpoint.read_longs_into(inp, self._markout_count)
            self._matured_fill_markouts = inp.read_long()
        else:
            self._markout_ewma[:] = 0
            self._markout_count[:] = 0
            self._matured_fill_markouts = 0
        self._pending_side[:] = 0
        self._pending_cursor[:] = 0
        self._pending_count = 0
