"""The full-checklist smart order router (port of Java
``execution.AdaptiveSor``).

Prices in everything a production SOR actually weighs, on top of the
all-in (fee-adjusted) price:

* **Probability of fill / venue reliability** -- a venue's
  :class:`~quantfinlib.execution.venue_scorecard.VenueScorecard` fill
  rate discounts its quote: expected cost adds ``(1 - fill_rate) x
  miss_penalty`` (the spread-ish cost of re-routing a faded child), and
  venues below a reliability floor are vetoed outright.
* **Latency** -- slower venues pay ``latency x urgency``: in a moving
  market, microseconds of delay are adverse selection. The scorecard's
  MEASURED latency overrides the advertised quote latency once observed.
* **Adverse selection** -- a venue whose fills are followed by
  reversion (negative post-fill markout) charges that reversion as a
  per-share cost.
* **Hidden liquidity** -- dark pools are probed with sizes learned from
  realized probe fills, seeded by a configurable default when a pool is
  still unknown.
* **Queue position** -- for the passive leg of a child (posting rather
  than taking), :func:`passive_fill_probability` delegates to
  :class:`~quantfinlib.microstructure.queue_model.QueueModel` so
  placement decisions can weigh the queue they would join.

Research/decision lane (allocates the plan); the per-decision cost is
microseconds, spent once per parent slice, not per tick. Venue identity
is by name, mapped once to dense scorecard ids at registration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from quantfinlib.execution.venue_quote import VenueQuote
from quantfinlib.execution.venue_scorecard import VenueScorecard
from quantfinlib.microstructure.execution import Side
from quantfinlib.microstructure.queue_model import QueueModel


@dataclass(frozen=True, slots=True)
class RouteLeg:
    """One child order of the routing plan. ``price`` is the quoted
    (pre-fee) price."""

    venue: str
    price: float
    quantity: int
    dark: bool


@dataclass(frozen=True, slots=True)
class Config:
    """Tunable penalties; :meth:`defaults` is a sane starting point."""

    miss_penalty_bps: float
    urgency_bps_per_ms: float
    min_fill_rate: float
    default_dark_probe_shares: int
    max_dark_fraction: float

    def __post_init__(self) -> None:
        if (self.miss_penalty_bps < 0 or self.urgency_bps_per_ms < 0
                or self.min_fill_rate < 0 or self.min_fill_rate > 1
                or self.default_dark_probe_shares < 0
                or self.max_dark_fraction < 0 or self.max_dark_fraction > 1):
            raise ValueError("invalid router config")

    @staticmethod
    def defaults() -> "Config":
        """missPenalty 2 bps (roughly half a spread re-cross), urgency
        1 bp/ms of latency, veto below 50% fill rate, 5,000-share
        default dark probe, dark capped at half the parent."""
        return Config(2.0, 1.0, 0.5, 5_000, 0.5)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The routed plan: ``lit`` legs cover up to the requested quantity
    (best-expected-cost first); ``probes`` are additive contingent dark
    legs sent alongside; ``unrouted`` is the shortfall no eligible lit
    venue could absorb (0 on a fully routed order)."""

    lit: List[RouteLeg]
    probes: List[RouteLeg]
    routed_qty: int
    unrouted: int


@dataclass(frozen=True, slots=True)
class _Scored:
    venue: str
    price: float
    expected: float
    size: int


class AdaptiveSor:
    """The full-checklist smart order router; see the module docstring."""

    def __init__(self, scorecard: VenueScorecard,
                config: Optional[Config] = None) -> None:
        self._scorecard = scorecard
        self._config = config if config is not None else Config.defaults()
        self._venue_ids: Dict[str, int] = {}

    def register(self, venue: str, scorecard_id: int) -> None:
        """Maps a venue name to its scorecard index. Register every
        venue once at setup; quotes from unregistered venues are
        routed on quote data alone (prior fill rate, advertised
        latency)."""
        if scorecard_id < 0 or scorecard_id >= self._scorecard.venue_count():
            raise ValueError(f"scorecardId out of range: {scorecard_id}")
        self._venue_ids[venue] = scorecard_id

    def scorecard(self) -> VenueScorecard:
        """The scorecard this router learns from (feed fills/misses/
        probes to it)."""
        return self._scorecard

    def route(self, side: Side, quantity: int,
             venues: Sequence[VenueQuote]) -> RoutingDecision:
        """Routes a marketable parent of ``quantity``. Lit venues are
        ranked by expected cost per share::

            all_in * [1 + (1 - fill_rate) * miss_penalty + latency * urgency
                      + adverse_selection]   (buys; the adjustments
                                              subtract for sells)

        where adverse_selection is the venue's negative post-fill
        markout as a fraction of price (a venue with favorable or
        unknown markout pays nothing extra), and swept best-first at
        displayed size. Every quoting dark venue gets a contingent
        probe leg at its midpoint, sized by learned hidden liquidity
        (or the configured default while unknown), capped at
        ``max_dark_fraction * quantity``.
        """
        buy = side == Side.BUY
        lit: List[RouteLeg] = []
        probes: List[RouteLeg] = []
        candidates: List[_Scored] = []

        for v in venues:
            vid = self._venue_ids.get(v.venue)
            if v.dark:
                probe = self._dark_probe_size(vid, quantity)
                if probe > 0 and not _is_nan(v.mid()):
                    probes.append(RouteLeg(v.venue, v.mid(), probe, True))
                continue
            px = v.ask if buy else v.bid
            size = v.ask_size if buy else v.bid_size
            if size <= 0 or _is_nan(px) or px <= 0:
                continue
            fill_rate = (self._scorecard.fill_rate(vid) if vid is not None
                        else self._scorecard.fill_rate_prior())
            if vid is not None and fill_rate < self._config.min_fill_rate:
                continue                    # reliability veto
            latency_nanos = v.latency_nanos
            if vid is not None and self._scorecard.measured_latency_nanos(vid) > 0:
                latency_nanos = self._scorecard.measured_latency_nanos(vid)
            all_in = px * (1 + (1 if buy else -1) * v.fee_bps / 1e4)
            miss_adj = (1 - fill_rate) * self._config.miss_penalty_bps / 1e4
            lat_adj = latency_nanos / 1e6 * self._config.urgency_bps_per_ms / 1e4
            # A venue whose fills revert charges that reversion per
            # share; favorable/unknown markout adds nothing (no bonus
            # for luck).
            adverse_adj = 0.0
            if vid is not None:
                markout = self._scorecard.post_fill_markout(vid)
                if markout < 0:
                    adverse_adj = -markout / px
            expected = all_in * (1 + (1 if buy else -1) * (miss_adj + lat_adj + adverse_adj))
            candidates.append(_Scored(v.venue, px, expected, size))

        candidates.sort(key=lambda c: c.expected if buy else -c.expected)

        remaining = quantity
        for c in candidates:
            if remaining <= 0:
                break
            take = min(remaining, c.size)
            lit.append(RouteLeg(c.venue, c.price, take, False))
            remaining -= take
        return RoutingDecision(lit, probes, quantity - remaining, remaining)

    def _dark_probe_size(self, vid: Optional[int], quantity: int) -> int:
        # Java Math.round: half-up (floor(x + 0.5)), not Python's
        # banker's-rounding round().
        learned = self._scorecard.expected_hidden_shares(vid) if vid is not None else 0.0
        base = (math.floor(learned + 0.5) if learned > 0
               else self._config.default_dark_probe_shares)
        return min(base, math.floor(self._config.max_dark_fraction * quantity + 0.5))

    @staticmethod
    def passive_fill_probability(qty_ahead: int, order_qty: int,
                                 expected_traded_qty: float) -> float:
        """Fill probability for a PASSIVE child joining a queue with
        ``qty_ahead`` ahead of it -- the queue-position leg of the
        routing checklist, delegated to
        :class:`~quantfinlib.microstructure.queue_model.QueueModel`.
        """
        return QueueModel.fill_probability(qty_ahead, order_qty, expected_traded_qty)


def _is_nan(x: float) -> bool:
    return x != x
