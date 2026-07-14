"""Streaming per-LP execution quality (port of Java
``com.quantfinlib.fx.LpScorecard``).

FX liquidity is quotes, not firm orders -- a provider may hold your
order and reject it -- so the practical measure of an LP is not its
displayed spread but its all-in behavior: how often it rejects, how
long it holds, what the market does right after a reject (the flow it
declined was the flow that was about to pay you), and the effective
spread it actually fills at. ``LpRouter`` consumes this card to price
rejects into the routing decision.

All statistics are exponentially weighted per event (configurable
alpha), so the card tracks current LP behavior, not the session
average. Post-reject markout is measured one horizon after each reject
against the mid fed via :meth:`on_mid`: positive markout = the market
moved the way you were trying to trade = the reject cost you real
money.

Pending markouts live in a small ring per LP (``PENDING_RING`` slots):
reject bursts -- which happen precisely when the market runs and
markouts are largest -- are sampled rather than overwritten, so the
stat cannot be biased low for exactly the LPs it must expose.

The Java ``persist.Checkpoint`` (over)night persistence is not ported
-- no ``persist`` lane in this Python port.
"""

from __future__ import annotations

import math


class LpScorecard:
    """Pending-markout slots per LP (bursts deeper than this overwrite
    the oldest)."""

    PENDING_RING = 4

    def __init__(self, lp_count: int, alpha: float = 0.05,
                 markout_horizon_nanos: int = 100_000_000):
        """``alpha``: EWMA weight per event, e.g. 0.05.
        ``markout_horizon_nanos``: how long after a reject the markout
        is read, e.g. 100ms = 100_000_000."""
        if lp_count < 1 or alpha <= 0 or alpha > 1 or markout_horizon_nanos <= 0:
            raise ValueError("need lp_count >= 1, alpha in (0,1], horizon > 0")
        self._lp_count = lp_count
        self._alpha = alpha
        self._markout_horizon_nanos = markout_horizon_nanos

        self._attempts = [0] * lp_count
        self._fills = [0] * lp_count
        self._rejects = [0] * lp_count
        self._reject_rate = [0.0] * lp_count
        self._hold_nanos_ewma = [0.0] * lp_count
        self._eff_spread_ewma = [0.0] * lp_count
        self._markout_ewma = [0.0] * lp_count
        self._markout_count = [0] * lp_count

        n = lp_count * self.PENDING_RING
        self._pending_side = [0] * n           # 0 empty, +1 buy, -1 sell
        self._pending_mid = [0.0] * n
        self._pending_ts = [0] * n
        self._pending_cursor = [0] * lp_count  # next write slot per LP
        self._pending_count = 0
        self._matured_markouts = 0

    # ------------------------------------------------------------------
    # Event feed
    # ------------------------------------------------------------------

    def on_fill(self, lp: int, buy: bool, price: float, mid_at_request: float,
               hold_nanos: int) -> None:
        """An accepted fill. ``buy``: our direction. ``price``: the
        all-in fill price. ``mid_at_request``: composite mid when the
        order was sent. ``hold_nanos``: time the LP held the order
        before accepting."""
        self._attempts[lp] += 1
        self._fills[lp] += 1
        self._reject_rate[lp] += self._alpha * (0 - self._reject_rate[lp])
        self._hold_nanos_ewma[lp] += self._alpha * (hold_nanos - self._hold_nanos_ewma[lp])
        eff = (price - mid_at_request) if buy else (mid_at_request - price)
        if math.isfinite(eff):
            # A NaN/Inf price or mid must not poison the EWMA permanently.
            self._eff_spread_ewma[lp] += self._alpha * (eff - self._eff_spread_ewma[lp])

    def on_reject(self, lp: int, buy: bool, mid_at_request: float,
                 timestamp_nanos: int, hold_nanos: int) -> None:
        """A last-look reject. The markout clock starts here: feed mids
        via :meth:`on_mid` and the move one horizon later is attributed
        to this reject."""
        self._attempts[lp] += 1
        self._rejects[lp] += 1
        self._reject_rate[lp] += self._alpha * (1 - self._reject_rate[lp])
        self._hold_nanos_ewma[lp] += self._alpha * (hold_nanos - self._hold_nanos_ewma[lp])
        if not math.isfinite(mid_at_request):
            # The reject still counts against the rate, but a NaN/Inf
            # reference mid can never start a markout: maturing against
            # it would poison the EWMA forever and silently de-route
            # this LP.
            return
        slot = lp * self.PENDING_RING + self._pending_cursor[lp]
        if self._pending_side[slot] == 0:
            self._pending_count += 1           # fresh slot; overwrite keeps count
        self._pending_side[slot] = 1 if buy else -1
        self._pending_mid[slot] = mid_at_request
        self._pending_ts[slot] = timestamp_nanos
        self._pending_cursor[lp] = (self._pending_cursor[lp] + 1) % self.PENDING_RING

    def on_mid(self, mid: float, timestamp_nanos: int) -> None:
        """Composite mid update: matures every pending reject markout
        whose horizon has elapsed. NaN mids (one-sided composite, feed
        gap) are ignored."""
        if self._pending_count == 0 or not math.isfinite(mid):
            return
        n = self._lp_count * self.PENDING_RING
        for slot in range(n):
            if (self._pending_side[slot] != 0
                    and timestamp_nanos - self._pending_ts[slot] >= self._markout_horizon_nanos):
                lp = slot // self.PENDING_RING
                move = self._pending_side[slot] * (mid - self._pending_mid[slot])
                # Seed from the first matured markout -- ramping from 0
                # under-penalized a toxic LP for its first ~1/alpha
                # rejects, exactly during the burst that revealed it.
                if self._markout_count[lp] == 0:
                    self._markout_ewma[lp] = move
                else:
                    self._markout_ewma[lp] += self._alpha * (move - self._markout_ewma[lp])
                self._markout_count[lp] += 1
                self._pending_side[slot] = 0
                self._pending_count -= 1
                self._matured_markouts += 1

    # ------------------------------------------------------------------
    # The card
    # ------------------------------------------------------------------

    def reject_rate(self, lp: int) -> float:
        """EWMA reject probability in [0, 1]; 0 before any events."""
        return self._reject_rate[lp]

    def avg_hold_nanos(self, lp: int) -> float:
        """EWMA hold time across fills and rejects, in nanos."""
        return self._hold_nanos_ewma[lp]

    def effective_spread(self, lp: int) -> float:
        """EWMA effective half-spread paid on fills, in price units."""
        return self._eff_spread_ewma[lp]

    def post_reject_markout(self, lp: int) -> float:
        """EWMA post-reject markout in price units -- positive means
        the market moved the way you were trying to trade after the LP
        declined."""
        return self._markout_ewma[lp]

    def matured_markouts(self) -> int:
        """Markouts matured across all LPs -- the router-degradation
        canary: zero while rejects accrue means :meth:`on_mid` is not
        wired and the routing penalty is silently zero."""
        return self._matured_markouts

    def attempts(self, lp: int) -> int:
        return self._attempts[lp]

    def fills(self, lp: int) -> int:
        return self._fills[lp]

    def rejects(self, lp: int) -> int:
        return self._rejects[lp]

    def lp_count(self) -> int:
        return self._lp_count
