"""Last-look-aware LP router (port of Java ``com.quantfinlib.fx.LpRouter``).

Chooses where to send an FX clip by EXPECTED all-in price, not
displayed price. The FX-specific insight it encodes: a tight quote from
an LP that rejects 20% of the time -- and whose rejects are followed by
adverse markout -- is more expensive than a wider firm quote. Expected
cost per LP::

    quoted price +/- reject_rate * max(post_reject_markout, 0)

(worse for the taker on either side), with LPs above a configurable
reject-rate cap vetoed outright. Quotes come from an ``FxTierBook``
(full-amount tiers), behavior from an ``LpScorecard``.

Full-amount only, one LP per clip -- the FX convention that avoids
spraying child orders. For deliberate multi-LP sweeps, use
``FxTierBook.sweep_plan`` directly and accept the signaling.
"""

from __future__ import annotations

import math

from quantfinlib.fx.fx_tier_book import FxTierBook
from quantfinlib.fx.lp_scorecard import LpScorecard


class LpRouter:
    """Wiring requirement: the scorecard's markout penalty only works if
    ``card.on_mid`` is fed composite mids on the same clock as
    ``on_reject`` -- without it markouts never mature, the penalty is
    silently zero, and routing degrades to displayed-price-plus-veto.
    Watch ``card.matured_markouts()`` in monitoring: zero while rejects
    accrue means the hook is missing.
    """

    def __init__(self, book: FxTierBook, card: LpScorecard, max_reject_rate: float,
                 hold_urgency_bps_per_ms: float = 0.0):
        """``max_reject_rate``: LPs whose EWMA reject rate exceeds this
        are vetoed regardless of price (e.g. 0.25). ``hold_urgency_bps_per_ms``:
        an LP's last-look hold is FX's latency dimension -- a positive
        value charges each LP's EWMA hold time against its quote (bps
        of price per millisecond held), so a slow-holding LP loses ties
        exactly like a high-latency venue does."""
        if book.lp_count() != card.lp_count():
            raise ValueError("book and scorecard LP counts differ")
        if max_reject_rate <= 0 or max_reject_rate > 1:
            raise ValueError("max_reject_rate must be in (0,1]")
        if hold_urgency_bps_per_ms < 0:
            raise ValueError("hold_urgency_bps_per_ms must be >= 0")
        self._book = book
        self._card = card
        self._max_reject_rate = max_reject_rate
        self._hold_urgency_bps_per_ms = hold_urgency_bps_per_ms
        self._last_quoted_price = math.nan
        self._last_expected_price = math.nan
        self._route_count = 0
        self._veto_count = 0

    def route(self, buy: bool, size: float) -> int:
        """Chooses the LP for a full-amount clip. Returns the LP index,
        or -1 when no eligible LP quotes the size (book too shallow or
        all vetoed). After a successful call, :meth:`last_quoted_price`
        is the LP's raw tier price and :meth:`last_expected_price` the
        reject-adjusted price the decision was made on."""
        self._route_count += 1
        self._last_quoted_price = math.nan
        self._last_expected_price = math.nan
        best_lp = -1
        best_expected = 0.0
        best_quoted = 0.0
        lps = self._book.lp_count()
        for lp in range(lps):
            rate = self._card.reject_rate(lp)
            if rate > self._max_reject_rate:
                self._veto_count += 1
                continue
            quoted = self._book.full_amount_price(lp, buy, size)
            penalty = rate * max(self._card.post_reject_markout(lp), 0.0)
            if self._hold_urgency_bps_per_ms > 0:
                # Hold time is priced like venue latency: bps/ms of quote.
                penalty += (quoted * (self._card.avg_hold_nanos(lp) / 1e6)
                           * self._hold_urgency_bps_per_ms / 1e4)
            expected = quoted + penalty if buy else quoted - penalty
            if not math.isfinite(expected):
                continue    # unquoted LP, or a poisoned stat: never routable
            if best_lp == -1 or (expected < best_expected if buy else expected > best_expected):
                best_lp = lp
                best_expected = expected
                best_quoted = quoted
        if best_lp >= 0:
            self._last_quoted_price = best_quoted
            self._last_expected_price = best_expected
        return best_lp

    def last_quoted_price(self) -> float:
        """Raw quoted price behind the last successful :meth:`route`;
        NaN otherwise."""
        return self._last_quoted_price

    def last_expected_price(self) -> float:
        """Reject-adjusted price behind the last successful
        :meth:`route`; NaN otherwise."""
        return self._last_expected_price

    def route_count(self) -> int:
        return self._route_count

    def veto_count(self) -> int:
        """LP-candidate evaluations skipped for exceeding the
        reject-rate cap."""
        return self._veto_count
