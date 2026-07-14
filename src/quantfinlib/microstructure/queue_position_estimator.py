"""Queue position estimation from L2 data (port of Java
``microstructure.QueuePositionEstimator``) -- for when you don't have
an L3 feed to track position exactly. With only aggregated level sizes
you can't see individual orders, so this estimator maintains a
probabilistic band on where a passive order sits, updated with the
standard assumptions:

* **On join** -- you rest at the back: shares-ahead = the level's
  current displayed size;
* **Executions at your level** -- trades hit the front (price-time
  priority), so they reduce shares-ahead one-for-one;
* **Cancels at your level** -- the hard part: a decrease in level size
  that isn't a trade is a cancel, and it could be ahead of you or
  behind. The literature's workable assumption is **pro-rata**: a
  cancel removes shares-ahead in proportion to the fraction of the
  queue that is ahead of you. That gives an unbiased estimate without
  L3 -- the ``ahead`` field is the expected value, and
  :meth:`fill_probability` turns it into a fill likelihood via
  :class:`~quantfinlib.microstructure.queue_model.QueueModel`.

One passive order per instance (cheap -- make one per working child).
**Feed ordering contract**: report each trade via :meth:`on_trade`
BEFORE the depth update that reflects it, and give
:meth:`on_level_resize` sizes net of trades already reported --
otherwise the same execution is counted once as a trade and again as a
cancel, and shares-ahead falls twice per fill.

Cross-asset: applies to any price-time-priority level (equity exchange
books and FX ECN/matching books alike).
"""

from __future__ import annotations

import math

from quantfinlib.microstructure.queue_model import QueueModel


class QueuePositionEstimator:
    """Pro-rata shares-ahead estimator for one resting order; see the
    module docstring."""

    __slots__ = ("_ahead", "_ahead_at_join", "_level_size", "_own_qty",
                 "_active")

    def __init__(self) -> None:
        self._ahead = 0.0          # expected shares ahead of us
        self._ahead_at_join = 0.0  # shares ahead when we joined (progress base)
        self._level_size = 0.0     # current displayed size at our level
        self._own_qty = 0
        self._active = False

    def join(self, level_size: float, own_qty: float) -> None:
        """Join the back of a level currently displaying
        ``level_size`` shares (before our order is added)."""
        if level_size < 0 or own_qty <= 0:
            raise ValueError("need levelSize >= 0, ownQty > 0")
        self._ahead = level_size
        self._ahead_at_join = level_size
        self._level_size = level_size + own_qty
        self._own_qty = own_qty
        self._active = True

    def on_trade(self, traded_qty: float) -> None:
        """A trade executed at our level: it consumed ``traded_qty``
        from the front, so shares-ahead drops by that much (clamped at
        0 -- once the front reaches us we start filling)."""
        if not self._active or traded_qty <= 0:
            return
        self._ahead = max(0.0, self._ahead - traded_qty)
        self._level_size = max(self._own_qty, self._level_size - traded_qty)

    def on_level_resize(self, new_level_size: float) -> None:
        """The level's displayed size changed to ``new_level_size``
        for a reason other than a trade -- i.e. cancels (net of any
        adds behind us). The removed quantity is attributed pro-rata:
        the fraction of the queue ahead of us is
        ``ahead / (level_size - own_qty)``, so that fraction of the
        cancel came from ahead."""
        if not self._active:
            return
        removed = (self._level_size - self._own_qty
                  - max(0.0, new_level_size - self._own_qty))
        # Only cancels (size shrank among the OTHER orders) move our
        # estimate; adds land behind us and don't change shares-ahead.
        if removed > 0:
            others = self._level_size - self._own_qty
            frac_ahead = self._ahead / others if others > 0 else 0.0
            self._ahead = max(0.0, self._ahead - removed * frac_ahead)
        self._level_size = max(self._own_qty, new_level_size)

    def shares_ahead(self) -> float:
        """Expected shares ahead of our order right now."""
        return self._ahead

    def fill_probability(self, expected_traded_qty: float) -> float:
        """Fill probability over a horizon in which
        ``expected_traded_qty`` shares are expected to execute at this
        level."""
        # Java Math.round: half-up (floor(x + 0.5)), not Python's
        # banker's-rounding round() -- the pro-rata estimate can land
        # exactly on a half share (e.g. ahead=5, others=10 -> a cancel
        # of 5 removes exactly 2.5), where round-half-to-even and
        # round-half-up disagree.
        return QueueModel.fill_probability(math.floor(self._ahead + 0.5),
                                           self._own_qty, expected_traded_qty)

    def queue_progress(self) -> float:
        """Queue progress since joining: 0 right after :meth:`join`, 1
        when the whole queue that was ahead of us has drained.
        Measured against the shares-ahead AT JOIN -- the only
        meaningful baseline."""
        if self._ahead_at_join <= 0:
            return 1.0
        return max(0.0, 1 - self._ahead / self._ahead_at_join)

    def active(self) -> bool:
        return self._active

    def own_qty(self) -> float:
        return self._own_qty

    def close(self) -> None:
        """Order left the book (filled/cancelled)."""
        self._active = False
