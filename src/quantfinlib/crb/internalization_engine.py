"""The internalize-or-route decision (port of Java
``com.quantfinlib.crb.InternalizationEngine``).

The economics that justify a central risk book's existence. Every
internalized unit of flow saves the street's spread AND market impact
twice over (the client's execution and the eventual hedge), so:

- risk-REDUCING flow (opposite sign to the book's net) is internalized
  up to the offsetting inventory, and the client is given back a share
  of the saved spread as price improvement -- the book was going to pay
  to shed that risk anyway;
- risk-ADDING flow is warehoused (internalized without improvement)
  only while the post-trade inventory stays inside the warehouse limit
  -- beyond that it routes out, because a warehouse limit that yields
  to one more trade is not a limit.

Flows and exposures are in the SAME factor units. Sign convention
matches ``CentralRiskBook``: the flow is what the book ABSORBS if it
internalizes.

The Java ``persist.Checkpoint`` overnight persistence is not ported --
no ``persist`` lane in this Python port.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Decision:
    """Where one flow went, and what the client got for it."""

    internalized: float
    routed: float
    improvement_bps: float


class InternalizationEngine:

    def __init__(self, warehouse_limit: float, improvement_share: float):
        """
        warehouse_limit: max |inventory| the book will hold on a factor
            after absorbing risk-adding flow, > 0
        improvement_share: fraction of the half spread returned to a
            risk-reducing client, in [0, 1]
        """
        if not (warehouse_limit > 0) or warehouse_limit == math.inf:
            raise ValueError("warehouse_limit must be positive and finite")
        if not (0 <= improvement_share <= 1):
            raise ValueError("improvement_share must be in [0, 1]")
        self._warehouse_limit = warehouse_limit
        self._improvement_share = improvement_share
        self._internalized_notional = 0.0
        self._routed_notional = 0.0

    def decide(self, book_net: float, flow: float, half_spread_bps: float) -> Decision:
        """Decides one flow against the book's current net on that
        factor.

        book_net: the book's net exposure on the factor (signed)
        flow: exposure change the book absorbs if it internalizes
            (signed, non-zero)
        half_spread_bps: the street's half spread for this risk -- what
            internalizing saves, > 0
        """
        if not math.isfinite(book_net) or not math.isfinite(flow) or flow == 0:
            raise ValueError("book_net/flow must be finite, flow non-zero")
        if not (half_spread_bps > 0) or half_spread_bps == math.inf:
            raise ValueError("half_spread_bps must be positive and finite")
        improvement = 0.0
        if book_net != 0 and _sign(flow) != _sign(book_net):
            # Risk-reducing: cross against inventory, share the saved spread.
            reducing = min(abs(flow), abs(book_net))
            improvement = self._improvement_share * half_spread_bps
            # Whatever exceeds the offset flips the book's sign -- that
            # excess is risk-ADDING and faces the warehouse test below.
            excess = abs(flow) - reducing
            warehoused = min(excess, self._warehouse_limit)
            internalized = _sign(flow) * (reducing + warehoused)
            if warehoused > 0 and excess > 0:
                # Blended improvement: only the reducing portion earned it.
                improvement = improvement * reducing / (reducing + warehoused)
        else:
            # Risk-adding: warehouse only inside the limit.
            headroom = max(0.0, self._warehouse_limit - abs(book_net))
            internalized = _sign(flow) * min(abs(flow), headroom)
        routed = flow - internalized
        self._internalized_notional += abs(internalized)
        self._routed_notional += abs(routed)
        return Decision(internalized, routed, improvement)

    def internalization_rate(self) -> float:
        """Fraction of decided notional the book kept (0 before any
        flow)."""
        total = self._internalized_notional + self._routed_notional
        return 0.0 if total <= 0 else self._internalized_notional / total

    def internalized_notional(self) -> float:
        return self._internalized_notional

    def routed_notional(self) -> float:
        return self._routed_notional


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0
