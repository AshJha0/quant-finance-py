"""The post-or-cross decision (port of Java
``execution.OrderPlacementPolicy``).

The smallest and most repeated choice in execution, made explicit as
expected-cost arithmetic instead of habit. Relative to the current mid,
for a buy::

    cross now:   pay the half spread                          -> h
    post at bid: filled (prob p):  earn h, pay adverse selection a,
                 collect the rebate r                          -> a - h - r
                 unfilled (1-p):   cross later after the
                 market drifted d against you                  -> h + d

    expected post cost = p*(a - h - r) + (1-p)*(h + d)
    POST iff that beats h

The inputs are the honest parts: ``fill_probability`` from a queue/fill
model, the adverse-selection cost ``a`` from post-fill markouts (a
passive fill happens exactly when the market comes THROUGH you -- free
money it is not), and the drift ``d`` from your alpha (positive =
expected to move against a waiting buyer). All in the same per-unit
currency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Placement:
    """The decision plus the arithmetic that made it."""

    post: bool
    expected_post_cost: float
    cross_cost: float


@dataclass(frozen=True, slots=True)
class PostRegion:
    """The fill-probability REGION where posting beats crossing.

    A single breakeven scalar cannot carry the answer:
    ``post_cost(p) = (h+d) - p*(2h+r+d-a)`` is linear in p, and when
    adverse selection exceeds ``2h+r+d`` the slope flips -- posting
    then pays only BELOW the threshold, not above. The region says it
    directly. Empty (:meth:`is_empty`) = never post under these
    conditions. Boundaries are indifference points (measure zero;
    :func:`decide` crosses on a tie).
    """

    from_: float
    to: float

    def is_empty(self) -> bool:
        return self.from_ > self.to

    def contains(self, fill_probability: float) -> bool:
        return self.from_ <= fill_probability <= self.to


_NEVER = PostRegion(1, 0)


def decide(half_spread: float, fill_probability: float, adverse_selection: float,
          adverse_drift: float, rebate: float) -> Placement:
    """
    Args:
        half_spread: half the touch spread, > 0.
        fill_probability: P(passive order fills within the horizon),
            in [0, 1].
        adverse_selection: expected cost WHEN passively filled
            (post-fill markout), >= 0.
        adverse_drift: expected move against the order while waiting
            unfilled (signed: negative = the market is expected to
            come to you).
        rebate: maker rebate per unit, >= 0.
    """
    if not (half_spread > 0) or half_spread == math.inf:
        raise ValueError("halfSpread must be positive and finite")
    if not (0 <= fill_probability <= 1):
        raise ValueError("fillProbability must be in [0, 1]")
    if not (adverse_selection >= 0) or adverse_selection == math.inf:
        raise ValueError("adverseSelection must be >= 0 and finite")
    if not math.isfinite(adverse_drift):
        raise ValueError("adverseDrift must be finite")
    if not (rebate >= 0) or rebate == math.inf:
        raise ValueError("rebate must be >= 0 and finite")
    post_cost = (fill_probability * (adverse_selection - half_spread - rebate)
                + (1 - fill_probability) * (half_spread + adverse_drift))
    return Placement(post_cost < half_spread, post_cost, half_spread)


def post_region(half_spread: float, adverse_selection: float, adverse_drift: float,
               rebate: float) -> PostRegion:
    """The desk's rule for these market conditions: post iff the fill
    probability lands inside the returned region of [0, 1]."""
    if not (half_spread > 0) or half_spread == math.inf:
        raise ValueError("halfSpread must be positive and finite")
    if not (adverse_selection >= 0) or adverse_selection == math.inf:
        raise ValueError("adverseSelection must be >= 0 and finite")
    if not math.isfinite(adverse_drift):
        raise ValueError("adverseDrift must be finite")
    if not (rebate >= 0) or rebate == math.inf:
        raise ValueError("rebate must be >= 0 and finite")
    # post_cost(p) = (h + d) - p*coef;  post iff post_cost < h  <=>  d < p*coef.
    coef = 2 * half_spread + rebate + adverse_drift - adverse_selection
    if coef > 0:
        p_star = adverse_drift / coef
        return _NEVER if p_star >= 1 else PostRegion(max(0.0, p_star), 1)
    if coef < 0:
        # The slope flipped: posting pays only BELOW the threshold.
        p_star = adverse_drift / coef
        return _NEVER if p_star <= 0 else PostRegion(0, min(1.0, p_star))
    # Flat in p: the sign of the drift decides for every p at once.
    return PostRegion(0, 1) if adverse_drift < 0 else _NEVER
