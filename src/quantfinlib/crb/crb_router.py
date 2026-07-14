"""The central risk book's order router (port of Java
``com.quantfinlib.crb.CrbRouter``).

Internal cross first, dark pools second, lit last, each leg priced
honestly:

- Internal -- crossing against the book's own offsetting inventory
  costs ZERO bps and leaks nothing: the CRB itself is the firm's first
  and best dark pool. Capped at the crossable inventory the caller
  reports;
- Dark pools -- midpoint fills pay no spread, but a venue whose fills
  systematically fade is not free: each venue carries an
  ADVERSE-SELECTION charge in bps (a post-fill markout estimate), and
  expected liquidity is discounted by fill probability. A dark venue is
  only used while its charge undercuts the lit cost;
- Lit -- pays the half spread plus expected impact, but fills. Whatever
  the dark legs are not EXPECTED to fill routes lit as well -- hedges
  that might fill are not hedges.

Allocation is greedy by expected cost, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DarkVenue:
    """A dark venue as the router sees it."""

    name: str
    expected_liquidity: float
    fill_probability: float
    adverse_selection_bps: float

    def __post_init__(self):
        if self.name is None or self.name.strip() == "":
            raise ValueError("venue must be named")
        if not (self.expected_liquidity >= 0) or self.expected_liquidity == math.inf:
            raise ValueError("expected_liquidity must be >= 0 and finite")
        if not (0 <= self.fill_probability <= 1):
            raise ValueError("fill_probability must be in [0, 1]")
        if not math.isfinite(self.adverse_selection_bps) or self.adverse_selection_bps < 0:
            raise ValueError("adverse_selection_bps must be >= 0 and finite")


@dataclass(frozen=True, slots=True)
class Allocation:
    """Where the notional went. ``dark[i]`` aligns with the venue list
    passed in; ``expected_cost_bps`` is the blended expected cost of the
    whole allocation."""

    internal: float
    dark: list[float]
    lit: float
    expected_cost_bps: float


class CrbRouter:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def route(notional: float, crossable_internal: float, venues: list[DarkVenue],
             half_spread_bps: float, impact_bps: float) -> Allocation:
        """Routes ``notional`` (positive, in book-currency units).

        notional: amount to execute, > 0
        crossable_internal: offsetting book inventory available to
            cross against, >= 0
        venues: dark venues with honest statistics
        half_spread_bps: lit half spread, > 0
        impact_bps: expected lit impact for this size, >= 0
        """
        if not (notional > 0) or notional == math.inf:
            raise ValueError("notional must be positive and finite")
        if not (crossable_internal >= 0) or crossable_internal == math.inf:
            raise ValueError("crossable_internal must be >= 0 and finite")
        if not (half_spread_bps > 0) or half_spread_bps == math.inf:
            raise ValueError("half_spread_bps must be positive and finite")
        if not (impact_bps >= 0) or impact_bps == math.inf:
            raise ValueError("impact_bps must be >= 0 and finite")
        lit_cost = half_spread_bps + impact_bps

        remaining = notional
        internal = min(remaining, crossable_internal)
        remaining -= internal
        cost_weighted = 0.0                              # internal leg costs 0

        # Dark venues in ascending adverse-selection order, used only
        # while they undercut lit; expected fill = liquidity x fillProb.
        m = len(venues)
        dark = [0.0] * m
        order = sorted(range(m), key=lambda i: venues[i].adverse_selection_bps)
        for k in order:
            if remaining <= 0:
                break
            v = venues[k]
            if v.adverse_selection_bps >= lit_cost:
                break                                     # ordered: the rest are worse
            expected_fill = min(remaining, v.expected_liquidity * v.fill_probability)
            if expected_fill <= 0:
                continue
            dark[k] = expected_fill
            cost_weighted += expected_fill * v.adverse_selection_bps
            remaining -= expected_fill

        lit = remaining
        cost_weighted += lit * lit_cost
        return Allocation(internal, dark, lit, cost_weighted / notional)
