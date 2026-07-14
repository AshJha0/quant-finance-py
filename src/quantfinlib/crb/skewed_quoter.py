"""Inventory-skewed two-way pricing (port of Java
``com.quantfinlib.crb.SkewedQuoter``).

The central risk book's quoting face. A CRB long inventory shades BOTH
quotes down: the ask gets more attractive (sell what we hold), the bid
less (stop accumulating). That is the Avellaneda-Stoikov
reservation-price intuition applied at book level, without the
vol/horizon machinery: skew is linear in inventory as a fraction of the
inventory limit, capped so the quote NEVER crosses itself::

    skew = -(inventory/limit)*skew_fraction*half_spread   (clamped +/-1)
    bid  = mid*(1 + (-half_spread + skew)/1e4)
    ask  = mid*(1 + (+half_spread + skew)/1e4)

``skew_fraction`` in [0, 1): at 1 a full-limit inventory would quote a
zero-width side, so construction stops just short.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantfinlib.util import math_utils as mu


@dataclass(frozen=True, slots=True)
class Quote:
    """A shaded two-way price."""

    bid: float
    ask: float
    skew_bps: float


class SkewedQuoter:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def quote(mid: float, half_spread_bps: float, inventory: float,
             inventory_limit: float, skew_fraction: float) -> Quote:
        """
        mid: current fair value, > 0
        half_spread_bps: unskewed half spread in bps, > 0
        inventory: book inventory in the quoted factor's units (signed)
        inventory_limit: inventory limit, > 0 (same units)
        skew_fraction: how much of the half spread a full-limit
            inventory shades, in [0, 1)
        """
        if not (mid > 0) or mid == math.inf:
            raise ValueError("mid must be positive and finite")
        if not (half_spread_bps > 0) or half_spread_bps == math.inf:
            raise ValueError("half_spread_bps must be positive and finite")
        if not (0 <= skew_fraction < 1):
            raise ValueError("skew_fraction must be in [0, 1)")
        # The worst-case downward shade is half_spread*(1 + skew_fraction):
        # past 10,000 bps that quotes a ZERO or NEGATIVE bid -- no market
        # this class serves has 100% half spreads, so reject loudly.
        if half_spread_bps * (1 + skew_fraction) >= 10_000:
            raise ValueError(
                f"half_spread_bps {half_spread_bps} with skew_fraction "
                f"{skew_fraction} could quote a non-positive bid")
        if not (inventory_limit > 0) or inventory_limit == math.inf:
            raise ValueError("inventory_limit must be positive and finite")
        if not math.isfinite(inventory):
            raise ValueError("inventory must be finite")
        load = mu.clamp(inventory / inventory_limit, -1, 1)
        skew_bps = -load * skew_fraction * half_spread_bps
        bid = mid * (1 + (-half_spread_bps + skew_bps) / 1e4)
        ask = mid * (1 + (half_spread_bps + skew_bps) / 1e4)
        return Quote(bid, ask, skew_bps)
