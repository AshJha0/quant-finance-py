"""Commodity futures curve (port of Java ``com.quantfinlib.commodities.CommodityCurve``).

Where the P&L of a commodity position mostly does NOT come from being
right about the spot price. A futures curve in CONTANGO (upward:
deferred contracts above spot) charges a long position negative ROLL
YIELD every month — selling the expiring cheap contract to buy the
deferred rich one — while BACKWARDATION (downward) pays the long for
rolling. Over a decade this roll term has dominated most commodity
index returns, which is the single most misunderstood fact about the
asset class (the USO oil fund's 2020 investors learned it the hard way:
spot oil recovered, the contango roll ate the fund anyway).

The numbers this class produces:

* annualized roll yield between two tenors:
  ``ln(F(near)/F(far)) / (far - near)`` — positive in backwardation
  (near above far: rolling down the curve pays the long);
* implied carry versus spot: from the storage-arbitrage relation
  ``F = S * exp((r + u - y) * t)``, the market-implied ``u - y``
  (storage cost minus convenience yield) is ``ln(F(t)/S)/t - r``. A
  deeply negative value means the market pays dearly to HOLD the
  physical (convenience yield — think heating oil before a cold snap);
* shape tests: ``is_contango()/is_backwardation()`` across the whole
  curve, strict at every adjacent pillar pair.

Linear interpolation between pillar prices, no extrapolation (asking
for a price beyond the pillars raises — a commodity curve's wings are
opinions, not data). Seasonality (natural gas winters) makes
whole-curve shape tests false for seasonal commodities by design — use
pairwise roll yields there, stated.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right


class CommodityCurve:
    """Futures curve over a spot with strict-shape tests and roll analytics."""

    def __init__(self, spot: float, tenors: list[float], prices: list[float]):
        # Internal: use the ``of`` factory.
        self._spot = spot
        self._tenors = tenors     # ascending
        self._prices = prices     # aligned futures prices

    @staticmethod
    def of(spot: float, tenor_years, prices) -> "CommodityCurve":
        """Builds the curve.

        Args:
            spot: spot price, > 0.
            tenor_years: ascending futures tenors, all > 0.
            prices: futures prices per tenor, all > 0.
        """
        if not (spot > 0) or spot == math.inf:
            raise ValueError(f"spot must be positive and finite, got {spot}")
        n = len(tenor_years)
        if n == 0 or len(prices) != n:
            raise ValueError("need aligned, non-empty tenors/prices")
        tenors_out: list[float] = []
        prices_out: list[float] = []
        prev = 0.0
        for i in range(n):
            if not (tenor_years[i] > prev) or tenor_years[i] == math.inf:
                raise ValueError("tenors must be ascending, positive, finite")
            if not (prices[i] > 0) or prices[i] == math.inf:
                raise ValueError(f"price must be positive and finite: {prices[i]}")
            tenors_out.append(float(tenor_years[i]))
            prices_out.append(float(prices[i]))
            prev = tenor_years[i]
        return CommodityCurve(spot, tenors_out, prices_out)

    def price(self, tenor_years: float) -> float:
        """Interpolated futures price; raises beyond the pillars (no extrapolation)."""
        keys, vals = self._tenors, self._prices
        i = bisect_right(keys, tenor_years) - 1   # floor
        j = bisect_left(keys, tenor_years)        # ceiling
        if j < len(keys) and keys[j] == tenor_years:
            return vals[j]                        # exact pillar
        if i < 0 or j >= len(keys):
            raise ValueError(f"tenor {tenor_years} outside pillars "
                             f"[{keys[0]}, {keys[-1]}]")
        w = (tenor_years - keys[i]) / (keys[j] - keys[i])
        return vals[i] + w * (vals[j] - vals[i])

    def annualized_roll_yield(self, near_years: float, far_years: float) -> float:
        """Annualized roll yield earned by a LONG rolling from ``near_years``
        to ``far_years``: positive in backwardation.
        """
        if not (far_years > near_years):
            raise ValueError("farYears must exceed nearYears")
        return (math.log(self.price(near_years) / self.price(far_years))
                / (far_years - near_years))

    def implied_carry(self, tenor_years: float, rate: float) -> float:
        """Market-implied storage-minus-convenience ``u - y`` (cc) at the
        tenor, from ``F = S * exp((r + u - y) t)``.

        Args:
            rate: the cc risk-free rate to the tenor.
        """
        if not math.isfinite(rate):
            raise ValueError("rate must be finite")
        return math.log(self.price(tenor_years) / self._spot) / tenor_years - rate

    def is_contango(self) -> bool:
        """Strictly upward at every adjacent pillar pair (deferred above near)."""
        prev = self._spot
        for px in self._prices:
            if px <= prev:
                return False
            prev = px
        return True

    def is_backwardation(self) -> bool:
        """Strictly downward at every adjacent pillar pair."""
        prev = self._spot
        for px in self._prices:
            if px >= prev:
                return False
            prev = px
        return True

    def spot(self) -> float:
        return self._spot
