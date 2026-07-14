"""Implied FX forward curve (port of Java
``com.quantfinlib.pricing.ForwardCurve``).

Construction from market outright forwards, with interpolation, implied
rate differentials, and covered-interest-parity arbitrage checks against
deposit rates. The Java TreeMap floor/ceiling lookups are reproduced
with bisect over a sorted tenor list.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort


class ForwardCurve:
    """Tenor -> outright forward curve anchored at spot (tenor 0)."""

    def __init__(self, spot: float) -> None:
        if spot <= 0:
            raise ValueError("spot must be positive")
        self._spot = spot
        self._tenors: list[float] = [0.0]
        self._outrights: dict[float, float] = {0.0: spot}

    def add_point(self, tenor_years: float, outright_forward: float) -> "ForwardCurve":
        if tenor_years <= 0:
            raise ValueError("tenor must be positive")
        if tenor_years not in self._outrights:
            insort(self._tenors, tenor_years)
        self._outrights[tenor_years] = outright_forward
        return self

    def spot(self) -> float:
        return self._spot

    def forward(self, tenor_years: float) -> float:
        """Interpolated outright forward at the tenor (linear in forward
        points between pillars; flat-slope extrapolation beyond the last
        pillar)."""
        if tenor_years <= 0:
            return self._spot
        i_hi = bisect_left(self._tenors, tenor_years)
        if i_hi == len(self._tenors):
            # Extrapolate using the slope of the last two pillars.
            last = self._tenors[-1]
            prev = self._tenors[-2]
            slope = ((self._outrights[last] - self._outrights[prev]) / (last - prev))
            return self._outrights[last] + slope * (tenor_years - last)
        i_lo = bisect_right(self._tenors, tenor_years) - 1
        lo = self._tenors[i_lo]
        hi = self._tenors[i_hi]
        if lo == hi:
            return self._outrights[lo]
        w = (tenor_years - lo) / (hi - lo)
        return self._outrights[lo] + w * (self._outrights[hi] - self._outrights[lo])

    def forward_points(self, tenor_years: float) -> float:
        """Forward points at the tenor (outright minus spot)."""
        return self.forward(tenor_years) - self._spot

    def implied_rate_differential(self, tenor_years: float) -> float:
        """Implied continuously-compounded rate differential (domestic minus
        foreign) from covered interest parity: ``F = S e^{(rd - rf) t}``.

        Convention note — this method is CONTINUOUS while
        ``theoretical_forward`` uses SIMPLE deposit rates, because each
        matches how its own input is quoted (a differential is usually
        consumed in cc form; deposits are quoted simple). Feeding this
        output back through ``theoretical_forward`` therefore shows a
        spurious ~12bp "basis" at 1y/5% that is pure compounding
        convention, not arbitrage — convert first.
        """
        return math.log(self.forward(tenor_years) / self._spot) / tenor_years

    @staticmethod
    def theoretical_forward(spot: float, domestic_rate: float, foreign_rate: float,
                            tenor_years: float) -> float:
        """CIP-theoretical forward from SIMPLE deposit rates (see the
        convention note on ``implied_rate_differential``)."""
        return spot * (1 + domestic_rate * tenor_years) / (1 + foreign_rate * tenor_years)

    def mispricing_bps(self, tenor_years: float, domestic_rate: float,
                       foreign_rate: float) -> float:
        """Covered-interest-parity arbitrage check: market forward versus the
        deposit-implied forward, in basis points (positive = market forward
        rich)."""
        theoretical = ForwardCurve.theoretical_forward(self._spot, domestic_rate,
                                                       foreign_rate, tenor_years)
        return (self.forward(tenor_years) - theoretical) / theoretical * 1e4
