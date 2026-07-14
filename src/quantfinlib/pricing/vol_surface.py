"""Implied volatility surface from pillar quotes (port of Java
``com.quantfinlib.pricing.VolSurface``).

Interpolation follows market practice:

* **Within a smile** — linear in vol across strikes, flat extrapolation
  beyond the quoted wings.
* **Across expiries** — linear in *total variance* (``w = vol^2 * T``)
  at fixed strike, which keeps the interpolated term structure
  calendar-consistent; flat vol extrapolation outside the quoted expiry
  range.

Immutable once built. The Java TreeMap floor/ceiling lookups are
reproduced with bisect over sorted key lists.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType


def _floor_key(keys: list[float], x: float) -> float | None:
    """Largest key <= x (Java TreeMap.floorKey)."""
    i = bisect_right(keys, x)
    return keys[i - 1] if i > 0 else None


def _ceiling_key(keys: list[float], x: float) -> float | None:
    """Smallest key >= x (Java TreeMap.ceilingKey)."""
    i = bisect_left(keys, x)
    return keys[i] if i < len(keys) else None


def _smile_vol(strikes: list[float], smile: dict[float, float], strike: float) -> float:
    """Linear in strike inside the smile; flat beyond the quoted wings."""
    lo = _floor_key(strikes, strike)
    hi = _ceiling_key(strikes, strike)
    if lo is None:
        return smile[hi]
    if hi is None:
        return smile[lo]
    if lo == hi:
        return smile[lo]
    w = (strike - lo) / (hi - lo)
    return smile[lo] + w * (smile[hi] - smile[lo])


class VolSurface:
    """Expiry -> (strike -> vol) pillar surface; build with ``VolSurface.builder()``."""

    def __init__(self, smiles: dict[float, dict[float, float]]) -> None:
        # Private in spirit: use the Builder.
        self._smiles = {t: dict(s) for t, s in sorted(smiles.items())}
        self._expiries = sorted(self._smiles)
        self._strikes = {t: sorted(s) for t, s in self._smiles.items()}

    @staticmethod
    def builder() -> "Builder":
        return Builder()

    def vol(self, expiry_years: float, strike: float) -> float:
        """Interpolated implied volatility at any (expiry, strike)."""
        lo = _floor_key(self._expiries, expiry_years)
        hi = _ceiling_key(self._expiries, expiry_years)
        if lo is None:
            return _smile_vol(self._strikes[hi], self._smiles[hi], strike)  # before first: flat
        if hi is None:
            return _smile_vol(self._strikes[lo], self._smiles[lo], strike)  # beyond last: flat
        if lo == hi:
            return _smile_vol(self._strikes[lo], self._smiles[lo], strike)
        t1, t2 = lo, hi
        v1 = _smile_vol(self._strikes[lo], self._smiles[lo], strike)
        v2 = _smile_vol(self._strikes[hi], self._smiles[hi], strike)
        w1 = v1 * v1 * t1
        w2 = v2 * v2 * t2
        w = w1 + (w2 - w1) * (expiry_years - t1) / (t2 - t1)
        return math.sqrt(w / expiry_years)

    def atm_vol(self, expiry_years: float, forward: float) -> float:
        """ATM vol, taking the forward (or spot) as the at-the-money strike."""
        return self.vol(expiry_years, forward)

    def price(self, option_type: OptionType, spot: float, strike: float,
              rate: float, carry: float, expiry_years: float) -> float:
        """Option price using the surface vol at (expiry, strike)."""
        return BlackScholes.price(option_type, spot, strike, rate, carry,
                                  self.vol(expiry_years, strike), expiry_years)

    def skew(self, expiry_years: float, strike_low: float, strike_high: float) -> float:
        """Smile slope between two strikes, in vol points per unit of strike."""
        return ((self.vol(expiry_years, strike_high) - self.vol(expiry_years, strike_low))
                / (strike_high - strike_low))

    def expiries(self) -> list[float]:
        return list(self._expiries)

    def strikes(self, expiry_years: float) -> list[float]:
        if expiry_years not in self._smiles:
            raise ValueError(f"no pillar expiry {expiry_years}")
        return list(self._strikes[expiry_years])


class Builder:
    """Accumulates pillar quotes; reusable after ``build`` (deep copy)."""

    def __init__(self) -> None:
        self._smiles: dict[float, dict[float, float]] = {}

    def add(self, expiry_years: float, strike: float, vol: float) -> "Builder":
        """Adds one pillar quote."""
        # not (x > 0) rejects NaN too: implied_vol returns NaN for
        # unattainable prices, and a NaN pillar poisons every
        # neighboring strike through interpolation.
        if (not (expiry_years > 0) or not (strike > 0) or not (vol > 0)
                or vol == math.inf):
            raise ValueError("expiry, strike and vol must be positive: "
                             f"{expiry_years}/{strike}/{vol}")
        self._smiles.setdefault(expiry_years, {})[strike] = vol
        return self

    def add_from_price(self, option_type: OptionType, market_price: float, spot: float,
                       strike: float, rate: float, carry: float,
                       expiry_years: float) -> "Builder":
        """Adds a pillar from a market option price via implied-vol inversion."""
        return self.add(expiry_years, strike,
                        BlackScholes.implied_vol(option_type, market_price, spot, strike,
                                                 rate, carry, expiry_years))

    def build(self) -> VolSurface:
        if not self._smiles:
            raise RuntimeError("no pillar quotes")
        return VolSurface(self._smiles)
