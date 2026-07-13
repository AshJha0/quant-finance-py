"""Zero-coupon yield curve (port of Java ``com.quantfinlib.rates.YieldCurve``).

The single most load-bearing object in fixed income: every bond price,
swap value, forward rate and DV01 is a function of it. The curve stores
continuously-compounded ZERO rates at pillar tenors; everything else is
derived: the discount factor ``DF(t) = e^{-z(t) * t}`` answers "what is
1 unit at time t worth today", and the implied forward between t1 and
t2 falls out of the ratio of discount factors — the market's own
break-even rate for that future period, no forecast involved.

You rarely observe zero rates directly; the market quotes PAR
instruments (deposits, swaps). Bootstrapping walks the quotes from
shortest to longest, at each pillar solving for the one discount factor
that reprices the quote given the factors already solved — for annual
par swaps, ``DF_n = (1 - parRate_n * A_{n-1}) / (1 + parRate_n)`` with
``A`` the annuity so far. The result reprices every input exactly
(tested), which is the definition of a usable curve.

Model choices, stated: linear interpolation ON ZERO RATES (simple,
fast, and the standard first choice; its known wart is small
forward-rate kinks at pillars — smoothing splines fix that at the cost
of locality), flat extrapolation beyond the pillars, and one curve for
both discounting and projection (pre-2008 style; a multi-curve
OIS/projection split is a composition of two of these).

Python port note: the Java TreeMap becomes a dict built key-by-key
(duplicate tenors: last one wins, matching ``TreeMap.put``) plus a
sorted key list searched with ``bisect`` for floor/ceiling lookups.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right


class YieldCurve:
    """Zero curve with linear interpolation and flat extrapolation."""

    def __init__(self, zeros: dict[float, float]):
        # Internal: use the factories of_zero_rates / bootstrap_annual_par_swaps.
        self._keys = sorted(zeros)                       # ascending tenors
        self._vals = [zeros[k] for k in self._keys]      # aligned cc zero rates

    @staticmethod
    def of_zero_rates(tenor_years, zero_rates_cc) -> "YieldCurve":
        """Curve from parallel arrays of tenors (years) and continuous zero rates."""
        if len(tenor_years) != len(zero_rates_cc) or len(tenor_years) == 0:
            raise ValueError("need equal-length, non-empty tenor/rate arrays")
        zeros: dict[float, float] = {}
        for t, z in zip(tenor_years, zero_rates_cc):
            if t <= 0:
                raise ValueError(f"tenor must be positive: {t}")
            zeros[float(t)] = float(z)
        return YieldCurve(zeros)

    @staticmethod
    def bootstrap_annual_par_swaps(tenor_years, par_rates) -> "YieldCurve":
        """Classic bootstrap from par swap rates with an annual fixed leg at
        integer-year pillars (missing years are filled by linear interpolation
        of the par rates): ``DF_n = (1 - parRate_n * A_{n-1}) / (1 + parRate_n)``.
        """
        n = len(tenor_years)
        if n != len(par_rates) or n == 0:
            raise ValueError("need equal-length, non-empty tenor/rate arrays")
        max_year = int(tenor_years[-1])
        # Interpolate par rates for every year 1..max_year.
        par = [0.0] * (max_year + 1)
        k = 0
        for y in range(1, max_year + 1):
            while k < n - 1 and tenor_years[k + 1] < y:
                k += 1
            if y <= tenor_years[0]:
                par[y] = par_rates[0]
            elif y >= tenor_years[-1]:
                par[y] = par_rates[-1]
            else:
                lo, hi = k, k + 1
                w = (y - tenor_years[lo]) / (tenor_years[hi] - tenor_years[lo])
                par[y] = par_rates[lo] + w * (par_rates[hi] - par_rates[lo])
        zeros: dict[float, float] = {}
        annuity = 0.0
        for y in range(1, max_year + 1):
            df = (1 - par[y] * annuity) / (1 + par[y])
            if df <= 0:
                raise ValueError(f"bootstrap produced non-positive DF at year {y}")
            annuity += df
            zeros[float(y)] = -math.log(df) / y
        return YieldCurve(zeros)

    def zero_rate(self, tenor_years: float) -> float:
        """Continuously-compounded zero rate (linear interpolation, flat extrapolation)."""
        keys, vals = self._keys, self._vals
        i = bisect_right(keys, tenor_years) - 1   # floor: largest key <= t
        j = bisect_left(keys, tenor_years)        # ceiling: smallest key >= t
        if i < 0:
            return vals[j]                        # flat short end
        if j >= len(keys):
            return vals[i]                        # flat long end
        if keys[i] == keys[j]:
            return vals[i]                        # exact pillar
        w = (tenor_years - keys[i]) / (keys[j] - keys[i])
        return vals[i] + w * (vals[j] - vals[i])

    def discount_factor(self, tenor_years: float) -> float:
        if tenor_years <= 0:
            return 1.0
        return math.exp(-self.zero_rate(tenor_years) * tenor_years)

    def forward_rate(self, from_years: float, to_years: float) -> float:
        """Implied continuously-compounded forward rate between two tenors."""
        if to_years <= from_years:
            raise ValueError("toYears must exceed fromYears")
        z1 = self.zero_rate(from_years) * from_years
        z2 = self.zero_rate(to_years) * to_years
        return (z2 - z1) / (to_years - from_years)

    def tenors(self) -> list[float]:
        """Pillar tenors, ascending (a copy — the curve stays immutable)."""
        return list(self._keys)
