"""Credit curve (port of Java ``com.quantfinlib.credit.CreditCurve``).

Piecewise-constant hazard rates bootstrapped from CDS par spreads, the
credit market's exact analogue of ``YieldCurve``'s bootstrap: walk the
quotes from shortest to longest, at each pillar solving for the one
hazard rate that reprices that maturity's CDS to zero upfront given
everything already solved.

The objects: the HAZARD RATE h(t) is the instantaneous default
intensity ("conditional on surviving to t, the annualized probability
of defaulting right now"); SURVIVAL is its exponential integral,
``Q(t) = exp(-integral of h)`` — piecewise-constant h makes that a
product of exponentials, evaluated exactly. The rule-of-thumb every
desk carries — the CREDIT TRIANGLE ``spread ~ h * (1 - R)`` — falls out
of setting premium = protection on a flat curve, and the tests pin this
class against it.

Leg discretization (stated): quarterly grid, premium leg
``S * sum 0.25 * DF(t_i) * Q(t_i)`` plus the standard
accrual-on-default half-period term, protection leg
``(1-R) * sum DF(t_i) * (Q(t_{i-1}) - Q(t_i))`` with discounting at
period end — the textbook discrete form (O(dt) bias vs the integral,
~0.1bp at these grids, stated not hidden). Recovery is a single number
for the whole curve, the market's quoting convention (40% senior
unsecured). Solving is bisection per pillar on h in [1e-9, 10] with an
explicit bracket check — a quote no hazard can explain raises rather
than returning the bound.
"""

from __future__ import annotations

import math

from quantfinlib.rates.yield_curve import YieldCurve

_DT = 0.25  # the quarterly grid step shared with CdsPricer


class CreditCurve:
    """Piecewise-constant hazard curve with a CDS par-spread bootstrap."""

    def __init__(self, pillar_times: list[float], hazards: list[float],
                 recovery: float):
        # Internal: use the ``bootstrap`` factory.
        self._pillar_times = pillar_times   # ascending, years
        self._hazards = hazards             # piecewise-constant on (prev, pillar]
        self._recovery = recovery

    @staticmethod
    def bootstrap(tenor_years, par_spreads, recovery: float,
                  discount: YieldCurve) -> "CreditCurve":
        """Bootstraps from CDS par spreads.

        Args:
            tenor_years: ascending integer-year pillars, >= 1.
            par_spreads: par CDS spreads (decimal: 0.01 = 100bp), > 0.
            recovery: assumed recovery rate in [0, 1).
            discount: risk-free discounting curve.
        """
        # Local import: CdsPricer prices off this curve and this bootstrap
        # prices with CdsPricer — the Java circularity, resolved lazily here.
        from quantfinlib.credit.cds_pricer import CdsPricer

        n = len(tenor_years)
        if n == 0 or len(par_spreads) != n:
            raise ValueError("need aligned, non-empty tenors/spreads")
        if not (recovery >= 0) or not (recovery < 1):
            raise ValueError(f"recovery must be in [0, 1), got {recovery}")
        prev = 0
        for i in range(n):
            if tenor_years[i] <= prev:
                raise ValueError("tenors must be ascending positive integers")
            prev = tenor_years[i]
            if not (par_spreads[i] > 0) or par_spreads[i] == math.inf:
                raise ValueError(
                    f"spread must be positive and finite: {par_spreads[i]}")
        times = [float(t) for t in tenor_years]
        hazards = [0.0] * n
        curve = CreditCurve(times, hazards, recovery)
        for k in range(n):
            spread = par_spreads[k]
            maturity = times[k]
            lo, hi = 1e-9, 10.0
            up_lo = curve._upfront_with_pillar(k, lo, spread, maturity, discount,
                                               CdsPricer)
            up_hi = curve._upfront_with_pillar(k, hi, spread, maturity, discount,
                                               CdsPricer)
            if up_lo * up_hi > 0:
                raise ValueError(f"pillar {tenor_years[k]}y spread {spread}"
                                 " has no hazard in [1e-9, 10]")
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                up = curve._upfront_with_pillar(k, mid, spread, maturity,
                                                discount, CdsPricer)
                if up * up_lo > 0:
                    lo = mid
                    up_lo = up
                else:
                    hi = mid
            hazards[k] = 0.5 * (lo + hi)
        return curve

    def _upfront_with_pillar(self, k: int, trial_hazard: float, spread: float,
                             maturity: float, discount: YieldCurve,
                             cds_pricer) -> float:
        saved = self._hazards[k]
        self._hazards[k] = trial_hazard
        up = (cds_pricer.protection_leg_pv(self, discount, maturity)
              - spread * cds_pricer.risky_annuity(self, discount, maturity))
        self._hazards[k] = saved
        return up

    def survival_probability(self, t: float) -> float:
        """Survival probability Q(t), exact under piecewise-constant hazards."""
        if not (t >= 0) or t == math.inf:
            raise ValueError(f"t must be >= 0 and finite, got {t}")
        integral = 0.0
        from_t = 0.0
        for i in range(len(self._pillar_times)):
            if from_t >= t:
                break
            to_t = min(self._pillar_times[i], t)
            if to_t > from_t:
                integral += self._hazards[i] * (to_t - from_t)
                from_t = to_t
        if from_t < t:  # flat extrapolation beyond the last pillar
            integral += self._hazards[-1] * (t - from_t)
        return math.exp(-integral)

    def default_probability(self, t: float) -> float:
        """Cumulative default probability 1 - Q(t)."""
        return 1 - self.survival_probability(t)

    def hazard(self, t: float) -> float:
        """The hazard rate in force at time t (flat beyond the last pillar)."""
        if not (t >= 0):
            raise ValueError(f"t must be >= 0, got {t}")
        for i in range(len(self._pillar_times)):
            if t <= self._pillar_times[i]:
                return self._hazards[i]
        return self._hazards[-1]

    def recovery(self) -> float:
        return self._recovery

    @staticmethod
    def grid_step() -> float:
        """The quarterly grid step shared with ``CdsPricer``."""
        return _DT
