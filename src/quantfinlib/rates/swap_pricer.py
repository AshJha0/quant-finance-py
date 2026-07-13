"""Vanilla interest-rate swap pricing (port of Java ``com.quantfinlib.rates.SwapPricer``).

Priced off the ``YieldCurve`` — the missing middle between the curve
(which the bootstrap builds FROM par swaps) and ``RatesOptions`` (which
prices options ON forward swaps): the PV, par rate and DV01 of an
actual swap position.

Single-curve identities (annual fixed leg, matching the bootstrap's
convention):

    annuity    = sum DF(t_i)                    i = 1..T (annual, tau = 1)
    parRate    = (1 - DF(T)) / annuity          spot-starting
    payerPv    = annuity * (parRate - K)        pay fixed K, receive float
    receiverPv = -payerPv

The float leg needs no forecasting in a single-curve world: it is worth
par at inception, i.e. ``1 - DF(T)`` per unit notional — which is
exactly why the par rate has that closed form. DV01 is the
bump-and-reprice sensitivity to a parallel 1bp shift of the zero curve:
for a fresh par swap on a flat cc curve it is
``annuity * e^z * 1bp * notional`` (the tests pin exactly that; the
desk shorthand ``annuity * 1bp`` is the sensitivity to the SIMPLE par
rate, a different derivative, ~e^z away). A swap struck at the par rate
must PV to zero — an identity, tested at 1e-12.

Stated simplifications: single curve (no OIS/projection split), annual
fixed leg, spot start.
"""

from __future__ import annotations

import math

from quantfinlib.rates.yield_curve import YieldCurve


class SwapPricer:
    """Static single-curve swap analytics."""

    @staticmethod
    def annuity(curve: YieldCurve, tenor_years: int) -> float:
        """PV of the annual fixed-leg annuity, per unit notional."""
        SwapPricer._require_tenor(tenor_years)
        a = 0.0
        for i in range(1, tenor_years + 1):
            a += curve.discount_factor(i)
        return a

    @staticmethod
    def par_rate(curve: YieldCurve, tenor_years: int) -> float:
        """The spot-starting par swap rate for ``tenor_years``."""
        return ((1 - curve.discount_factor(tenor_years))
                / SwapPricer.annuity(curve, tenor_years))

    @staticmethod
    def payer_pv(curve: YieldCurve, tenor_years: int, fixed_rate: float) -> float:
        """PV per unit notional of a PAYER swap (pay fixed ``fixed_rate``,
        receive float). Negate for the receiver.
        """
        if not math.isfinite(fixed_rate):
            raise ValueError("fixedRate must be finite")
        return (SwapPricer.annuity(curve, tenor_years)
                * (SwapPricer.par_rate(curve, tenor_years) - fixed_rate))

    @staticmethod
    def dv01(curve: YieldCurve, tenor_years: int, fixed_rate: float) -> float:
        """DV01 per unit notional: the payer swap's PV change for a +1bp
        parallel shift of the zero curve (positive — rates up helps the
        fixed payer).
        """
        base = SwapPricer.payer_pv(curve, tenor_years, fixed_rate)
        bumped = SwapPricer.payer_pv(SwapPricer.parallel_bump(curve, 1e-4),
                                     tenor_years, fixed_rate)
        return bumped - base

    @staticmethod
    def parallel_bump(curve: YieldCurve, bump: float) -> YieldCurve:
        """A copy of the curve with every pillar zero rate shifted by ``bump``."""
        tenors = curve.tenors()
        zeros = [curve.zero_rate(t) + bump for t in tenors]
        return YieldCurve.of_zero_rates(tenors, zeros)

    @staticmethod
    def _require_tenor(tenor_years: int) -> None:
        if tenor_years < 1:
            raise ValueError(f"tenorYears must be >= 1, got {tenor_years}")
