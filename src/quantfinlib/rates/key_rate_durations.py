"""Key-rate durations (port of Java ``com.quantfinlib.rates.KeyRateDurations``).

WHERE on the curve a bond's rate risk lives. ``BondPricer.dv01``
answers "what does one parallel basis point cost?"; a curve rarely
moves in parallel, so risk desks slice that DV01 across the curve's
tenors: bump ONE node, hold the rest, reprice. The vector of per-node
sensitivities is the hedging recipe (each KRD is offset with the
instrument that drives that node), and its sum recovers the parallel
duration — a consistency check the tests pin.

Mechanics: for each curve tenor, the zero rate at that node is bumped
+/-1bp (all other nodes fixed — the curve's own interpolation spreads
the bump between neighbors, which IS the standard convention), the bond
is repriced off each bumped curve, and the central difference gives the
sensitivity. Prices per 100 face follow the ``BondPricer`` convention.
"""

from __future__ import annotations

from quantfinlib.rates.bond_pricer import BondPricer
from quantfinlib.rates.yield_curve import YieldCurve

_BUMP = 1e-4    # one basis point


class KeyRateDurations:
    """Static per-node bump-and-reprice sensitivities."""

    @staticmethod
    def key_rate_dv01s(face: float, coupon_rate: float, frequency: int,
                       maturity_years: float, curve: YieldCurve) -> list[float]:
        """Per-node price sensitivities of a fixed-coupon bond to a 1bp bump
        of each curve tenor, in price units per 100 face (positive = the
        bond LOSES that much when the node rises 1bp — DV01 sign
        convention). Index i corresponds to ``curve.tenors()[i]`` ascending.
        """
        tenors = curve.tenors()
        base_rates = [curve.zero_rate(t) for t in tenors]
        krd = []
        bumped = list(base_rates)
        for i in range(len(tenors)):
            bumped[i] = base_rates[i] + _BUMP
            up = BondPricer.price_from_curve(face, coupon_rate, frequency,
                                             maturity_years,
                                             YieldCurve.of_zero_rates(tenors, bumped))
            bumped[i] = base_rates[i] - _BUMP
            down = BondPricer.price_from_curve(face, coupon_rate, frequency,
                                               maturity_years,
                                               YieldCurve.of_zero_rates(tenors, bumped))
            bumped[i] = base_rates[i]
            krd.append((down - up) / 2)     # positive when rates-up hurts
        return krd

    @staticmethod
    def parallel_dv01(face: float, coupon_rate: float, frequency: int,
                      maturity_years: float, curve: YieldCurve) -> float:
        """The parallel DV01 off the curve (every node bumped together) — the
        number the key-rate slices must add back up to, within the
        curve-interpolation tolerance the tests document.
        """
        tenors = curve.tenors()
        up = [curve.zero_rate(t) + _BUMP for t in tenors]
        down = [curve.zero_rate(t) - _BUMP for t in tenors]
        price_up = BondPricer.price_from_curve(face, coupon_rate, frequency,
                                               maturity_years,
                                               YieldCurve.of_zero_rates(tenors, up))
        price_down = BondPricer.price_from_curve(face, coupon_rate, frequency,
                                                 maturity_years,
                                                 YieldCurve.of_zero_rates(tenors, down))
        return (price_down - price_up) / 2
