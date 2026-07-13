"""Bond credit-spread measures (port of Java ``com.quantfinlib.credit.CreditSpreads``).

The translation layer between a bond's PRICE and how much of it is
credit.

The Z-SPREAD is the single constant shift z added to every point of the
risk-free zero curve that makes the bond's discounted cash flows equal
its dirty price:

    price = sum cf_i * exp(-(z(t_i) + z) * t_i)

It is the honest successor to "yield spread over the 10y": a yield
spread compares one bond's YTM to one government point and mixes curve
shape into the number; the Z-spread strips the entire risk-free curve
out first, so what remains is compensation for credit and liquidity. A
desk triangulates it against the same name's CDS:
``zSpread - cdsParSpread`` is the CDS-BOND BASIS, the classic
relative-value trade (negative basis: buy the bond, buy CDS protection,
collect the difference — a trade that famously blew through funding
constraints in 2008, which is why the basis is not free money).

Solving is bisection on z in [-50%, +500%] with an explicit bracket
check: a price outside what that range can explain raises rather than
returning an endpoint (the house rule for every solver since the YTM
incident). Annual-fraction period grid ``i/frequency``, cash flows of a
standard fixed-coupon bond.
"""

from __future__ import annotations

import math

from quantfinlib.rates.yield_curve import YieldCurve


class CreditSpreads:
    """Static Z-spread solver and shifted-curve pricer."""

    @staticmethod
    def z_spread(dirty_price: float, face: float, coupon_rate: float,
                 frequency: int, years_to_maturity: float,
                 curve: YieldCurve) -> float:
        """The Z-spread (continuously compounded, decimal) of a fixed-coupon
        bond over ``curve``.

        Args:
            dirty_price: market dirty price per ``face``.
            face: face value, > 0.
            coupon_rate: annual coupon rate (decimal).
            frequency: coupons per year, >= 1.
            years_to_maturity: whole periods assumed, > 0.
        """
        if not (dirty_price > 0) or dirty_price == math.inf:
            raise ValueError("dirtyPrice must be positive and finite")
        if not (face > 0) or face == math.inf:
            raise ValueError("face must be positive and finite")
        if not (coupon_rate >= 0) or coupon_rate == math.inf:
            raise ValueError("couponRate must be >= 0 and finite")
        if frequency < 1:
            raise ValueError(f"frequency must be >= 1, got {frequency}")
        if not (years_to_maturity > 0) or years_to_maturity == math.inf:
            raise ValueError("yearsToMaturity must be positive and finite")
        lo, hi = -0.5, 5.0
        pv_lo = _pv(face, coupon_rate, frequency, years_to_maturity, curve, lo)
        pv_hi = _pv(face, coupon_rate, frequency, years_to_maturity, curve, hi)
        # PV is decreasing in z: pv(lo) is the maximum, pv(hi) the minimum.
        if not (dirty_price <= pv_lo) or not (dirty_price >= pv_hi):
            raise ValueError(f"price {dirty_price} has no Z-spread in "
                             f"[-50%, 500%] (attainable [{pv_hi}, {pv_lo}])")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _pv(face, coupon_rate, frequency, years_to_maturity, curve,
                   mid) > dirty_price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def price_with_z_spread(face: float, coupon_rate: float, frequency: int,
                            years_to_maturity: float, curve: YieldCurve,
                            z: float) -> float:
        """Bond PV under the curve shifted by a constant z (cc)."""
        if not math.isfinite(z):
            raise ValueError("z must be finite")
        return _pv(face, coupon_rate, frequency, years_to_maturity, curve, z)


def _pv(face: float, coupon_rate: float, frequency: int,
        years_to_maturity: float, curve: YieldCurve, z: float) -> float:
    n = math.floor(years_to_maturity * frequency + 0.5)  # Java Math.round
    coupon = face * coupon_rate / frequency
    pv = 0.0
    for i in range(1, n + 1):
        t = i / frequency
        cf = coupon + (face if i == n else 0.0)
        pv += cf * math.exp(-(curve.zero_rate(t) + z) * t)
    return pv
