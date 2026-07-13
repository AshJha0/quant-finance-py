"""Fixed-coupon bond analytics (port of Java ``com.quantfinlib.rates.BondPricer``).

Price/yield conversion, Macaulay and modified duration, convexity, and
DV01. Yields are per-annum with ``frequency`` compounding (the bond's
coupon frequency); coupons are assumed on a regular schedule with the
last payment at maturity.

The risk numbers and what each is FOR: Macaulay duration is the
cash-flow-weighted average time to payment (years — an intuition
number); modified duration = Macaulay / (1 + y/f) is the first-order
price sensitivity, ``dP/P ~ -modDur * dy``; DV01 is the same derivative
in money terms per 1bp — the number the desk actually hedges, because
DV01s ADD across positions while durations must be value-weighted.
Convexity is the second-order term that makes the duration hedge wrong
for large moves — always in the LONG bondholder's favor, which is why
long-convexity positions cost carry.

Simplifications, stated: whole coupon periods (no accrued-interest
split of dirty into clean price), no settlement-lag discounting, and
yield-to-maturity as the single rate — the classic textbook bond, exact
for pricing OFF a yield quote. Pricing off a full curve (each cash flow
at its own zero rate) is ``YieldCurve``'s job; hedging against
non-parallel curve moves is ``KeyRateDurations``'.

Python port note: the Java class also carries date-based pricing with
real day-count/business-calendar conventions; those depend on
``DayCount``/``BusinessCalendar``, which are outside this port's scope,
so only the period-grid analytics are ported.
"""

from __future__ import annotations

import math

from quantfinlib.rates.yield_curve import YieldCurve


class BondPricer:
    """Static bond analytics on a regular whole-period schedule."""

    @staticmethod
    def price_from_yield(face: float, coupon_rate: float, frequency: int,
                         years_to_maturity: float, yield_: float) -> float:
        """Dirty price per ``face`` from a yield (regular schedule, whole periods)."""
        n = BondPricer._periods(frequency, years_to_maturity)
        coupon = face * coupon_rate / frequency
        y = yield_ / frequency
        price = 0.0
        for i in range(1, n + 1):
            price += coupon / (1 + y) ** i
        return price + face / (1 + y) ** n

    @staticmethod
    def price_from_curve(face: float, coupon_rate: float, frequency: int,
                         years_to_maturity: float, curve: YieldCurve) -> float:
        """Price by discounting each cash flow on a zero curve."""
        n = BondPricer._periods(frequency, years_to_maturity)
        coupon = face * coupon_rate / frequency
        price = 0.0
        for i in range(1, n + 1):
            price += coupon * curve.discount_factor(i / frequency)
        return price + face * curve.discount_factor(n / frequency)

    @staticmethod
    def yield_to_maturity(price: float, face: float, coupon_rate: float,
                          frequency: int, years_to_maturity: float) -> float:
        """Yield to maturity by bisection (price must be positive)."""
        lo, hi = -0.9, 10.0
        # Bracket check: a price outside [PV(10), PV(-0.9)] has no yield in
        # the search range, and bisection would silently converge to an
        # endpoint and hand back -89.99% or 1000% as if it were a market
        # yield. Refuse loudly instead.
        max_price = BondPricer.price_from_yield(face, coupon_rate, frequency,
                                                years_to_maturity, lo)
        min_price = BondPricer.price_from_yield(face, coupon_rate, frequency,
                                                years_to_maturity, hi)
        if not (price >= min_price) or not (price <= max_price):
            raise ValueError(f"price {price} has no yield in [-90%, 1000%]"
                             f" (attainable price range [{min_price}, {max_price}])")
        for _ in range(200):
            mid = (lo + hi) / 2
            if BondPricer.price_from_yield(face, coupon_rate, frequency,
                                           years_to_maturity, mid) > price:
                lo = mid    # price too high -> yield higher
            else:
                hi = mid
        return (lo + hi) / 2

    @staticmethod
    def macaulay_duration(face: float, coupon_rate: float, frequency: int,
                          years_to_maturity: float, yield_: float) -> float:
        """Macaulay duration in years: PV-weighted average time to cash flow."""
        n = BondPricer._periods(frequency, years_to_maturity)
        coupon = face * coupon_rate / frequency
        y = yield_ / frequency
        weighted = 0.0
        price = 0.0
        for i in range(1, n + 1):
            t = i / frequency
            cf = coupon + (face if i == n else 0.0)
            pv = cf / (1 + y) ** i
            weighted += t * pv
            price += pv
        return weighted / price

    @staticmethod
    def modified_duration(face: float, coupon_rate: float, frequency: int,
                          years_to_maturity: float, yield_: float) -> float:
        """Modified duration: price sensitivity per unit yield change."""
        return (BondPricer.macaulay_duration(face, coupon_rate, frequency,
                                             years_to_maturity, yield_)
                / (1 + yield_ / frequency))

    @staticmethod
    def convexity(face: float, coupon_rate: float, frequency: int,
                  years_to_maturity: float, yield_: float) -> float:
        """Convexity (numeric second derivative of price w.r.t. yield, normalized)."""
        h = 1e-4
        p0 = BondPricer.price_from_yield(face, coupon_rate, frequency,
                                         years_to_maturity, yield_)
        up = BondPricer.price_from_yield(face, coupon_rate, frequency,
                                         years_to_maturity, yield_ + h)
        dn = BondPricer.price_from_yield(face, coupon_rate, frequency,
                                         years_to_maturity, yield_ - h)
        return (up + dn - 2 * p0) / (p0 * h * h)

    @staticmethod
    def dv01(face: float, coupon_rate: float, frequency: int,
             years_to_maturity: float, yield_: float) -> float:
        """Price change for a one-basis-point yield move (positive number)."""
        price = BondPricer.price_from_yield(face, coupon_rate, frequency,
                                            years_to_maturity, yield_)
        return (BondPricer.modified_duration(face, coupon_rate, frequency,
                                             years_to_maturity, yield_)
                * price * 1e-4)

    @staticmethod
    def _periods(frequency: int, years_to_maturity: float) -> int:
        if frequency < 1 or years_to_maturity <= 0:
            raise ValueError("need positive frequency and maturity")
        # Java Math.round: half-up. Python round() is banker's rounding, so
        # spell out floor(x + 0.5) to keep the two ports on the same grid.
        return math.floor(years_to_maturity * frequency + 0.5)
