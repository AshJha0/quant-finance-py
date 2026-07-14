"""Black-76 — options on FORWARDS and futures (port of Java
``com.quantfinlib.pricing.Black76``).

The Black-Scholes sibling for rates caps/floors and swaptions, commodity
futures options, bond futures options. A forward has no carry (it costs
nothing to hold), so the underlying drift drops out and the price is the
discounted Black formula on the forward itself. Equivalent to
``BlackScholes`` with ``carry = 0`` and spot = forward — a test pins
that identity — but the market quotes these instruments IN Black-76
terms, so the model deserves its own front door. NaN-transparent inputs
produce NaN outputs (research-lane pricing convention).
"""

from __future__ import annotations

import math

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType
from quantfinlib.util import math_utils as mu


class Black76:
    """Static pricer namespace, mirroring the Java final class."""

    @staticmethod
    def price(option_type: OptionType, forward: float, strike: float,
              rate: float, vol: float, time_years: float) -> float:
        """Discounted Black-76 price of a call/put on a forward."""
        if time_years <= 0 or vol <= 0:
            return (math.exp(-rate * max(time_years, 0.0))
                    * BlackScholes.intrinsic(option_type, forward, strike))
        df = math.exp(-rate * time_years)
        sqrt_t = math.sqrt(time_years)
        d1 = (math.log(forward / strike) + 0.5 * vol * vol * time_years) / (vol * sqrt_t)
        d2 = d1 - vol * sqrt_t
        if option_type is OptionType.CALL:
            return df * (forward * mu.norm_cdf(d1) - strike * mu.norm_cdf(d2))
        return df * (strike * mu.norm_cdf(-d2) - forward * mu.norm_cdf(-d1))

    @staticmethod
    def delta(option_type: OptionType, forward: float, strike: float,
              rate: float, vol: float, time_years: float) -> float:
        """Sensitivity to the FORWARD (not spot): df * N(d1) for calls."""
        if time_years <= 0 or vol <= 0:
            if option_type is OptionType.CALL:
                intrinsic_delta = 1.0 if forward > strike else 0.0
            else:
                intrinsic_delta = -1.0 if forward < strike else 0.0
            return math.exp(-rate * max(time_years, 0.0)) * intrinsic_delta
        d1 = ((math.log(forward / strike) + 0.5 * vol * vol * time_years)
              / (vol * math.sqrt(time_years)))
        df = math.exp(-rate * time_years)
        if option_type is OptionType.CALL:
            return df * mu.norm_cdf(d1)
        return df * (mu.norm_cdf(d1) - 1)

    @staticmethod
    def vega(forward: float, strike: float, rate: float,
             vol: float, time_years: float) -> float:
        """Vega per 1.00 of vol (divide by 100 for per-point). Same for calls and puts."""
        if time_years <= 0 or vol <= 0:
            return 0.0
        sqrt_t = math.sqrt(time_years)
        d1 = (math.log(forward / strike) + 0.5 * vol * vol * time_years) / (vol * sqrt_t)
        return math.exp(-rate * time_years) * forward * mu.norm_pdf(d1) * sqrt_t

    @staticmethod
    def implied_vol(option_type: OptionType, market_price: float, forward: float,
                    strike: float, rate: float, time_years: float) -> float:
        """Black-76 implied vol from a price, via bisection (NaN if unattainable)."""
        lo = 1e-6
        hi = 5.0
        if (market_price <= Black76.price(option_type, forward, strike, rate, lo, time_years)
                or market_price >= Black76.price(option_type, forward, strike, rate, hi,
                                                 time_years)):
            return math.nan
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if Black76.price(option_type, forward, strike, rate, mid, time_years) < market_price:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
