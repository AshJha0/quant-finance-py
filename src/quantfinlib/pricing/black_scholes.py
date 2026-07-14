"""Black-Scholes-Merton pricing and Greeks (port of Java
``com.quantfinlib.pricing.BlackScholes``).

Continuous carry yield ``carry``: the dividend yield for equities or the
foreign interest rate for FX (Garman-Kohlhagen). All rates and
volatility are annualized; theta is per year; vega/rho are per 1.00
change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from quantfinlib.util import math_utils as mu


class OptionType(Enum):
    """Call/put flag; ``sign`` is +1 for a call, -1 for a put."""

    CALL = 1
    PUT = -1

    def sign(self) -> int:
        return self.value


@dataclass(frozen=True, slots=True)
class Greeks:
    """Full Greek set for one option (port of the Java record)."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class BlackScholes:
    """Static pricer namespace, mirroring the Java final class."""

    @staticmethod
    def price(option_type: OptionType, spot: float, strike: float, rate: float,
              carry: float, vol: float, time_years: float) -> float:
        if time_years <= 0:
            return BlackScholes.intrinsic(option_type, spot, strike)
        if vol <= 0:
            # Deterministic world: discounted forward intrinsic. Without
            # this branch the ATM-forward case is 0/0 in d1 and the price
            # comes back NaN (off-ATM it survives only because +/-Inf
            # happens to hit the right CDF tail).
            return max(0.0, option_type.sign() * (spot * math.exp(-carry * time_years)
                                                  - strike * math.exp(-rate * time_years)))
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        d2 = d1 - vol * math.sqrt(time_years)
        df = math.exp(-rate * time_years)
        cf = math.exp(-carry * time_years)
        if option_type is OptionType.CALL:
            return spot * cf * mu.norm_cdf(d1) - strike * df * mu.norm_cdf(d2)
        return strike * df * mu.norm_cdf(-d2) - spot * cf * mu.norm_cdf(-d1)

    @staticmethod
    def delta(option_type: OptionType, spot: float, strike: float, rate: float,
              carry: float, vol: float, time_years: float) -> float:
        if time_years <= 0:
            return _intrinsic_delta(option_type, spot, strike)
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        cf = math.exp(-carry * time_years)
        if option_type is OptionType.CALL:
            return cf * mu.norm_cdf(d1)
        return cf * (mu.norm_cdf(d1) - 1)

    @staticmethod
    def gamma(spot: float, strike: float, rate: float, carry: float,
              vol: float, time_years: float) -> float:
        if time_years <= 0:
            return 0.0
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        return (math.exp(-carry * time_years) * mu.norm_pdf(d1)
                / (spot * vol * math.sqrt(time_years)))

    @staticmethod
    def vega(spot: float, strike: float, rate: float, carry: float,
             vol: float, time_years: float) -> float:
        """Per 1.00 change in volatility (divide by 100 for per-vol-point)."""
        if time_years <= 0:
            return 0.0
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        return spot * math.exp(-carry * time_years) * mu.norm_pdf(d1) * math.sqrt(time_years)

    @staticmethod
    def theta(option_type: OptionType, spot: float, strike: float, rate: float,
              carry: float, vol: float, time_years: float) -> float:
        """Per year (divide by 365 for per-calendar-day)."""
        if time_years <= 0:
            return 0.0
        sqrt_t = math.sqrt(time_years)
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        d2 = d1 - vol * sqrt_t
        cf = math.exp(-carry * time_years)
        df = math.exp(-rate * time_years)
        common = -spot * cf * mu.norm_pdf(d1) * vol / (2 * sqrt_t)
        if option_type is OptionType.CALL:
            return (common - rate * strike * df * mu.norm_cdf(d2)
                    + carry * spot * cf * mu.norm_cdf(d1))
        return (common + rate * strike * df * mu.norm_cdf(-d2)
                - carry * spot * cf * mu.norm_cdf(-d1))

    @staticmethod
    def rho(option_type: OptionType, spot: float, strike: float, rate: float,
            carry: float, vol: float, time_years: float) -> float:
        """Per 1.00 change in the domestic rate."""
        if time_years <= 0:
            return 0.0
        d2 = _d1(spot, strike, rate, carry, vol, time_years) - vol * math.sqrt(time_years)
        df = math.exp(-rate * time_years)
        if option_type is OptionType.CALL:
            return strike * time_years * df * mu.norm_cdf(d2)
        return -strike * time_years * df * mu.norm_cdf(-d2)

    @staticmethod
    def greeks(option_type: OptionType, spot: float, strike: float, rate: float,
               carry: float, vol: float, time_years: float) -> Greeks:
        return Greeks(
            BlackScholes.price(option_type, spot, strike, rate, carry, vol, time_years),
            BlackScholes.delta(option_type, spot, strike, rate, carry, vol, time_years),
            BlackScholes.gamma(spot, strike, rate, carry, vol, time_years),
            BlackScholes.vega(spot, strike, rate, carry, vol, time_years),
            BlackScholes.theta(option_type, spot, strike, rate, carry, vol, time_years),
            BlackScholes.rho(option_type, spot, strike, rate, carry, vol, time_years))

    @staticmethod
    def implied_vol(option_type: OptionType, market_price: float, spot: float,
                    strike: float, rate: float, carry: float, time_years: float) -> float:
        """Implied volatility by bisection.

        Returns NaN when the price is not attainable inside the
        [1e-4, 5.0] vol bracket (below intrinsic, or above the maximum
        BS price) — a stale or rounded market price must surface as
        "no vol", not silently come back as the 500% search bound and
        poison a smile.
        """
        lo, hi = 1e-4, 5.0
        if (market_price < BlackScholes.price(option_type, spot, strike, rate, carry, lo, time_years)
                or market_price > BlackScholes.price(option_type, spot, strike, rate, carry, hi,
                                                     time_years)):
            return math.nan
        for _ in range(200):
            mid = (lo + hi) / 2
            if BlackScholes.price(option_type, spot, strike, rate, carry, mid,
                                  time_years) < market_price:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    @staticmethod
    def intrinsic(option_type: OptionType, spot: float, strike: float) -> float:
        return max(0.0, option_type.sign() * (spot - strike))


def _intrinsic_delta(option_type: OptionType, spot: float, strike: float) -> float:
    if option_type is OptionType.CALL:
        return 1.0 if spot > strike else 0.0
    return -1.0 if spot < strike else 0.0


def _d1(spot: float, strike: float, rate: float, carry: float,
        vol: float, time_years: float) -> float:
    return ((math.log(spot / strike) + (rate - carry + 0.5 * vol * vol) * time_years)
            / (vol * math.sqrt(time_years)))
