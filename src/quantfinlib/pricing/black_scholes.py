"""Black-Scholes-Merton pricing and Greeks (port of Java
``com.quantfinlib.pricing.BlackScholes``).

Continuous carry yield ``carry``: the dividend yield for equities or the
foreign interest rate for FX (Garman-Kohlhagen). All rates and
volatility are annualized; theta is per year; vega/rho are per 1.00
change.

Input contract: ``spot``, ``strike`` must be finite and > 0; ``rate``,
``carry``, ``vol``, ``time_years`` must be finite; ``vol`` and
``time_years`` may be <= 0 (the deterministic / expired limits are
returned in closed form and every Greek agrees with ``price``). Any
NaN or infinity raises ``ValueError`` — a NaN input must never come
back as a NaN price that survives into a P&L.
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
        _require_inputs(spot, strike, rate, carry, vol, time_years)
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
        _require_inputs(spot, strike, rate, carry, vol, time_years)
        if time_years <= 0:
            return _intrinsic_delta(option_type, spot, strike)
        cf = math.exp(-carry * time_years)
        if vol <= 0:
            # Deterministic limit of the price branch: cf * 1{F > K}
            # (put: -cf * 1{F < K}); ATM-forward takes the call side of
            # the step, consistent with ``price`` returning 0 there.
            return cf * _intrinsic_delta(option_type, _forward(spot, rate, carry, time_years),
                                         strike)
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        if option_type is OptionType.CALL:
            return cf * mu.norm_cdf(d1)
        return cf * (mu.norm_cdf(d1) - 1)

    @staticmethod
    def gamma(spot: float, strike: float, rate: float, carry: float,
              vol: float, time_years: float) -> float:
        _require_inputs(spot, strike, rate, carry, vol, time_years)
        if time_years <= 0 or vol <= 0:
            return 0.0      # a step payoff has no curvature away from the strike
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        return (math.exp(-carry * time_years) * mu.norm_pdf(d1)
                / (spot * vol * math.sqrt(time_years)))

    @staticmethod
    def vega(spot: float, strike: float, rate: float, carry: float,
             vol: float, time_years: float) -> float:
        """Per 1.00 change in volatility (divide by 100 for per-vol-point)."""
        _require_inputs(spot, strike, rate, carry, vol, time_years)
        if time_years <= 0 or vol <= 0:
            return 0.0
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        return spot * math.exp(-carry * time_years) * mu.norm_pdf(d1) * math.sqrt(time_years)

    @staticmethod
    def theta(option_type: OptionType, spot: float, strike: float, rate: float,
              carry: float, vol: float, time_years: float) -> float:
        """Per year (divide by 365 for per-calendar-day)."""
        _require_inputs(spot, strike, rate, carry, vol, time_years)
        if time_years <= 0:
            return 0.0
        cf = math.exp(-carry * time_years)
        df = math.exp(-rate * time_years)
        if vol <= 0:
            # Carry terms only (the diffusion term vanishes with vol):
            # d/dT of the discounted forward intrinsic.
            fwd = _forward(spot, rate, carry, time_years)
            if option_type is OptionType.CALL:
                return (-rate * strike * df + carry * spot * cf) if fwd > strike else 0.0
            return (rate * strike * df - carry * spot * cf) if fwd < strike else 0.0
        sqrt_t = math.sqrt(time_years)
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        d2 = d1 - vol * sqrt_t
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
        _require_inputs(spot, strike, rate, carry, vol, time_years)
        if time_years <= 0:
            return 0.0
        df = math.exp(-rate * time_years)
        if vol <= 0:
            fwd = _forward(spot, rate, carry, time_years)
            if option_type is OptionType.CALL:
                return strike * time_years * df if fwd > strike else 0.0
            return -strike * time_years * df if fwd < strike else 0.0
        d2 = _d1(spot, strike, rate, carry, vol, time_years) - vol * math.sqrt(time_years)
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

        Raises:
            ValueError: for a NaN/negative ``market_price``, for
                ``time_years <= 0`` (an expired option has no implied
                vol — every vol gives the intrinsic price, so the
                bisection would collapse onto the lower bracket bound
                and return it), or non-finite/non-positive spot/strike.
        """
        _require_inputs(spot, strike, rate, carry, 1.0, time_years)
        if not (market_price >= 0) or market_price == math.inf:
            raise ValueError(f"market_price must be finite and >= 0, got {market_price}")
        if not (time_years > 0):
            raise ValueError("time_years must be > 0 for an implied vol")
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


def _forward(spot: float, rate: float, carry: float, time_years: float) -> float:
    return spot * math.exp((rate - carry) * time_years)


def _require_inputs(spot: float, strike: float, rate: float, carry: float,
                    vol: float, time_years: float) -> None:
    if not (spot > 0) or spot == math.inf:
        raise ValueError(f"spot must be finite and > 0, got {spot}")
    if not (strike > 0) or strike == math.inf:
        raise ValueError(f"strike must be finite and > 0, got {strike}")
    for name, x in (("rate", rate), ("carry", carry), ("vol", vol),
                    ("time_years", time_years)):
        if not math.isfinite(x):
            raise ValueError(f"{name} must be finite, got {x}")


def _d1(spot: float, strike: float, rate: float, carry: float,
        vol: float, time_years: float) -> float:
    return ((math.log(spot / strike) + (rate - carry + 0.5 * vol * vol) * time_years)
            / (vol * math.sqrt(time_years)))
