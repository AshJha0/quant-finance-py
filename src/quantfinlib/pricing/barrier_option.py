"""Single-barrier vanilla options in closed form (port of Java
``com.quantfinlib.pricing.BarrierOption``).

Continuously monitored knock-in / knock-out for the *regular* barrier
configurations, priced by the reflection principle (Reiner-Rubinstein,
as in Hull):

* **Down** barriers on **calls** with ``H <= K`` (barrier in the OTM
  region): down-and-in from the reflection formula, down-and-out from
  in-out parity ``KO = vanilla - KI``;
* **Up** barriers on **puts** with ``H >= K``, the mirror case.

**Reverse** barriers (a barrier in the ITM region, e.g. an up-and-out
call) knock out exactly where the payoff is largest, need the full
eight-case decomposition, and their risk is dominated by the barrier
gamma — this class rejects them explicitly rather than pricing them
subtly wrong. No rebates. Conventions match ``BlackScholes``.
"""

from __future__ import annotations

import math

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType
from quantfinlib.util import math_utils as mu


class BarrierOption:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def down_and_in_call(spot: float, strike: float, barrier: float,
                         rate: float, carry: float, vol: float, time_years: float) -> float:
        """Down-and-in call, ``H <= min(S, K)``: alive only after the barrier trades."""
        _validate_down_call(spot, strike, barrier)
        if time_years <= 0:
            return 0.0  # never touched, expires worthless as a knock-in
        return _reflection_in(OptionType.CALL, spot, strike, barrier, rate, carry,
                              vol, time_years)

    @staticmethod
    def down_and_out_call(spot: float, strike: float, barrier: float,
                          rate: float, carry: float, vol: float, time_years: float) -> float:
        """Down-and-out call, ``H <= min(S, K)``: dies if the barrier trades."""
        _validate_down_call(spot, strike, barrier)
        # In-out parity: holding both KI and KO replicates the vanilla.
        return (BlackScholes.price(OptionType.CALL, spot, strike, rate, carry, vol, time_years)
                - BarrierOption.down_and_in_call(spot, strike, barrier, rate, carry,
                                                 vol, time_years))

    @staticmethod
    def up_and_in_put(spot: float, strike: float, barrier: float,
                      rate: float, carry: float, vol: float, time_years: float) -> float:
        """Up-and-in put, ``H >= max(S, K)``: the mirror of the down-and-in call."""
        _validate_up_put(spot, strike, barrier)
        if time_years <= 0:
            return 0.0
        return _reflection_in(OptionType.PUT, spot, strike, barrier, rate, carry,
                              vol, time_years)

    @staticmethod
    def up_and_out_put(spot: float, strike: float, barrier: float,
                       rate: float, carry: float, vol: float, time_years: float) -> float:
        """Up-and-out put, ``H >= max(S, K)``."""
        _validate_up_put(spot, strike, barrier)
        return (BlackScholes.price(OptionType.PUT, spot, strike, rate, carry, vol, time_years)
                - BarrierOption.up_and_in_put(spot, strike, barrier, rate, carry,
                                              vol, time_years))


def _reflection_in(option_type: OptionType, spot: float, strike: float,
                   barrier: float, rate: float, carry: float,
                   vol: float, t: float) -> float:
    """The Reiner-Rubinstein knock-in value: the vanilla priced on the
    barrier-reflected path measure. Sign symmetry handles the put mirror."""
    s = option_type.sign()  # +1 call (down barrier), -1 put (up barrier)
    lam = (rate - carry + 0.5 * vol * vol) / (vol * vol)
    sq = vol * math.sqrt(t)
    y = math.log(barrier * barrier / (spot * strike)) / sq + lam * sq
    hs = barrier / spot
    return s * (spot * math.exp(-carry * t) * hs ** (2 * lam) * mu.norm_cdf(s * y)
                - strike * math.exp(-rate * t) * hs ** (2 * lam - 2)
                * mu.norm_cdf(s * (y - sq)))


def _validate_down_call(spot: float, strike: float, barrier: float) -> None:
    _validate_common(spot, strike, barrier)
    if barrier >= spot:
        raise ValueError(f"down barrier {barrier} already breached at spot {spot}")
    if barrier > strike:
        raise ValueError(
            "reverse barrier (H > K on a call) is not supported in closed form here — "
            "price it with Monte Carlo or a barrier-aware tree")


def _validate_up_put(spot: float, strike: float, barrier: float) -> None:
    _validate_common(spot, strike, barrier)
    if barrier <= spot:
        raise ValueError(f"up barrier {barrier} already breached at spot {spot}")
    if barrier < strike:
        raise ValueError(
            "reverse barrier (H < K on a put) is not supported in closed form here — "
            "price it with Monte Carlo or a barrier-aware tree")


def _validate_common(spot: float, strike: float, barrier: float) -> None:
    if spot <= 0 or strike <= 0 or barrier <= 0:
        raise ValueError("spot, strike, barrier must be > 0")
