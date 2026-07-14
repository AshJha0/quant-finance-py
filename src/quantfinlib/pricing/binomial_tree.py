"""Cox-Ross-Rubinstein binomial tree (port of Java
``com.quantfinlib.pricing.BinomialTree``).

European and American options with a continuous carry yield
(dividends / foreign rate). Converges to Black-Scholes for European
payoffs; prices the early-exercise premium for American ones.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType


class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class BinomialTree:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def price(option_type: OptionType, style: ExerciseStyle, spot: float, strike: float,
              rate: float, carry: float, vol: float, time_years: float, steps: int) -> float:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if time_years <= 0:
            return BlackScholes.intrinsic(option_type, spot, strike)
        dt = time_years / steps
        u = math.exp(vol * math.sqrt(dt))
        d = 1 / u
        growth = math.exp((rate - carry) * dt)
        p = (growth - d) / (u - d)
        if p <= 0 or p >= 1:
            raise ValueError(f"degenerate tree (p={p}): increase steps or check inputs")
        discount = math.exp(-rate * dt)
        sign = option_type.sign()

        # Terminal payoffs at nodes j = 0..steps: s = spot * u^j * d^(steps-j).
        j = np.arange(steps + 1)
        terminal = spot * u ** j * d ** (steps - j)
        values = np.maximum(0.0, sign * (terminal - strike))
        # Backward induction with optional early exercise (vectorized over
        # the layer; arithmetic per node identical to the Java loop).
        for i in range(steps - 1, -1, -1):
            values = discount * (p * values[1:i + 2] + (1 - p) * values[:i + 1])
            if style is ExerciseStyle.AMERICAN:
                ji = np.arange(i + 1)
                s = spot * u ** ji * d ** (i - ji)
                values = np.maximum(values, np.maximum(0.0, sign * (s - strike)))
        return float(values[0])

    @staticmethod
    def early_exercise_premium(option_type: OptionType, spot: float, strike: float,
                               rate: float, carry: float, vol: float,
                               time_years: float, steps: int) -> float:
        """Early-exercise premium: American price minus European price."""
        return (BinomialTree.price(option_type, ExerciseStyle.AMERICAN, spot, strike,
                                   rate, carry, vol, time_years, steps)
                - BinomialTree.price(option_type, ExerciseStyle.EUROPEAN, spot, strike,
                                     rate, carry, vol, time_years, steps))

    @staticmethod
    def delta(option_type: OptionType, style: ExerciseStyle, spot: float, strike: float,
              rate: float, carry: float, vol: float, time_years: float, steps: int) -> float:
        """Delta by central difference (bump size 1e-4 of spot, as in Java)."""
        h = spot * 1e-4
        up = BinomialTree.price(option_type, style, spot + h, strike, rate, carry,
                                vol, time_years, steps)
        dn = BinomialTree.price(option_type, style, spot - h, strike, rate, carry,
                                vol, time_years, steps)
        return (up - dn) / (2 * h)
