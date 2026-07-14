"""Asian (average-price) options (port of Java
``com.quantfinlib.pricing.AsianOption``).

Two averages, two methods:

* **Geometric average — exact.** A product of lognormals is lognormal,
  so the geometric Asian has a Black-Scholes-style closed form
  (Kemna-Vorst 1990, discrete-fixing version). For n fixings equally
  spaced at ``t_i = i T / n`` (last fixing AT expiry)::

      E[ln G]   = ln S + (r - q - vol^2/2) * T (n+1)/(2n)
      Var[ln G] = vol^2 * T * (n+1)(2n+1)/(6 n^2)

  As n grows, Var goes to ``vol^2 T / 3``; at n = 1 the formula IS
  vanilla Black-Scholes (tested exact).
* **Arithmetic average — Turnbull-Wakeman (1991) moment matching.** TW
  computes the arithmetic average's first two moments EXACTLY under
  GBM, then prices Black-76 style on the lognormal with those moments
  (``Var[ln A] = ln(M2/M1^2)``). The double sum is O(n^2): fine for
  real fixing schedules, not a Monte Carlo replacement.

AM-GM guarantees ``A >= G`` pathwise, so an arithmetic CALL is always
worth at least the geometric call. Fixings strictly after inception
(``t_1 = T/n > 0``): decompose a seasoned Asian before calling.
Research lane.
"""

from __future__ import annotations

import math

from quantfinlib.pricing.black_scholes import OptionType
from quantfinlib.util import math_utils as mu


class AsianOption:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def geometric_price(option_type: OptionType, spot: float, strike: float,
                        rate: float, carry: float, vol: float, time_years: float,
                        averaging_points: int) -> float:
        """Exact discrete geometric-average Asian price (Kemna-Vorst).

        Args:
            averaging_points: number of equally spaced fixings n >= 1 at
                ``t_i = i T / n``; n = 1 is vanilla BS.
        """
        _validate(spot, strike, rate, carry, vol, time_years, averaging_points)
        n = float(averaging_points)
        mean_log = (math.log(spot)
                    + (rate - carry - 0.5 * vol * vol) * time_years * (n + 1) / (2 * n))
        var_log = vol * vol * time_years * (n + 1) * (2 * n + 1) / (6 * n * n)
        forward = math.exp(mean_log + 0.5 * var_log)
        return _black_on_lognormal(option_type, forward, strike, var_log, rate, time_years)

    @staticmethod
    def arithmetic_price(option_type: OptionType, spot: float, strike: float,
                         rate: float, carry: float, vol: float, time_years: float,
                         averaging_points: int) -> float:
        """Arithmetic-average Asian price via Turnbull-Wakeman two-moment
        lognormal matching (see module doc; O(n^2) in the fixing count)."""
        _validate(spot, strike, rate, carry, vol, time_years, averaging_points)
        n = averaging_points
        h = time_years / n
        g = rate - carry
        m1 = 0.0
        for i in range(1, n + 1):
            m1 += math.exp(g * i * h)
        m1 *= spot / n
        m2 = 0.0
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                m2 += math.exp(g * (i + j) * h + vol * vol * min(i, j) * h)
        m2 *= spot * spot / (float(n) * n)
        # Matched log-variance; guard the deterministic edge (vol = 0 makes
        # M2 exactly M1^2 and the log a rounding-noise 0/0).
        var_log = math.log(m2 / (m1 * m1)) if m2 > m1 * m1 else 0.0
        return _black_on_lognormal(option_type, m1, strike, var_log, rate, time_years)


def _black_on_lognormal(option_type: OptionType, f: float, k: float,
                        v2: float, rate: float, time_years: float) -> float:
    """Discounted Black-style price on a lognormal with mean f, log-variance v2."""
    df = math.exp(-rate * time_years)
    if v2 <= 0:
        return df * max(0.0, option_type.sign() * (f - k))  # zero-vol intrinsic on the forward
    sd = math.sqrt(v2)
    d1 = (math.log(f / k) + 0.5 * v2) / sd
    d2 = d1 - sd
    s = option_type.sign()
    return df * s * (f * mu.norm_cdf(s * d1) - k * mu.norm_cdf(s * d2))


def _validate(spot: float, strike: float, rate: float, carry: float,
              vol: float, time_years: float, averaging_points: int) -> None:
    if not (spot > 0) or spot == math.inf:
        raise ValueError(f"spot must be positive and finite, got {spot}")
    if not (strike > 0) or strike == math.inf:
        raise ValueError(f"strike must be positive and finite, got {strike}")
    if not (math.isfinite(rate) and math.isfinite(carry)):
        raise ValueError("rate and carry must be finite")
    if not (vol >= 0) or vol == math.inf:
        raise ValueError(f"vol must be >= 0 and finite, got {vol}")
    if not (time_years > 0) or time_years == math.inf:
        raise ValueError(f"timeYears must be positive and finite, got {time_years}")
    if averaging_points < 1:
        raise ValueError(f"averagingPoints must be >= 1, got {averaging_points}")
