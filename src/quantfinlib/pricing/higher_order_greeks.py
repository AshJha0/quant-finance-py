"""Second-order Greeks a vol book actually hedges with (port of Java
``com.quantfinlib.pricing.HigherOrderGreeks``).

* **Vanna** d2V/dS d(sigma) — how delta drifts when vol moves; THE
  skew-hedging Greek.
* **Volga** d2V/d(sigma)2 (vomma) — vega convexity. Vanna and volga are
  the two Greeks the ``VannaVolga`` pricing method charges the smile for.
* **Cross-gamma** — the two-asset P&L term d2V/dS1 dS2 for the Margrabe
  exchange-option case.

Same conventions as ``BlackScholes``: ``carry`` is the continuous YIELD
q. Vanna/volga are identical for calls and puts (put-call parity kills
the sign difference in the second order).
"""

from __future__ import annotations

import math

from quantfinlib.util import math_utils as mu


class HigherOrderGreeks:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def vanna(spot: float, strike: float, rate: float, carry: float,
              vol: float, time_years: float) -> float:
        """Vanna d2V/dS d(sigma): per 1.00 spot x 1.00 vol. Same for calls and puts."""
        if time_years <= 0 or vol <= 0 or spot <= 0:
            return 0.0
        sqrt_t = math.sqrt(time_years)
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        d2 = d1 - vol * sqrt_t
        return -math.exp(-carry * time_years) * mu.norm_pdf(d1) * d2 / vol

    @staticmethod
    def volga(spot: float, strike: float, rate: float, carry: float,
              vol: float, time_years: float) -> float:
        """Volga (vomma) d2V/d(sigma)2: vega convexity. Same for calls and puts."""
        if time_years <= 0 or vol <= 0 or spot <= 0:
            return 0.0
        sqrt_t = math.sqrt(time_years)
        d1 = _d1(spot, strike, rate, carry, vol, time_years)
        d2 = d1 - vol * sqrt_t
        vega = spot * math.exp(-carry * time_years) * mu.norm_pdf(d1) * sqrt_t
        return vega * d1 * d2 / vol

    @staticmethod
    def exchange_cross_gamma(spot1: float, spot2: float, vol1: float,
                             vol2: float, correlation: float,
                             time_years: float) -> float:
        """Cross-gamma of a Margrabe exchange option.

        ``d2V/dS1 dS2 = -phi(d1)/(S2 * sigma_hat * sqrt(T))`` with
        ``sigma_hat^2 = vol1^2 + vol2^2 - 2 rho vol1 vol2``. Negative:
        the exchange option loses convexity when the two legs move
        together. For a generic basket, differentiate YOUR pricer
        numerically instead — this is the closed form worth having, not
        a universal answer.
        """
        if time_years <= 0 or spot1 <= 0 or spot2 <= 0:
            return 0.0
        sigma_hat = math.sqrt(max(1e-12,
                                  vol1 * vol1 + vol2 * vol2 - 2 * correlation * vol1 * vol2))
        sqrt_t = math.sqrt(time_years)
        d1 = ((math.log(spot1 / spot2) + 0.5 * sigma_hat * sigma_hat * time_years)
              / (sigma_hat * sqrt_t))
        return -mu.norm_pdf(d1) / (spot2 * sigma_hat * sqrt_t)


def _d1(spot: float, strike: float, rate: float, carry: float,
        vol: float, time_years: float) -> float:
    return ((math.log(spot / strike) + (rate - carry + 0.5 * vol * vol) * time_years)
            / (vol * math.sqrt(time_years)))
