"""Two-asset closed forms: Margrabe and Kirk (port of Java
``com.quantfinlib.pricing.ExchangeOption``).

**Margrabe (1978)** — the right to exchange asset 2 for asset 1 (payoff
``max(0, S1 - S2)``) is a Black-Scholes call in disguise: price asset 1
IN UNITS OF asset 2 and the strike becomes 1, the rate drops out
entirely and the vol is the vol of the RATIO,
``vol^2 = vol1^2 + vol2^2 - 2 rho vol1 vol2``.

**Kirk (1995)** — a spread call ``max(0, F1 - F2 - K)`` has no exact
lognormal closed form; Kirk approximates ``F2 + K`` as one lognormal
asset with vol scaled by its F2 share. Exact in both limits — K = 0
collapses to Margrabe, F2 = 0 collapses to Black-76 (both pinned).

Research lane, deterministic, no smile (single flat vol per leg).
"""

from __future__ import annotations

import math

from quantfinlib.util import math_utils as mu


class ExchangeOption:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def margrabe(s1: float, s2: float, q1: float, q2: float,
                 vol1: float, vol2: float, rho: float, time_years: float) -> float:
        """Margrabe: receive asset 1, deliver asset 2 at expiry.

        Args:
            s1: spot of the asset received, > 0.
            s2: spot of the asset delivered, > 0.
            q1: asset 1 continuous yield.
            q2: asset 2 continuous yield.
            rho: correlation of the two log-returns, in [-1, 1].
        """
        _require_positive(s1, "s1")
        _require_positive(s2, "s2")
        _require_vols(vol1, vol2, rho)
        if not (math.isfinite(q1) and math.isfinite(q2)):
            raise ValueError("yields must be finite")
        if time_years <= 0:
            return max(0.0, s1 - s2)
        variance = vol1 * vol1 + vol2 * vol2 - 2 * rho * vol1 * vol2
        f1 = s1 * math.exp(-q1 * time_years)
        f2 = s2 * math.exp(-q2 * time_years)
        if variance <= 0:
            # Perfectly correlated equal-vol legs: the ratio cannot move.
            return max(0.0, f1 - f2)
        sigma = math.sqrt(variance)
        sqrt_t = math.sqrt(time_years)
        d1 = ((math.log(s1 / s2) + (q2 - q1 + 0.5 * variance) * time_years)
              / (sigma * sqrt_t))
        d2 = d1 - sigma * sqrt_t
        return f1 * mu.norm_cdf(d1) - f2 * mu.norm_cdf(d2)

    @staticmethod
    def kirk_spread_call(f1: float, f2: float, strike: float, rate: float,
                         vol1: float, vol2: float, rho: float, time_years: float) -> float:
        """Kirk's approximation for a spread CALL on two forwards:
        ``max(0, F1 - F2 - K)`` paid at expiry, discounted at ``rate``.
        ``strike`` may be 0 (Margrabe limit) but not negative — flip the
        legs instead."""
        _require_positive(f1, "f1")
        if not (f2 >= 0) or f2 == math.inf:
            raise ValueError(f"f2 must be >= 0 and finite, got {f2}")
        if not (strike >= 0) or strike == math.inf:
            raise ValueError(f"strike must be >= 0 and finite, got {strike}")
        if f2 + strike <= 0:
            raise ValueError("f2 + strike must be > 0")
        if not math.isfinite(rate):
            raise ValueError("rate must be finite")
        _require_vols(vol1, vol2, rho)
        df = math.exp(-rate * max(time_years, 0.0))
        if time_years <= 0:
            return df * max(0.0, f1 - f2 - strike)
        f = f2 / (f2 + strike)
        variance = vol1 * vol1 - 2 * rho * vol1 * vol2 * f + vol2 * vol2 * f * f
        if variance <= 0:
            return df * max(0.0, f1 - f2 - strike)
        sigma = math.sqrt(variance)
        sqrt_t = math.sqrt(time_years)
        d1 = ((math.log(f1 / (f2 + strike)) + 0.5 * variance * time_years)
              / (sigma * sqrt_t))
        d2 = d1 - sigma * sqrt_t
        return df * (f1 * mu.norm_cdf(d1) - (f2 + strike) * mu.norm_cdf(d2))


def _require_positive(v: float, name: str) -> None:
    if not (v > 0) or v == math.inf:
        raise ValueError(f"{name} must be positive and finite, got {v}")


def _require_vols(vol1: float, vol2: float, rho: float) -> None:
    if (not (vol1 >= 0) or vol1 == math.inf
            or not (vol2 >= 0) or vol2 == math.inf):
        raise ValueError("vols must be >= 0 and finite")
    if not (rho >= -1) or not (rho <= 1):
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
