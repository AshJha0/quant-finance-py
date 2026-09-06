"""SABR stochastic volatility model (port of Java
``com.quantfinlib.pricing.SabrModel``).

Hagan et al. (2002) lognormal implied volatility approximation and smile
calibration. With beta fixed (market convention), calibrates
(alpha, rho, nu) to observed strike/vol quotes — turning
``VolSurface``-style pillar smiles into a parametric, arbitrage-aware
fit that inter/extrapolates sensibly.

Port note: the Java calibration seeds ``java.util.SplittableRandom(42)``
for its random search; the bit-exact transcription of that generator
lives in :mod:`quantfinlib.sim._java_random` (``SplittableRandom64``,
pinned against JVM output in ``tests/test_sim.py``) and is shared here,
so the calibrated parameters are bit-identical across the two ports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantfinlib.sim._java_random import SplittableRandom64 as _SplittableRandom


@dataclass(frozen=True, slots=True)
class SabrParams:
    """Calibrated SABR parameters and the fit's RMSE in vol points."""

    alpha: float
    beta: float
    rho: float
    nu: float
    rmse: float


class SabrModel:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def implied_vol(f: float, k: float, t: float,
                    alpha: float, beta: float, rho: float, nu: float) -> float:
        """Hagan lognormal implied vol for forward ``f``, strike ``k``,
        expiry ``t``."""
        if f <= 0 or k <= 0 or t <= 0 or alpha <= 0:
            raise ValueError("f, k, t, alpha must be positive")
        if not (abs(rho) < 1):
            raise ValueError(f"rho must be in (-1, 1), got {rho}")
        if not (nu >= 0):
            raise ValueError(f"nu must be >= 0, got {nu}")
        one_minus_beta = 1 - beta
        log_fk = math.log(f / k)
        fk_pow = (f * k) ** (one_minus_beta / 2)

        # Correction term (common to ATM and non-ATM).
        correction = 1 + t * (
            one_minus_beta * one_minus_beta * alpha * alpha / (24 * fk_pow * fk_pow)
            + rho * beta * nu * alpha / (4 * fk_pow)
            + (2 - 3 * rho * rho) * nu * nu / 24)

        if abs(log_fk) < 1e-10:
            return alpha / f ** one_minus_beta * correction

        z = nu / alpha * fk_pow * log_fk
        if abs(z) < 1e-12:
            # nu -> 0 is the CEV limit: z/x(z) -> 1 (x(z) = z + O(z^2)),
            # not the 0/0 the literal formula produces.
            z_over_x = 1.0
        else:
            x = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
            z_over_x = z / x
        denom = fk_pow * (1
                          + one_minus_beta * one_minus_beta / 24 * log_fk * log_fk
                          + one_minus_beta ** 4 / 1920 * log_fk ** 4)
        return alpha / denom * z_over_x * correction

    @staticmethod
    def calibrate(f: float, t: float, beta: float,
                  strikes, market_vols) -> SabrParams:
        """Calibrates (alpha, rho, nu) with beta fixed, by seeded random
        search plus shrinking coordinate refinement (derivative-free,
        deterministic)."""
        strikes = [float(k) for k in strikes]
        market_vols = [float(v) for v in market_vols]
        if len(strikes) != len(market_vols) or len(strikes) < 3:
            raise ValueError("need >= 3 aligned strike/vol quotes")
        # Initial alpha from the closest-to-ATM quote: vol_atm ~ alpha / f^(1-beta).
        atm_vol = market_vols[0]
        best_dist = math.inf
        for k, v in zip(strikes, market_vols):
            d = abs(k - f)
            if d < best_dist:
                best_dist = d
                atm_vol = v
        alpha0 = atm_vol * f ** (1 - beta)

        rnd = _SplittableRandom(42)
        best = [alpha0, 0.0, 0.5]
        best_sse = _sse(f, t, beta, strikes, market_vols, best[0], best[1], best[2])

        for _ in range(4_000):
            alpha = alpha0 * (0.4 + 1.6 * rnd.next_double())
            rho = -0.98 + 1.96 * rnd.next_double()
            nu = 0.01 + 2.99 * rnd.next_double()
            err = _sse(f, t, beta, strikes, market_vols, alpha, rho, nu)
            if err < best_sse:
                best_sse = err
                best = [alpha, rho, nu]
        # Coordinate refinement with shrinking steps.
        steps = [alpha0 * 0.1, 0.1, 0.1]
        for _ in range(200):
            improved = False
            for p in range(3):
                for direction in (-1, 1):
                    trial = list(best)
                    trial[p] += direction * steps[p]
                    if trial[0] <= 0 or abs(trial[1]) >= 0.999 or trial[2] <= 0:
                        continue
                    err = _sse(f, t, beta, strikes, market_vols,
                               trial[0], trial[1], trial[2])
                    if err < best_sse:
                        best_sse = err
                        best = trial
                        improved = True
            if not improved:
                for p in range(3):
                    steps[p] /= 2
                if steps[1] < 1e-7:
                    break
        return SabrParams(best[0], beta, best[1], best[2],
                          math.sqrt(best_sse / len(strikes)))


def _sse(f: float, t: float, beta: float, strikes, vols,
         alpha: float, rho: float, nu: float) -> float:
    total = 0.0
    for k, v in zip(strikes, vols):
        d = SabrModel.implied_vol(f, k, t, alpha, beta, rho, nu) - v
        total += d * d
    return total
