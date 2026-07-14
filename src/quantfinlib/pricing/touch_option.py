"""One-touch and no-touch options, pay-at-expiry (port of Java
``com.quantfinlib.pricing.TouchOption``).

Continuously monitored GBM, with the barrier-hitting probability itself
exposed, since desks quote one-touches AS (roughly) discounted hit
probabilities. With log-drift ``m = r - q - vol^2/2`` and barrier
log-distance ``h = ln(H/S)``, the reflection principle gives the
probability of touching an **upper** barrier (``h > 0``) by expiry::

    P = N((-h + mT)/(vol sqrt(T))) + e^{2mh/vol^2} N((-h - mT)/(vol sqrt(T)))

and symmetrically for a lower barrier. A one-touch paying at expiry is
``payout * e^{-rT} * P``; a no-touch is the complement. Conventions
match ``BlackScholes``.
"""

from __future__ import annotations

import math

from quantfinlib.util import math_utils as mu


class TouchOption:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def hit_probability(spot: float, barrier: float, rate: float,
                        carry: float, vol: float, time_years: float) -> float:
        """Probability that spot touches ``barrier`` at least once before expiry."""
        _validate(spot, barrier, vol, time_years)
        if spot == barrier:
            return 1.0  # already touching
        if time_years == 0:
            return 0.0
        m = rate - carry - 0.5 * vol * vol
        h = math.log(barrier / spot)
        sq = vol * math.sqrt(time_years)
        drift = m * time_years
        if h > 0:
            # Upper barrier: reflect around +h.
            return (mu.norm_cdf((-h + drift) / sq)
                    + math.exp(2 * m * h / (vol * vol)) * mu.norm_cdf((-h - drift) / sq))
        # Lower barrier: mirror image.
        return (mu.norm_cdf((h - drift) / sq)
                + math.exp(2 * m * h / (vol * vol)) * mu.norm_cdf((h + drift) / sq))

    @staticmethod
    def one_touch(spot: float, barrier: float, rate: float, carry: float,
                  vol: float, time_years: float, payout: float) -> float:
        """Pays ``payout`` at expiry if the barrier traded at any point."""
        return (payout * math.exp(-rate * time_years)
                * TouchOption.hit_probability(spot, barrier, rate, carry, vol, time_years))

    @staticmethod
    def no_touch(spot: float, barrier: float, rate: float, carry: float,
                 vol: float, time_years: float, payout: float) -> float:
        """Pays ``payout`` at expiry if the barrier never traded."""
        return (payout * math.exp(-rate * time_years)
                * (1 - TouchOption.hit_probability(spot, barrier, rate, carry, vol,
                                                   time_years)))


def _validate(spot: float, barrier: float, vol: float, t: float) -> None:
    if spot <= 0 or barrier <= 0 or vol <= 0 or t < 0:
        raise ValueError("spot, barrier, vol must be > 0 and timeYears >= 0")
