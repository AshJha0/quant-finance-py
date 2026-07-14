"""Variance swap analytics (port of Java
``com.quantfinlib.pricing.VarianceSwap``).

At expiry the swap pays ``notional x (realized variance - strike)``. The
two quantities a desk actually books:

* **Variance vs vega notional** — dealers quote in VEGA (P&L per vol
  point) but settle in VARIANCE units; the bridge is
  ``variance_notional = vega_notional / (2 K_vol)``.
* **Mark-to-market of a seasoned swap** — variance is ADDITIVE in time,
  so a swap part-way through its life is realized variance so far
  (locked in) blended with the fair strike for the remaining leg,
  discounted.

Conventions: variance in annualized decimal^2 (0.04 = 20 vol), time in
years, realized variance supplied by the caller.

Port note: the Java ``fairVariance`` (model-free strike replicated from
an option chain) delegates to ``volatility.VolatilityIndex``, a domain
not yet ported — it is omitted here and will arrive with the volatility
package.
"""

from __future__ import annotations

import math


class VarianceSwap:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def vol_swap_strike(fair_variance: float, variance_of_variance: float) -> float:
        """VOLATILITY swap fair strike via the Brockhaus-Long convexity
        correction: ``E[sqrt(V)] ~ sqrt(E[V]) - Var(V) / (8 E[V]^{3/2})``.

        A vol swap strike is always BELOW the square root of the
        variance strike (Jensen: sqrt is concave), and by how much
        depends on the variance of variance — which is a model input,
        not chain-readable; that is exactly why vol swaps are not
        model-free while variance swaps are.

        Args:
            fair_variance: E[V], the variance-swap strike, > 0.
            variance_of_variance: Var(V) under your vol-of-vol model, >= 0.
        """
        if not (fair_variance > 0) or fair_variance == math.inf:
            raise ValueError("fairVariance must be positive and finite")
        if not (variance_of_variance >= 0) or variance_of_variance == math.inf:
            raise ValueError("varianceOfVariance must be >= 0 and finite")
        return (math.sqrt(fair_variance)
                - variance_of_variance / (8 * fair_variance ** 1.5))

    @staticmethod
    def variance_notional(vega_notional: float, strike_vol: float) -> float:
        """Variance notional from a vega-notional quote:
        ``vega_notional / (2 * strike_vol)``.

        Args:
            vega_notional: P&L per 1.00 of volatility (per "100 vol points").
            strike_vol: the strike in VOL terms (0.20, not 0.04), > 0.
        """
        if not (strike_vol > 0) or strike_vol == math.inf:
            raise ValueError(f"strikeVol must be positive and finite, got {strike_vol}")
        if not math.isfinite(vega_notional):
            raise ValueError("vegaNotional must be finite")
        return vega_notional / (2 * strike_vol)

    @staticmethod
    def mark_to_market(strike_variance: float, realized_variance: float,
                       remaining_fair: float, t_elapsed_years: float,
                       t_total_years: float, rate: float) -> float:
        """Mark-to-market of a seasoned variance swap per unit of variance
        notional (multiply by ``variance_notional`` for money).

        Args:
            strike_variance: original strike K0 (variance units, > 0).
            realized_variance: annualized variance realized over [0, t], >= 0.
            remaining_fair: current fair strike for [t, T] (variance), >= 0.
            t_elapsed_years: elapsed time t, >= 0.
            t_total_years: total life T, > 0, >= t.
            rate: cc discount rate to expiry.
        """
        if not (strike_variance > 0) or strike_variance == math.inf:
            raise ValueError("strikeVariance must be positive and finite")
        if (not (realized_variance >= 0) or realized_variance == math.inf
                or not (remaining_fair >= 0) or remaining_fair == math.inf):
            raise ValueError("variances must be >= 0 and finite")
        if not (t_total_years > 0) or t_total_years == math.inf:
            raise ValueError("tTotalYears must be positive and finite")
        if not (t_elapsed_years >= 0) or t_elapsed_years > t_total_years:
            raise ValueError(
                f"tElapsedYears must be in [0, {t_total_years}], got {t_elapsed_years}")
        if not math.isfinite(rate):
            raise ValueError("rate must be finite")
        weight_elapsed = t_elapsed_years / t_total_years
        expected = (weight_elapsed * realized_variance
                    + (1 - weight_elapsed) * remaining_fair)
        return math.exp(-rate * (t_total_years - t_elapsed_years)) * (expected - strike_variance)
