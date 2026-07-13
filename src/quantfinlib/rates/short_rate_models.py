"""Classic short-rate models (port of Java ``com.quantfinlib.rates.ShortRateModels``).

Each answers "what is a zero-coupon bond worth if the short rate
follows this SDE?" in closed form — the workhorse trio of rates risk
factor modeling:

* Vasicek ``dr = a(b - r)dt + sigma dW`` — Gaussian, tractable
  everywhere, and honest about its flaw: rates can go negative (which,
  post-2015, is a feature as much as a bug);
* CIR ``dr = a(b - r)dt + sigma sqrt(r) dW`` — the square-root
  diffusion keeps rates non-negative (strictly positive when the Feller
  condition ``2ab >= sigma^2`` holds), at the cost of fatter formulas;
* Hull-White — Vasicek with a time-dependent drift fitted so the model
  reprices TODAY'S curve exactly: the standard production choice,
  because a rates model that disagrees with the discount curve it
  hedges against is wrong by construction. Bond prices come from the
  curve plus a Gaussian convexity adjustment; no explicit theta(t) is
  needed for pricing.

All prices are for unit face. Simulation steps (exact Gaussian for
Vasicek, full-truncation Euler for CIR) are provided for Monte Carlo
scenario generation. Calibration of (a, sigma) to market instruments is
the caller's optimization exercise; these functions price and simulate
given parameters.
"""

from __future__ import annotations

import math

from quantfinlib.rates.yield_curve import YieldCurve


class ShortRateModels:
    """Static closed-form short-rate pricers and simulation steps."""

    # ------------------------------------------------------------------
    # Vasicek
    # ------------------------------------------------------------------

    @staticmethod
    def vasicek_bond(short_rate: float, a: float, b: float, sigma: float,
                     maturity_years: float) -> float:
        """Vasicek zero-coupon bond price P(t, t+T) given the short rate now."""
        _require_positive(a, "a")
        _require_non_negative(sigma, "sigma")
        _require_positive(maturity_years, "maturityYears")
        b_t = (1 - math.exp(-a * maturity_years)) / a
        log_a = ((b_t - maturity_years) * (b - sigma * sigma / (2 * a * a))
                 - sigma * sigma * b_t * b_t / (4 * a))
        return math.exp(log_a - b_t * short_rate)

    @staticmethod
    def vasicek_yield(short_rate: float, a: float, b: float, sigma: float,
                      maturity_years: float) -> float:
        """The continuously-compounded zero yield implied by ``vasicek_bond``."""
        return -math.log(ShortRateModels.vasicek_bond(
            short_rate, a, b, sigma, maturity_years)) / maturity_years

    @staticmethod
    def vasicek_step(short_rate: float, a: float, b: float, sigma: float,
                     dt_years: float, gaussian: float) -> float:
        """One EXACT Vasicek simulation step (the transition is Gaussian, so
        no discretization error): ``r <- b + (r-b)e^{-a dt} + stdev * z``.
        """
        decay = math.exp(-a * dt_years)
        stdev = sigma * math.sqrt((1 - decay * decay) / (2 * a))
        return b + (short_rate - b) * decay + stdev * gaussian

    # ------------------------------------------------------------------
    # CIR
    # ------------------------------------------------------------------

    @staticmethod
    def cir_bond(short_rate: float, a: float, b: float, sigma: float,
                 maturity_years: float) -> float:
        """CIR zero-coupon bond price P(t, t+T)."""
        _require_positive(a, "a")
        _require_positive(sigma, "sigma")
        _require_positive(maturity_years, "maturityYears")
        _require_non_negative(short_rate, "shortRate")
        h = math.sqrt(a * a + 2 * sigma * sigma)
        exp_ht = math.exp(h * maturity_years)
        denom = 2 * h + (a + h) * (exp_ht - 1)
        b_t = 2 * (exp_ht - 1) / denom
        a_t = (2 * h * math.exp((a + h) * maturity_years / 2) / denom) \
            ** (2 * a * b / (sigma * sigma))
        return a_t * math.exp(-b_t * short_rate)

    @staticmethod
    def cir_feller(a: float, b: float, sigma: float) -> float:
        """The Feller ratio 2ab/sigma^2; >= 1 keeps the CIR rate strictly positive."""
        return 2 * a * b / (sigma * sigma)

    @staticmethod
    def cir_step(short_rate: float, a: float, b: float, sigma: float,
                 dt_years: float, gaussian: float) -> float:
        """One full-truncation Euler CIR step (never sources vol from a negative rate)."""
        r_plus = max(short_rate, 0.0)
        return (short_rate + a * (b - r_plus) * dt_years
                + sigma * math.sqrt(r_plus * dt_years) * gaussian)

    # ------------------------------------------------------------------
    # Hull-White (one factor, fitted to a market curve)
    # ------------------------------------------------------------------

    @staticmethod
    def hull_white_bond(curve: YieldCurve, t_years: float, maturity_years: float,
                        short_rate: float, a: float, sigma: float) -> float:
        """Hull-White zero-coupon bond price P(t, t+T) given the market curve
        and the short rate now. By construction, at ``t = 0`` with
        ``short_rate = f(0, 0)`` this reprices the curve exactly; away from
        it, the Gaussian convexity adjustment applies:

            P(t,T) = [P(0,T)/P(0,t)] * exp(B(f(0,t) - r) - sigma^2 B^2 (1-e^{-2at})/(4a))

        Args:
            curve: today's discount curve.
            t_years: valuation time (0 = today).
            maturity_years: time FROM t to the bond's maturity.
            short_rate: the simulated short rate at t.
            a: mean-reversion speed.
            sigma: short-rate vol.
        """
        _require_positive(a, "a")
        _require_non_negative(sigma, "sigma")
        _require_positive(maturity_years, "maturityYears")
        _require_non_negative(t_years, "tYears")
        b_t = (1 - math.exp(-a * maturity_years)) / a
        p_t_mat = curve.discount_factor(t_years + maturity_years)
        p_t = 1.0 if t_years == 0 else curve.discount_factor(t_years)
        fwd = ShortRateModels.instantaneous_forward(curve, t_years)
        variance = sigma * sigma * (1 - math.exp(-2 * a * t_years)) / (4 * a)
        return (p_t_mat / p_t) * math.exp(b_t * (fwd - short_rate)
                                          - variance * b_t * b_t)

    @staticmethod
    def instantaneous_forward(curve: YieldCurve, t_years: float) -> float:
        """The instantaneous forward rate f(0, t) off the curve, by symmetric
        finite difference of ln P (the curve carries no analytic derivative).
        """
        _require_non_negative(t_years, "tYears")
        h = 1.0 / 365
        if t_years < h:
            # A centered window would clamp into a one-sided average of
            # f near (t+h)/2, not t — biased on a steep money-market end.
            # Second-order one-sided stencil of g(t) = -ln P(t) instead
            # (exact whenever g is locally quadratic).
            g0 = -math.log(curve.discount_factor(t_years))
            g1 = -math.log(curve.discount_factor(t_years + h))
            g2 = -math.log(curve.discount_factor(t_years + 2 * h))
            return (-3 * g0 + 4 * g1 - g2) / (2 * h)
        lo = t_years - h
        hi = t_years + h
        return -(math.log(curve.discount_factor(hi))
                 - math.log(curve.discount_factor(lo))) / (hi - lo)


def _require_positive(x: float, name: str) -> None:
    if not (x > 0) or x == math.inf:
        raise ValueError(f"{name} must be positive and finite")


def _require_non_negative(x: float, name: str) -> None:
    if not (x >= 0) or x == math.inf:
        raise ValueError(f"{name} must be >= 0 and finite")
