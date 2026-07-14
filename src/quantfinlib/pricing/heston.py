"""Heston (1993) stochastic-volatility pricing (port of Java
``com.quantfinlib.pricing.Heston``).

Variance follows its own mean-reverting square-root process, correlated
with spot::

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + sigma_v sqrt(v) dW2,  d<W1,W2> = rho dt

Pricing is semi-analytic: the European call is two probabilities
recovered from the model's characteristic function by numerical
integration. This implementation uses the "little Heston trap"
formulation (Albrecher et al. 2007), which is numerically stable for
long maturities where the original 1993 branch-cut form explodes, and
fixed-step Simpson integration on a damped integrand. The integration
window and step count stretch with the parameters — a fixed window
silently truncates a 1-week 4%-vol option's integral.

The Feller condition ``2 kappa theta >= sigma_v^2`` keeps the variance
strictly positive; parameters violating it are accepted (markets
calibrate there constantly). No calibration is shipped. Research lane.

Port note: the Simpson sum is evaluated with numpy over the u-grid —
the same arithmetic elementwise as the Java loop; only the summation
order differs (well inside test tolerance). ``call_monte_carlo`` draws
from ``numpy.random.default_rng`` instead of ``java.util.Random``; it
is deterministic per seed and only ever compared statistically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HestonParams:
    """Model parameters. ``feller()`` tells you which regime you are in."""

    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        if (not (self.kappa > 0) or not (self.theta > 0) or not (self.sigma_v > 0)
                or not (self.v0 > 0) or not (-1 <= self.rho <= 1)
                or self.kappa == math.inf or self.theta == math.inf
                or self.sigma_v == math.inf or self.v0 == math.inf):
            raise ValueError(
                "need kappa, theta, sigmaV, v0 > 0 (finite) and rho in [-1, 1]")

    def feller(self) -> float:
        """The Feller ratio 2 kappa theta / sigma_v^2; >= 1 keeps variance positive."""
        return 2 * self.kappa * self.theta / (self.sigma_v * self.sigma_v)


_BASE_INTEGRATION_STEPS = 4_096   # Simpson: must be even
# Window sized for the 20%-vol / 1y reference; the integrand's decay
# scale is ~1/(sigma_eff * sqrt(T)), NOT parameter-free, so _probability
# stretches both the window and the step count for short-dated / low-vol
# inputs (a fixed 200 silently truncates a 1-week 4%-vol option's integral).
_BASE_INTEGRATION_LIMIT = 200.0
_REFERENCE_SIGMA_SQRT_T = 0.2
_MAX_STRETCH = 64.0


class Heston:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def call(spot: float, strike: float, rate: float, div_yield: float,
             time_years: float, p: HestonParams) -> float:
        """European call under Heston (semi-analytic)."""
        _require_market_inputs(spot, strike, rate, div_yield, time_years)
        forward = spot * math.exp((rate - div_yield) * time_years)
        df = math.exp(-rate * time_years)
        p1 = _probability(forward, strike, time_years, p, True)
        p2 = _probability(forward, strike, time_years, p, False)
        return df * (forward * p1 - strike * p2)

    @staticmethod
    def put(spot: float, strike: float, rate: float, div_yield: float,
            time_years: float, p: HestonParams) -> float:
        """European put via put-call parity."""
        forward = spot * math.exp((rate - div_yield) * time_years)
        df = math.exp(-rate * time_years)
        return (Heston.call(spot, strike, rate, div_yield, time_years, p)
                - df * (forward - strike))

    @staticmethod
    def call_monte_carlo(spot: float, strike: float, rate: float,
                         div_yield: float, time_years: float, p: HestonParams,
                         steps: int, paths: int, seed: int) -> float:
        """Full-truncation Euler Monte Carlo — the pricing cross-check (used
        by the tests to validate the semi-analytic integral, and usable for
        payoffs the closed form cannot reach). Deterministic per seed."""
        _require_market_inputs(spot, strike, rate, div_yield, time_years)
        if steps < 1 or paths < 1:
            raise ValueError("need steps >= 1 and paths >= 1")
        rng = np.random.default_rng(seed)
        dt = time_years / steps
        sqrt_dt = math.sqrt(dt)
        rho_bar = math.sqrt(1 - p.rho * p.rho)
        drift = (rate - div_yield) * dt
        log_s = np.full(paths, math.log(spot))
        v = np.full(paths, p.v0)
        for _ in range(steps):
            v_plus = np.maximum(v, 0.0)              # full truncation
            z1 = rng.standard_normal(paths)
            z2 = p.rho * z1 + rho_bar * rng.standard_normal(paths)
            log_s += drift - 0.5 * v_plus * dt + np.sqrt(v_plus) * sqrt_dt * z1
            v += (p.kappa * (p.theta - v_plus) * dt
                  + p.sigma_v * np.sqrt(v_plus) * sqrt_dt * z2)
        payoff = np.maximum(np.exp(log_s) - strike, 0.0)
        return math.exp(-rate * time_years) * float(np.sum(payoff)) / paths


def _require_market_inputs(spot: float, strike: float, rate: float,
                           div_yield: float, time_years: float) -> None:
    if (not (spot > 0) or not (strike > 0) or not (time_years > 0)
            or spot == math.inf or strike == math.inf
            or not (math.isfinite(rate) and math.isfinite(div_yield))
            or time_years == math.inf):
        raise ValueError("invalid market inputs")


def _probability(forward: float, strike: float, t: float,
                 p: HestonParams, first: bool) -> float:
    """P1 (delta-measure) or P2 (risk-neutral) via Simpson integration of
    the little-trap characteristic function."""
    log_moneyness = math.log(forward / strike)
    sigma_eff = math.sqrt(max(p.v0, p.theta))
    stretch = min(_MAX_STRETCH,
                  max(1.0, _REFERENCE_SIGMA_SQRT_T / (sigma_eff * math.sqrt(t))))
    # Scale the step count with the window so resolution is preserved
    # (an even multiple of an even base stays Simpson-legal).
    steps = _BASE_INTEGRATION_STEPS * int(math.ceil(stretch))
    h = _BASE_INTEGRATION_LIMIT * stretch / steps
    i = np.arange(steps + 1)
    u = i * h + 1e-9          # dodge the u=0 singularity of the integrand
    weight = np.where((i == 0) | (i == steps), 1.0, np.where(i % 2 == 1, 4.0, 2.0))
    total = float(np.sum(weight * _integrand(u, log_moneyness, t, p, first)))
    return 0.5 + (h / 3) * total / math.pi


def _integrand(u: np.ndarray, x: float, t: float, p: HestonParams,
               first: bool) -> np.ndarray:
    """Re{e^{iu x} phi(u)/(iu)} with the little-trap phi. All complex math
    inlined exactly as in the Java source, vectorized over u."""
    kappa = p.kappa
    theta = p.theta
    sv = p.sigma_v
    rho = p.rho
    v0 = p.v0
    # u' and b depend on which probability we are computing.
    u_sign = 0.5 if first else -0.5
    b = kappa - rho * sv if first else kappa

    # d^2 = (b - i rho sigma u)^2 + sigma^2 (u^2 - 2i u_sign u)
    re_a = b
    im_a = -rho * sv * u                       # (b - i rho sigma u)
    re_a2 = re_a * re_a - im_a * im_a
    im_a2 = 2 * re_a * im_a
    re_b = sv * sv * (u * u)                   # sigma^2 u^2  (real part add)
    im_b = sv * sv * (-2 * u_sign * u)         # sigma^2 (-2i u_sign u)
    re_d2 = re_a2 + re_b
    im_d2 = im_a2 + im_b
    # Complex sqrt of (re_d2 + i im_d2), principal branch — STABLE form:
    # re_d2 = b^2 + sigma^2 u^2 (1 - rho^2) >= 0 always, so the real part
    # is the large component; the imaginary part MUST come from
    # im_d2/(2 re_d), not from sqrt((mod - re)/2), which underflows to 0
    # when |im_d2| << re_d2 and silently zeroes the phase slope near u = 0
    # (a first-node corruption worth ~0.5% of an ATM price — caught by the
    # BS-limit test).
    mod_d2 = np.hypot(re_d2, im_d2)
    re_d = np.sqrt((mod_d2 + re_d2) / 2)
    im_d = im_d2 / (2 * re_d)

    # g2 = (b - i rho sigma u - d)/(b - i rho sigma u + d) — the LITTLE-TRAP ratio
    re_num = re_a - re_d
    im_num = im_a - im_d
    re_den = re_a + re_d
    im_den = im_a + im_d
    den2 = re_den * re_den + im_den * im_den
    re_g = (re_num * re_den + im_num * im_den) / den2
    im_g = (im_num * re_den - re_num * im_den) / den2

    # e^{-d t}
    exp_re = np.exp(-re_d * t)
    re_edt = exp_re * np.cos(-im_d * t)
    im_edt = exp_re * np.sin(-im_d * t)

    # 1 - g e^{-dt}  and  1 - g
    re_one_minus_ge = 1 - (re_g * re_edt - im_g * im_edt)
    im_one_minus_ge = -(re_g * im_edt + im_g * re_edt)
    re_one_minus_g = 1 - re_g
    im_one_minus_g = -im_g

    # C = (kappa theta / sigma^2) [(b - i rho sigma u - d) t
    #                              - 2 ln((1 - g e^{-dt})/(1 - g))]
    d2 = re_one_minus_g * re_one_minus_g + im_one_minus_g * im_one_minus_g
    ratio_re = (re_one_minus_ge * re_one_minus_g + im_one_minus_ge * im_one_minus_g) / d2
    ratio_im = (im_one_minus_ge * re_one_minus_g - re_one_minus_ge * im_one_minus_g) / d2
    log_mod_ratio = 0.5 * np.log(ratio_re * ratio_re + ratio_im * ratio_im)
    # Principal-branch arg is valid ONLY for the little-trap ratio with
    # Re(d) >= 0 (Lord & Kahl: its winding number is zero for all t) —
    # the 1993 g or a flipped d branch would need phase tracking here.
    arg_ratio = np.arctan2(ratio_im, ratio_re)
    coef = kappa * theta / (sv * sv)
    re_c = coef * ((re_a - re_d) * t - 2 * log_mod_ratio)
    im_c = coef * ((im_a - im_d) * t - 2 * arg_ratio)

    # D = ((b - i rho sigma u - d)/sigma^2) (1 - e^{-dt})/(1 - g e^{-dt})
    re_one_minus_edt = 1 - re_edt
    im_one_minus_edt = -im_edt
    d2b = re_one_minus_ge * re_one_minus_ge + im_one_minus_ge * im_one_minus_ge
    re_frac = (re_one_minus_edt * re_one_minus_ge + im_one_minus_edt * im_one_minus_ge) / d2b
    im_frac = (im_one_minus_edt * re_one_minus_ge - re_one_minus_edt * im_one_minus_ge) / d2b
    re_dd = ((re_a - re_d) * re_frac - (im_a - im_d) * im_frac) / (sv * sv)
    im_dd = ((re_a - re_d) * im_frac + (im_a - im_d) * re_frac) / (sv * sv)

    # phi = exp(C + D v0 + iu x)   (forward-measure form: no drift term)
    re_exp = re_c + re_dd * v0
    im_exp = im_c + im_dd * v0 + u * x
    mod = np.exp(re_exp)
    # integrand = Re{ phi/(iu) } = Re{ phi (-i)/u }; Re{(a+bi)(-i/u)} = b/u.
    return mod * np.sin(im_exp) / u
