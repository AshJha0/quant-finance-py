"""Port of Java ``com.quantfinlib.volatility.Egarch11``.

EGARCH(1,1) — Nelson's exponential GARCH, the LOG-variance dynamics the
plain family cannot express:

    ln h_t = omega + beta * ln h_{t-1}
             + alpha * (|z_{t-1}| - sqrt(2/pi)) + gamma * z_{t-1}

with standardized shocks ``z = r / sqrt(h)``. Two things the log form
buys: no positivity constraints AT ALL (any parameter signs give a valid
variance — the exp does the work Garch11's omega/alpha/beta >= 0
constraints do), and leverage as a SIGN — ``gamma < 0`` means a down
move raises tomorrow's volatility more than an equal up move. The only
formal stationarity constraint is ``|beta| < 1``.

Estimation mirrors the family: Gaussian MLE over a coarse-to-fine grid
spanning the EMPIRICALLY PLAUSIBLE box — alpha in [0, 0.9], gamma in
[-0.9, 0.9], beta in [0, 0.995]. Negative alpha or beta, while formally
admissible in the log form, are not searched (stated, not hidden: they
describe oscillating log-variance no asset-return series exhibits).
omega is targeted to the sample's log variance (``omega = (1 - beta) *
ln(sample_var)`` — an approximation, since E[ln h] <= ln E[h] by
Jensen; stated, not hidden). One-step-ahead ``next_variance`` is exact;
multi-step forecasts are deliberately REFUSED — iterating the log
recursion forecasts the MEDIAN variance, not the mean, and quietly
returning it as "the forecast" is the kind of lie this library refuses
(``Garch11``/``GjrGarch11`` forecast multi-step honestly; use them when
you need horizons).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.util import math_utils as mu

_LOG_2PI = math.log(2 * math.pi)
_E_ABS_Z = math.sqrt(2 / math.pi)


class Egarch11:
    """Static EGARCH(1,1) fit / conditional variance / one-step forecast."""

    @dataclass(frozen=True)
    class Params:
        omega: float
        alpha: float
        gamma: float
        beta: float
        log_likelihood: float

        def unconditional_log_variance(self) -> float:
            """Long-run (unconditional) LOG variance omega / (1 - beta)."""
            return self.omega / (1 - self.beta)

    @staticmethod
    def fit(returns) -> "Egarch11.Params":
        """Fits EGARCH(1,1) to (demeaned) returns by grid MLE.

        Raises:
            ValueError: for fewer than 100 returns, non-finite returns,
                or a series that carries no variance.
        """
        returns = np.asarray(returns, dtype=float)
        if returns.shape[0] < 100:
            raise ValueError(f"need at least 100 returns, got {returns.shape[0]}")
        if not np.all(np.isfinite(returns)):
            raise ValueError("returns must be finite")
        r = returns - mu.mean(returns)
        sample_var = mu.variance(r)
        if not (sample_var > 0):
            raise ValueError("returns carry no variance")
        ln_var = math.log(sample_var)

        best_alpha = 0.1
        best_gamma = 0.0
        best_beta = 0.9
        best_ll = -math.inf
        # The empirically plausible box (see module doc): |beta| < 1 is
        # the only formal constraint, but negative alpha/beta describe
        # oscillating log-variance no return series exhibits.
        a_lo, a_hi = 0.0, 0.9
        g_lo, g_hi = -0.9, 0.9
        b_lo, b_hi = 0.0, 0.995

        for _ in range(3):
            grid = 12
            alphas = a_lo + (a_hi - a_lo) * np.arange(grid + 1) / grid
            gammas = g_lo + (g_hi - g_lo) * np.arange(grid + 1) / grid
            betas = b_lo + (b_hi - b_lo) * np.arange(grid + 1) / grid
            ll = _log_likelihood_grid(r, ln_var, alphas[:, None, None],
                                      gammas[None, :, None], betas[None, None, :])
            flat = ll.ravel()  # C order == Java's i/j/k scan order
            idx = int(np.argmax(flat))
            if flat[idx] > best_ll:
                best_ll = float(flat[idx])
                n1 = grid + 1
                best_alpha = float(alphas[idx // (n1 * n1)])
                best_gamma = float(gammas[(idx // n1) % n1])
                best_beta = float(betas[idx % n1])
            a_step = (a_hi - a_lo) / grid
            g_step = (g_hi - g_lo) / grid
            b_step = (b_hi - b_lo) / grid
            a_lo = max(0.0, best_alpha - 2 * a_step)
            a_hi = min(0.9, best_alpha + 2 * a_step)
            g_lo = max(-0.9, best_gamma - 2 * g_step)
            g_hi = min(0.9, best_gamma + 2 * g_step)
            b_lo = max(0.0, best_beta - 2 * b_step)
            b_hi = min(0.995, best_beta + 2 * b_step)
        omega = (1 - best_beta) * ln_var
        return Egarch11.Params(omega, best_alpha, best_gamma, best_beta, best_ll)

    @staticmethod
    def conditional_variances(returns, p: "Egarch11.Params") -> np.ndarray:
        """Conditional variance series under the fitted parameters
        (positive BY CONSTRUCTION — the log form needs no clamps)."""
        returns = _require_series(returns)
        mean = mu.mean(returns)
        h = np.empty(returns.shape[0])
        ln_h = math.log(mu.variance(returns))
        h[0] = math.exp(ln_h)
        for t in range(1, returns.shape[0]):
            x = returns[t - 1] - mean
            z = x / math.sqrt(h[t - 1])
            ln_h = (p.omega + p.beta * ln_h
                    + p.alpha * (abs(z) - _E_ABS_Z) + p.gamma * z)
            h[t] = math.exp(ln_h)
        return h

    @staticmethod
    def next_variance(returns, p: "Egarch11.Params") -> float:
        """One-step-ahead variance — EXACT (tomorrow's ln h is
        deterministic today)."""
        returns = np.asarray(returns, dtype=float)
        h = Egarch11.conditional_variances(returns, p)
        mean = mu.mean(returns)
        x = returns[-1] - mean
        last = h[-1]
        z = x / math.sqrt(last)
        return math.exp(p.omega + p.beta * math.log(last)
                        + p.alpha * (abs(z) - _E_ABS_Z) + p.gamma * z)

    @staticmethod
    def forecast_variance(returns, p: "Egarch11.Params", horizon: int) -> float:
        """REFUSED for every horizon: iterating the EGARCH log recursion
        forecasts the MEDIAN variance, not the mean, and quietly
        returning it as "the forecast" would be a lie. Use ``Garch11``
        or ``GjrGarch11`` when you need multi-step horizons; one exact
        step ahead is ``next_variance``.

        (The Java reference expresses the same refusal by not offering
        the method at all; the Python port raises so a caller reaching
        for the family-wide name gets the explanation, not an
        AttributeError.)

        Raises:
            RuntimeError: always.
        """
        raise RuntimeError(
            "EGARCH multi-step variance forecasts are refused: iterating the "
            "log recursion forecasts the MEDIAN variance, not the mean — use "
            "Garch11/GjrGarch11 for horizons, or next_variance for the exact "
            "one-step-ahead value")


def _require_series(returns) -> np.ndarray:
    """The recursion needs a positive, finite starting variance — a
    constant or NaN-bearing series would return h = 0 or NaN silently,
    contradicting the "positive by construction" guarantee."""
    returns = np.asarray(returns, dtype=float)
    if returns.shape[0] < 2:
        raise ValueError("need >= 2 returns")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must be finite")
    if not (mu.variance(returns) > 0):
        raise ValueError("returns carry no variance")
    return returns


def _log_likelihood_grid(r: np.ndarray, ln_sample_var: float, alpha: np.ndarray,
                         gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Gaussian log-likelihood per (alpha, gamma, beta) grid cell.

    A runaway log-variance (non-finite or |ln h| > 100) means those
    parameters are junk for this data — the cell is rejected (-inf)
    rather than allowed to overflow exp(), exactly as in Java.
    """
    shape = np.broadcast_shapes(alpha.shape, gamma.shape, beta.shape)
    ln_h = np.full(shape, ln_sample_var)
    omega = (1 - beta) * ln_sample_var
    ll = np.zeros(shape)
    dead = np.zeros(shape, dtype=bool)
    for x in r:
        dead |= ~np.isfinite(ln_h) | (np.abs(ln_h) > 100)
        lnh_s = np.where(dead, 0.0, ln_h)  # frozen benign value for dead cells
        h = np.exp(lnh_s)
        ll += -0.5 * (_LOG_2PI + lnh_s + x * x / h)
        z = x / np.sqrt(h)
        ln_h = omega + beta * lnh_s + alpha * (np.abs(z) - _E_ABS_Z) + gamma * z
    return np.where(dead, -math.inf, ll)
