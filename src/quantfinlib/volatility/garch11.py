"""Port of Java ``com.quantfinlib.volatility.Garch11``.

GARCH(1,1) volatility model with Gaussian maximum-likelihood fitting:
``h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}``.

Estimation uses variance targeting (``omega = sample_var * (1 - alpha
- beta)``, which pins the unconditional variance to the sample
variance) and a coarse-to-fine grid search over (alpha, beta) —
derivative-free, deterministic, and robust for a two-parameter surface.

The grid sweep is vectorized over cells with NumPy (masking invalid
cells to -inf, exactly the cells the Java ``continue`` prunes), while
the variance recurrence itself is stepped in time order so each cell's
likelihood is the same recursion Java evaluates. Cell selection keeps
Java's strictly-greater/first-wins semantics: C-order flattening of
the (alpha, beta) grid matches the i-outer / j-inner loop nesting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.util import math_utils as mu

_LOG_2PI = math.log(2 * math.pi)


class Garch11:
    """Static GARCH(1,1) fit / conditional variance / forecast (Java parity)."""

    @dataclass(frozen=True)
    class Params:
        omega: float
        alpha: float
        beta: float
        log_likelihood: float

        def persistence(self) -> float:
            return self.alpha + self.beta

        def unconditional_variance(self) -> float:
            return self.omega / (1 - self.alpha - self.beta)

    @staticmethod
    def fit(returns) -> "Garch11.Params":
        """Fits GARCH(1,1) to (demeaned) returns by MLE with variance targeting.

        Raises:
            ValueError: for fewer than 100 returns.
        """
        returns = np.asarray(returns, dtype=float)
        if returns.shape[0] < 100:
            raise ValueError(f"need at least 100 returns, got {returns.shape[0]}")
        r = returns - mu.mean(returns)
        sample_var = mu.variance(r)

        best_alpha, best_beta = 0.05, 0.9
        best_ll = -math.inf
        # Full admissible box (the persistence check prunes invalid cells):
        # a narrower start is a hard cap refinement can barely creep past,
        # pinning low-persistence/high-ARCH series at the box edge.
        alpha_lo, alpha_hi, beta_lo, beta_hi = 0.005, 0.99, 1e-6, 0.999

        for _ in range(3):
            grid = 25
            alphas = alpha_lo + (alpha_hi - alpha_lo) * np.arange(grid + 1) / grid
            betas = beta_lo + (beta_hi - beta_lo) * np.arange(grid + 1) / grid
            a = alphas[:, None]
            b = betas[None, :]
            invalid = (a + b >= 0.9995) | (a <= 0) | (b <= 0)
            ll = _log_likelihood_grid(r, sample_var, a, b)
            ll = np.where(invalid, -math.inf, ll)
            flat = ll.ravel()  # C order == Java's i-outer/j-inner scan
            idx = int(np.argmax(flat))
            if flat[idx] > best_ll:
                best_ll = float(flat[idx])
                best_alpha = float(alphas[idx // (grid + 1)])
                best_beta = float(betas[idx % (grid + 1)])
            # Zoom into the best cell for the next pass.
            alpha_step = (alpha_hi - alpha_lo) / grid
            beta_step = (beta_hi - beta_lo) / grid
            alpha_lo = max(1e-6, best_alpha - 2 * alpha_step)
            alpha_hi = min(0.99, best_alpha + 2 * alpha_step)
            beta_lo = max(1e-6, best_beta - 2 * beta_step)
            beta_hi = min(0.999, best_beta + 2 * beta_step)
        omega = sample_var * (1 - best_alpha - best_beta)
        return Garch11.Params(omega, best_alpha, best_beta, best_ll)

    @staticmethod
    def conditional_variances(returns, params: "Garch11.Params") -> np.ndarray:
        """Conditional variance series under the fitted parameters
        (seeded at the sample variance)."""
        returns = np.asarray(returns, dtype=float)
        mean = mu.mean(returns)
        h = np.empty(returns.shape[0])
        h[0] = mu.variance(returns)
        for t in range(1, returns.shape[0]):
            x = returns[t - 1] - mean
            h[t] = params.omega + params.alpha * x * x + params.beta * h[t - 1]
        return h

    @staticmethod
    def forecast_variance(returns, params: "Garch11.Params", horizon: int) -> float:
        """k-step-ahead variance forecast.

        ``h_{T+k} = uncond + persistence^{k-1} * (h_{T+1} - uncond)`` —
        mean-reverts to the unconditional variance at the persistence rate.

        Raises:
            ValueError: if horizon < 1.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        returns = np.asarray(returns, dtype=float)
        mean = mu.mean(returns)
        h = Garch11.conditional_variances(returns, params)
        last_r = returns[-1] - mean
        next_ = (params.omega + params.alpha * last_r * last_r
                 + params.beta * h[-1])
        uncond = params.unconditional_variance()
        return uncond + params.persistence() ** (horizon - 1) * (next_ - uncond)


def _log_likelihood_grid(r: np.ndarray, sample_var: float,
                         alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Gaussian log-likelihood per (alpha, beta) grid cell.

    Same recursion as the Java scalar ``logLikelihood`` evaluated for
    every cell simultaneously; a cell whose variance path hits h <= 0
    is dead (Java returns -inf immediately, we mask at the end).
    """
    omega = sample_var * (1 - alpha - beta)
    shape = np.broadcast_shapes(alpha.shape, beta.shape)
    h = np.full(shape, sample_var)
    ll = np.zeros(shape)
    dead = np.zeros(shape, dtype=bool)
    for x in r:
        dead |= h <= 0
        hs = np.where(dead, 1.0, h)  # benign stand-in; masked to -inf below
        ll += -0.5 * (_LOG_2PI + np.log(hs) + x * x / hs)
        h = omega + alpha * x * x + beta * h
    return np.where(dead, -math.inf, ll)
