"""Port of Java ``com.quantfinlib.volatility.GjrGarch11``.

GJR-GARCH(1,1,1) — GARCH with the LEVERAGE term equity markets demand:

    h_t = omega + (alpha + gamma * 1{r<0}) * r_{t-1}^2 + beta * h_{t-1}

A down move raises tomorrow's variance by ``alpha + gamma``; an equal
up move by only ``alpha``. That asymmetry (Glosten-Jagannathan-Runkle,
1993) is not a refinement — on equity indices gamma is typically LARGER
than alpha, and a symmetric ``Garch11`` systematically underestimates
post-selloff volatility. Fitting a GJR and finding gamma ~ 0 is itself
information: the series has no asymmetry and the simpler model suffices.

Estimation mirrors ``Garch11``: Gaussian MLE with variance targeting
(``omega = sample_var * (1 - alpha - gamma/2 - beta)``, since negative
returns carry half the mass under symmetry) and a coarse-to-fine grid
over (alpha, gamma, beta) spanning the FULL admissible box — a narrower
starting box is a hard cap the refinement passes can only creep past by
~2 steps per pass, silently pinning extreme fits at the box edge.
Persistence is ``alpha + gamma/2 + beta``. Grid vectorization and cell
selection follow the same scheme as ``garch11`` (C-order flatten of the
(alpha, gamma, beta) grid matches Java's i/j/k loop nesting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.util import math_utils as mu

_LOG_2PI = math.log(2 * math.pi)


class GjrGarch11:
    """Static GJR-GARCH(1,1) fit / conditional variance / forecast."""

    @dataclass(frozen=True)
    class Params:
        omega: float
        alpha: float
        gamma: float
        beta: float
        log_likelihood: float

        def persistence(self) -> float:
            """alpha + gamma/2 + beta — mean-reversion persistence under
            symmetric returns."""
            return self.alpha + self.gamma / 2 + self.beta

        def unconditional_variance(self) -> float:
            return self.omega / (1 - self.persistence())

    @staticmethod
    def fit(returns) -> "GjrGarch11.Params":
        """Fits GJR-GARCH(1,1) to (demeaned) returns by MLE with variance
        targeting.

        Raises:
            ValueError: for fewer than 100 returns.
        """
        returns = np.asarray(returns, dtype=float)
        if returns.shape[0] < 100:
            raise ValueError(f"need at least 100 returns, got {returns.shape[0]}")
        r = returns - mu.mean(returns)
        sample_var = mu.variance(r)

        best_alpha = 0.03
        best_gamma = 0.05
        best_beta = 0.9
        best_ll = -math.inf
        a_lo, a_hi = 0.0, 0.99
        g_lo, g_hi = 0.0, 1.9
        b_lo, b_hi = 1e-6, 0.999

        for _ in range(3):
            grid = 12  # 13^3 ~ 2,200 cells per pass
            alphas = a_lo + (a_hi - a_lo) * np.arange(grid + 1) / grid
            gammas = g_lo + (g_hi - g_lo) * np.arange(grid + 1) / grid
            betas = b_lo + (b_hi - b_lo) * np.arange(grid + 1) / grid
            a = alphas[:, None, None]
            g = gammas[None, :, None]
            b = betas[None, None, :]
            invalid = ((a < 0) | (g < 0) | (b <= 0)
                       | (a + g / 2 + b >= 0.9995))
            ll = _log_likelihood_grid(r, sample_var, a, g, b)
            ll = np.where(invalid, -math.inf, ll)
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
            a_hi = min(0.99, best_alpha + 2 * a_step)
            g_lo = max(0.0, best_gamma - 2 * g_step)
            g_hi = min(1.9, best_gamma + 2 * g_step)
            b_lo = max(1e-6, best_beta - 2 * b_step)
            b_hi = min(0.999, best_beta + 2 * b_step)
        omega = sample_var * (1 - best_alpha - best_gamma / 2 - best_beta)
        return GjrGarch11.Params(omega, best_alpha, best_gamma, best_beta, best_ll)

    @staticmethod
    def conditional_variances(returns, params: "GjrGarch11.Params") -> np.ndarray:
        """Conditional variance series under the fitted parameters
        (seeded at the sample variance)."""
        returns = np.asarray(returns, dtype=float)
        mean = mu.mean(returns)
        h = np.empty(returns.shape[0])
        h[0] = mu.variance(returns)
        for t in range(1, returns.shape[0]):
            x = returns[t - 1] - mean
            arch = params.alpha + params.gamma if x < 0 else params.alpha
            h[t] = params.omega + arch * x * x + params.beta * h[t - 1]
        return h

    @staticmethod
    def forecast_variance(returns, params: "GjrGarch11.Params",
                          horizon: int) -> float:
        """k-step-ahead variance forecast — mean-reverts to the
        unconditional variance at the persistence rate, exactly as
        ``Garch11`` but with the asymmetric first step.

        Raises:
            ValueError: if horizon < 1.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        returns = np.asarray(returns, dtype=float)
        mean = mu.mean(returns)
        h = GjrGarch11.conditional_variances(returns, params)
        last_r = returns[-1] - mean
        arch = params.alpha + params.gamma if last_r < 0 else params.alpha
        next_ = params.omega + arch * last_r * last_r + params.beta * h[-1]
        uncond = params.unconditional_variance()
        return uncond + params.persistence() ** (horizon - 1) * (next_ - uncond)


def _log_likelihood_grid(r: np.ndarray, sample_var: float, alpha: np.ndarray,
                         gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Gaussian log-likelihood per (alpha, gamma, beta) grid cell.

    Same recursion as the Java scalar ``logLikelihood``; cells with a
    non-positive targeted omega or an h <= 0 variance path are dead
    (Java returns -inf, we mask at the end).
    """
    omega = sample_var * (1 - alpha - gamma / 2 - beta)
    shape = np.broadcast_shapes(alpha.shape, gamma.shape, beta.shape)
    dead = np.broadcast_to(omega <= 0, shape).copy()
    h = np.full(shape, sample_var)
    ll = np.zeros(shape)
    arch_neg = alpha + gamma
    for x in r:
        dead |= h <= 0
        hs = np.where(dead, 1.0, h)
        ll += -0.5 * (_LOG_2PI + np.log(hs) + x * x / hs)
        arch = arch_neg if x < 0 else alpha
        h = omega + arch * x * x + beta * h
    return np.where(dead, -math.inf, ll)
