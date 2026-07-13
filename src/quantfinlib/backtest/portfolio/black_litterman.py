"""Black-Litterman expected returns (port of Java ``optimization.BlackLitterman``).

Start from the market-implied equilibrium (reverse optimization of the
market portfolio) and blend in investor views with explicit confidences
— the standard cure for mean-variance optimizers' hypersensitivity to
raw return estimates.

Posterior: ``mu = [(tau*S)^-1 + P' O^-1 P]^-1 [(tau*S)^-1 Pi + P' O^-1 Q]``
with pick matrix P (one row per view), view returns Q, and diagonal view
variances O (smaller = more confident).
"""

from __future__ import annotations

import numpy as np

from quantfinlib.util import math_utils as mu


class BlackLitterman:
    """Static Black-Litterman blending; see the module docstring."""

    @staticmethod
    def implied_equilibrium_returns(risk_aversion: float, covariance,
                                    market_weights) -> np.ndarray:
        """Equilibrium (implied) returns from the market portfolio:
        ``Pi = delta * Sigma * w_mkt``."""
        return risk_aversion * mu.mat_vec(np.asarray(covariance, dtype=float),
                                          np.asarray(market_weights, dtype=float))

    @staticmethod
    def posterior_returns(tau: float, covariance, equilibrium_returns,
                          p, q, omega_diag) -> np.ndarray:
        """Posterior expected returns blending equilibrium and views.

        Args:
            tau: Uncertainty scaling of the prior (typically 0.01-0.05).
            covariance: Asset covariance matrix Sigma.
            equilibrium_returns: Prior (equilibrium) returns Pi.
            p: Pick matrix [views][assets]; empty = no views.
            q: Expected return of each view.
            omega_diag: Variance (uncertainty) of each view.

        Raises:
            ValueError: if q/omega don't align with the views or a view
                variance is non-positive.
        """
        pi = np.asarray(equilibrium_returns, dtype=float)
        n = pi.shape[0]
        p = np.asarray(p, dtype=float).reshape(-1, n) if len(p) else np.empty((0, n))
        q = np.asarray(q, dtype=float)
        omega_diag = np.asarray(omega_diag, dtype=float)
        views = p.shape[0]
        if views == 0:
            return pi.copy()
        if q.shape[0] != views or omega_diag.shape[0] != views:
            raise ValueError("q and omega must have one entry per view")
        cov = np.asarray(covariance, dtype=float)

        # (tau*Sigma)^-1
        prior_precision = mu.inverse(tau * cov)

        # A = (tau*S)^-1 + P' O^-1 P ;  b = (tau*S)^-1 Pi + P' O^-1 Q
        a = prior_precision.copy()
        b = mu.mat_vec(prior_precision, pi)
        for v in range(views):
            if omega_diag[v] <= 0:
                raise ValueError("view variance must be positive")
            inv_omega = 1.0 / omega_diag[v]
            b += p[v] * inv_omega * q[v]
            a += inv_omega * np.outer(p[v], p[v])
        return mu.solve_linear(a, b)
