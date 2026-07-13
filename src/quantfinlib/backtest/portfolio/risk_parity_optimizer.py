"""Risk parity (port of Java ``optimization.RiskParityOptimizer``).

The portfolio where every asset contributes *equally* to total risk
(``w_i * (Sigma w)_i`` equal across assets). Solved by the standard
multiplicative fixed-point iteration on marginal risk contributions —
deterministic and long-only.
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.backtest.portfolio.portfolio_optimizer import Allocation
from quantfinlib.util import math_utils as mu


class RiskParityOptimizer:
    """Static equal-risk-contribution solver; see the module docstring."""

    @staticmethod
    def equal_risk_contribution(expected_returns, covariance) -> Allocation:
        """Equal-risk-contribution weights (expected returns used only for
        reporting).

        Raises:
            ValueError: on a dimension mismatch or a non-positive-variance
                asset (the ERC fixed point does not exist and the
                multiplicative update thrashes on it — refuse, don't spin).
            RuntimeError: if the fixed point does not converge in 10,000
                iterations (refuse rather than hand back a stalled iterate
                as if it were the ERC solution).
        """
        er = np.asarray(expected_returns, dtype=float)
        cov = np.asarray(covariance, dtype=float)
        n = cov.shape[0]
        if er.shape[0] != n:
            raise ValueError(f"expectedReturns ({er.shape[0]}) must align "
                             f"with covariance ({n})")
        if cov.ndim != 2 or cov.shape[1] != n:
            raise ValueError(f"covariance is not {n} x {n}")
        for i in range(n):
            # NaN gate: not (v > 0) is True for NaN, exactly as in Java.
            if not (cov[i, i] > 0):
                raise ValueError(
                    f"asset {i} has non-positive variance: {cov[i, i]}")
        w = np.full(n, 1.0 / n)

        converged = False
        for _ in range(10_000):
            marginal = mu.mat_vec(cov, w)
            port_var = mu.dot(w, marginal)
            target = port_var / n
            rc = w * marginal
            max_deviation = float(np.max(np.abs(rc - target))) / port_var
            # Multiplicative update pulls each contribution toward the target.
            w = w * np.sqrt(target / np.maximum(rc, 1e-16))
            w /= float(np.sum(w))
            if max_deviation < 1e-10:
                converged = True
                break
        if not converged:
            raise RuntimeError("equal-risk-contribution fixed point did not "
                               "converge in 10000 iterations")
        ret = mu.dot(er, w)
        vol = math.sqrt(mu.quadratic_form(w, cov))
        return Allocation(w, ret, vol, 0.0 if vol == 0 else ret / vol)

    @staticmethod
    def risk_contributions(w, covariance) -> np.ndarray:
        """Each asset's fractional contribution to portfolio variance under
        weights ``w``."""
        w = np.asarray(w, dtype=float)
        marginal = mu.mat_vec(np.asarray(covariance, dtype=float), w)
        port_var = mu.dot(w, marginal)
        if port_var == 0:
            return np.zeros(w.shape[0])
        return w * marginal / port_var
