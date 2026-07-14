"""COMPONENT VaR — Euler allocation of delta-normal VaR.

Port of Java ``com.quantfinlib.risk.ComponentVar``. Under the
delta-normal model the Euler allocation makes the split exact and
additive::

    sigma_p     = sqrt(w' S w)
    marginal_i  = (Sw)_i / sigma_p          d sigma_p / d w_i
    component_i = w_i * marginal_i          sum_i component_i = sigma_p (exact)

Scaled by the same z-score, component VaRs SUM EXACTLY to portfolio VaR
— no "diversification residual" bucket. The three numbers:

* component VaR — how much of today's risk this position owns;
* marginal VaR — how fast VaR moves per unit added;
* incremental VaR — how much VaR disappears if the position is CLOSED
  entirely (NOT component VaR: a large position that hedges the book
  has POSITIVE size, NEGATIVE component, and closing it RAISES VaR).

Sign convention: VaR is reported positive; components carry sign (a
hedge's component is negative). Delta-normal only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.util import math_utils as mu


@dataclass(frozen=True)
class Allocation:
    """portfolio_var: total delta-normal VaR (positive).
    components: per-position component VaR; sums exactly to portfolio_var.
    marginals: per-position marginal VaR (z * (Sw)_i / sigma_p)."""

    portfolio_var: float
    components: np.ndarray
    marginals: np.ndarray


def allocate(weights, covariance, confidence: float) -> Allocation:
    """Euler allocation of delta-normal VaR.

    Args:
        weights: position exposures (currency units), signed.
        covariance: return covariance matrix, symmetric, n x n.
        confidence: e.g. 0.99; z = norm_inv(confidence).

    Raises:
        ValueError: misaligned shapes, non-finite entries, confidence
            outside (0.5, 1), or non-positive portfolio variance (all
            NaN-rejecting gates).
    """
    w = np.asarray(weights, dtype=float)
    c = np.asarray(covariance, dtype=float)
    n = w.shape[0]
    if n == 0 or c.shape[0] != n:
        raise ValueError(
            f"weights ({n}) and covariance ({c.shape[0] if c.ndim else 0}) must align")
    if not (confidence > 0.5) or not (confidence < 1):
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence}")
    if not np.all(np.isfinite(w)):
        raise ValueError("non-finite weight")
    if c.ndim != 2 or c.shape[1] != n:
        raise ValueError(f"covariance rows are not length {n}")
    if not np.all(np.isfinite(c)):
        raise ValueError("non-finite covariance entry")
    sw = c @ w
    variance = float(w @ sw)
    if not (variance > 0):
        raise ValueError(
            "portfolio variance must be > 0 (flat or perfectly hedged book), "
            f"got {variance}")
    sigma = math.sqrt(variance)
    z = mu.norm_inv(confidence)
    marginals = z * sw / sigma
    components = w * marginals
    return Allocation(z * sigma, components, marginals)


def incremental(weights, covariance, confidence: float, i: int) -> float:
    """Incremental VaR of position ``i``: portfolio VaR now minus VaR with
    the position closed (weight zeroed). Positive means closing it REDUCES
    risk; negative means the position is a hedge and closing it raises VaR."""
    w = np.asarray(weights, dtype=float)
    if i < 0 or i >= w.shape[0]:
        raise ValueError(f"position index {i} out of range")
    full = allocate(w, covariance, confidence).portfolio_var
    without = w.copy()
    without[i] = 0.0
    sw = mu.quadratic_form(without, covariance)
    # A book that is flat without this position has zero remaining VaR.
    rest = mu.norm_inv(confidence) * math.sqrt(sw) if sw > 0 else 0.0
    return full - rest
