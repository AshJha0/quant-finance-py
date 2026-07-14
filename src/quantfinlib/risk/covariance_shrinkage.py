"""LEDOIT-WOLF covariance shrinkage (2004) toward the scaled identity.

Port of Java ``com.quantfinlib.risk.CovarianceShrinkage``. The sample
covariance matrix is the MAXIMALLY overfit estimate; the estimator
shrinks toward ``mu * I`` (mu = average sample variance)::

    Sigma* = delta * mu*I + (1 - delta) * S

with intensity ``delta = b2 / d2`` chosen FROM THE DATA:
``d2 = ||S - mu*I||_F^2`` measures how far the sample matrix is from
the target and ``b2`` estimates how much of that distance is pure
sampling noise (average Frobenius distance of single-observation outer
products from S, over T^2, clamped to d2). The sample covariance uses
the POPULATION (/T) convention — the Ledoit-Wolf convention, kept so
pins transfer from the Java port. The result is positive-definite for
delta > 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Result:
    """matrix: the shrunk covariance Sigma*.
    intensity: delta in [0, 1] — how far toward the target.
    target: mu, the average sample variance (the target's diagonal)."""

    matrix: np.ndarray
    intensity: float
    target: float


def ledoit_wolf(returns) -> Result:
    """Shrinks the sample covariance of a T x N return matrix.

    Args:
        returns: ``returns[t][j]`` = period-t return of asset j;
            T >= 2, N >= 1, rectangular, finite.

    Raises:
        ValueError: too few observations, no assets, a ragged matrix,
            or non-finite entries.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("ragged matrix")
    t, n = r.shape
    if t < 2:
        raise ValueError(f"need >= 2 observations, got {t}")
    if n < 1:
        raise ValueError("need >= 1 asset")
    if not np.all(np.isfinite(r)):
        raise ValueError("non-finite return")

    # Demeaned data and the sample covariance S (population, /T — the
    # Ledoit-Wolf convention).
    x = r - np.mean(r, axis=0)
    s = (x.T @ x) / t
    m = float(np.trace(s)) / n

    # d^2 = ||S - mu I||_F^2 / n (LW normalize by n; cancels in the ratio
    # as long as both d2 and b2 use the same normalization).
    dev = s - m * np.eye(n)
    d2 = float(np.sum(dev * dev)) / n

    # b^2 = (1/T^2) sum_t ||x_t x_t' - S||_F^2 / n, clamped to d^2.
    outer = x[:, :, None] * x[:, None, :] - s     # T x N x N
    b2 = float(np.sum(outer * outer)) / (float(t) * t * n)
    b2 = min(b2, d2)

    delta = b2 / d2 if d2 > 0 else 1.0  # degenerate S == mu I: any delta identical
    shrunk = delta * m * np.eye(n) + (1 - delta) * s
    return Result(shrunk, delta, m)


def shrink(returns) -> np.ndarray:
    """Convenience: the shrunk matrix only."""
    return ledoit_wolf(returns).matrix
