"""Principal component analysis of a covariance matrix — the risk-factor
compressor.

Port of Java ``com.quantfinlib.risk.Pca``: the cyclic Jacobi eigenvalue
algorithm — exact for symmetric matrices, dependency-free, O(n^3) per
sweep, converging in a handful of sweeps at risk dimensions.
Eigenvalues clip at zero: a covariance matrix is PSD in exact
arithmetic, and a tiny negative eigenvalue is numerical noise, not an
imaginary risk factor.
"""

from __future__ import annotations

import math

import numpy as np


class Pca:
    """Decomposes a symmetric covariance matrix (n x n).

    Asymmetry beyond floating-point noise is rejected — a typo'd
    covariance must not silently symmetrize into a different matrix.

    Raises:
        ValueError: empty/non-square/non-finite/asymmetric input.
        RuntimeError: Jacobi non-convergence after 100 sweeps (a
            symmetric matrix that has not converged by then is a
            signal, not a diagonal to report).
    """

    def __init__(self, covariance):
        cov = np.asarray(covariance, dtype=float)
        n = cov.shape[0]
        if n < 1:
            raise ValueError("empty matrix")
        if cov.ndim != 2 or cov.shape[1] != n:
            raise ValueError("matrix must be square")
        if not np.all(np.isfinite(cov)):
            raise ValueError("non-finite covariance entry")
        if np.any(np.abs(cov - cov.T) > 1e-9 * (1 + np.abs(cov))):
            raise ValueError("matrix must be symmetric")
        m = cov.copy()
        # Normalize to unit scale before Jacobi: entries near 1e155 would
        # overflow the squared-norm accumulators to Infinity, and entries
        # near 1e-155 would underflow them to zero — either way corrupting
        # the convergence thresholds. Eigenvalues scale back linearly.
        scale = float(np.max(np.abs(m)))
        if scale > 0:
            m /= scale
        else:
            scale = 1.0                    # zero matrix: nothing to scale
        v = np.eye(n)
        _jacobi(m, v)

        # Extract diagonal eigenvalues + columns, sort descending (stable).
        eig = np.diagonal(m).copy()
        order = np.argsort(-eig, kind="stable")
        self._eigenvalues = np.maximum(0.0, eig[order]) * scale  # PSD clip
        self._eigenvectors = v[:, order].T   # [component][factor], unit length
        self._total_variance = float(np.sum(self._eigenvalues))

    def eigenvalue(self, c: int) -> float:
        """Variance carried by component ``c`` (descending order)."""
        return float(self._eigenvalues[c])

    def loading(self, c: int, f: int) -> float:
        """Unit loading of factor ``f`` on component ``c``."""
        return float(self._eigenvectors[c, f])

    def explained_variance(self, k: int) -> float:
        """Fraction of total variance the first ``k`` components explain."""
        if k < 1 or k > self._eigenvalues.shape[0]:
            raise ValueError("k out of range")
        if self._total_variance <= 0:
            return 0.0
        return float(np.sum(self._eigenvalues[:k])) / self._total_variance

    def size(self) -> int:
        return self._eigenvalues.shape[0]


def _jacobi(m: np.ndarray, v: np.ndarray) -> None:
    n = m.shape[0]
    # Convergence must be RELATIVE to the matrix's own scale — the caller
    # chooses the units (return fractions vs currency^2, ~1e16 apart), so
    # absolute thresholds either never trigger or trigger too early.
    norm_sq = float(np.sum(m * m))
    if norm_sq == 0:
        return                             # the zero matrix is already diagonal
    off_tol = 1e-24 * norm_sq
    # The per-element skip threshold must be CONSISTENT with the total
    # off-diagonal stop: if every element sits below skip_tol, their summed
    # squares must already satisfy off < off_tol — otherwise all rotations
    # get skipped while the stop never fires, and the sweep loop spins to
    # the non-convergence raise on a converged matrix.
    pairs = n * (n - 1) / 2.0 if n > 1 else 1.0
    skip_tol = math.sqrt(off_tol / pairs)
    for _sweep in range(100):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += m[p, q] * m[p, q]
        if off < off_tol:
            return
        for p in range(n):
            for q in range(p + 1, n):
                if abs(m[p, q]) < skip_tol:
                    continue
                theta = (m[q, q] - m[p, p]) / (2 * m[p, q])
                t = math.copysign(1.0, theta) / (abs(theta)
                                                 + math.sqrt(theta * theta + 1))
                if theta == 0:
                    t = 1.0
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                _rotate(m, v, p, q, c, s)
    # Jacobi converges quadratically — a symmetric matrix that has not
    # converged in 100 sweeps is a signal, not a diagonal to report.
    raise RuntimeError("Jacobi failed to converge in 100 sweeps")


def _rotate(m: np.ndarray, v: np.ndarray, p: int, q: int,
            c: float, s: float) -> None:
    mip = m[:, p].copy()
    miq = m[:, q].copy()
    m[:, p] = c * mip - s * miq
    m[:, q] = s * mip + c * miq
    mpi = m[p, :].copy()
    mqi = m[q, :].copy()
    m[p, :] = c * mpi - s * mqi
    m[q, :] = s * mpi + c * mqi
    vip = v[:, p].copy()
    viq = v[:, q].copy()
    v[:, p] = c * vip - s * viq
    v[:, q] = s * vip + c * viq
