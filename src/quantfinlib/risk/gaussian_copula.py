"""Gaussian and Student-t copula samplers.

Port of Java ``com.quantfinlib.risk.GaussianCopula``: dependence
separated from marginals. Both samplers emit correlated UNIFORMS — feed
them through your marginals' inverse CDFs. The Gaussian copula has NO
tail dependence; the Student-t copula with few degrees of freedom has
strong SYMMETRIC tail dependence and converges to the Gaussian as
``df -> inf``.

Deviations from Java, documented:
    * The random source is a ``numpy.random.Generator`` (caller-owned,
      deterministic per seed) instead of ``java.util.Random`` — a
      different stream, so seeded pins are statistical, not
      value-exact.
    * The Java sampler rejects ``out == scratch`` because its in-place
      ``correlate`` loop reads ``scratch`` while writing ``out``. The
      NumPy matrix product materializes a temporary, so aliasing would
      not actually corrupt the draw here — but the contract is kept
      (``ValueError`` on the same array or overlapping views) so code
      that is wrong against the Java API stays wrong here too instead
      of silently depending on a NumPy implementation detail.
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils as mu

_MIN_NORMAL = 2.2250738585072014e-308  # Java Double.MIN_NORMAL


def cholesky(matrix) -> np.ndarray:
    """Cholesky factorization; fails loudly on a non-positive-definite input.

    The pivot tolerance is RELATIVE to the matrix's own diagonal scale:
    correlation matrices (diag 1) factor exactly as before, and
    genuinely positive-definite covariances quoted in small units are
    not rejected (a 0.5bp-daily-vol factor has variance ~2.5e-9; an
    absolute 1e-12 floor called a valid pair of them "not
    positive-definite").

    Raises:
        ValueError: non-square input, or a pivot at or below the
            relative floor (NaN-rejecting ``not (pivot > floor)`` gate).
    """
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    if m.ndim != 2 or m.shape[1] != n:
        raise ValueError("matrix must be square")
    max_diag = 0.0
    for i in range(n):
        max_diag = max(max_diag, abs(m[i, i]))
    pivot_floor = 1e-12 * max(max_diag, _MIN_NORMAL)
    l = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = float(m[i, j])
            for k in range(j):
                s -= l[i, k] * l[j, k]
            if i == j:
                if not (s > pivot_floor):
                    raise ValueError(
                        f"matrix is not positive-definite (pivot {i} = {s})")
                l[i, j] = math.sqrt(s)
            else:
                l[i, j] = s / l[j, j]
    return l


class GaussianCopula:
    """Copula sampler over a fixed correlation matrix.

    The correlation matrix is Cholesky-factored at construction — a
    borderline matrix fails loudly here, not as NaN samples later.
    """

    def __init__(self, correlation):
        self._chol = cholesky(correlation)
        self._dim = self._chol.shape[0]

    @property
    def dimension(self) -> int:
        return self._dim

    def sample(self, rng: np.random.Generator, out: np.ndarray,
               scratch: np.ndarray) -> None:
        """One Gaussian-copula draw: ``out[i]`` are correlated uniforms in (0, 1).

        Arrays are caller-owned; ``scratch`` must be a second array of
        the same length (kept separate so the sampler allocates only
        the draw itself — and to keep the Java aliasing contract).
        """
        self._require_distinct(out, scratch)
        scratch[:self._dim] = rng.standard_normal(self._dim)
        self._correlate(scratch, out)
        for i in range(self._dim):
            out[i] = mu.norm_cdf(out[i])

    def sample_t(self, rng: np.random.Generator, df: int, out: np.ndarray,
                 scratch: np.ndarray) -> None:
        """One Student-t-copula draw with ``df`` degrees of freedom.

        The same correlated Gaussians divided by a shared sqrt(chi2/df)
        — the SHARED shock is what creates tail dependence. Uniforms
        come from the EXACT t-CDF (:func:`math_utils.t_cdf`); a
        moment-matched normal approximation here would distort exactly
        the tail quantiles this sampler exists to model.
        """
        if df < 1:
            raise ValueError("df must be >= 1")
        self._require_distinct(out, scratch)
        scratch[:self._dim] = rng.standard_normal(self._dim)
        self._correlate(scratch, out)
        z = rng.standard_normal(df)
        chi_sq = float(np.sum(z * z))
        scale = math.sqrt(df / max(chi_sq, 1e-12))
        for i in range(self._dim):
            out[i] = mu.t_cdf(out[i] * scale, df)

    def _correlate(self, z: np.ndarray, out: np.ndarray) -> None:
        out[:self._dim] = self._chol @ z[:self._dim]

    def _require_distinct(self, out: np.ndarray, scratch: np.ndarray) -> None:
        self._require_length(out)
        self._require_length(scratch)
        # Java's correlate() reads scratch while writing out; keep the
        # contract (see module docstring for why NumPy would survive it).
        if out is scratch or np.shares_memory(out, scratch):
            raise ValueError("out and scratch must be distinct arrays")

    def _require_length(self, a: np.ndarray) -> None:
        if a.shape[0] < self._dim:
            raise ValueError(
                f"array has {a.shape[0]} entries, copula has {self._dim}")
