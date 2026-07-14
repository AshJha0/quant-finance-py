"""Lo-MacKinlay variance ratio test (port of Java
``microstructure.VarianceRatio``).

The question that comes before every strategy choice: is this series
trending, mean-reverting, or a random walk? Under a random walk,
variance grows LINEARLY with horizon, so the ratio

    VR(q) = Var(q-period returns) / (q * Var(1-period returns))

is 1. Positive return autocorrelation (momentum) compounds — VR > 1;
negative autocorrelation (mean reversion) cancels — VR < 1. The
z-statistic says whether the deviation is signal or sampling noise.

Overlapping q-period sums with the SIMPLIFIED denominator —
Lo-MacKinlay's small-sample unbiased correction is omitted (bias
~(q-1)/n, negligible for n >> q; the length gate enforces n >= 10q) —
and the homoskedastic z-statistic; the heteroskedasticity-robust
variant is likewise omitted. Stated, not hidden. VR(1) is identically 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.util import math_utils as mu


class VarianceRatio:
    """Static Lo-MacKinlay test; see the module docstring."""

    @dataclass(frozen=True)
    class Result:
        """Test outcome.

        Attributes:
            ratio: VR(q): 1 = random walk, > 1 trending, < 1 reverting.
            z_stat: Homoskedastic z; |z| > ~2 rejects the random walk.
        """

        ratio: float
        z_stat: float

        def rejects_random_walk(self) -> bool:
            """|z| >= 2: the deviation from a random walk is not noise."""
            return abs(self.z_stat) >= 2

    @staticmethod
    def test(returns, q: int) -> "VarianceRatio.Result":
        """Runs VR(q) on 1-period returns.

        Args:
            returns: 1-period returns, >= 10*q finite observations.
            q: Aggregation horizon, >= 2 (VR(1) == 1 needs no test).
        """
        if q < 2:
            raise ValueError("q must be >= 2 (VR(1) is identically 1)")
        r = np.asarray(returns, dtype=float)
        n = r.shape[0]
        if n < 10 * q:
            raise ValueError(
                f"need >= {10 * q} observations for VR({q}), got {n}")
        if not np.all(np.isfinite(r)):
            raise ValueError("returns must be finite")
        m_mu = mu.mean(r)
        # 1-period variance (sample).
        d1 = r - m_mu
        var1 = float(np.sum(d1 * d1)) / (n - 1)
        if not var1 > 0:
            raise ValueError("returns carry no variance")
        # Overlapping q-period sums around q*mu.
        m = n - q + 1
        window = np.convolve(r, np.ones(q), mode="valid")  # length m
        dq = window - q * m_mu
        var_q = float(np.sum(dq * dq)) / (m - 1)

        vr = var_q / (q * var1)
        # Homoskedastic asymptotic:
        # (VR-1) * sqrt(n) ~ N(0, 2(2q-1)(q-1)/(3q)).
        asymptotic_var = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q)
        z = (vr - 1) * math.sqrt(n / asymptotic_var)
        return VarianceRatio.Result(vr, z)
