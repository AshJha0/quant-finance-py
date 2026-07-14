"""Streaming EWMA covariance matrix (port of Java
``microstructure.EwmaCovariance``) -- the multi-asset risk picture that
single-symbol volatility cannot see. RiskMetrics-style: one return
vector per sampling interval, ``cov <- lambda*cov + (1-lambda)*r_i*r_j``,
with the classic zero-mean convention (intraday returns have
negligible mean at these horizons; carrying decayed means would double
the state for a correction smaller than the estimation noise -- a
documented choice, not an oversight).

:meth:`marginal_contribution` is the diagonal-approximation-plus-cross
piece for capacity allocation across a basket; feed the matrix in and
scarce liquidity flows to the symbols whose remaining position
contributes most to BASKET risk, not just to their own.
:meth:`min_variance_hedge_ratio` is the live hedge beta (cov/var) for
cross hedging.

**Discipline.** The matrix stays positive-semidefinite because every
update is a full-vector rank-1 outer product: a sample containing ANY
non-finite return is dropped whole (updating only the clean pairs
would break PSD and silently skew correlations). Each pair seeds from
its first observation rather than ramping from 0. The lower triangle
lives in one flat array -- O(n^2) work, which at basket sizes (tens of
symbols) on an interval cadence is microseconds. Single writer.
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils

#: 46,340 is where the triangle index i*(i+1)/2 leaves 32-bit int
#: range in the Java source -- far beyond the basket sizes this
#: streaming design is for, but the same upper bound is kept here for
#: parity (Python ints don't overflow, but a basket this large is a
#: modeling error, not a feature).
_MAX_SYMBOLS = 46_340


class EwmaCovariance:
    """Streaming, zero-mean EWMA covariance matrix over a fixed
    basket; see the module docstring."""

    __slots__ = ("_symbols", "_lambda", "_tri", "_samples")

    def __init__(self, symbols: int, lam: float = 0.94) -> None:
        """
        Args:
            symbols: basket size (dense indices, fixed at
                construction).
            lam: decay per sample, e.g. 0.94 (RiskMetrics daily
                convention; intraday intervals often want 0.97-0.99).
        """
        if symbols < 1 or symbols > _MAX_SYMBOLS or lam <= 0 or lam >= 1:
            raise ValueError(
                f"need symbols in [1, {_MAX_SYMBOLS}], lambda in (0,1)")
        self._symbols = symbols
        self._lambda = lam
        self._tri = np.zeros(symbols * (symbols + 1) // 2)
        self._samples = 0

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def on_returns(self, returns) -> None:
        """One sampling interval: every symbol's return over the
        interval that just closed (0 for a symbol that did not move --
        that IS its return). A vector containing any non-finite entry
        is dropped entirely: a partial update would break
        positive-semidefiniteness, so a bad print on one symbol must
        not corrupt the whole matrix.

        Args:
            returns: length >= symbols; entries beyond the basket are
                ignored.
        """
        r = np.asarray(returns, dtype=float)
        if r.shape[0] < self._symbols:
            raise ValueError(
                f"returns has {r.shape[0]} entries, basket needs "
                f"{self._symbols}")
        if not np.all(np.isfinite(r[:self._symbols])):
            return                          # gap: drop the whole sample
        seed = self._samples == 0
        k = 0
        for i in range(self._symbols):
            ri = r[i]
            for j in range(i + 1):
                prod = ri * r[j]
                self._tri[k] = (prod if seed
                               else self._tri[k]
                               + (1 - self._lambda) * (prod - self._tri[k]))
                k += 1
        self._samples += 1

    # ------------------------------------------------------------------
    # The matrix
    # ------------------------------------------------------------------

    def covariance(self, i: int, j: int) -> float:
        """Decayed covariance between two symbols (order-free)."""
        if i >= j:
            return float(self._tri[i * (i + 1) // 2 + j])
        return float(self._tri[j * (j + 1) // 2 + i])

    def variance(self, i: int) -> float:
        """Decayed variance of one symbol."""
        return float(self._tri[i * (i + 1) // 2 + i])

    def volatility(self, i: int) -> float:
        """Decayed volatility (per sqrt(interval)), 0 until learned."""
        return math.sqrt(max(self.variance(i), 0.0))

    def correlation(self, i: int, j: int) -> float:
        """Decayed correlation in [-1, 1]; 0 while either variance is
        0."""
        denom = math.sqrt(self.variance(i) * self.variance(j))
        return (math_utils.clamp(self.covariance(i, j) / denom, -1, 1)
                if denom > 0 else 0.0)

    # ------------------------------------------------------------------
    # Portfolio arithmetic
    # ------------------------------------------------------------------

    def portfolio_variance(self, weights) -> float:
        """``w' Sigma w``: portfolio variance of the (signed) weight
        vector."""
        w = np.asarray(weights, dtype=float)
        self._require_length(w)
        total = 0.0
        k = 0
        for i in range(self._symbols):
            for j in range(i + 1):
                term = w[i] * w[j] * self._tri[k]
                total += term if i == j else 2 * term
                k += 1
        return total

    def marginal_contribution(self, weights, out) -> float:
        """Marginal contribution to portfolio risk:
        ``out[i] = w_i * (Sigma w)_i / (w' Sigma w)`` -- the fraction
        of total basket variance symbol i's position is responsible
        for (contributions sum to 1; a natural hedge contributes
        negatively). All zeros while the portfolio variance is not
        positive -- no risk picture, no signal.

        Returns:
            The portfolio variance ``w' Sigma w``, so a caller gating
            on "is there a risk picture?" needs exactly one call.
        """
        w = np.asarray(weights, dtype=float)
        self._require_length(w)
        self._require_length(np.asarray(out, dtype=float))
        total = self.portfolio_variance(w)
        if total <= 0:
            for i in range(self._symbols):
                out[i] = 0.0
            return total
        for i in range(self._symbols):
            sigma_w = 0.0
            for j in range(self._symbols):
                sigma_w += self.covariance(i, j) * w[j]
            out[i] = w[i] * sigma_w / total
        return total

    def min_variance_hedge_ratio(self, target: int, hedge: int) -> float:
        """The live minimum-variance hedge ratio: hedge ``target`` with
        ``cov(target,hedge)/var(hedge)`` units of ``hedge``. 0 while
        the hedge instrument's variance is unlearned."""
        v = self.variance(hedge)
        return self.covariance(target, hedge) / v if v > 0 else 0.0

    def symbols(self) -> int:
        return self._symbols

    def samples(self) -> int:
        return self._samples

    def _require_length(self, a: np.ndarray) -> None:
        if a.shape[0] < self._symbols:
            raise ValueError(
                f"array has {a.shape[0]} entries, basket needs "
                f"{self._symbols}")
