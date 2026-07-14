"""Streaming cross-asset lead-lag estimation (port of Java
``microstructure.LeadLagEstimator``).

Does instrument A's return *now* predict instrument B's return a few
intervals from now? The classic pairs — EURUSD leads EURJPY, index
futures lead the cash basket, the liquid large-cap leads its sector
peers — are exactly this structure, and it is the basis of cross
hedging and cross pricing.

Feed one :meth:`LeadLagEstimator.on_sample` per fixed sampling interval
with the two instruments' returns over that interval (the caller owns
the sampling clock). The estimator keeps a small ring of the leader's
recent returns and, for each candidate lag ``k = 0..max_lag``, a
time-decayed correlation between the leader's return k intervals ago
and the follower's return now.

Reading the output: :meth:`LeadLagEstimator.best_lag` is the ``k > 0``
with the largest |correlation| — a genuine lead, because the leader's
return was observable before the follower's. Compare it against
``correlation_at_lag(0)``: if the contemporaneous correlation dominates
every lagged one, the pair co-moves but neither side is tradeably
ahead. Treat a lead that appears and disappears with the decay window
as noise; only a persistent best lag with stable sign is structure.
(The Java checkpoint persistence is not ported.)
"""

from __future__ import annotations

import math

import numpy as np


class LeadLagEstimator:
    """Streaming lead-lag correlations; see the module docstring."""

    __slots__ = ("_max_lag", "_alpha", "_leader_ring", "_head", "_ring_fill",
                 "_samples", "_mean_lead", "_mean_follow", "_var_lead",
                 "_var_follow", "_covar")

    def __init__(self, max_lag: int = 10, alpha: float = 0.01) -> None:
        """``max_lag``: largest lead to test, in sampling intervals;
        ``alpha``: EWMA weight of the correlation statistics (0.01 ~ a
        few-hundred-sample memory)."""
        if max_lag < 1 or alpha <= 0 or alpha > 1:
            raise ValueError("need max_lag >= 1, alpha in (0,1]")
        self._max_lag = max_lag
        self._alpha = alpha
        # Ring of the leader's most recent max_lag+1 returns; head = newest.
        self._leader_ring = np.zeros(max_lag + 1)
        self._head = 0
        self._ring_fill = 0
        self._samples = 0
        n = max_lag + 1
        # Per-lag time-decayed moments of (leader[t-k], follower[t]).
        self._mean_lead = np.zeros(n)
        self._mean_follow = np.zeros(n)
        self._var_lead = np.zeros(n)
        self._var_follow = np.zeros(n)
        self._covar = np.zeros(n)

    def on_sample(self, leader_return: float, follower_return: float) -> None:
        """One sampling interval: the leader's and follower's returns over
        the interval that just closed. Non-finite inputs are a gap — the
        sample is dropped entirely (no moment updates, ring untouched) so
        a bad print can't poison the correlations. Across gaps, lag k
        means "k valid samples ago", not k wall-clock intervals."""
        if not math.isfinite(leader_return) or not math.isfinite(follower_return):
            return
        ring = self._leader_ring
        self._head = 0 if self._head + 1 == ring.shape[0] else self._head + 1
        ring[self._head] = leader_return
        if self._ring_fill <= self._max_lag:
            self._ring_fill += 1
        self._samples += 1

        a = self._alpha
        lags = min(self._ring_fill - 1, self._max_lag)
        for k in range(lags + 1):
            lead = ring[self._index(k)]
            self._mean_lead[k] += a * (lead - self._mean_lead[k])
            self._mean_follow[k] += a * (follower_return - self._mean_follow[k])
            dl = lead - self._mean_lead[k]
            df = follower_return - self._mean_follow[k]
            self._var_lead[k] += a * (dl * dl - self._var_lead[k])
            self._var_follow[k] += a * (df * df - self._var_follow[k])
            self._covar[k] += a * (dl * df - self._covar[k])

    def correlation_at_lag(self, lag: int) -> float:
        """Time-decayed correlation between the leader's return ``lag``
        intervals ago and the follower's return now. 0 until enough
        samples exist at that lag."""
        denom = math.sqrt(self._var_lead[lag] * self._var_follow[lag])
        return float(self._covar[lag] / denom) if denom > 0 else 0.0

    def best_lag(self) -> int:
        """The lag ``k >= 1`` with the largest |correlation| — the
        estimated lead time in sampling intervals. 0 when no lagged
        correlation has been measured yet (fewer than 2 samples)."""
        best = 0
        best_abs = 0.0
        lags = int(min(self._samples - 1, self._max_lag))
        for k in range(1, lags + 1):
            c = abs(self.correlation_at_lag(k))
            if c > best_abs:
                best_abs = c
                best = k
        return best

    def best_correlation(self) -> float:
        """The signed correlation at :meth:`best_lag`; 0 when best_lag()
        is 0."""
        k = self.best_lag()
        return 0.0 if k == 0 else self.correlation_at_lag(k)

    def expected_follower_return(self) -> float:
        """The regression prediction of the follower's next-interval
        return from the leader's return at the best lag:
        ``beta(k) * leader_return[t-k+1]`` with ``beta = cov/var_lead``.
        0 when no lead has been measured."""
        k = self.best_lag()
        if k == 0 or self._var_lead[k] <= 0:
            return 0.0
        # The leader return that is k intervals before the follower's
        # NEXT interval is the one observed k-1 intervals before now.
        return float(self._covar[k] / self._var_lead[k]
                     * self._leader_ring[self._index(k - 1)])

    def max_lag(self) -> int:
        return self._max_lag

    def samples(self) -> int:
        return self._samples

    def _index(self, k: int) -> int:
        """Ring slot holding the leader's return from ``k`` intervals ago."""
        i = self._head - k
        return i + self._leader_ring.shape[0] if i < 0 else i
