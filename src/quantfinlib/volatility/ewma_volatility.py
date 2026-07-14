"""Port of Java ``com.quantfinlib.volatility.EwmaVolatility``.

Exponentially weighted moving average variance (RiskMetrics-style):
``h_t = lambda * h_{t-1} + (1 - lambda) * r_{t-1}^2``, seeded with the
sample variance. Reacts to volatility regime changes far faster than a
rolling window.
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils as mu


class EwmaVolatility:
    """RiskMetrics-style EWMA variance estimator with decay ``lambda_``."""

    def __init__(self, lambda_: float) -> None:
        if lambda_ <= 0 or lambda_ >= 1:
            raise ValueError(f"lambda must be in (0,1): {lambda_}")
        self._lambda = lambda_

    @classmethod
    def risk_metrics(cls) -> "EwmaVolatility":
        """The classic RiskMetrics daily decay (lambda = 0.94)."""
        return cls(0.94)

    def variances(self, returns) -> np.ndarray:
        """Conditional variance series aligned with returns.

        ``variances[i]`` is the estimate for period i, formed from
        information up to i-1. Seeded with the unconditional sample
        variance; the recurrence is transcribed exactly from Java.

        Raises:
            ValueError: for fewer than 2 returns.
        """
        returns = np.asarray(returns, dtype=float)
        if returns.shape[0] < 2:
            raise ValueError("need at least 2 returns")
        h = np.empty(returns.shape[0])
        h[0] = mu.variance(returns)  # unconditional seed
        lam = self._lambda
        for t in range(1, returns.shape[0]):
            h[t] = lam * h[t - 1] + (1 - lam) * returns[t - 1] * returns[t - 1]
        return h

    def latest_vol(self, returns) -> float:
        """One-step-ahead volatility forecast (per period)."""
        returns = np.asarray(returns, dtype=float)
        h = self.variances(returns)
        last = h[-1]
        next_ = (self._lambda * last
                 + (1 - self._lambda) * returns[-1] * returns[-1])
        return math.sqrt(next_)

    def annualized_vol(self, returns, periods_per_year: int) -> float:
        """One-step-ahead volatility, annualized by ``sqrt(periods_per_year)``."""
        return self.latest_vol(returns) * math.sqrt(periods_per_year)
