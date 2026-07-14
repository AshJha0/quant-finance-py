"""Port of Java ``com.quantfinlib.volatility.VolatilityDecomposition``.

Systematic vs IDIOSYNCRATIC volatility — the decomposition behind "how
much of this stock's risk is the market, and how much is the company?"
A single-factor (CAPM-style) regression of asset returns on market
returns splits total variance EXACTLY:

    Var(asset) = beta^2 * Var(market)  +  Var(residual)
                 '-- systematic --'       '- idiosyncratic -'

with ``beta = Cov(asset, market) / Var(market)`` — the split is exact
(not approximate) because OLS residuals are uncorrelated with the
regressor by construction. The two halves behave differently:
systematic volatility cannot be diversified away and is hedgeable with
index instruments; idiosyncratic volatility diversifies across names
and is exactly what single-name hedges and pairs trades carry.

R^2 is the systematic SHARE — a utility at 0.7 is mostly a market
proxy; a biotech at 0.05 is mostly its own story. All variances are
per-period (annualize with ``* periods_per_year``, vols with the square
root — the record has helpers). Sample moments (n-1) throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.util import math_utils as mu


class VolatilityDecomposition:
    """Static systematic/idiosyncratic variance split (Java parity)."""

    @dataclass(frozen=True)
    class Decomposition:
        """OLS variance split.

        Attributes:
            beta: Cov(a,m) / Var(m).
            total_variance: per-period Var(asset).
            systematic_variance: beta^2 * Var(market).
            idiosyncratic_variance: the residual: total - systematic (>= 0).
            r_squared: systematic share of total, in [0, 1].
        """
        beta: float
        total_variance: float
        systematic_variance: float
        idiosyncratic_variance: float
        r_squared: float

        def systematic_vol(self, periods_per_year: int) -> float:
            """Annualized systematic volatility."""
            return math.sqrt(self.systematic_variance * periods_per_year)

        def idiosyncratic_vol(self, periods_per_year: int) -> float:
            """Annualized idiosyncratic volatility."""
            return math.sqrt(self.idiosyncratic_variance * periods_per_year)

        def total_vol(self, periods_per_year: int) -> float:
            """Annualized total volatility."""
            return math.sqrt(self.total_variance * periods_per_year)

    @staticmethod
    def decompose(asset_returns, market_returns) -> "VolatilityDecomposition.Decomposition":
        """Decomposes an asset's variance against a market/benchmark series.

        Args:
            asset_returns: per-period returns, >= 30 finite observations.
            market_returns: aligned benchmark returns (must carry variance
                — a flat benchmark decomposes nothing).

        Raises:
            ValueError: on misaligned/short/non-finite inputs or a
                zero-variance benchmark.
        """
        asset_returns = np.asarray(asset_returns, dtype=float)
        market_returns = np.asarray(market_returns, dtype=float)
        if (asset_returns.shape[0] != market_returns.shape[0]
                or asset_returns.shape[0] < 30):
            raise ValueError(
                f"need >= 30 aligned observations, got {asset_returns.shape[0]}")
        if (not np.all(np.isfinite(asset_returns))
                or not np.all(np.isfinite(market_returns))):
            raise ValueError("returns must be finite")
        market_var = mu.variance(market_returns)
        if not (market_var > 0):
            raise ValueError(
                "the benchmark carries no variance — nothing to decompose against")
        cov = mu.covariance(asset_returns, market_returns)
        beta = cov / market_var
        total_var = mu.variance(asset_returns)
        systematic = beta * beta * market_var  # = cov^2 / Var(m)
        # Exact by the OLS identity; the max() only absorbs float dust
        # (Cauchy-Schwarz guarantees systematic <= total in exact math).
        idiosyncratic = max(0.0, total_var - systematic)
        r_squared = min(1.0, systematic / total_var) if total_var > 0 else 0.0
        return VolatilityDecomposition.Decomposition(
            beta, total_var, systematic, idiosyncratic, r_squared)
