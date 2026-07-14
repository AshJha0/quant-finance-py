"""Liquidity estimated from BARS ALONE (port of Java
``microstructure.LiquidityMeasures``) -- the estimators for every
market where you have prices but no quotes: history before your tick
capture started, less-developed markets, bonds marked once a day, or a
20-year backtest that would otherwise pretend spreads were zero.

* **Roll (1984)** -- the effective spread implied by bid-ask BOUNCE:
  trade prices ping-ponging between bid and ask create negative
  autocovariance in price changes, and
  ``s = 2*sqrt(-cov(dp_t, dp_(t-1)))``. When the autocovariance is
  POSITIVE (trending sample, no bounce signature) the estimator is
  undefined and returns NaN -- not zero, because "zero spread" is a
  claim and NaN is an honest shrug;
* **Corwin-Schultz (2012)** -- the spread from two days' HIGH-LOW
  ranges: variance grows with time but the spread does not, so
  comparing one 2-day range against two 1-day ranges isolates the
  spread. Negative estimates clamp to 0 (standard practice, stated);
* **Amihud (2002)** -- price impact per currency unit traded:
  ``mean(|return| / dollarVolume)``. The cross-sectional illiquidity
  ranker -- multiply by 1e6 for the conventional "per million"
  quotation.

Static, deterministic, research lane. These are ESTIMATORS with real
sampling error on short windows -- rank with them, do not mark books
with them.
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils


class LiquidityMeasures:
    """Static bar-based liquidity estimators; see the module
    docstring."""

    @staticmethod
    def roll_spread(prices) -> float:
        """Roll's implied effective spread from trade/close prices
        (same units as the prices). NaN when the bounce signature is
        absent -- see the module doc.

        Args:
            prices: >= 3 finite positive prices.
        """
        p = np.asarray(prices, dtype=float)
        if p.shape[0] < 3:
            raise ValueError("need >= 3 prices")
        _require_positive_finite(p)
        n = p.shape[0] - 1
        d = p[1:] - p[:-1]
        # cov(dp_t, dp_(t-1)) over the n-1 adjacent pairs.
        mean = math_utils.mean(d)
        cov = 0.0
        for i in range(1, n):
            cov += (d[i] - mean) * (d[i - 1] - mean)
        cov /= (n - 1)
        return 2 * math.sqrt(-cov) if cov < 0 else math.nan

    @staticmethod
    def corwin_schultz_spread(high1: float, low1: float, high2: float,
                              low2: float) -> float:
        """Corwin-Schultz high-low spread estimate as a FRACTION of
        price, from two consecutive periods' highs and lows. Negative
        estimates clamp to 0 (stated standard practice)."""
        _require_positive_finite(np.array([high1, low1, high2, low2]))
        if high1 < low1 or high2 < low2:
            raise ValueError("high must be >= low")
        b = _square(math.log(high1 / low1)) + _square(math.log(high2 / low2))
        gamma = _square(math.log(max(high1, high2) / min(low1, low2)))
        k = 3 - 2 * math.sqrt(2)
        alpha = ((math.sqrt(2 * b) - math.sqrt(b)) / k
                - math.sqrt(gamma / k))
        spread = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
        return max(0.0, spread)

    @staticmethod
    def amihud_illiquidity(returns, dollar_volumes) -> float:
        """Amihud illiquidity: ``mean(|return| / dollarVolume)`` --
        return per currency unit traded. Zero-volume periods are a
        data problem, not an infinity: they throw.

        Args:
            returns: per-period returns (fractions), finite.
            dollar_volumes: per-period traded value, > 0, aligned.
        """
        r = np.asarray(returns, dtype=float)
        v = np.asarray(dollar_volumes, dtype=float)
        if r.shape[0] != v.shape[0] or r.shape[0] < 1:
            raise ValueError("need aligned, non-empty arrays")
        total = 0.0
        for i in range(r.shape[0]):
            if not math.isfinite(r[i]):
                raise ValueError("returns must be finite")
            if not (v[i] > 0) or v[i] == math.inf:
                raise ValueError(
                    "dollarVolumes must be positive and finite (zero "
                    f"volume at {i} is a data gap, not infinite "
                    "illiquidity)")
            total += abs(r[i]) / v[i]
        return total / r.shape[0]


def _square(x: float) -> float:
    return x * x


def _require_positive_finite(a: np.ndarray) -> None:
    for x in a:
        if not (x > 0) or x == math.inf:
            raise ValueError("prices must be positive and finite")
