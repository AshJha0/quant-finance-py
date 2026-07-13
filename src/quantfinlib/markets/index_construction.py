"""Index construction (port of Java ``com.quantfinlib.markets.IndexConstruction``).

The arithmetic behind "the market was up 1%". Three weighting schemes
produce three different markets from the same stocks:

* cap-weighted (S&P 500 style): weight = float-adjusted market cap
  share. Self-rebalancing under price moves (a stock that doubles
  doubles its own weight — no trading needed), which is why cap-weight
  is the lowest-turnover scheme and the natural benchmark;
* price-weighted (Dow style): weight = price share. A $400 stock moves
  the index 8x as much as a $50 stock regardless of company size — a
  historical accident kept alive by tradition;
* equal-weighted: constant 1/N. Systematically tilts small and must
  TRADE every rebalance to stay equal — the turnover is the price of
  the tilt.

The DIVISOR is how a level series survives membership and share
changes: ``level = sum(price * shares * float) / divisor``. When a
member is added, dropped or re-floated, the divisor is rescaled so the
level is CONTINUOUS through the change —
``newDivisor = oldDivisor * newAggregate / oldAggregate`` — so the
index only ever moves for price reasons (pinned by test: a member swap
leaves the level unchanged at the instant of the swap).
``turnover(w1, w2) = 0.5 * sum |w1 - w2|`` is the one-way fraction of
the portfolio that must trade between two weight vectors — the
index-tracking cost driver.
"""

from __future__ import annotations

import math

import numpy as np


class IndexConstruction:
    """Static weighting, divisor and turnover arithmetic."""

    @staticmethod
    def cap_weights(prices, shares, float_factors) -> np.ndarray:
        """Float-adjusted cap weights: ``w_i ~ price_i * shares_i * float_i``."""
        n = _validate(prices, shares, float_factors)
        w = np.empty(n)
        total = 0.0
        for i in range(n):
            w[i] = prices[i] * shares[i] * float_factors[i]
            total += w[i]
        return w / total

    @staticmethod
    def price_weights(prices) -> np.ndarray:
        """Price weights: ``w_i ~ price_i`` (the Dow's accident)."""
        n = _validate_prices(prices)
        w = np.asarray(prices, dtype=float)
        return w / float(np.sum(w))

    @staticmethod
    def equal_weights(n: int) -> np.ndarray:
        """Equal weights, 1/N."""
        if n < 1:
            raise ValueError(f"need n >= 1, got {n}")
        return np.full(n, 1.0 / n)

    @staticmethod
    def level(prices, shares, float_factors, divisor: float) -> float:
        """Index level from an aggregate and a divisor."""
        _validate(prices, shares, float_factors)
        if not (divisor > 0) or divisor == math.inf:
            raise ValueError("divisor must be positive and finite")
        agg = 0.0
        for i in range(len(prices)):
            agg += prices[i] * shares[i] * float_factors[i]
        return agg / divisor

    @staticmethod
    def adjust_divisor(old_divisor: float, old_aggregate: float,
                       new_aggregate: float) -> float:
        """The rescaled divisor that keeps the level CONTINUOUS through a
        membership/share/float change: pass the aggregate cap before and
        after the change (both at the same instant).
        """
        if not (old_divisor > 0) or not (old_aggregate > 0) or not (new_aggregate > 0):
            raise ValueError("divisor and aggregates must be positive")
        return old_divisor * new_aggregate / old_aggregate

    @staticmethod
    def turnover(from_weights, to_weights) -> float:
        """One-way turnover between two aligned weight vectors: ``0.5 * sum|w1-w2|``."""
        if len(from_weights) != len(to_weights):
            raise ValueError("weight vectors must align")
        total = 0.0
        for i in range(len(from_weights)):
            if not math.isfinite(from_weights[i]) or not math.isfinite(to_weights[i]):
                raise ValueError(f"non-finite weight at {i}")
            total += abs(from_weights[i] - to_weights[i])
        return 0.5 * total


def _validate(prices, shares, float_factors) -> int:
    n = _validate_prices(prices)
    if len(shares) != n or len(float_factors) != n:
        raise ValueError("prices/shares/floats must align")
    for i in range(n):
        if not (shares[i] > 0) or shares[i] == math.inf:
            raise ValueError(f"shares must be positive and finite: {shares[i]}")
        if not (float_factors[i] > 0) or float_factors[i] > 1:
            raise ValueError(f"float factor must be in (0, 1]: {float_factors[i]}")
    return n


def _validate_prices(prices) -> int:
    if len(prices) == 0:
        raise ValueError("need at least one constituent")
    for p in prices:
        if not (p > 0) or p == math.inf:
            raise ValueError(f"price must be positive and finite: {p}")
    return len(prices)
