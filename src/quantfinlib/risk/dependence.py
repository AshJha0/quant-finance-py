"""Rank-based dependence measures — what Pearson correlation misses.

Port of Java ``com.quantfinlib.risk.Dependence``:

* Spearman's rho — Pearson on the RANKS, robust to outliers and any
  monotone transform (also the correlation FRTB's PLAT is defined on);
* Kendall's tau — concordant minus discordant pair probability, with
  the elliptical-copula property ``rho_pearson = sin(pi*tau/2)``.

Ties get midranks (Spearman) / count as neither concordant nor
discordant (Kendall tau-a — adequate for continuous return data;
heavy-tie categorical data wants tau-b, out of scope and said so).
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.risk import risk_metrics


def spearman(a, b) -> float:
    """Spearman rank correlation in [-1, 1]."""
    a, b = _require_same_length(a, b)
    return risk_metrics.correlation(ranks(a), ranks(b))


def kendall_tau(a, b) -> float:
    """Kendall's tau (tau-a) in [-1, 1]; O(n^2) pairs (vectorized)."""
    a, b = _require_same_length(a, b)
    n = a.shape[0]
    iu = np.triu_indices(n, 1)
    prod = (a[:, None] - a[None, :])[iu] * (b[:, None] - b[None, :])[iu]
    concordant = int(np.count_nonzero(prod > 0))
    discordant = int(np.count_nonzero(prod < 0))
    # ties: neither — tau-a convention, documented above
    pairs = n * (n - 1) // 2
    return (concordant - discordant) / pairs


def pearson_from_kendall(tau: float) -> float:
    """The elliptical-copula bridge: Pearson rho implied by a Kendall tau."""
    if not (-1 <= tau <= 1):
        raise ValueError("tau must be in [-1, 1]")
    return math.sin(math.pi * tau / 2)


def ranks(values) -> np.ndarray:
    """Midranks (average rank for ties), 1-based."""
    v = np.asarray(values, dtype=float)
    n = v.shape[0]
    order = np.argsort(v, kind="stable")
    sv = v[order]
    new_run = np.ones(n, dtype=bool)
    new_run[1:] = sv[1:] != sv[:-1]
    run_id = np.cumsum(new_run) - 1
    first = np.flatnonzero(new_run)
    counts = np.diff(np.append(first, n))
    mid = first + (counts - 1) / 2.0 + 1          # 1-based midrank
    out = np.empty(n)
    out[order] = mid[run_id]
    return out


def _require_same_length(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        raise ValueError("need two equal-length series of >= 2")
    return a, b
