"""VWAP schedule design (port of Java ``execution.VwapScheduler``).

Allocates child slices proportionally to an expected intraday volume
profile (e.g. from an intraday liquidity forecaster), so participation
tracks the market's own volume curve.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from quantfinlib.execution.slice import Slice

#: Python ints never overflow the way Java's ``long`` does, but a
#: pathological ``duration_millis * i`` for a huge horizon is still a
#: modeling error (placing slices centuries apart), not a feature. The
#: same sanity bound Java's overflow guard enforced is kept here for
#: parity: durations wide enough to overflow a 64-bit millis offset are
#: rejected rather than silently accepted.
_MAX_DURATION_MILLIS = (1 << 63) - 1


def _check_duration(duration_millis: int, n: int) -> None:
    if duration_millis < 0 or n < 1 or duration_millis > _MAX_DURATION_MILLIS // n:
        raise ValueError(
            f"durationMillis out of range for {n} slices")


def schedule(total_qty: int, volume_profile: Sequence[float],
            duration_millis: int) -> List[Slice]:
    """
    Args:
        total_qty: parent order quantity.
        volume_profile: expected volume per bucket (any positive scale).
        duration_millis: total execution window; slice i starts at
            ``i * duration / buckets``.
    """
    profile = np.asarray(volume_profile, dtype=float)
    if profile.shape[0] == 0 or total_qty <= 0:
        raise ValueError("need positive quantity and a non-empty profile")
    _check_duration(duration_millis, profile.shape[0])
    quantities = allocate_proportionally(total_qty, profile)
    n = profile.shape[0]
    return [Slice(duration_millis * i // n, int(quantities[i])) for i in range(n)]


def allocate_proportionally(total: int, weights: Sequence[float]) -> np.ndarray:
    """Largest-remainder proportional allocation: integer quantities
    that sum exactly to ``total``, proportional to ``weights``."""
    w = np.asarray(weights, dtype=float)
    n = w.shape[0]
    if np.any(w < 0):
        raise ValueError(f"negative weight: {float(w[w < 0][0])}")
    total_weight = float(np.sum(w))
    out = np.zeros(n, dtype=np.int64)
    if total_weight == 0:
        out[0] = total
        return out
    fractions = np.zeros(n)
    allocated = 0
    for i in range(n):
        raw = total * w[i] / total_weight
        out[i] = int(np.floor(raw))
        fractions[i] = raw - out[i]
        allocated += out[i]
    # Distribute the remainder to the largest fractional parts (stable
    # tie-break by index, matching Java's Comparator with equal keys
    # preserving encounter order via a stable sort).
    remainder = int(total - allocated)
    order = np.argsort(-fractions, kind="stable")
    for k in range(remainder):
        out[order[k % n]] += 1
    return out
