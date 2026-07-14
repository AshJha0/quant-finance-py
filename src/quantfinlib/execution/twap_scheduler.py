"""TWAP (time-weighted average price) schedule design (port of Java
``execution.TwapScheduler``).

Splits a parent order into evenly spaced child slices, optionally with
randomized sizes to reduce schedule predictability (anti-gaming). Slice
quantities always sum exactly to the parent quantity.
"""

from __future__ import annotations

from typing import List

import numpy as np

from quantfinlib.execution import vwap_scheduler
from quantfinlib.execution.slice import Slice

_MAX_DURATION_MILLIS = (1 << 63) - 1


def schedule(total_qty: int, duration_millis: int, num_slices: int) -> List[Slice]:
    """Equal slices at equal intervals starting at t=0."""
    weights = np.ones(num_slices)
    return _to_slices(total_qty, duration_millis, weights)


def schedule_randomized(total_qty: int, duration_millis: int, num_slices: int,
                        jitter_pct: float, seed: int) -> List[Slice]:
    """Randomized TWAP: slice sizes jittered by up to ``jitter_pct``
    (e.g. 0.3 = +-30%), deterministic for a given seed.

    Port note: uses ``numpy.random.default_rng`` rather than Java's
    ``SplittableRandom``, so the exact jittered values differ bit-for-bit
    from the Java reference for the same seed; only the structural
    guarantee (slices sum exactly to ``total_qty``) is pinned across
    ports.
    """
    rng = np.random.default_rng(seed)
    weights = 1 + jitter_pct * (2 * rng.random(num_slices) - 1)
    return _to_slices(total_qty, duration_millis, weights)


def _to_slices(total_qty: int, duration_millis: int, weights: np.ndarray) -> List[Slice]:
    n = weights.shape[0]
    if n == 0 or total_qty <= 0:
        raise ValueError("need positive quantity and at least one slice")
    # Same overflow guard WmrFixingScheduler carries: a pathological
    # horizon must not wrap duration_millis * i negative and place
    # slices before the window opens.
    if duration_millis < 0 or duration_millis > _MAX_DURATION_MILLIS // n:
        raise ValueError(f"durationMillis out of range for {n} slices")
    quantities = vwap_scheduler.allocate_proportionally(total_qty, weights)
    return [Slice(duration_millis * i // n, int(quantities[i])) for i in range(n)]
