"""Anti-gaming randomization for schedule-driven algos (port of Java
``execution.AntiGamingJitter``).

A TWAP that fires identical children on a metronome is a gift to
anyone watching the tape: predators detect the clock in a handful of
intervals and lean on every child. The counter-measure is controlled
jitter that keeps the SCHEDULE honest while killing the pattern:

* **Size jitter** -- each child +/- ``size_fraction``, with the
  differences redistributed so the TOTAL is preserved exactly (the
  parent must complete; anti-gaming never changes what gets done, only
  how recognizable it looks).
* **Time jitter** -- each firing time +/- ``time_fraction`` of its own
  interval, monotonicity preserved (children never reorder) and end
  time never exceeded.

Deterministic per seed -- replayable in backtests, auditable in
production. Relationship to
:func:`~quantfinlib.execution.twap_scheduler.schedule_randomized`: that
function jitters TWAP slice SIZES at construction; this class is the
generic overlay -- it jitters any existing child plan (VWAP,
benchmark-executor, hand-built) and adds the TIME dimension the
schedulers do not randomize.

Port note: uses ``numpy.random.default_rng`` rather than Java's
``java.util.Random``, so the exact jittered values differ bit-for-bit
from the Java reference for the same seed; the structural guarantees
(exact total preservation, monotonic times, same-seed reproducibility,
different-seed divergence) are what's pinned across ports.
"""

from __future__ import annotations

import math

import numpy as np


class AntiGamingJitter:
    """Size/time jitter overlay for execution schedules; see the
    module docstring."""

    def __init__(self, seed: int, size_fraction: float, time_fraction: float) -> None:
        """
        Args:
            seed: deterministic seed (replayable, auditable).
            size_fraction: max relative size perturbation, in [0, 0.5].
            time_fraction: max relative time perturbation within each
                interval, in [0, 0.5].
        """
        if not (0 <= size_fraction <= 0.5):
            raise ValueError("sizeFraction must be in [0, 0.5]")
        if not (0 <= time_fraction <= 0.5):
            raise ValueError("timeFraction must be in [0, 0.5]")
        self._rng = np.random.default_rng(seed)
        self._size_fraction = size_fraction
        self._time_fraction = time_fraction

    def jitter_sizes(self, child_qty) -> np.ndarray:
        """Jitters child sizes +/- ``size_fraction``, preserving the
        total EXACTLY and never producing a negative child.
        Perturbations are paired (child i gives what child i+1 takes),
        so the completion curve wanders inside a one-child envelope of
        the original."""
        out = np.array(child_qty, dtype=np.int64, copy=True)
        if np.any(out < 0):
            raise ValueError("child quantities must be >= 0")
        for i in range(out.shape[0] - 1):
            # Transfer between neighbors: total invariant by construction.
            max_shift = math.floor(min(out[i], out[i + 1]) * self._size_fraction)
            if max_shift <= 0:
                continue
            shift = math.floor((2 * self._rng.random() - 1) * max_shift + 0.5)
            out[i] += shift
            out[i + 1] -= shift
        return out

    def jitter_times(self, times_nanos, start_nanos: int = 0) -> np.ndarray:
        """Jitters firing times within their intervals: each time moves
        +/- ``time_fraction`` of the gap to its neighbors, strict
        monotonicity preserved, first/last never escape ``[start_nanos,
        original end]``."""
        original = np.array(times_nanos, dtype=np.int64)
        out = original.copy()
        n = out.shape[0]
        for i in range(1, n):
            if out[i] <= out[i - 1]:
                raise ValueError("times must be strictly increasing")
        if n > 0 and out[0] < start_nanos:
            raise ValueError("first time is before start")
        for i in range(n):
            lo = start_nanos if i == 0 else int(out[i - 1]) + 1
            # next entry not yet jittered: safe bound
            hi = int(original[-1]) if i == n - 1 else int(out[i + 1]) - 1
            gap_before = int(original[i]) - (start_nanos if i == 0 else int(original[i - 1]))
            max_shift = math.floor(gap_before * self._time_fraction)
            shifted = int(original[i]) + math.floor(
                (2 * self._rng.random() - 1) * max_shift + 0.5)
            out[i] = max(lo, min(hi, shifted))
        return out
