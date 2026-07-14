"""Iceberg order state machine (port of Java ``execution.IcebergOrder``).

Shows only a small display tranche of the full quantity and reloads
automatically when the visible portion fills. Display sizes can be
randomized to make the iceberg harder to detect.
"""

from __future__ import annotations

import math

import numpy as np


class IcebergOrder:
    """Iceberg display/reload state machine; see the module docstring."""

    def __init__(self, total_qty: int, display_qty: int,
                randomize_pct: float = 0.0, seed: int = 0) -> None:
        """
        Args:
            randomize_pct: display-size jitter, e.g. 0.2 = +-20% (0 = fixed).

        Port note: uses ``numpy.random.default_rng`` rather than Java's
        ``SplittableRandom``, so jittered tranche sizes differ
        bit-for-bit from the Java reference for the same seed; only the
        structural guarantees (bounds, exact total) are pinned across
        ports.
        """
        if total_qty <= 0 or display_qty <= 0:
            raise ValueError("quantities must be positive")
        self._display_qty = display_qty
        self._randomize_pct = randomize_pct
        self._rng = np.random.default_rng(seed)
        self._remaining = total_qty     # total unexecuted (visible + hidden)
        self._visible = self._next_tranche()

    def _next_tranche(self) -> int:
        base = self._display_qty
        if self._randomize_pct > 0:
            # Java Math.round: half-up (floor(x + 0.5)).
            raw = self._display_qty * (
                1 + self._randomize_pct * (2 * self._rng.random() - 1))
            base = max(1, math.floor(raw + 0.5))
        return min(base, self._remaining)

    def on_fill(self, qty: int) -> bool:
        """Records a fill against the visible tranche. Returns True
        when the tranche was exhausted and a fresh one was loaded (i.e.
        the working order should be re-submitted at the back of the
        queue)."""
        if qty <= 0 or qty > self._visible:
            raise ValueError(f"fill {qty} exceeds visible {self._visible}")
        self._visible -= qty
        self._remaining -= qty
        if self._visible == 0 and self._remaining > 0:
            self._visible = self._next_tranche()
            return True
        return False

    def visible_qty(self) -> int:
        return self._visible

    def hidden_qty(self) -> int:
        return self._remaining - self._visible

    def remaining_qty(self) -> int:
        return self._remaining

    def is_complete(self) -> bool:
        return self._remaining == 0
