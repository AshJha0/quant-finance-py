"""Hidden-liquidity / iceberg detection from the lit tape (port of Java
``microstructure.HiddenLiquidityDetector``; also covers the "iceberg
detector" item in this port's task list -- the Java source has one
class, not two, and :meth:`HiddenLiquidityDetector.is_iceberg` is its
iceberg-detection surface).

Displayed size is only part of what rests at a price -- icebergs show
a small tip and reload, and hidden/midpoint orders don't show at all.
You can't see them, but you can *infer* them from a tell: **a level
that trades more than it ever displayed, and keeps quoting.**

The sound per-print signature: **a single execution larger than the
size displayed at that moment.** Displayed liquidity cannot fill more
than it shows, so the excess in that one print necessarily executed
against hidden size at the level. (A cumulative executed-vs-displayed
comparison is NOT sound at L2: a busy level legitimately trades many
times its instantaneous display through ordinary adds -- that
formulation false-flags normal flow.) Per level, the detector keeps an
EWMA of the print/displayed ratio at those hidden events;
:meth:`hidden_multiplier` ~= 1 means "what you see is what's there," 3
means "~=3x the tip is likely lurking."

Cross-asset: the trades-more-than-displayed-and-keeps-quoting tell is
the same on an equity exchange level and an FX ECN level (icebergs are
standard on both). Single writer.
"""

from __future__ import annotations

import numpy as np


class HiddenLiquidityDetector:
    """Per-level iceberg/hidden-size inference from lit prints; see
    the module docstring."""

    __slots__ = ("_levels", "_alpha", "_displayed", "_refill_ratio_ewma",
                 "_refill_obs")

    def __init__(self, levels: int, alpha: float = 0.2) -> None:
        """
        Args:
            levels: number of price levels tracked (dense tick
                indices).
            alpha: EWMA weight on each refill observation, e.g. 0.2.
        """
        if levels < 1 or alpha <= 0 or alpha > 1:
            raise ValueError("need levels >= 1, alpha in (0,1]")
        self._levels = levels
        self._alpha = alpha
        self._displayed = np.zeros(levels)
        self._refill_ratio_ewma = np.zeros(levels)
        self._refill_obs = np.zeros(levels, dtype=np.int64)

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def on_displayed(self, level: int, size: float) -> None:
        """The displayed size now standing at ``level``."""
        self._displayed[level] = max(0.0, size)

    def on_execution(self, level: int, qty: float) -> None:
        """One trade print of ``qty`` at ``level``, compared against
        the size displayed at that moment. A print exceeding the
        display is the hidden-liquidity event: the overflow could only
        have filled against unseen size. The EWMA seeds from the first
        observation -- a ratio's meaningful floor is 1.0, so ramping
        up from 0 would under-register a genuine single event."""
        if qty <= 0:
            return
        shown = self._displayed[level]
        if shown > 0 and qty > shown:
            ratio = qty / shown
            self._refill_ratio_ewma[level] = (
                ratio if self._refill_obs[level] == 0
                else self._refill_ratio_ewma[level]
                + self._alpha * (ratio - self._refill_ratio_ewma[level]))
            self._refill_obs[level] += 1

    def on_level_cleared(self, level: int) -> None:
        """The level fully cleared (best moved away / all pulled)."""
        self._displayed[level] = 0.0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def hidden_multiplier(self, level: int) -> float:
        """Estimated ratio of true resting size to displayed size at a
        level: 1 = no hidden liquidity detected, ``>1`` = likely
        iceberg. Uses the EWMA refill ratio, falling back to 1 before
        any evidence."""
        if self._refill_obs[level] == 0:
            return 1.0
        return max(1.0, float(self._refill_ratio_ewma[level]))

    def estimated_true_depth(self, level: int) -> float:
        """Estimated total resting size at a level = displayed x
        hidden multiplier -- the depth an execution algo should size
        against, not the visible tip."""
        return self._displayed[level] * self.hidden_multiplier(level)

    def is_iceberg(self, level: int) -> bool:
        """True once a level has shown iceberg behavior at least
        once."""
        return (self._refill_obs[level] > 0
                and self._refill_ratio_ewma[level] > 1.0)

    def refill_observations(self, level: int) -> int:
        return int(self._refill_obs[level])

    def levels(self) -> int:
        return self._levels
