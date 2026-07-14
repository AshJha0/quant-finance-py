"""The futures roll (port of Java ``execution.FuturesRollAlgo``).

The trade every futures position must do and most do badly: move from
the expiring front contract to the back over the roll window,
following the LIQUIDITY MIGRATION rather than fighting it. Rolling
everything on day one pays wide back-month spreads; waiting for expiry
pays the congestion of everyone else's last day. The algo tracks a
migration curve -- the cumulative fraction of open interest that has
moved by each day -- and rolls in step::

    target(day) = round(position * curve[day]);  due = target - rolled

The default curve is the classic roll S-shape (slow start,
concentrated middle, complete before the final day's scramble). Each
day's due quantity executes as a CALENDAR SPREAD -- sell front / buy
back for a long -- which is exactly a
:class:`~quantfinlib.execution.spread_execution_algo.SpreadExecutionAlgo`
with ratio 1 and the calendar spread's own legging cap.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class FuturesRollAlgo:
    """Migration-curve-following futures roll; see the module docstring."""

    def __init__(self, position_contracts: int,
                cumulative_migration: Sequence[float]) -> None:
        """
        Args:
            position_contracts: contracts to roll (positive; direction
                is the caller's ticket), > 0.
            cumulative_migration: cumulative fraction migrated by end
                of each roll day: non-decreasing, in [0, 1] (a zero
                early entry = nothing due yet), final entry exactly 1.
        """
        if position_contracts <= 0:
            raise ValueError("positionContracts must be > 0")
        curve = np.asarray(cumulative_migration, dtype=float)
        if curve.shape[0] == 0:
            raise ValueError("need at least one roll day")
        prev = 0.0
        for c in curve:
            if not (c >= prev) or c > 1:
                raise ValueError(
                    "migration curve must be non-decreasing within [0, 1]")
            prev = c
        if curve[-1] != 1.0:
            raise ValueError(
                "the curve must END at exactly 1 -- a roll that does not "
                "complete is a delivery notice waiting to happen")
        self._position = position_contracts
        self._cumulative_curve = curve.copy()
        self._rolled = 0

    @staticmethod
    def default_migration(days: int) -> np.ndarray:
        """The classic S-curve over ``days`` roll days: slow start,
        concentrated middle, fully complete at the end -- smoothstep
        ``3x^2 - 2x^3`` sampled at each day's close."""
        if days < 1:
            raise ValueError("need >= 1 roll day")
        curve = np.zeros(days)
        for d in range(days):
            x = (d + 1) / days
            curve[d] = x * x * (3 - 2 * x)
        curve[days - 1] = 1.0                  # exact, not 0.999999
        return curve

    def due_on_day(self, day: int) -> int:
        """Contracts due on ``day`` (0-based): the migration target
        minus what has already rolled. Falling behind earlier days
        simply makes later days' due larger -- the roll always catches
        up to the curve."""
        if day < 0 or day >= self._cumulative_curve.shape[0]:
            raise ValueError(f"day {day} outside the roll window")
        # Java Math.round: half-up (floor(x + 0.5)), not Python's
        # banker's-rounding round().
        target = math.floor(self._position * self._cumulative_curve[day] + 0.5)
        return max(0, target - self._rolled)

    def on_rolled(self, contracts: int) -> None:
        """Records rolled contracts (calendar-spread fills)."""
        if contracts < 0 or self._rolled + contracts > self._position:
            raise ValueError(f"rolled {contracts} would exceed the position")
        self._rolled += contracts

    def rolled(self) -> int:
        return self._rolled

    def remaining(self) -> int:
        return self._position - self._rolled

    def done(self) -> bool:
        return self._rolled == self._position

    def roll_days(self) -> int:
        return self._cumulative_curve.shape[0]
