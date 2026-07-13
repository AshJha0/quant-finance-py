"""Drawdown structure (port of Java ``backtest.DrawdownAnalytics``).

"Max drawdown 18%" hides the number that actually fires clients: how
LONG the pain lasted. A strategy that loses 18% and recovers in three
weeks and one that spends two years under water have the same max
drawdown and completely different survival odds. Redemptions,
risk-committee reviews and career risk are all functions of drawdown
DURATION, not just depth.

The walk: track the running peak; a drawdown episode opens the first
period equity dips below it and closes when equity regains the peak
(recovery) or the series ends (still open — ``recovery_index = -1``, a
fact worth surfacing, not hiding: an open drawdown at the end of a
backtest is often the honest state of the strategy today).

* **depth** — ``1 - trough / peak`` per episode;
* **duration** — periods from the peak to recovery (or to the last bar
  for an open episode);
* **time under water** — the fraction of ALL periods spent below the
  running peak. A strategy under water 60% of the time is painful to
  hold even when each individual dip is shallow;
* **episodes** — the full chronological list, so callers can take the
  top-k by depth, histogram durations, or line episodes up against
  market events.

Equity must be positive throughout — a ratio-of-peak drawdown is
meaningless through zero or negative equity. Static, deterministic,
research lane. The max depth agrees exactly with the plain max-drawdown
estimator (tested).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class DrawdownAnalytics:
    """Static drawdown-episode analysis; see the module docstring."""

    @dataclass(frozen=True)
    class Drawdown:
        """One peak-to-recovery episode.

        ``recovery_index`` is ``-1`` while the drawdown is still open at
        series end; duration then runs to the last bar.
        """

        peak_index: int
        trough_index: int
        recovery_index: int
        depth: float

        def duration(self, series_length: int) -> int:
            """Periods from peak to recovery, or to series end if open."""
            end = self.recovery_index if self.recovery_index >= 0 else series_length - 1
            return end - self.peak_index

    @dataclass(frozen=True)
    class Result:
        """Aggregate drawdown structure.

        Attributes:
            max_depth: Deepest episode's depth (0 if equity never dips).
            max_duration: Longest episode duration in periods.
            time_under_water: Fraction of periods below the running peak.
            episodes: Chronological drawdown episodes (tuple).
        """

        max_depth: float
        max_duration: int
        time_under_water: float
        episodes: tuple

    @staticmethod
    def analyze(equity) -> "DrawdownAnalytics.Result":
        """Analyzes an equity curve of >= 2 finite, strictly positive points.

        Raises:
            ValueError: on fewer than 2 points or any non-finite / <= 0 value.
        """
        e = np.asarray(equity, dtype=float)
        n = e.shape[0]
        if n < 2:
            raise ValueError(f"need >= 2 equity points, got {n}")
        for v in e:
            # NaN gate: not (v > 0) is True for NaN, exactly as in Java.
            if not (v > 0) or v == math.inf:
                raise ValueError(f"equity must be finite and > 0, got {v}")

        episodes: list[DrawdownAnalytics.Drawdown] = []
        peak = float(e[0])
        peak_idx = 0
        under_water = 0
        open_ = False
        trough = 0.0
        trough_idx = -1

        for i in range(1, n):
            if e[i] >= peak:
                if open_:
                    episodes.append(DrawdownAnalytics.Drawdown(
                        peak_idx, trough_idx, i, 1 - trough / peak))
                    open_ = False
                peak = float(e[i])
                peak_idx = i
            else:
                under_water += 1
                if not open_:
                    open_ = True
                    trough = float(e[i])
                    trough_idx = i
                elif e[i] < trough:
                    trough = float(e[i])
                    trough_idx = i
        if open_:
            episodes.append(DrawdownAnalytics.Drawdown(
                peak_idx, trough_idx, -1, 1 - trough / peak))

        max_depth = 0.0
        max_duration = 0
        for d in episodes:
            max_depth = max(max_depth, d.depth)
            max_duration = max(max_duration, d.duration(n))
        return DrawdownAnalytics.Result(max_depth, max_duration,
                                        under_water / n, tuple(episodes))
