"""Cross-sectional momentum (port of the classic lane of Java
``backtest.portfolio.CrossSectionalMomentum``).

The classic equity factor: at every rebalance the strategy ranks the
tradeable stocks, goes long the best trailing performers and short the
worst. Momentum definition is the academic standard "12-1":
``close(i - skip) / close(i - lookback) - 1`` — trailing ``lookback``
bars with the most recent ``skip`` bars excluded (short-term reversal
contaminates raw 12-month momentum; Jegadeesh-Titman 1993). Each side
is equal-weighted at ``gross_per_side`` total, so the default book is
dollar-neutral with 2x gross-per-side exposure.

The Java class ranks over a ``PointInTimeUniverse`` to kill
survivorship bias; that data type is not in the Python port yet, so
``universe`` accepts ``None`` (everything always tradeable — the naive
behavior) or any object exposing ``is_member(symbol, timestamp)`` for
forward compatibility.

Candidates need ``lookback`` bars of history at the rebalance bar;
earlier bars produce an empty book (the engine holds cash).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from quantfinlib.backtest.portfolio.portfolio_strategy import PortfolioStrategy
from quantfinlib.data.bar_series import BarSeries


class CrossSectionalMomentum(PortfolioStrategy):
    """Point-in-time cross-sectional momentum; see the module docstring."""

    @dataclass(frozen=True)
    class Config:
        """Factor construction parameters.

        Attributes:
            lookback_bars: Momentum window (252 ~ 12 months of dailies).
            skip_bars: Most-recent bars excluded (21 ~ 1 month).
            per_side: Names held long and short (shrinks when the member
                count can't fill both sides without overlap).
            gross_per_side: Total absolute weight per side (0.5 -> 1x
                gross, dollar-neutral).
        """

        lookback_bars: int
        skip_bars: int
        per_side: int
        gross_per_side: float

        def __post_init__(self) -> None:
            if (self.lookback_bars <= 0 or self.skip_bars < 0
                    or self.skip_bars >= self.lookback_bars
                    or self.per_side <= 0 or self.gross_per_side <= 0):
                raise ValueError("need lookback > skip >= 0, per_side > 0, "
                                 "gross_per_side > 0")

        @staticmethod
        def twelve_minus_one(per_side: int
                             ) -> "CrossSectionalMomentum.Config":
            """The academic 12-1 monthly-rebalance setup on daily bars."""
            return CrossSectionalMomentum.Config(252, 21, per_side, 0.5)

    def __init__(self, universe,
                 config: "CrossSectionalMomentum.Config") -> None:
        self._universe = universe   # None = everything always tradeable
        self._config = config
        self._data: Optional[Dict[str, BarSeries]] = None
        self._symbols: List[str] = []
        self._clock: Optional[BarSeries] = None

    def name(self) -> str:
        c = self._config
        return (f"XS_MOMENTUM({c.lookback_bars}-{c.skip_bars}, "
                f"{c.per_side}/side)")

    def init(self, data: Dict[str, BarSeries]) -> None:
        self._data = data
        self._symbols = list(data.keys())
        # Aligned series: any one keeps time.
        self._clock = next(iter(data.values()))

    def target_weights(self, index: int) -> Dict[str, float]:
        c = self._config
        if index < c.lookback_bars:
            return {}   # not enough history: hold cash
        now = self._clock.timestamp(index)

        # The point-in-time step: rank ONLY the names that are members
        # at this bar. Dead and dropped stocks never enter the
        # cross-section.
        momentum: List[float] = []
        candidate: List[str] = []
        for symbol in self._symbols:
            if (self._universe is not None
                    and not self._universe.is_member(symbol, now)):
                continue
            s = self._data[symbol]
            past = s.close(index - c.lookback_bars)
            recent = s.close(index - c.skip_bars)
            momentum.append(recent / past - 1)
            candidate.append(symbol)
        n = len(candidate)
        # Both sides need at least one name each without overlapping.
        side = min(c.per_side, n // 2)
        if side == 0:
            return {}

        # Ascending stable sort: losers at the front, winners at the back.
        order = np.argsort(np.asarray(momentum), kind="stable")

        weights: Dict[str, float] = {}
        w = c.gross_per_side / side
        for i in range(side):
            weights[candidate[order[n - 1 - i]]] = w    # winners long
            weights[candidate[order[i]]] = -w           # losers short
        return weights
