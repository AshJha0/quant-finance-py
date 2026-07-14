"""The opportunistic execution archetype (port of Java
``execution.LiquiditySeekingAlgo``).

The counterpart to :class:`~quantfinlib.execution.benchmark_executor.BenchmarkExecutor`'s
schedule-driven family. A schedule algo asks "am I behind the curve?";
a liquidity seeker asks "is the market cheap RIGHT NOW?" and trades in
bursts when it is: spread tighter than its time-of-day forecast, calm
volatility regime, low estimated impact. Between opportunities it sits
still -- which is why it needs the one discipline every seek-style algo
ships with: a completion floor that ramps in over the final stretch of
the horizon, so patience can never become a missed parent.

Decision per interval: score the moment -- spread at-or-under forecast,
vol regime below the calm threshold, impact below the cap -- and if
ALL hold, take an aggressive clip (``max_depth_fraction *
displayed_depth``). If not, send only the completion floor: 0 until
``force_complete_from``, then the remaining quantity spread linearly
over what is left of the horizon (at f = 1 the floor IS the
remainder). NaN inputs degrade honestly: an unknown forecast means the
moment cannot be judged cheap (no opportunistic burst), but the floor
still guarantees completion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from quantfinlib.execution.benchmark_executor import MarketState
from quantfinlib.util import math_utils


@dataclass(frozen=True, slots=True)
class Config:
    """
    Attributes:
        spread_tolerance: how far above forecast still counts as cheap,
            as a fraction (0.1 = up to 10% over); 0 = strict.
        max_vol_regime: opportunistic bursts only below this normalized
            vol regime (0..1), e.g. 0.5.
        max_impact_bps: opportunistic bursts only below this estimated
            impact, e.g. 5 bps.
        max_depth_fraction: burst clip as a fraction of displayed
            depth, in (0, 1].
        force_complete_from: schedule fraction where the completion
            floor starts ramping, in [0, 1), e.g. 0.7.
    """

    spread_tolerance: float
    max_vol_regime: float
    max_impact_bps: float
    max_depth_fraction: float
    force_complete_from: float

    def __post_init__(self) -> None:
        if (self.spread_tolerance < 0 or self.max_vol_regime < 0
                or self.max_impact_bps < 0
                or self.max_depth_fraction <= 0 or self.max_depth_fraction > 1
                or self.force_complete_from < 0 or self.force_complete_from >= 1):
            raise ValueError("invalid seek config")

    @staticmethod
    def defaults() -> "Config":
        """10% spread tolerance, vol regime < 0.5, impact < 5 bps, 25%
        clips, floor from 70%."""
        return Config(0.10, 0.5, 5.0, 0.25, 0.7)


class LiquiditySeekingAlgo:
    """Opportunistic liquidity-seeking executor; see the module docstring."""

    def __init__(self, parent_qty: int, config: Optional[Config] = None) -> None:
        if parent_qty <= 0:
            raise ValueError("parentQty must be > 0")
        self._parent_qty = parent_qty
        self._config = config if config is not None else Config.defaults()
        self._executed = 0

    def due_quantity(self, schedule_fraction: float, m: MarketState,
                     forecast_spread: float) -> int:
        """Shares to send now. An opportunistic burst when the moment
        is cheap (see class doc); otherwise the completion floor.
        Always capped at the remainder.

        Args:
            schedule_fraction: elapsed fraction of the horizon in [0, 1].
            m: the live market snapshot (:class:`MarketState` units contract).
            forecast_spread: the expected spread for this time of day;
                NaN = cheapness unknowable, floor only.
        """
        remaining = self._parent_qty - self._executed
        if remaining <= 0:
            return 0
        f = math_utils.clamp(schedule_fraction, 0, 1)

        due = 0
        if self._is_cheap(m, forecast_spread):
            depth = m.displayed_depth
            due = (remaining if depth == math.inf
                  else math.floor(self._config.max_depth_fraction * max(depth, 0)))

        # The completion floor: from force_complete_from, the remainder
        # spread linearly over the horizon left; at f = 1 the floor IS
        # the remainder.
        if f > self._config.force_complete_from:
            ramp = ((f - self._config.force_complete_from)
                    / (1 - self._config.force_complete_from))
            floor = remaining if f >= 1 else math.ceil(remaining * ramp)
            due = max(due, floor)
        return min(due, remaining)

    def _is_cheap(self, m: MarketState, forecast_spread: float) -> bool:
        """All three opportunity gates at once -- spread at/under its
        forecast (within tolerance), calm regime, low impact. NaN
        anywhere fails the gate it appears in: an unknowable moment is
        never "cheap"."""
        spread_cheap = (m.spread >= 0 and forecast_spread > 0
                       and forecast_spread != math.inf
                       and m.spread <= forecast_spread * (1 + self._config.spread_tolerance))
        # NaN fails BOTH remaining gates (NaN <= x is False): an
        # unknowable regime or impact must never authorize a burst -- a
        # vol-feed outage during a spike is exactly when firing would
        # hurt most.
        calm = m.volatility <= self._config.max_vol_regime
        low_impact = m.impact_bps <= self._config.max_impact_bps
        return spread_cheap and calm and low_impact

    def on_fill(self, qty: int) -> None:
        """Our own child fill."""
        if qty > 0:
            self._executed += qty

    def executed(self) -> int:
        return self._executed

    def remaining(self) -> int:
        return self._parent_qty - self._executed

    def done(self) -> bool:
        return self._executed >= self._parent_qty
