"""The dynamic benchmark execution algorithm (port of Java
``execution.BenchmarkExecutor``).

One stateful executor that works a parent order toward any of the
standard benchmarks -- VWAP, TWAP, Arrival Price, Implementation
Shortfall, Closing Price, Opening Price, and Participation (POV) --
and, unlike a precomputed slice list (TWAP/VWAP/IS schedulers),
re-decides every interval from live market state. Cross-asset:
prices/sizes are floats, so it serves equities and FX identically.

Two layers:

1. **The benchmark curve** -- each :class:`Benchmark` defines the
   fraction of the parent that SHOULD be complete by now. Time-driven
   benchmarks (TWAP linear, Arrival/IS front-loaded, Close
   back-loaded, Open aggressively front-loaded) use elapsed schedule
   fraction; volume-driven benchmarks (VWAP, POV) use the realized
   volume curve.
2. **The dynamic adjustment** -- the raw "behind schedule" quantity is
   then shaped by alpha, spread/volatility, liquidity, and schedule
   drift.

Usage: construct with the parent, benchmark and horizon; each interval
call :meth:`BenchmarkExecutor.due_quantity` with the current
:class:`MarketState` and the elapsed schedule fraction (and, for volume
benchmarks, feed market prints via :meth:`BenchmarkExecutor.on_market_volume`);
send the returned child; report fills via :meth:`BenchmarkExecutor.on_fill`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

from quantfinlib.microstructure.execution import Side
from quantfinlib.util import math_utils

#: How hard a 1% relative trading cost (spread fraction of mid plus
#: impact as a fraction) damps aggression: the term is
#: ``1 / (1 + cost * SPREAD_SENSITIVITY)``, so a 1% cost halves the
#: pace and a 2-pip FX spread (~0.002%) barely registers. A deliberate
#: calibration constant, not a unit conversion.
SPREAD_SENSITIVITY = 100.0


class Benchmark(Enum):
    """The benchmark this parent is measured against."""

    #: Equal participation over time.
    TWAP = auto()
    #: Follow the expected volume curve.
    VWAP = auto()
    #: Minimize slippage vs the price when the order arrived (front-loaded).
    ARRIVAL_PRICE = auto()
    #: Almgren-Chriss shortfall: front-loaded, volatility raises urgency.
    IMPLEMENTATION_SHORTFALL = auto()
    #: Track the closing price: back-loaded toward the close.
    CLOSING_PRICE = auto()
    #: Track the opening price: aggressively front-loaded at the open.
    OPENING_PRICE = auto()
    #: Percentage-of-volume: a fixed share of realized volume (time-agnostic).
    PARTICIPATION = auto()


@dataclass(frozen=True, slots=True)
class MarketState:
    """A snapshot of the real-time inputs a benchmark algo evaluates.

    All fields are optional in the sense that a neutral value disables
    that input: ``spread`` = 0, ``volatility`` = 0, ``alpha`` = 0,
    ``displayed_depth`` = +inf (no liquidity cap),
    ``expected_volume_fraction_elapsed`` = schedule fraction (VWAP
    falls back to TWAP), ``impact_bps`` = 0. NaN in any field is
    treated as its neutral value -- a transient bad input must degrade
    the input, never silently stall the parent.

    Units contract (normalized, so the shipped models plug in directly):

    * ``volatility`` -- a normalized volatility-REGIME signal, ~0 calm
      to ~1 extreme.
    * ``alpha`` -- a normalized expected-move signal in [-1, 1] (+ = up).
    * ``displayed_depth`` -- size dealable NOW (displayed top of book).

    Attributes:
        mid: current mid (for reference/markout).
        spread: bid/ask spread in price units (cost).
        volatility: normalized vol-regime signal, ~0..1.
        displayed_depth: size available to take now (liquidity cap).
        expected_volume_fraction_elapsed: VWAP curve: fraction of the
            day's volume expected to have traded by now.
        alpha: normalized expected-move signal in [-1, 1] (+ = up).
        impact_bps: estimated impact of a full child in bps; damps
            aggression alongside the spread.
    """

    mid: float
    spread: float
    volatility: float
    displayed_depth: float
    expected_volume_fraction_elapsed: float
    alpha: float
    impact_bps: float

    @staticmethod
    def neutral(mid: float, schedule_fraction: float) -> "MarketState":
        """A neutral state: no spread/vol/alpha/impact, unlimited
        depth, VWAP=TWAP."""
        return MarketState(mid, 0, 0, math.inf, schedule_fraction, 0, 0)


def _clamp01(x: float) -> float:
    return math_utils.clamp(x, 0, 1)


class BenchmarkExecutor:
    """Live-adaptive benchmark executor; see the module docstring."""

    def __init__(self, side: Side, parent_qty: int, benchmark: Benchmark,
                participation_rate: float, alpha_urgency: float,
                max_depth_fraction: float) -> None:
        """
        Args:
            side: buy or sell (sets the alpha sign convention).
            parent_qty: total quantity to execute.
            benchmark: the benchmark to track.
            participation_rate: POV target in (0,1] (ignored unless PARTICIPATION).
            alpha_urgency: how hard alpha shifts the pace (0 disables; ~5-20 typical).
            max_depth_fraction: cap each child at this fraction of displayed depth, (0,1].
        """
        if parent_qty <= 0:
            raise ValueError("parentQty must be > 0")
        if (benchmark == Benchmark.PARTICIPATION
                and (participation_rate <= 0 or participation_rate > 1)):
            raise ValueError("PARTICIPATION needs participationRate in (0,1]")
        if alpha_urgency < 0:
            raise ValueError("alphaUrgency must be >= 0")
        if max_depth_fraction <= 0 or max_depth_fraction > 1:
            raise ValueError("maxDepthFraction must be in (0,1]")
        self._side = side
        self._parent_qty = parent_qty
        self._benchmark = benchmark
        self._participation_rate = participation_rate
        self._alpha_urgency = alpha_urgency
        self._max_depth_fraction = max_depth_fraction
        self._executed = 0
        self._market_volume = 0

    @staticmethod
    def of(side: Side, parent_qty: int, benchmark: Benchmark) -> "BenchmarkExecutor":
        """Sensible defaults: alpha urgency 1 (a full-scale normalized
        alpha of +-1 doubles/halves the pace -- smooth, never
        rail-pinned), child capped at 25% of displayed depth.
        PARTICIPATION must state its rate -- use :meth:`pov`."""
        if benchmark == Benchmark.PARTICIPATION:
            raise ValueError(
                "PARTICIPATION needs an explicit rate: use pov(side, qty, rate)")
        return BenchmarkExecutor(side, parent_qty, benchmark, 0.1, 1, 0.25)

    @staticmethod
    def pov(side: Side, parent_qty: int, participation_rate: float) -> "BenchmarkExecutor":
        """POV convenience."""
        return BenchmarkExecutor(side, parent_qty, Benchmark.PARTICIPATION,
                                 participation_rate, 1, 0.25)

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def on_market_volume(self, qty: int) -> None:
        """A market print that was NOT our fill (drives VWAP/POV
        realized volume)."""
        if qty > 0:
            self._market_volume += qty

    def on_fill(self, qty: int) -> None:
        """Our own child fill."""
        if qty > 0:
            self._executed += qty

    # ------------------------------------------------------------------
    # The decision
    # ------------------------------------------------------------------

    def due_quantity(self, schedule_fraction: float, m: MarketState) -> int:
        """Shares to send now, given the current market and how far
        through the schedule we are. Returns 0 when on/ahead of
        schedule or done; caps at the parent remainder and at
        ``max_depth_fraction * displayed_depth``.

        Args:
            schedule_fraction: elapsed fraction of the execution
                horizon in [0, 1] (wall-clock progress); ignored by
                PARTICIPATION, which is volume-driven.
            m: the live market snapshot.
        """
        remaining = self._parent_qty - self._executed
        if remaining <= 0:
            return 0
        f = _clamp01(schedule_fraction)

        if self._benchmark == Benchmark.PARTICIPATION:
            # Volume-driven: target = participation x others' volume.
            target = int(self._participation_rate * self._market_volume)
            behind = target - self._executed
        else:
            # Target completion by now = curve(progress) x parent. A
            # NaN volume-curve input degrades VWAP to the time fraction
            # -- int(nan) would raise in Python, and even a silent 0
            # would read as "nothing due" and stall the parent, so the
            # NaN is caught explicitly.
            target_frac = self._target_completion(f, m)
            if math.isnan(target_frac):
                target_frac = f
            # Java Math.round: half-up (floor(x + 0.5)), not Python's
            # banker's-rounding round().
            behind = (math.floor(target_frac * self._parent_qty + 0.5)
                     - self._executed)
        if behind <= 0:
            return 0

        # Dynamic shaping: alpha pulls the pace, spread/vol damp it.
        # For PARTICIPATION the multiplier may only DAMP (clamp to <=
        # 1): the participation rate is a hard promise to the client,
        # and letting alpha push a POV child to 4x "behind" realizes
        # participation far above the configured rate.
        urgency = self._urgency_multiplier(m)
        if self._benchmark == Benchmark.PARTICIPATION:
            urgency = min(urgency, 1.0)
        shaped = math.ceil(behind * urgency)

        # Liquidity cap and parent remainder.
        cap = remaining
        if m.displayed_depth < math.inf:
            cap = min(cap, math.floor(self._max_depth_fraction * m.displayed_depth))
        return max(0, min(shaped, cap))

    def _target_completion(self, f: float, m: MarketState) -> float:
        """Fraction of the parent that SHOULD be complete at schedule
        progress ``f`` under this benchmark's curve."""
        b = self._benchmark
        if b == Benchmark.TWAP:
            return f
        if b == Benchmark.VWAP:
            # NOTE: Python's builtin min/max (used by math_utils.clamp)
            # do not propagate NaN the way Java's Math.min/Math.max do
            # (min(1, nan) == 1, not nan), so NaN must be preserved
            # explicitly here -- the caller relies on a NaN target
            # fraction to fall back to the time fraction f.
            v = m.expected_volume_fraction_elapsed
            return math.nan if math.isnan(v) else _clamp01(v)
        if b in (Benchmark.ARRIVAL_PRICE, Benchmark.IMPLEMENTATION_SHORTFALL):
            # Front-loaded: 1-(1-f)^2 trades more early (cuts timing risk).
            return 1 - (1 - f) * (1 - f)
        if b == Benchmark.CLOSING_PRICE:
            # Back-loaded: f^2 keeps weight near the close.
            return f * f
        if b == Benchmark.OPENING_PRICE:
            # Aggressively front-loaded: sqrt(f) is near-done early.
            return math.sqrt(f)
        # PARTICIPATION is volume-driven, branched off before this
        # dispatch: a silent fallback value here would make a
        # mis-routed POV behave like TWAP with no error -- fail loud
        # instead.
        raise AssertionError(
            "PARTICIPATION is volume-driven and handled before the curve")

    def _urgency_multiplier(self, m: MarketState) -> float:
        """The real-time pace multiplier around the schedule. Alpha in
        the trading direction speeds up (you're racing an adverse
        move); a wide spread or estimated impact slows down (cost);
        volatility slows passive benchmarks but speeds urgency-driven
        ones (timing risk). Bounded to [0.25, 4]."""
        # NaN inputs are neutral (0): a transient bad signal must
        # weaken the input, never poison the multiplier into a silent
        # stall.
        alpha = 0.0 if math.isnan(m.alpha) else m.alpha
        signed_alpha = alpha if self._side == Side.BUY else -alpha
        u = 1 + self._alpha_urgency * signed_alpha

        # Cost of trading NOW: relative spread plus the estimated
        # impact of a full child (both dimensionless fractions of
        # price), damping aggression together -- NaN in either fails
        # the > 0 test = neutral.
        cost = 0.0
        if m.mid > 0 and m.spread > 0:
            cost += m.spread / m.mid
        if m.impact_bps > 0:
            cost += m.impact_bps / 1e4
        if cost > 0:
            u *= 1.0 / (1 + cost * SPREAD_SENSITIVITY)

        # Volatility (normalized regime, ~0..1): raises urgency for
        # shortfall/arrival (timing risk), lowers it for passive
        # benchmarks.
        if m.volatility > 0:
            if self._benchmark in (Benchmark.IMPLEMENTATION_SHORTFALL,
                                   Benchmark.ARRIVAL_PRICE):
                u *= 1 + m.volatility
            else:
                u *= 1.0 / (1 + m.volatility)
        return math_utils.clamp(u, 0.25, 4.0)

    # ------------------------------------------------------------------
    # Progress / diagnostics
    # ------------------------------------------------------------------

    def executed(self) -> int:
        return self._executed

    def remaining(self) -> int:
        return self._parent_qty - self._executed

    def parent_qty(self) -> int:
        return self._parent_qty

    def market_volume(self) -> int:
        return self._market_volume

    def done(self) -> bool:
        return self._executed >= self._parent_qty

    def realized_participation(self) -> float:
        """Realized participation vs other-flow volume (NaN before any
        market print)."""
        return (math.nan if self._market_volume == 0
               else self._executed / self._market_volume)

    def schedule_drift(self, schedule_fraction: float, m: MarketState) -> float:
        """Schedule drift: executed fraction minus the benchmark's
        target fraction at ``schedule_fraction``. Positive = ahead,
        negative = behind. For PARTICIPATION this compares against the
        participation target instead of the time curve."""
        executed_frac = self._executed / self._parent_qty
        if self._benchmark == Benchmark.PARTICIPATION:
            target = (0.0 if self._market_volume == 0
                     else min(1.0, self._participation_rate * self._market_volume
                              / self._parent_qty))
        else:
            target = self._target_completion(_clamp01(schedule_fraction), m)
        return executed_frac - target

    def side(self) -> Side:
        return self._side

    def benchmark(self) -> Benchmark:
        return self._benchmark
