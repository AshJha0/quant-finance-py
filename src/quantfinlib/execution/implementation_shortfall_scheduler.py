"""Implementation-shortfall (arrival-price) schedule (port of Java
``execution.ImplementationShortfallScheduler``).

Turns the Almgren-Chriss optimal trajectory into executable
:class:`~quantfinlib.execution.slice.Slice` s. Where TWAP spreads evenly
and VWAP follows the volume curve, IS front-loads with urgency kappa --
trade more now to cut exposure to price drift, but not so fast that
temporary impact dominates; risk aversion lambda sets the balance
(lambda -> 0 degrades to TWAP, exactly as the math says).

Slice quantities are integer-allocated largest-remainder style so they
always sum exactly to the parent quantity.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from quantfinlib.execution import vwap_scheduler
from quantfinlib.execution.slice import Slice
from quantfinlib.microstructure.almgren_chriss import AlmgrenChriss

_MAX_DURATION_MILLIS = (1 << 63) - 1


def _optimal_trajectory_safe(params: "AlmgrenChriss.Params") -> "AlmgrenChriss.Trajectory":
    """``AlmgrenChriss.optimal_trajectory``, but tolerant of the sinh
    overflow a huge risk aversion triggers.

    Port note: Java's ``Math.sinh`` saturates to ``Double.POSITIVE_INFINITY``
    past ~710, so an extreme ``risk_aversion`` degrades the Java
    trajectory to NaN holdings/trades (inf/inf) rather than throwing --
    exactly the condition this scheduler's overflow guard below is
    built to catch. Python's ``math.sinh`` instead raises
    ``OverflowError`` at that same boundary, so it is caught here and
    mapped to the all-NaN trajectory Java would have produced, keeping
    the overflow guard and the front-load calibrator's "NaN reads as
    infinitely front-loaded" logic identical across ports.
    """
    try:
        return AlmgrenChriss.optimal_trajectory(params)
    except OverflowError:
        n = params.intervals
        return AlmgrenChriss.Trajectory(
            np.full(n + 1, math.nan), np.full(n, math.nan),
            math.inf, math.nan, math.nan)


def schedule(params: "AlmgrenChriss.Params", duration_millis: int) -> List[Slice]:
    """The optimal IS schedule for the given market parameters.

    Args:
        params: Almgren-Chriss inputs (``total_shares`` is the parent).
        duration_millis: wall-clock execution window the horizon maps onto.
    """
    t = _optimal_trajectory_safe(params)
    weights = t.trades
    if not np.all(np.isfinite(weights)):
        # sinh overflows around kappa*T ~ 710; without this guard the
        # NaN weights would silently degrade the integer allocation
        # into an O(parentShares) loop yielding a garbage schedule.
        raise ValueError(
            "risk aversion too high for these parameters (sinh overflow "
            "in the AC trajectory) -- reduce riskAversion or shorten "
            "the horizon")
    n = weights.shape[0]
    if duration_millis < 0 or duration_millis > _MAX_DURATION_MILLIS // n:
        raise ValueError(f"durationMillis out of range for {n} slices")
    # Java Math.round: half-up (floor(x + 0.5)), not Python's
    # banker's-rounding round().
    parent = math.floor(params.total_shares + 0.5)
    quantities = vwap_scheduler.allocate_proportionally(parent, weights)
    return [Slice(duration_millis * i // n, int(quantities[i])) for i in range(n)]


def risk_aversion_for_front_load(base: "AlmgrenChriss.Params",
                                 front_load_fraction: float) -> float:
    """Convenience urgency calibration: the risk aversion whose first
    slice is roughly ``front_load_fraction`` of the parent (e.g. 0.3 =
    "30% up front"), found by bisection on lambda. Useful when traders
    think in front-load, not in lambda.
    """
    if (front_load_fraction <= 1.0 / base.intervals
            or front_load_fraction >= 1):
        raise ValueError(
            "frontLoadFraction must exceed the TWAP slice and be below 1")
    lo = 0.0
    hi = 1.0
    # Grow hi until it front-loads enough. A NaN fraction (sinh overflow
    # at huge lambda) counts as "more than enough": it stops the growth
    # and bisection then converges back below the overflow boundary.
    for _ in range(60):
        if _front_load_of(base, hi) >= front_load_fraction:
            break
        hi *= 4
    for _ in range(80):
        mid = (lo + hi) / 2
        if _front_load_of(base, mid) < front_load_fraction:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    # Never return silently-wrong urgency: verify the calibration landed.
    achieved = _front_load_of(base, lam)
    if not abs(achieved - front_load_fraction) < 0.01:
        raise ValueError(
            f"front-load {front_load_fraction:.2f} unreachable for these "
            f"parameters (best achievable near {achieved:.3f}) -- the AC "
            "trajectory overflows before front-loading that hard")
    return lam


def _front_load_of(base: "AlmgrenChriss.Params", lam: float) -> float:
    """First-slice fraction, NaN mapped to +inf so overflow reads as
    "too front-loaded"."""
    t = _optimal_trajectory_safe(base.with_risk_aversion(lam))
    f = t.trades[0] / base.total_shares
    return math.inf if math.isnan(f) else f
