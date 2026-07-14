"""The central risk book's hedging loop (port of Java
``com.quantfinlib.crb.CrbAutoHedger``).

Per-factor exposure BANDS, a cost-aware hedge when breached, and a
cooldown so the book does not chase its own hedges. The policy is
deliberately two-speed:

- inside the bands, warehouse -- the whole point of a CRB is that
  inventory nets against future flow for free;
- on a breach, hedge the breached factors back to ``reset_fraction`` of
  the limit through ``HedgeOptimizer`` -- cost-aware first, but if the
  cost-aware hedge leaves any factor still OUTSIDE its limit the hedge
  reruns at zero cost weight: a hard limit outranks transaction-cost
  thrift, always.

Time is a caller-supplied interval counter (no wall clock --
deterministic, replayable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantfinlib.crb.hedge_optimizer import HedgeOptimizer


@dataclass(frozen=True, slots=True)
class HedgeOrder:
    """One instrument's hedge instruction."""

    instrument: int
    notional: float


class CrbAutoHedger:

    def __init__(self, limits: list[float], reset_fraction: float,
                cooldown_intervals: int):
        """
        limits: per-factor |exposure| limits (registry order; > 0 each)
        reset_fraction: post-hedge target as a fraction of the limit,
            in (0, 1]
        cooldown_intervals: min intervals between hedges, >= 0
        """
        for l in limits:
            if not (l > 0) or l == math.inf:
                raise ValueError("every limit must be positive and finite")
        if not (0 < reset_fraction <= 1):
            raise ValueError("reset_fraction must be in (0, 1]")
        if cooldown_intervals < 0:
            raise ValueError("cooldown_intervals must be >= 0")
        self._limits = list(limits)
        self._reset_fraction = reset_fraction
        self._cooldown_intervals = cooldown_intervals
        self._has_hedged = False
        self._last_hedge_interval = 0
        self._hedges_emitted = 0

    def breached(self, exposures: list[float]) -> bool:
        """True when any factor sits outside its band."""
        self._require_length(exposures)
        return any(abs(exposures[f]) > self._limits[f] for f in range(len(self._limits)))

    def check(self, exposures: list[float], covariance: list[list[float]],
             loadings: list[list[float]], cost_per_unit: list[float],
             cost_weight: float, now_interval: int) -> list[HedgeOrder]:
        """The hedging decision for this interval. Empty when inside all
        bands or still cooling down; otherwise the cheapest hedge of
        the EXCESS -- only what sits beyond ``reset_fraction*limit`` on
        the breached factors is hedged. If the cost-aware hedge still
        leaves a factor outside its hard limit, the excess is re-hedged
        cost-blind.

        exposures: current factor exposures (registry order)
        covariance: factor covariance for the optimizer
        loadings: instrument factor loadings [factor][instrument]
        cost_per_unit: per-unit hedge costs
        cost_weight: lambda for the cost-aware first attempt
        now_interval: caller's interval counter (monotone)
        """
        self._require_length(exposures)
        if not self.breached(exposures):
            return []
        # has_hedged guards the subtraction: an unguarded first check
        # would suppress the very FIRST hedge if the cooldown compare
        # underflowed.
        if self._has_hedged and now_interval - self._last_hedge_interval < self._cooldown_intervals:
            return []
        # Hedge target: the excess beyond the reset band on breached
        # factors, zero elsewhere -- flattening the whole book would
        # throw away exactly the inventory the CRB exists to warehouse.
        excess = [0.0] * len(self._limits)
        for f in range(len(self._limits)):
            band = self._limits[f] * self._reset_fraction
            if abs(exposures[f]) > self._limits[f]:
                excess[f] = _sign(exposures[f]) * (abs(exposures[f]) - band)
        h = HedgeOptimizer.hedge(excess, covariance, loadings, cost_per_unit, cost_weight)
        if self._still_breached(HedgeOptimizer.residual(exposures, loadings, h)):
            # The limit is hard; the cost preference is not.
            h = HedgeOptimizer.hedge(excess, covariance, loadings, cost_per_unit, 0.0)
        # Dust filter: coordinate descent converges instruments that
        # belong at zero only to ~tolerance of the largest notional --
        # no desk sends a sub-cent hedge order to the street.
        max_h = max((abs(v) for v in h), default=0.0)
        dust = 1e-6 * max_h
        orders = [HedgeOrder(i, v) for i, v in enumerate(h) if abs(v) > dust]
        if orders:
            self._has_hedged = True
            self._last_hedge_interval = now_interval
            self._hedges_emitted += 1
        return orders

    def target_band(self, factor: int) -> float:
        """The band the book hedges back INTO for a factor."""
        return self._limits[factor] * self._reset_fraction

    def hedges_emitted(self) -> int:
        return self._hedges_emitted

    def _still_breached(self, residual: list[float]) -> bool:
        return any(abs(residual[f]) > self._limits[f] for f in range(len(self._limits)))

    def _require_length(self, exposures: list[float]) -> None:
        if len(exposures) != len(self._limits):
            raise ValueError(
                f"exposures length {len(exposures)} != limits length {len(self._limits)}")
        for e in exposures:
            # abs(NaN) > limit is FALSE: an unguarded NaN exposure would
            # read as "inside the band" and silently disable the
            # auto-hedger for that factor forever.
            if not math.isfinite(e):
                raise ValueError("exposures must be finite")


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0
