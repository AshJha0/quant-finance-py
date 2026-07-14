"""Two-legged spread execution with LEGGING-RISK control (port of Java
``execution.SpreadExecutionAlgo``).

Pairs trades, cash-vs-futures basis, stub-vs-hedge: the trade is the
SPREAD, and the risk is the moment you own one leg without the other.
The discipline every spread desk runs:

* the LEAD leg (the illiquid one) is worked patiently, because it is
  the constraint;
* the HEDGE leg (the liquid one) CHASES the lead leg's fills at the
  spread ratio, because liquidity is cheap there;
* the legging imbalance ``|executed_lead * ratio - executed_hedge|``
  is capped: at the cap the algo stops adding lead risk entirely and
  the hedge child becomes the full imbalance -- cross it, pay the
  spread, get flat. An imbalance cap that yields is not a cap.

Quantities are positive per leg (the buy/sell directions are the
caller's order tickets); the ratio is hedge units per lead unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Children:
    """This interval's child quantities, per leg."""

    lead_qty: int
    hedge_qty: int
    at_risk_cap: bool


class SpreadExecutionAlgo:
    """Two-legged spread execution with a hard legging-imbalance cap;
    see the module docstring."""

    def __init__(self, lead_parent_qty: int, hedge_per_lead_unit: float,
                legging_limit_hedge_units: int, lead_child_max: int) -> None:
        """
        Args:
            lead_parent_qty: total lead-leg quantity to execute, > 0.
            hedge_per_lead_unit: spread ratio: hedge units per lead
                unit, > 0.
            legging_limit_hedge_units: max |lead*ratio - hedge|
                imbalance tolerated, >= 1 hedge unit.
            lead_child_max: largest lead child per decision, > 0 (the
                patience knob -- small children, passive fills).
        """
        if lead_parent_qty <= 0:
            raise ValueError("leadParentQty must be > 0")
        if not (hedge_per_lead_unit > 0) or hedge_per_lead_unit == math.inf:
            raise ValueError("ratio must be positive and finite")
        if legging_limit_hedge_units < 1:
            raise ValueError("leggingLimit must be >= 1 hedge unit")
        if legging_limit_hedge_units < hedge_per_lead_unit:
            # floor(limit/ratio) would be 0 at zero imbalance: decide()
            # could never emit a lead child and the algo would livelock.
            raise ValueError(
                f"leggingLimit {legging_limit_hedge_units} cannot cover even "
                f"one lead unit's hedge (ratio {hedge_per_lead_unit}) -- "
                "execution would be impossible")
        if lead_child_max <= 0:
            raise ValueError("leadChildMax must be > 0")
        self._lead_parent = lead_parent_qty
        self._ratio = hedge_per_lead_unit
        self._legging_limit = legging_limit_hedge_units
        self._lead_child_max = lead_child_max
        self._lead_executed = 0
        self._hedge_executed = 0

    def decide(self) -> Children:
        """The next children. The hedge chases the CURRENT imbalance;
        the lead child is sized so even a full fill cannot push the
        projected imbalance past the cap (assuming the hedge child also
        fills -- the cap protects against the hedge NOT filling, which
        is why the lead stops entirely at the cap)."""
        imbalance = self.imbalance_hedge_units()
        hedge_qty = max(0, imbalance)
        lead_remaining = self._lead_parent - self._lead_executed
        at_cap = imbalance >= self._legging_limit
        if at_cap:
            # Stop adding lead risk; the hedge child is the whole
            # imbalance -- cross it and get flat.
            lead_qty = 0
        else:
            # Even a full lead fill with NO hedge fill stays inside the cap.
            headroom_lead_units = math.floor((self._legging_limit - imbalance) / self._ratio)
            lead_qty = min(min(lead_remaining, self._lead_child_max), headroom_lead_units)
        return Children(lead_qty, hedge_qty, at_cap)

    def on_lead_fill(self, qty: int) -> None:
        """Lead-leg fill."""
        if qty < 0 or self._lead_executed + qty > self._lead_parent:
            raise ValueError(f"lead fill {qty} overfills parent")
        self._lead_executed += qty

    def on_hedge_fill(self, qty: int) -> None:
        """Hedge-leg fill. Bounded by the full spread's hedge target --
        a fill beyond it is a duplicate report, the same upstream bug
        the lead-leg guard exists to catch."""
        # Java Math.round: half-up. Python round() is banker's
        # rounding, so spell out floor(x + 0.5) to keep the two ports
        # on the same grid (the "round(4.5) == 5" pin depends on it).
        target = math.floor(self._lead_parent * self._ratio + 0.5)
        if qty < 0 or self._hedge_executed + qty > target:
            raise ValueError(
                f"hedge fill {qty} would exceed the spread's hedge target {target}")
        self._hedge_executed += qty

    def imbalance_hedge_units(self) -> int:
        """Current legging imbalance in hedge units (positive = hedge
        behind)."""
        return (math.floor(self._lead_executed * self._ratio + 0.5)
               - self._hedge_executed)

    def lead_executed(self) -> int:
        return self._lead_executed

    def hedge_executed(self) -> int:
        return self._hedge_executed

    def done(self) -> bool:
        """Done when the lead is complete AND the hedge has caught up."""
        return self._lead_executed == self._lead_parent and self.imbalance_hedge_units() <= 0
