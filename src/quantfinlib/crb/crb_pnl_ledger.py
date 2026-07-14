"""The central risk book's economics ledger (port of Java
``com.quantfinlib.crb.CrbPnlLedger``).

The number the desk head actually asks for at the close: did the
spread captured by internalizing pay for the hedging done? A CRB that
nets beautifully but hedges expensively is a cost center with good
graphics.

Accounting model (realized flow economics, deliberately simple and
stated): internalized client flow captures the street half spread
minus whatever improvement was given back
(``notional*(half_spread - improvement)/1e4``); routed flow captures
nothing (it went to the street); hedge executions cost their all-in
bps; router allocations cost their blended expected bps. Inventory
MARK-TO-MARKET P&L is deliberately out of scope -- that is the risk
report's domain (``CentralRiskBook.report``), and mixing realized
spread economics with unrealized inventory marks is how desks fool
themselves. All notionals positive, all bps non-negative, book-currency
units.

The Java ``persist.Checkpoint`` overnight persistence (writeState/
readState) is not ported -- no ``persist`` lane in this Python port.
"""

from __future__ import annotations

import math

from quantfinlib.crb.crb_router import Allocation
from quantfinlib.crb.internalization_engine import Decision


class CrbPnlLedger:

    def __init__(self):
        self._spread_captured = 0.0
        self._improvement_paid = 0.0
        self._hedge_cost = 0.0
        self._router_cost = 0.0
        self._internalizations = 0
        self._hedges = 0

    def on_internalized(self, internalized_notional: float, half_spread_bps: float,
                        improvement_bps: float) -> None:
        """Records one internalization decision's economics.

        internalized_notional: |notional| kept on the book, >= 0
        half_spread_bps: the street half spread saved, > 0
        improvement_bps: improvement given to the client, >= 0 and
            <= half_spread_bps
        """
        _require_non_negative(internalized_notional, "internalizedNotional")
        if not (half_spread_bps > 0) or half_spread_bps == math.inf:
            raise ValueError("half_spread_bps must be positive and finite")
        _require_non_negative(improvement_bps, "improvementBps")
        if improvement_bps > half_spread_bps:
            raise ValueError(
                f"improvement {improvement_bps} exceeds the half spread "
                f"{half_spread_bps} — the desk would be paying clients to trade")
        if internalized_notional == 0:
            return                          # fully routed decision: no economics
        self._spread_captured += internalized_notional * (half_spread_bps - improvement_bps) / 1e4
        self._improvement_paid += internalized_notional * improvement_bps / 1e4
        self._internalizations += 1

    def on_decision(self, decision: Decision, half_spread_bps: float) -> None:
        """Convenience: books a whole ``InternalizationEngine.Decision``."""
        self.on_internalized(abs(decision.internalized), half_spread_bps,
                             decision.improvement_bps)

    def on_hedge(self, notional: float, cost_bps: float) -> None:
        """Records a hedge execution's all-in cost."""
        _require_non_negative(cost_bps, "costBps")
        if not math.isfinite(notional):
            raise ValueError("notional must be finite")
        n = abs(notional)
        if n == 0:
            return
        self._hedge_cost += n * cost_bps / 1e4
        self._hedges += 1

    def on_route(self, notional: float, allocation: Allocation) -> None:
        """Records a router allocation's blended expected cost."""
        _require_non_negative(notional, "notional")
        self._router_cost += notional * allocation.expected_cost_bps / 1e4

    def spread_captured(self) -> float:
        """Spread captured by internalizing, net of improvement given
        back."""
        return self._spread_captured

    def improvement_paid(self) -> float:
        """Improvement handed to clients -- the cost of being worth
        trading with."""
        return self._improvement_paid

    def hedge_cost(self) -> float:
        return self._hedge_cost

    def router_cost(self) -> float:
        return self._router_cost

    def net_economics(self) -> float:
        """The desk's realized economics: captured spread minus hedging
        and routing costs. POSITIVE means the netting engine paid for
        its own risk management -- the CRB's entire commercial
        argument."""
        return self._spread_captured - self._hedge_cost - self._router_cost

    def internalizations(self) -> int:
        return self._internalizations

    def hedges(self) -> int:
        return self._hedges


def _require_non_negative(x: float, name: str) -> None:
    if not (x >= 0) or x == math.inf:
        raise ValueError(f"{name} must be >= 0 and finite")
