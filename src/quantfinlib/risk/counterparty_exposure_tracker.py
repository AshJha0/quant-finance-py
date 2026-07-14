"""Counterparty credit exposure modeling with netting.

Port of Java ``com.quantfinlib.risk.CounterpartyExposureTracker``:

* Current exposure — max(0, net mark-to-market) per netting set.
* Potential future exposure — notional add-ons by tenor bucket (BIS
  current-exposure-method style FX factors: <1y 1%, 1-5y 5%, >5y 7.5%)
  with the CEM net-to-gross adjustment
  ``PFE = (0.4 + 0.6 * NGR) * gross add-on``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CounterpartyTrade:
    """One trade in a counterparty netting set."""

    counterparty: str
    product: str
    notional: float
    mark_to_market: float
    tenor_years: float


def add_on_factor(tenor_years: float) -> float:
    """BIS CEM-style FX add-on factor by residual tenor."""
    if tenor_years < 1:
        return 0.01
    if tenor_years <= 5:
        return 0.05
    return 0.075


class CounterpartyExposureTracker:
    """Tracks trades per counterparty netting set (insertion-ordered)."""

    def __init__(self) -> None:
        self._by_counterparty: dict[str, list[CounterpartyTrade]] = {}

    def add_trade(self, trade: CounterpartyTrade) -> "CounterpartyExposureTracker":
        self._by_counterparty.setdefault(trade.counterparty, []).append(trade)
        return self

    def current_exposure(self, counterparty: str) -> float:
        """Net current exposure (MTM netted within the counterparty netting
        set, floored at 0)."""
        net = sum(t.mark_to_market for t in self._trades(counterparty))
        return max(0.0, net)

    def potential_future_exposure(self, counterparty: str) -> float:
        """Potential future exposure with the CEM net-to-gross adjustment:
        ``PFE = (0.4 + 0.6 * NGR) * sum |notional| * addOn``, where
        NGR = net current exposure / gross positive MTM. A well-hedged
        netting set (offsetting MTMs) earns up to a 60% add-on reduction.
        NGR is 1 (no relief) when there is no positive MTM to net against."""
        gross_add_on = 0.0
        gross_positive_mtm = 0.0
        for t in self._trades(counterparty):
            gross_add_on += abs(t.notional) * add_on_factor(t.tenor_years)
            gross_positive_mtm += max(0.0, t.mark_to_market)
        ngr = (self.current_exposure(counterparty) / gross_positive_mtm
               if gross_positive_mtm > 0 else 1.0)
        return (0.4 + 0.6 * ngr) * gross_add_on

    def total_exposure(self, counterparty: str) -> float:
        """Total exposure = current + potential future."""
        return (self.current_exposure(counterparty)
                + self.potential_future_exposure(counterparty))

    def all_exposures(self) -> dict[str, float]:
        """Total exposure per counterparty (insertion order preserved)."""
        return {cp: self.total_exposure(cp) for cp in self._by_counterparty}

    def _trades(self, counterparty: str) -> list[CounterpartyTrade]:
        return self._by_counterparty.get(counterparty, [])
