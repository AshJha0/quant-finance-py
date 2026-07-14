"""Settlement (Herstatt) risk: the exposure created when you pay away one
currency before receiving the other leg.

Port of Java ``com.quantfinlib.risk.SettlementRiskAnalyzer``. Named for
Bankhaus Herstatt (1974), closed after receiving DEM but before paying
out USD.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettlementLeg:
    """One settlement instruction pair: we pay one leg and receive the other."""

    counterparty: str
    pay_currency: str
    pay_amount: float
    pay_time_millis: int
    receive_currency: str
    receive_amount: float
    receive_time_millis: int

    def has_herstatt_window(self) -> bool:
        """True when we pay before we receive — the Herstatt window."""
        return self.pay_time_millis < self.receive_time_millis


def herstatt_exposure(legs: list[SettlementLeg]) -> dict[str, float]:
    """Total at-risk receive amounts per counterparty: sum of legs where
    payment goes out before the countervalue arrives."""
    out: dict[str, float] = {}
    for leg in legs:
        if leg.has_herstatt_window():
            out[leg.counterparty] = out.get(leg.counterparty, 0.0) + leg.receive_amount
    return out


def peak_exposure(legs: list[SettlementLeg], counterparty: str) -> float:
    """Peak intraday settlement exposure to one counterparty: the maximum
    total receive-amount outstanding (paid but not yet received) at any
    point in time."""
    events: list[tuple[int, float]] = []
    for leg in legs:
        if leg.counterparty != counterparty or not leg.has_herstatt_window():
            continue
        events.append((leg.pay_time_millis, leg.receive_amount))
        events.append((leg.receive_time_millis, -leg.receive_amount))
    # At equal timestamps apply PAYMENTS (positive deltas) before
    # receipts: for a worst-case exposure metric the conservative reading
    # of a simultaneous pay/receive is that your money left first.
    # Receipts-first would quietly understate the peak — the wrong
    # direction for a number named after Herstatt.
    events.sort(key=lambda ev: (ev[0], -ev[1]))
    outstanding = 0.0
    peak = 0.0
    for _, delta in events:
        outstanding += delta
        peak = max(peak, outstanding)
    return peak
