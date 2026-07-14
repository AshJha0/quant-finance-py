"""MiFID II-style best execution analytics (port of Java
``regulatory.BestExecutionAnalyzer``), RTS 27/28 spirit: slippage
versus arrival mid, latency-to-fill distribution, fraction executed at
or better than arrival, and per-venue slippage breakdown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from quantfinlib.microstructure.execution import Side


@dataclass(frozen=True, slots=True)
class OrderOutcome:
    """One parent order outcome. Unfilled orders: ``filled=False``,
    price/latency ignored."""

    order_id: str
    venue: str
    side: Side
    quantity: int
    arrival_mid: float
    execution_price: float
    latency_to_fill_nanos: int
    filled: bool


@dataclass(frozen=True, slots=True)
class BestExecutionReport:
    total_orders: int
    fill_rate: float
    avg_slippage_bps: float
    median_latency_to_fill_millis: float
    at_or_better_than_arrival_pct: float
    avg_slippage_bps_by_venue: Dict[str, float]


class BestExecutionAnalyzer:
    def __init__(self) -> None:
        self._outcomes: List[OrderOutcome] = []

    def add(self, outcome: OrderOutcome) -> "BestExecutionAnalyzer":
        self._outcomes.append(outcome)
        return self

    def report(self) -> BestExecutionReport:
        if not self._outcomes:
            raise RuntimeError("no order outcomes recorded")
        filled = 0
        slippage_sum = 0.0
        at_or_better = 0
        latencies: List[float] = []
        venue_agg: Dict[str, List[float]] = {}  # venue -> [sum, count]

        for o in self._outcomes:
            if not o.filled:
                continue
            filled += 1
            slip = o.side.sign() * (o.execution_price - o.arrival_mid) / o.arrival_mid * 1e4
            slippage_sum += slip
            if slip <= 0:
                at_or_better += 1
            latencies.append(o.latency_to_fill_nanos / 1e6)
            agg = venue_agg.setdefault(o.venue, [0.0, 0.0])
            agg[0] += slip
            agg[1] += 1

        by_venue = {venue: agg[0] / agg[1] for venue, agg in venue_agg.items()}

        return BestExecutionReport(
            total_orders=len(self._outcomes),
            fill_rate=filled / len(self._outcomes),
            avg_slippage_bps=math.nan if filled == 0 else slippage_sum / filled,
            median_latency_to_fill_millis=_median(latencies),
            at_or_better_than_arrival_pct=math.nan if filled == 0 else at_or_better / filled,
            avg_slippage_bps_by_venue=by_venue,
        )


def _median(values: List[float]) -> float:
    if not values:
        return math.nan
    a = np.sort(np.asarray(values, dtype=float))
    n = a.shape[0]
    if n % 2 == 1:
        return float(a[n // 2])
    return (float(a[n // 2 - 1]) + float(a[n // 2])) / 2
