"""Transaction Cost Analysis (port of Java
``microstructure.TransactionCostAnalyzer``).

Benchmarks matched trades against the arrival mid, the interval market
VWAP, and the prevailing mid at each fill (effective spread). All costs
are signed so positive = cost to the trader, for both buys and sells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from quantfinlib.microstructure.execution import Execution


class TransactionCostAnalyzer:
    """Static TCA; see the module docstring."""

    @dataclass(frozen=True)
    class TcaReport:
        """One parent order's cost decomposition (costs in bps,
        positive = cost to the trader)."""

        total_quantity: int
        avg_execution_price: float
        arrival_mid: float
        market_vwap: float
        implementation_shortfall_bps: float
        slippage_vs_vwap_bps: float
        avg_effective_spread_bps: float

    @staticmethod
    def analyze(fills: Sequence[Execution], arrival_mid: float,
                market_vwap: float, mid_at_fill
                ) -> "TransactionCostAnalyzer.TcaReport":
        """Analyzes the child fills of one parent order (all same side).

        Args:
            fills: The child fills of one parent order.
            arrival_mid: Market mid when the parent order was created.
            market_vwap: Market VWAP over the execution interval (or a
                synthetic forward benchmark for FX forwards/swaps).
            mid_at_fill: Prevailing mid at each fill, aligned with
                ``fills``.
        """
        fills = list(fills)
        mids = list(mid_at_fill)
        if not fills:
            raise ValueError("no fills")
        if len(mids) != len(fills):
            raise ValueError("mid_at_fill must align with fills")
        # Benchmark prices divide every headline number: a single stale
        # zero mid would put an inf in the report with no warning.
        if (not (arrival_mid > 0) or arrival_mid == math.inf
                or not (market_vwap > 0) or market_vwap == math.inf):
            raise ValueError("arrival_mid and market_vwap must be positive"
                             f" and finite: {arrival_mid}, {market_vwap}")
        for i, m in enumerate(mids):
            if not (m > 0) or m == math.inf:
                raise ValueError(
                    f"mid_at_fill[{i}] must be positive and finite: {m}")
        side = fills[0].side
        sign = side.sign()

        qty = 0
        notional = 0.0
        eff_spread_weighted = 0.0
        for f, m in zip(fills, mids):
            qty += f.quantity
            notional += f.notional()
            eff_spread_weighted += (2.0 * sign * (f.price - m) / m * 1e4
                                    * f.quantity)
        vwap_exec = notional / qty

        return TransactionCostAnalyzer.TcaReport(
            qty,
            vwap_exec,
            arrival_mid,
            market_vwap,
            sign * (vwap_exec - arrival_mid) / arrival_mid * 1e4,
            sign * (vwap_exec - market_vwap) / market_vwap * 1e4,
            eff_spread_weighted / qty)
