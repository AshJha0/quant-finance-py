"""Parent order record (port of Java ``backtest.ParentOrder``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from quantfinlib.microstructure.execution import Execution, Side


@dataclass(frozen=True)
class ParentOrder:
    """One parent order worked by the execution-aware backtester: the
    signal that created it, the arrival price (close at signal time —
    the TCA benchmark), and the child fills with the bar index each
    filled on. ``reason`` is ``"ENTRY"`` for entries, or the exit reason
    (SIGNAL / STOP_LOSS / TAKE_PROFIT / END_OF_DATA).

    Attributes:
        side: BUY (entry) or SELL (exit).
        signal_index: Bar index where the signal fired.
        arrival_price: Close at the signal bar — the TCA benchmark.
        reason: ``REASON_ENTRY`` or an exit reason.
        fills: The child fills, in execution order.
        fill_bar_indices: Bar index of each fill, aligned with ``fills``.
    """

    side: Side
    signal_index: int
    arrival_price: float
    reason: str
    fills: Tuple[Execution, ...]
    fill_bar_indices: Tuple[int, ...]

    REASON_ENTRY = "ENTRY"

    def filled_qty(self) -> int:
        return sum(f.quantity for f in self.fills)

    def avg_fill_price(self) -> float:
        qty = 0
        notional = 0.0
        for f in self.fills:
            qty += f.quantity
            notional += f.notional()
        return math.nan if qty == 0 else notional / qty

    def fill_duration_bars(self) -> int:
        """Bars from first to last fill (0 for unfilled or single-bar
        parents)."""
        if not self.fills:
            return 0
        return self.fill_bar_indices[-1] - self.fill_bar_indices[0]
