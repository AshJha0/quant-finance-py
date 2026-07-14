"""Trading signal (port of Java ``backtest.Signal``)."""

from __future__ import annotations

from enum import Enum


class Signal(Enum):
    """Trading signal emitted by a strategy for a single bar."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
