"""Completed round-trip trade record (port of Java ``backtest.Trade``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round-trip trade.

    ``pnl`` is net of commissions; ``return_pct`` is relative to capital
    committed at entry.

    Attributes:
        symbol: Instrument identifier.
        entry_index: Bar index of the entry fill.
        exit_index: Bar index of the exit fill.
        entry_time: Epoch timestamp of the entry.
        exit_time: Epoch timestamp of the exit.
        entry_price: Fill price at entry.
        exit_price: Fill price at exit.
        quantity: Position size.
        pnl: Realized profit and loss, net of commissions.
        return_pct: Return on capital committed at entry.
        exit_reason: One of the ``REASON_*`` constants.
    """

    symbol: str
    entry_index: int
    exit_index: int
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    exit_reason: str

    REASON_SIGNAL = "SIGNAL"
    REASON_STOP_LOSS = "STOP_LOSS"
    REASON_TAKE_PROFIT = "TAKE_PROFIT"
    REASON_END_OF_DATA = "END_OF_DATA"

    @property
    def is_win(self) -> bool:
        """True when the trade closed with strictly positive P&L."""
        return self.pnl > 0

    @property
    def bars_held(self) -> int:
        """Holding period in bars: ``exit_index - entry_index``."""
        return self.exit_index - self.entry_index
