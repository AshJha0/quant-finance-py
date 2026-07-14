"""Bar-driven strategy interface (port of Java ``backtest.TradingStrategy``)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from quantfinlib.backtest.signal import Signal
from quantfinlib.data.bar_series import BarSeries


class TradingStrategy(ABC):
    """A bar-driven trading strategy.

    :meth:`init` is called once by the backtester (or live engine) to
    precompute indicators; :meth:`on_bar` is then invoked for each bar
    in order and must be allocation-free for low-latency execution.
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""

    @abstractmethod
    def init(self, series: BarSeries) -> None:
        """Precomputes indicators over the full series."""

    @abstractmethod
    def on_bar(self, index: int) -> Signal:
        """The signal decided at bar ``index``'s close."""

    def stop_loss_pct(self) -> float:
        """Optional per-trade stop loss as a fraction (0 = disabled)."""
        return 0.0

    def take_profit_pct(self) -> float:
        """Optional per-trade take profit as a fraction (0 = disabled)."""
        return 0.0
