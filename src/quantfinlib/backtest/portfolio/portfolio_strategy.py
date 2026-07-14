"""Multi-asset weight-based strategy interface (port of Java
``backtest.portfolio.PortfolioStrategy``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from quantfinlib.data.bar_series import BarSeries


class PortfolioStrategy(ABC):
    """A multi-asset, weight-based strategy for the portfolio
    backtester. All series must be index-aligned (same length, same bar
    times). Weights are fractions of current equity: positive = long,
    negative = short, missing symbol = flat; |weights| may sum above 1
    for leverage."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""

    @abstractmethod
    def init(self, data: Dict[str, BarSeries]) -> None:
        """Called once with the aligned per-symbol series."""

    @abstractmethod
    def target_weights(self, index: int) -> Dict[str, float]:
        """Target weights by symbol as of bar ``index`` (decided at that
        bar's close)."""
