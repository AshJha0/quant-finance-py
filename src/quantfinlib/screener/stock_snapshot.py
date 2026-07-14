"""One screening candidate (port of Java ``screener.StockSnapshot``)."""

from __future__ import annotations

from dataclasses import dataclass

from quantfinlib.data.bar_series import BarSeries
from quantfinlib.screener.fundamentals import Fundamentals


@dataclass(frozen=True, slots=True)
class StockSnapshot:
    """One screening candidate: symbol, price history, and fundamentals."""

    symbol: str
    series: BarSeries
    fundamentals: Fundamentals

    def last_close(self) -> float:
        return self.series.last_close()
