"""OHLCV market-data bar (port of Java com.quantfinlib.core.Bar)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Bar:
    """Immutable OHLCV bar. Timestamp is epoch milliseconds.

    Construction validates the one relation a bar cannot violate:
    high >= low (a NaN high/low pair slips through, exactly as in the
    Java record — the loaders reject NaN rows upstream). Open and close
    are NOT forced inside [low, high]: some venues publish auction
    prints outside the intrabar range, and the Java reference
    deliberately accepts them.

    Attributes:
        timestamp: Epoch milliseconds of the bar close.
        open: First traded price of the interval.
        high: Highest traded price.
        low: Lowest traded price.
        close: Last traded price.
        volume: Total quantity traded.
    """

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")

    def typical_price(self) -> float:
        """(high + low + close) / 3 — the pivot-style price proxy."""
        return (self.high + self.low + self.close) / 3.0

    def range(self) -> float:
        """High minus low: the intrabar travel used by range-vol estimators."""
        return self.high - self.low

    def is_bullish(self) -> bool:
        """True when the bar closed strictly above its open."""
        return self.close > self.open
