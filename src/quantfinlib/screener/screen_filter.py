"""Composable screening predicate (port of Java ``screener.ScreenFilter``).

Java's ``ScreenFilter`` is a ``@FunctionalInterface`` with default
``and``/``or``/``negate`` combinators. Python has no functional
interfaces and ``and``/``or``/``not`` are keywords, so this port wraps
a plain predicate callable in a small class exposing ``and_``/``or_``/
``negate`` instead.
"""

from __future__ import annotations

from typing import Callable

from quantfinlib.screener.stock_snapshot import StockSnapshot

_Predicate = Callable[[StockSnapshot], bool]


class ScreenFilter:
    """Composable screening predicate over a :class:`StockSnapshot`."""

    def __init__(self, predicate: _Predicate) -> None:
        self._predicate = predicate

    def matches(self, stock: StockSnapshot) -> bool:
        return bool(self._predicate(stock))

    def __call__(self, stock: StockSnapshot) -> bool:
        return self.matches(stock)

    def and_(self, other: "ScreenFilter") -> "ScreenFilter":
        return ScreenFilter(lambda s: self.matches(s) and other.matches(s))

    def or_(self, other: "ScreenFilter") -> "ScreenFilter":
        return ScreenFilter(lambda s: self.matches(s) or other.matches(s))

    def negate(self) -> "ScreenFilter":
        return ScreenFilter(lambda s: not self.matches(s))
