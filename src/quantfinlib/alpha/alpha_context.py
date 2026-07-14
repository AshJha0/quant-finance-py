"""Alpha research panel (port of Java ``alpha.AlphaContext`` plus the
``screener.Fundamentals`` / ``data.PointInTimeUniverse`` fragments it
needs).

Everything in :mod:`quantfinlib.alpha` works cross-sectionally: at a bar
index, a factor scores every symbol, and downstream steps (evaluation,
construction, backtest) consume those scores as arrays aligned with
:meth:`AlphaContext.symbols`. Freezing the symbol order once here is
what makes a plain float array the interchange type for the whole
pipeline — no per-step dict lookups, no ordering ambiguity.

Survivorship: alpha research is the stage survivorship bias flatters
most — the delisted losers a short book would have held are the exact
names a today's-constituents panel lacks. Attach a point-in-time
universe via :meth:`AlphaContext.with_universe` and every built-in
factor scores non-members/dead names as NaN at each bar
(:meth:`AlphaContext.is_active`), so ICs, validation and constructed
weights only ever see the point-in-time cross-section. Without a
universe the panel is survivorship-blind — fine for methodology work,
dishonest for performance claims.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from quantfinlib.data.bar_series import BarSeries


@dataclass(frozen=True)
class Fundamentals:
    """Static fundamentals snapshot for one symbol (the subset of the
    Java ``screener.Fundamentals`` the value/quality factors read).
    NaN fields mean "unknown"."""

    pe_ratio: float = math.nan
    pb_ratio: float = math.nan
    roe: float = math.nan
    debt_to_equity: float = math.nan


class PointInTimeUniverse(ABC):
    """Point-in-time index membership: ``is_member(symbol, timestamp)``
    must answer with information available AT that timestamp (port of
    the Java ``data.PointInTimeUniverse`` contract)."""

    @abstractmethod
    def is_member(self, symbol: str, timestamp: int) -> bool:
        """Whether ``symbol`` was in the universe at ``timestamp``."""


class AlphaContext:
    """The research dataset an alpha factor operates on: an
    index-aligned panel of price series over a fixed symbol order, with
    optional fundamentals.

    Series must be index-aligned (same length, same bar times); the
    constructor enforces equal length, the timestamp discipline is the
    caller's (documented) responsibility. Fundamentals are an optional
    static snapshot: factors that need them return NaN for symbols
    without entries.
    """

    __slots__ = ("_symbols", "_series", "_fundamentals", "_universe",
                 "_bars")

    def __init__(self, symbols: Tuple[str, ...], series: Tuple[BarSeries, ...],
                 fundamentals: Tuple[Optional[Fundamentals], ...],
                 universe: Optional[PointInTimeUniverse], bars: int) -> None:
        self._symbols = symbols
        self._series = series
        self._fundamentals = fundamentals
        self._universe = universe
        self._bars = bars

    @staticmethod
    def of(data: Dict[str, BarSeries],
           fundamentals: Optional[Dict[str, Fundamentals]] = None
           ) -> "AlphaContext":
        """Panel from a symbol -> series mapping, with an optional
        fundamentals snapshot for value/quality factors.

        Symbols are SORTED for a deterministic panel axis regardless of
        the caller's dict insertion order — results must not depend on
        hash/insertion order.
        """
        if not data:
            raise ValueError("no series supplied")
        fundamentals = fundamentals or {}
        symbols = sorted(data.keys())
        n = data[symbols[0]].size()
        series: List[BarSeries] = []
        funda: List[Optional[Fundamentals]] = []
        for sym in symbols:
            s = data[sym]
            if s.size() != n:
                raise ValueError(
                    "series must be index-aligned: "
                    f"{sym} has {s.size()} bars, expected {n}")
            series.append(s)
            funda.append(fundamentals.get(sym))
        return AlphaContext(tuple(symbols), tuple(series), tuple(funda),
                            None, n)

    def with_universe(self, universe: PointInTimeUniverse) -> "AlphaContext":
        """The same panel with a point-in-time universe attached:
        built-in factors then score non-members as NaN per bar (see the
        module doc). Universe timestamps must be in the same units as
        the bar timestamps."""
        return AlphaContext(self._symbols, self._series, self._fundamentals,
                            universe, self._bars)

    def is_active(self, i: int, bar_index: int) -> bool:
        """Whether symbol ``i`` is in the tradeable cross-section at
        ``bar_index``: always True without a universe, otherwise
        point-in-time membership (dead and dropped names excluded)."""
        return (self._universe is None
                or self._universe.is_member(self._symbols[i],
                                            self.timestamp(bar_index)))

    def symbols(self) -> Tuple[str, ...]:
        """The frozen symbol order every score/weight array aligns with."""
        return self._symbols

    def symbol_count(self) -> int:
        return len(self._symbols)

    def bars(self) -> int:
        """Panel length in bars (every series has exactly this many)."""
        return self._bars

    def series(self, i: int) -> BarSeries:
        """Price series for symbol index ``i`` (the panel axis)."""
        return self._series[i]

    def fundamentals(self, i: int) -> Optional[Fundamentals]:
        """Fundamentals for symbol index ``i``, or None when unknown."""
        return self._fundamentals[i]

    def timestamp(self, index: int) -> int:
        """Bar timestamp at ``index`` (taken from the first series)."""
        return self._series[0].timestamp(index)

    def return_over(self, i: int, from_index: int, to_index: int) -> float:
        """Simple return of symbol ``i`` over ``(from_index, to_index]``
        — the forward-return building block evaluation and backtesting
        share."""
        s = self._series[i]
        return s.close(to_index) / s.close(from_index) - 1.0
