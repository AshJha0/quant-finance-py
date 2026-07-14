"""Immutable OHLCV time series (port of Java ``core.BarSeries``).

Structure-of-arrays layout on NumPy arrays: the Java class keeps
primitive arrays for cache-friendly, boxing-free access, and the
Python port keeps the same shape with read-only ``np.ndarray`` views.
Array accessors (:attr:`opens`, :attr:`closes`, ...) return the
internal arrays without copying; they are flagged non-writeable.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from quantfinlib.data.bar import Bar


class BarSeries:
    """Immutable OHLCV series for a single symbol.

    Construct via :meth:`builder`, :meth:`of` or :meth:`from_bars`.
    """

    __slots__ = ("_symbol", "_timestamps", "_open", "_high", "_low",
                 "_close", "_volume", "_size")

    def __init__(self, symbol: str, timestamps, opens, highs, lows,
                 closes, volumes) -> None:
        ts = np.asarray(timestamps, dtype=np.int64)
        o = np.asarray(opens, dtype=float)
        h = np.asarray(highs, dtype=float)
        lo = np.asarray(lows, dtype=float)
        c = np.asarray(closes, dtype=float)
        v = np.asarray(volumes, dtype=float)
        n = ts.shape[0]
        if n == 0:
            raise RuntimeError(f"empty series: {symbol}")
        for a in (o, h, lo, c, v):
            if a.shape[0] != n:
                raise ValueError("all arrays must share one length")
        for a in (ts, o, h, lo, c, v):
            a.setflags(write=False)
        self._symbol = symbol
        self._timestamps = ts
        self._open = o
        self._high = h
        self._low = lo
        self._close = c
        self._volume = v
        self._size = n

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @staticmethod
    def builder(symbol: str) -> "BarSeries.Builder":
        return BarSeries.Builder(symbol)

    @staticmethod
    def of(symbol: str, closes) -> "BarSeries":
        """Builds a series from close prices only (open=high=low=close)."""
        c = np.asarray(closes, dtype=float)
        n = c.shape[0]
        ts = np.arange(n, dtype=np.int64) * 86_400_000
        return BarSeries(symbol, ts, c, c, c, c, np.full(n, 1_000_000.0))

    @staticmethod
    def from_bars(symbol: str, bars: Iterable[Bar]) -> "BarSeries":
        b = BarSeries.builder(symbol)
        for bar in bars:
            b.add(bar.timestamp, bar.open, bar.high, bar.low,
                  bar.close, bar.volume)
        return b.build()

    # ------------------------------------------------------------------
    # Scalar accessors (Java-parity names)
    # ------------------------------------------------------------------

    def symbol(self) -> str:
        return self._symbol

    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def timestamp(self, i: int) -> int:
        return int(self._timestamps[i])

    def open(self, i: int) -> float:
        return float(self._open[i])

    def high(self, i: int) -> float:
        return float(self._high[i])

    def low(self, i: int) -> float:
        return float(self._low[i])

    def close(self, i: int) -> float:
        return float(self._close[i])

    def volume(self, i: int) -> float:
        return float(self._volume[i])

    def last_close(self) -> float:
        return float(self._close[self._size - 1])

    def bar(self, i: int) -> Bar:
        return Bar(self.timestamp(i), self.open(i), self.high(i),
                   self.low(i), self.close(i), self.volume(i))

    # ------------------------------------------------------------------
    # Zero-copy array accessors (read-only views)
    # ------------------------------------------------------------------

    def timestamps(self) -> np.ndarray:
        return self._timestamps

    def opens(self) -> np.ndarray:
        return self._open

    def highs(self) -> np.ndarray:
        return self._high

    def lows(self) -> np.ndarray:
        return self._low

    def closes(self) -> np.ndarray:
        return self._close

    def volumes(self) -> np.ndarray:
        return self._volume

    # ------------------------------------------------------------------

    def slice(self, from_: int, to_exclusive: int) -> "BarSeries":
        """Copy of the bar range [from_, to_exclusive) as a new series."""
        if from_ < 0 or to_exclusive > self._size or from_ >= to_exclusive:
            raise ValueError(
                f"bad slice [{from_},{to_exclusive}) of {self._size}")
        s = slice(from_, to_exclusive)
        return BarSeries(self._symbol, self._timestamps[s].copy(),
                         self._open[s].copy(), self._high[s].copy(),
                         self._low[s].copy(), self._close[s].copy(),
                         self._volume[s].copy())

    def returns(self) -> np.ndarray:
        """Simple (arithmetic) returns; length = size - 1."""
        c = self._close
        return c[1:] / c[:-1] - 1.0

    def log_returns(self) -> np.ndarray:
        """Log returns; length = size - 1."""
        c = self._close
        return np.log(c[1:] / c[:-1])

    # ------------------------------------------------------------------

    class Builder:
        """Accumulates bars; :meth:`build` freezes them into a series."""

        def __init__(self, symbol: str) -> None:
            self._symbol = symbol
            self._ts: List[int] = []
            self._o: List[float] = []
            self._h: List[float] = []
            self._l: List[float] = []
            self._c: List[float] = []
            self._v: List[float] = []

        def add(self, timestamp: int, open_: float, high: float, low: float,
                close: float, volume: float) -> "BarSeries.Builder":
            self._ts.append(timestamp)
            self._o.append(open_)
            self._h.append(high)
            self._l.append(low)
            self._c.append(close)
            self._v.append(volume)
            return self

        def add_bar(self, bar: Bar) -> "BarSeries.Builder":
            return self.add(bar.timestamp, bar.open, bar.high, bar.low,
                            bar.close, bar.volume)

        def build(self) -> "BarSeries":
            if not self._ts:
                raise RuntimeError(f"empty series: {self._symbol}")
            return BarSeries(self._symbol, self._ts, self._o, self._h,
                             self._l, self._c, self._v)
