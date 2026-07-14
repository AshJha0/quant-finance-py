"""Latency-adjusted fair value for rapidly updating order books (port of
Java ``com.quantfinlib.pricing.FairValueEngine``).

Maintains the size-weighted microprice plus a short-window mid drift
estimate, so a consumer that is ``latency_nanos`` behind the market can
project the "true mid" at the moment its order would actually arrive.

Fixed ring of samples; single writer.
"""

from __future__ import annotations

import math


def _highest_one_bit(x: int) -> int:
    """Java Integer.highestOneBit: the highest power of two <= x (0 for 0)."""
    return 0 if x <= 0 else 1 << (x.bit_length() - 1)


class FairValueEngine:
    """Size-weighted microprice + drift projection over a lookback window."""

    def __init__(self, capacity: int = 256, window_nanos: int = 500_000_000) -> None:
        """
        Args:
            capacity: ring capacity (rounded up to a power of two).
            window_nanos: lookback window for the drift estimate.

        The no-argument default is the Java no-arg constructor: a
        256-sample ring over a 500 ms drift window.
        """
        cap = _highest_one_bit(max(2, capacity - 1)) * 2
        self._mids = [0.0] * cap
        self._times = [0] * cap
        self._mask = cap - 1
        self._window_nanos = window_nanos
        self._head = 0                 # number of samples ever written
        self._microprice = math.nan

    def on_quote(self, bid: float, ask: float, bid_size: float, ask_size: float,
                 timestamp_nanos: int) -> None:
        """Feed a top-of-book update."""
        self._microprice = FairValueEngine.microprice(bid, ask, bid_size, ask_size)
        i = self._head & self._mask
        self._mids[i] = (bid + ask) / 2
        self._times[i] = timestamp_nanos
        self._head += 1

    @staticmethod
    def microprice(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
        """Size-weighted microprice: ``I*ask + (1-I)*bid``,
        ``I = bid_size/(bid_size + ask_size)``."""
        total = bid_size + ask_size
        if total <= 0 or math.isnan(bid) or math.isnan(ask):
            return math.nan
        i = bid_size / total
        return i * ask + (1 - i) * bid

    def latest_microprice(self) -> float:
        """Latest microprice (NaN before the first quote)."""
        return self._microprice

    def drift_per_second(self) -> float:
        """Estimated mid drift in price units per second over the lookback window."""
        if self._head < 2:
            return 0.0
        newest = (self._head - 1) & self._mask
        cutoff = self._times[newest] - self._window_nanos
        oldest_seq = max(0, self._head - len(self._mids))
        # Walk back to the oldest sample still inside the window.
        oldest = newest
        for seq in range(self._head - 2, oldest_seq - 1, -1):
            i = seq & self._mask
            if self._times[i] < cutoff:
                break
            oldest = i
        dt = self._times[newest] - self._times[oldest]
        if dt <= 0:
            return 0.0
        return (self._mids[newest] - self._mids[oldest]) / (dt / 1e9)

    def latency_adjusted_fair(self, latency_nanos: int) -> float:
        """Fair price projected ``latency_nanos`` into the future:
        microprice plus drift over the latency horizon."""
        return self._microprice + self.drift_per_second() * latency_nanos / 1e9
