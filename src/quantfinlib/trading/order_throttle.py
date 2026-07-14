"""Order-rate throttle (port of Java ``trading.OrderThrottle``).

A nanosecond token bucket for exchange message-rate limits (every real
venue enforces one; exceeding it earns disconnects or fines, so the
gateway must self-limit). Sustained rate ``rate_per_sec`` with bursts
up to ``burst`` -- a quiet spell banks up to one burst of headroom,
then the bucket refills continuously.

Single-writer (call from the order-entry thread); no clock reads of
its own -- the caller passes the current nanosecond timestamp so tests
are deterministic and the hot path controls its own syscalls.
"""

from __future__ import annotations

import math
from typing import Optional


class OrderThrottle:
    """Nanosecond token-bucket order-rate throttle; see the module
    docstring."""

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        """
        Args:
            rate_per_sec: sustained messages per second (> 0).
            burst: bucket depth: messages allowed back-to-back (>= 1).
        """
        if rate_per_sec <= 0 or burst < 1:
            raise ValueError("need ratePerSec > 0, burst >= 1")
        self._tokens_per_nano = rate_per_sec / 1e9
        self._burst = float(burst)
        self._tokens = float(burst)         # start full: allow an opening burst
        self._last_nanos: Optional[int] = None
        self._acquired = 0
        self._throttled = 0

    def try_acquire(self, now_nanos: int) -> bool:
        """Attempts to take one send permit at ``now_nanos``. False =
        do not send (queue or drop per your policy -- this class only
        counts)."""
        self._refill(now_nanos)
        # The 1e-9 slack absorbs floating-point drift from split
        # refills (a+b in two steps can sum a hair under one step): it
        # grants at most sub-nanosecond-early permits, never extra ones.
        if self._tokens >= 1 - 1e-9:
            self._tokens = max(0.0, self._tokens - 1)
            self._acquired += 1
            return True
        self._throttled += 1
        return False

    def nanos_until_available(self, now_nanos: int) -> int:
        """Nanoseconds until a permit would be available (0 when one
        already is) -- for pacing loops that would rather sleep than
        spin-fail."""
        self._refill(now_nanos)
        if self._tokens >= 1:
            return 0
        return math.ceil((1 - self._tokens) / self._tokens_per_nano)

    def _refill(self, now_nanos: int) -> None:
        if self._last_nanos is None:
            self._last_nanos = now_nanos
            return
        dt = now_nanos - self._last_nanos
        if dt > 0:
            self._tokens = min(self._burst, self._tokens + dt * self._tokens_per_nano)
            self._last_nanos = now_nanos

    def acquired_count(self) -> int:
        """Permits granted so far."""
        return self._acquired

    def throttled_count(self) -> int:
        """Denials so far -- a persistent nonzero rate means the
        strategy outruns the venue limit."""
        return self._throttled
