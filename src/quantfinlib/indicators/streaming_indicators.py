"""Port of Java ``com.quantfinlib.indicators.StreamingIndicators``.

Incremental O(1)-per-tick indicators for live strategies: update state
with each new value instead of recomputing arrays. Numerically
identical to the batch ``Indicators`` implementations (same seeding and
smoothing), so a strategy backtested on batch arrays behaves the same
when run live on the streaming versions.

Instances are single-threaded by design — one per strategy/consumer
thread — matching the single-consumer dispatch model of the Java bus.
"""

from __future__ import annotations

import math


class StreamingIndicators:
    """Namespace for the streaming indicator classes (mirrors Java)."""

    class Sma:
        """Simple moving average over a fixed window; NaN until the
        window fills."""

        def __init__(self, period: int) -> None:
            self._period = period
            self._window = [0.0] * period
            self._sum = 0.0
            self._count = 0
            self._idx = 0
            self._value = math.nan

        def update(self, v: float) -> float:
            if self._count >= self._period:
                self._sum -= self._window[self._idx]
            self._window[self._idx] = v
            self._sum += v
            self._idx = 0 if self._idx + 1 == self._period else self._idx + 1
            self._count += 1
            self._value = (self._sum / self._period
                           if self._count >= self._period else math.nan)
            return self._value

        def value(self) -> float:
            return self._value

    class Ema:
        """Exponential moving average seeded with the SMA of the first
        ``period`` values."""

        def __init__(self, period: int) -> None:
            self._period = period
            self._k = 2.0 / (period + 1)
            self._seed_sum = 0.0
            self._count = 0
            self._value = math.nan

        def update(self, v: float) -> float:
            self._count += 1
            if self._count < self._period:
                self._seed_sum += v
                return math.nan
            if self._count == self._period:
                self._value = (self._seed_sum + v) / self._period
            else:
                self._value += (v - self._value) * self._k
            return self._value

        def value(self) -> float:
            return self._value

    class Rsi:
        """Wilder RSI; NaN until ``period`` price changes have been
        observed."""

        def __init__(self, period: int) -> None:
            self._period = period
            self._prev = math.nan
            self._changes = 0
            self._gain_sum = 0.0
            self._loss_sum = 0.0
            self._avg_gain = 0.0
            self._avg_loss = 0.0
            self._value = math.nan

        def update(self, v: float) -> float:
            if math.isnan(self._prev):
                self._prev = v
                return math.nan
            d = v - self._prev
            self._prev = v
            self._changes += 1
            period = self._period
            if self._changes < period:
                self._gain_sum += max(d, 0.0)
                self._loss_sum += max(-d, 0.0)
                return math.nan
            if self._changes == period:
                self._avg_gain = (self._gain_sum + max(d, 0.0)) / period
                self._avg_loss = (self._loss_sum + max(-d, 0.0)) / period
            else:
                self._avg_gain = (self._avg_gain * (period - 1) + max(d, 0.0)) / period
                self._avg_loss = (self._avg_loss * (period - 1) + max(-d, 0.0)) / period
            self._value = _to_rsi(self._avg_gain, self._avg_loss)
            return self._value

        def value(self) -> float:
            return self._value

    class Macd:
        """MACD line, signal and histogram, matching the batch seeding
        exactly (the signal EMA only ever sees valid MACD line values —
        no NaN pre-history bias)."""

        def __init__(self, fast_period: int, slow_period: int,
                     signal_period: int) -> None:
            self._fast = StreamingIndicators.Ema(fast_period)
            self._slow = StreamingIndicators.Ema(slow_period)
            self._signal_ema = StreamingIndicators.Ema(signal_period)
            self._line = math.nan
            self._signal = math.nan
            self._histogram = math.nan

        def update(self, v: float) -> float:
            """Returns the MACD line (NaN during warm-up)."""
            f = self._fast.update(v)
            s = self._slow.update(v)
            if math.isnan(s):
                return math.nan
            self._line = f - s
            self._signal = self._signal_ema.update(self._line)
            self._histogram = (math.nan if math.isnan(self._signal)
                               else self._line - self._signal)
            return self._line

        def line(self) -> float:
            return self._line

        def signal(self) -> float:
            return self._signal

        def histogram(self) -> float:
            return self._histogram

    class Vwap:
        """Cumulative volume-weighted average price."""

        def __init__(self) -> None:
            self._cum_pv = 0.0
            self._cum_vol = 0.0
            self._value = math.nan

        def update(self, price: float, volume: float) -> float:
            self._cum_pv += price * volume
            self._cum_vol += volume
            self._value = price if self._cum_vol == 0 else self._cum_pv / self._cum_vol
            return self._value

        def value(self) -> float:
            return self._value


def _to_rsi(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)
