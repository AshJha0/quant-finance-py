"""Streaming order-flow signals (port of Java
``microstructure.FlowSignals``) for short-horizon execution decisions:
the three imbalances an execution engine reads before crossing a
spread -- cross-asset (equity ticks or raw FX rates; see the two
``on_quote`` entry points) --

* **Order-flow imbalance (OFI)** -- Cont/Kukanov/Stoikov best-level
  formulation: a bid price/size increase or ask decrease is buying
  pressure; the mirror is selling pressure. Exponentially time-decayed
  so the signal is a "recent net flow" with a configurable memory;
* **Queue imbalance** -- ``(bidSize - askSize)/(bidSize + askSize)`` at
  the inside, the classic next-tick-direction predictor;
* **Trade imbalance** -- time-decayed signed aggressor volume over
  time-decayed total volume (+1 = all buying, -1 = all selling).

Single-writer, primitives only -- feed it from the same thread as your
book builder or bus callback.

Deliberate gap semantics (preserved from the Java source): one-sided
quotes (either size <= 0) and non-dealable prices (NaN, zero,
negative, infinite -- placeholder sentinels, e.g. from an NBBO after
the last venue on a side drops) are treated as a signal GAP --
:meth:`queue_imbalance` reads 0 and no OFI contribution is booked, so a
feed artifact must not look like an aggressive sweep. The next
two-sided dealable quote re-seeds the OFI baseline.
"""

from __future__ import annotations

import math


class FlowSignals:
    """Streaming OFI / queue-imbalance / trade-imbalance signals; see
    the module docstring."""

    __slots__ = ("_tau_nanos", "_prev_bid", "_prev_bid_size", "_prev_ask",
                 "_prev_ask_size", "_has_quote", "_bid_size", "_ask_size",
                 "_ofi", "_ofi_time", "_signed_volume", "_total_volume",
                 "_trade_time", "_quote_count", "_trade_count")

    def __init__(self, half_life_nanos: int = 500_000_000) -> None:
        """``half_life_nanos``: decay half-life for OFI and trade
        imbalance; e.g. 500ms. Shorter = twitchier."""
        if half_life_nanos <= 0:
            raise ValueError("halfLifeNanos must be positive")
        self._tau_nanos = half_life_nanos / math.log(2)
        self._prev_bid = 0.0
        self._prev_bid_size = 0
        self._prev_ask = 0.0
        self._prev_ask_size = 0
        self._has_quote = False
        self._bid_size = 0
        self._ask_size = 0
        self._ofi = 0.0
        self._ofi_time = 0
        self._signed_volume = 0.0
        self._total_volume = 0.0
        self._trade_time = 0
        self._quote_count = 0
        self._trade_count = 0

    def on_quote_ticks(self, bid_tick: int, bid_sz: int, ask_tick: int,
                       ask_sz: int, timestamp_nanos: int) -> None:
        """Inside-quote update (ticks + sizes)."""
        self.on_quote(float(bid_tick), bid_sz, float(ask_tick), ask_sz,
                     timestamp_nanos)

    def on_quote(self, bid: float, bid_sz: float, ask: float, ask_sz: float,
                timestamp_nanos: int) -> None:
        """Inside-quote update on raw float prices -- the cross-asset
        entry point (FX rates, or anything not tick-gridded). Same
        semantics as the tick overload: price comparisons drive the
        OFI legs, so any monotonic price representation works."""
        self._quote_count += 1
        # Dealable-price gate (!(x > 0) also catches NaN): a
        # zero/infinite placeholder price with positive sizes must not
        # book phantom OFI legs or latch its sizes into
        # queue_imbalance.
        if (bid_sz <= 0 or ask_sz <= 0 or not (bid > 0) or not (ask > 0)
                or bid == math.inf or ask == math.inf):
            self._bid_size = 0
            self._ask_size = 0
            self._has_quote = False         # gap: don't book flow off a sentinel
            return
        self._bid_size = bid_sz
        self._ask_size = ask_sz
        if self._has_quote:
            e = 0.0
            if bid > self._prev_bid:
                e += bid_sz
            elif bid == self._prev_bid:
                e += bid_sz - self._prev_bid_size
            else:
                e -= self._prev_bid_size
            if ask < self._prev_ask:
                e -= ask_sz
            elif ask == self._prev_ask:
                e -= ask_sz - self._prev_ask_size
            else:
                e += self._prev_ask_size
            self._ofi = self._decayed(self._ofi, self._ofi_time,
                                      timestamp_nanos) + e
            self._ofi_time = timestamp_nanos
        else:
            self._has_quote = True
            self._ofi_time = timestamp_nanos
        self._prev_bid = bid
        self._prev_bid_size = bid_sz
        self._prev_ask = ask
        self._prev_ask_size = ask_sz

    def on_trade(self, buy_aggressor: bool, quantity: float,
                timestamp_nanos: int) -> None:
        """Trade print with aggressor side: ``buy_aggressor`` true when
        the buyer crossed the spread (trade at/above ask under
        Lee-Ready)."""
        self._trade_count += 1
        decay = self._decay_factor(self._trade_time, timestamp_nanos)
        self._signed_volume = (self._signed_volume * decay
                               + (quantity if buy_aggressor else -quantity))
        self._total_volume = self._total_volume * decay + quantity
        self._trade_time = timestamp_nanos

    def ofi(self, now_nanos: "int | None" = None) -> float:
        """Time-decayed net order-flow imbalance in shares (+ =
        buying pressure). With ``now_nanos``, the decay-adjusted OFI
        as of that time without adding an event."""
        if now_nanos is None:
            return self._ofi
        return self._decayed(self._ofi, self._ofi_time, now_nanos)

    def queue_imbalance(self) -> float:
        """Inside-queue imbalance in [-1, 1]; 0 when either side is
        empty/unset."""
        if self._bid_size <= 0 or self._ask_size <= 0:
            return 0.0                      # one-sided book: no imbalance signal
        return ((self._bid_size - self._ask_size)
                / (self._bid_size + self._ask_size))

    def trade_imbalance(self) -> float:
        """Signed/total decayed aggressor volume in [-1, 1]; 0 before
        any trade."""
        return (0.0 if self._total_volume <= 0
                else self._signed_volume / self._total_volume)

    def quote_count(self) -> int:
        return self._quote_count

    def trade_count(self) -> int:
        return self._trade_count

    def _decayed(self, value: float, last_time: int, now: int) -> float:
        return value * self._decay_factor(last_time, now)

    def _decay_factor(self, last_time: int, now: int) -> float:
        dt = now - last_time
        return 1.0 if dt <= 0 else math.exp(-dt / self._tau_nanos)
