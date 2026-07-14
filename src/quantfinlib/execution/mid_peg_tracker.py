"""Mid-rate pegging model (port of Java ``execution.MidPegTracker``).

Tracks the target price of a mid-pegged order with an offset and
optional limit cap, and decides when the peg has drifted far enough to
justify a reprice (each reprice costs queue priority and a message, so
small moves are ignored).
"""

from __future__ import annotations

import math

from quantfinlib.microstructure.execution import Side


class MidPegTracker:
    """Mid-peg reprice decision; see the module docstring."""

    def __init__(self, side: Side, offset: float, limit_price: float,
                reprice_threshold: float) -> None:
        """
        Args:
            side: BUY or SELL.
            offset: signed offset from the mid (negative = more passive
                for a buy).
            limit_price: hard cap: never price through this (NaN to disable).
            reprice_threshold: minimum absolute peg move before repricing.
        """
        if not math.isfinite(offset):
            raise ValueError(f"offset must be finite, got {offset}")
        # NaN disables the limit by contract; anything else must be finite.
        if limit_price in (math.inf, -math.inf):
            raise ValueError("limitPrice must be finite or NaN to disable")
        # A negative or NaN threshold makes every quote "beyond
        # threshold": the peg reprices constantly and burns the queue
        # priority it exists to protect.
        if not (reprice_threshold >= 0) or reprice_threshold == math.inf:
            raise ValueError(
                f"repriceThreshold must be >= 0 and finite, got {reprice_threshold}")
        self._side = side
        self._offset = offset
        self._limit_price = limit_price
        self._reprice_threshold = reprice_threshold
        self._current_price = math.nan

    def on_quote(self, bid: float, ask: float) -> float:
        """Feed a top-of-book update. Returns the new order price when
        a reprice is warranted, or NaN when the current price should be
        left alone."""
        target = (bid + ask) / 2 + self._offset
        if not math.isnan(self._limit_price):
            target = (min(target, self._limit_price) if self._side == Side.BUY
                     else max(target, self._limit_price))
        if (math.isnan(self._current_price)
                or abs(target - self._current_price) >= self._reprice_threshold):
            self._current_price = target
            return target
        return math.nan

    def current_price(self) -> float:
        """Current working price (NaN before the first quote)."""
        return self._current_price
