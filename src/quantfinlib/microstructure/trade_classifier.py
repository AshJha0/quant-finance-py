"""Trade aggressor classification (port of Java
``microstructure.TradeClassifier``, Lee-Ready, 1991): the missing glue
for feeds that print trades without saying who initiated.

1. **Quote rule** -- a trade at or above the ask was buyer-initiated
   (someone lifted the offer); at or below the bid, seller-initiated.
   Between the quotes, above the mid leans buy, below leans sell;
2. **Tick test** (exactly at the mid, or no quote) -- an uptick from
   the previous trade price is a buy, a downtick a sell, an equal
   price repeats the last classification (the "zero-tick" rule).

Classification accuracy of this scheme is ~85% on modern equity data
and similar on FX ECN prints -- imperfect by construction (that's the
literature's number, not a defect). One instance per symbol;
cross-asset (raw float prices), single writer.
"""

from __future__ import annotations

import math

#: Classification results.
BUY = 1
SELL = -1
UNKNOWN = 0


class TradeClassifier:
    """Lee-Ready trade classifier; see the module docstring."""

    __slots__ = ("_bid", "_ask", "_last_trade_price", "_last_classification")

    def __init__(self) -> None:
        self._bid = math.nan
        self._ask = math.nan
        self._last_trade_price = math.nan
        self._last_classification = UNKNOWN

    def on_quote(self, bid: float, ask: float) -> None:
        """The current inside quote (NaN sides are treated as
        absent)."""
        self._bid = bid
        self._ask = ask

    def classify(self, trade_price: float) -> int:
        """Classifies a trade print and remembers it for the tick
        test. Returns :data:`BUY`, :data:`SELL` or :data:`UNKNOWN` (no
        quote, no prior trade, or a non-finite price)."""
        if not (trade_price > 0) or trade_price == math.inf:
            return UNKNOWN                  # non-dealable print: classify nothing
        result = self._quote_rule(trade_price)
        if result == UNKNOWN:
            result = self._tick_test(trade_price)
        self._last_trade_price = trade_price
        if result != UNKNOWN:
            self._last_classification = result
        return result

    def is_buy_aggressor(self, trade_price: float) -> bool:
        """Convenience: UNKNOWN maps to the last known side."""
        c = self.classify(trade_price)
        return (self._last_classification if c == UNKNOWN else c) == BUY

    def _quote_rule(self, price: float) -> int:
        has_bid = not math.isnan(self._bid)
        has_ask = not math.isnan(self._ask)
        if has_ask and price >= self._ask:
            return BUY
        if has_bid and price <= self._bid:
            return SELL
        if has_bid and has_ask:
            mid = 0.5 * (self._bid + self._ask)
            if price > mid:
                return BUY
            if price < mid:
                return SELL
        return UNKNOWN                      # exactly at mid, or no usable quote

    def _tick_test(self, price: float) -> int:
        if math.isnan(self._last_trade_price):
            return UNKNOWN
        if price > self._last_trade_price:
            return BUY
        if price < self._last_trade_price:
            return SELL
        return self._last_classification    # zero-tick: repeat the last side
