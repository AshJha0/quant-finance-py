"""Order side and matched-trade record (ports of Java ``orderbook.Side``
and ``microstructure.Execution``)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"

    def sign(self) -> int:
        """+1 for BUY, -1 for SELL — for signed cost/slippage arithmetic."""
        return 1 if self is Side.BUY else -1

    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(frozen=True, slots=True)
class Execution:
    """A matched trade (fill) for TCA and venue analytics.

    Attributes:
        symbol: Instrument identifier.
        side: BUY or SELL.
        price: All-in fill price.
        quantity: Filled quantity.
        timestamp_nanos: Fill time.
        venue: Venue tag (e.g. "PRIMARY", "LASTLOOK").
    """

    symbol: str
    side: Side
    price: float
    quantity: int
    timestamp_nanos: int
    venue: str

    def notional(self) -> float:
        return self.price * self.quantity
