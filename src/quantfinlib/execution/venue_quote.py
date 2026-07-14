"""A venue's dealable top of book for routing (port of Java
``execution.VenueQuote``).

For dark venues, bid/ask sizes are the estimated executable liquidity
and fills are assumed at the venue's midpoint.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VenueQuote:
    venue: str
    bid: float
    bid_size: int
    ask: float
    ask_size: int
    fee_bps: float
    latency_nanos: int
    dark: bool

    def mid(self) -> float:
        return (self.bid + self.ask) / 2
