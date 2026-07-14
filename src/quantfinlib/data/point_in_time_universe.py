"""Point-in-time universe membership (port of Java ``data.PointInTimeUniverse``).

The engine-side half of survivorship-bias-free backtesting.

Survivorship bias enters when a backtest's universe is built from
TODAY's constituents: every bankruptcy, acquisition and delisting has
already been silently removed, so the strategy only ever "picks" from
winners. Removing the bias needs two things:

1. Data -- historical membership including dead tickers, and delisting
   returns (what a holder actually received). This class cannot
   conjure that; it comes from CRSP-style datasets.
2. Engine -- screens and rebalances that only see members AS OF EACH
   DATE, and positions that terminate correctly when a security dies.
   That is what this class provides, consumed by
   ``StockScreener.members_as_of`` and a universe-aware portfolio
   backtester.

Per symbol it records membership intervals (a symbol can leave and
rejoin an index) and at most one terminal event:

* DELISTING with a delisting return -- the final-day return relative
  to the last close (-1 = shareholders got nothing). When the true
  value is unknown for an involuntary delisting, the literature's
  convention is ``DEFAULT_INVOLUNTARY_DELISTING_RETURN`` (Shumway 1997).
* MERGER with per-share deal terms: cash and/or shares of the acquirer.

Timestamps use the same epoch units as the ``BarSeries`` being
backtested.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class EventType(Enum):
    """How a security's life ends."""

    DELISTING = "DELISTING"
    MERGER = "MERGER"


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    """A security's terminal event.

    :param timestamp: when the event takes effect (first bar at/after
        it applies the event)
    :param type: delisting or merger
    :param delisting_return: final-day return on the last close
        (delistings; 0 for mergers)
    :param cash_per_share: merger cash component per share
    :param acquirer_shares_per_share: merger stock component per share
    :param acquirer: acquirer symbol (``None`` for delistings and
        all-cash deals)
    """

    timestamp: int
    type: EventType
    delisting_return: float
    cash_per_share: float
    acquirer_shares_per_share: float
    acquirer: Optional[str]


@dataclass(frozen=True, slots=True)
class _Interval:
    from_: int  # inclusive
    to: int  # inclusive


class PointInTimeUniverse:
    """Point-in-time universe membership and terminal-event lifecycle."""

    #: The standard haircut for involuntary delistings with unknown
    #: proceeds: -30% on the last traded price (Shumway, Journal of
    #: Finance 1997).
    DEFAULT_INVOLUNTARY_DELISTING_RETURN: float = -0.30

    def __init__(self) -> None:
        self._memberships: Dict[str, List[_Interval]] = {}
        self._events: Dict[str, TerminalEvent] = {}

    def add_membership(
        self, symbol: str, from_timestamp: int, to_timestamp_inclusive: Optional[int] = None
    ) -> "PointInTimeUniverse":
        """Adds a membership interval (inclusive of both endpoints). A
        symbol may hold several disjoint intervals -- index drop and
        later re-add. Omit ``to_timestamp_inclusive`` for an open-ended
        (current) membership."""
        to_ts = (2**63 - 1) if to_timestamp_inclusive is None else to_timestamp_inclusive
        if to_ts < from_timestamp:
            raise ValueError(f"membership ends before it starts: {symbol}")
        self._memberships.setdefault(symbol, []).append(_Interval(from_timestamp, to_ts))
        return self

    def record_delisting(self, symbol: str, timestamp: int, delisting_return: float) -> "PointInTimeUniverse":
        """Records a delisting: membership (if any) is truncated at the
        event and the position terminates at
        ``last_close * (1 + delisting_return)``. Use
        ``DEFAULT_INVOLUNTARY_DELISTING_RETURN`` when the true proceeds
        are unknown."""
        if delisting_return < -1:
            raise ValueError(f"delisting return cannot be below -100%: {delisting_return}")
        self._put_event(
            symbol,
            TerminalEvent(timestamp, EventType.DELISTING, delisting_return, 0.0, 0.0, None),
        )
        return self

    def record_merger(
        self,
        symbol: str,
        timestamp: int,
        cash_per_share: float,
        acquirer_shares_per_share: float,
        acquirer: Optional[str],
    ) -> "PointInTimeUniverse":
        """Records a merger/acquisition: at the event each held share
        converts to ``cash_per_share`` cash plus
        ``acquirer_shares_per_share`` shares of ``acquirer``. All-cash
        deals pass 0 shares and ``acquirer=None``."""
        if cash_per_share < 0 or acquirer_shares_per_share < 0:
            raise ValueError("deal terms cannot be negative")
        if acquirer_shares_per_share > 0 and acquirer is None:
            raise ValueError("stock component needs an acquirer symbol")
        self._put_event(
            symbol,
            TerminalEvent(timestamp, EventType.MERGER, 0.0, cash_per_share, acquirer_shares_per_share, acquirer),
        )
        return self

    def _put_event(self, symbol: str, event: TerminalEvent) -> None:
        if symbol in self._events:
            raise ValueError(f"{symbol} already has a terminal event")
        self._events[symbol] = event

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_member(self, symbol: str, timestamp: int) -> bool:
        """Whether the symbol is a universe member at ``timestamp``:
        inside a membership interval and not past its terminal event."""
        event = self._events.get(symbol)
        if event is not None and timestamp >= event.timestamp:
            return False  # dead securities are never members
        intervals = self._memberships.get(symbol)
        if intervals is None:
            return False
        return any(iv.from_ <= timestamp <= iv.to for iv in intervals)

    def members_as_of(self, timestamp: int) -> List[str]:
        """All members as of a timestamp, sorted for determinism."""
        return sorted(symbol for symbol in self._memberships if self.is_member(symbol, timestamp))

    def terminal_event(self, symbol: str) -> Optional[TerminalEvent]:
        """The symbol's terminal event, or ``None`` while it lives."""
        return self._events.get(symbol)

    def all_symbols(self) -> List[str]:
        """Every symbol that ever appears in this universe (living and
        dead), sorted for determinism."""
        return sorted(set(self._memberships.keys()) | set(self._events.keys()))
