"""Business-day calendar (minimal transcription of Java
``com.quantfinlib.rates.BusinessCalendar``).

The rates port lives on the year-fraction grid and did not need the
conventions layer; the FX package does — spot lags, tenor rolls and
NDF fixing lags are all calendar walks. This transcribes exactly the
subset ``CurrencyPair``/``Ndf`` consume: weekend+holiday membership,
the roll conventions, business-day add/subtract, and the two-center
union. The Java coupon ``schedule`` helpers are not ported.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum


class Roll(Enum):
    """Date roll conventions for dates landing on non-business days."""

    NONE = "none"
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"


class BusinessCalendar:
    """Weekends plus a holiday set, with roll conventions and lag walks."""

    def __init__(self, holidays: frozenset[dt.date]):
        self._holidays = frozenset(holidays)

    @staticmethod
    def weekends_only() -> "BusinessCalendar":
        return BusinessCalendar(frozenset())

    @staticmethod
    def with_holidays(*holidays: dt.date) -> "BusinessCalendar":
        return BusinessCalendar(frozenset(holidays))

    def union(self, other: "BusinessCalendar") -> "BusinessCalendar":
        """Joint calendar of two centers: business day in BOTH (holiday
        union) — the FX settlement rule."""
        return BusinessCalendar(self._holidays | other._holidays)

    def is_business_day(self, date: dt.date) -> bool:
        return date.weekday() < 5 and date not in self._holidays

    def roll(self, date: dt.date, convention: Roll) -> dt.date:
        """Applies the roll convention to a date."""
        if convention is Roll.NONE or self.is_business_day(date):
            return date
        if convention is Roll.FOLLOWING:
            return self._next_business_day(date)
        if convention is Roll.PRECEDING:
            return self._previous_business_day(date)
        # MODIFIED_FOLLOWING: forward unless that crosses month-end.
        following = self._next_business_day(date)
        if following.month == date.month:
            return following
        return self._previous_business_day(date)

    def add_business_days(self, date: dt.date, n: int) -> dt.date:
        """Adds ``n >= 0`` business days — e.g. T+2 settlement."""
        if n < 0:
            raise ValueError("n must be >= 0")
        d = date
        for _ in range(n):
            d = self._next_business_day(d)
        return d

    def subtract_business_days(self, date: dt.date, n: int) -> dt.date:
        """Walks back ``n >= 0`` business days — e.g. an NDF fixing lag."""
        if n < 0:
            raise ValueError("n must be >= 0")
        d = date
        for _ in range(n):
            d = self._previous_business_day(d)
        return d

    def _next_business_day(self, date: dt.date) -> dt.date:
        d = date + dt.timedelta(days=1)
        while not self.is_business_day(d):
            d += dt.timedelta(days=1)
        return d

    def _previous_business_day(self, date: dt.date) -> dt.date:
        d = date - dt.timedelta(days=1)
        while not self.is_business_day(d):
            d -= dt.timedelta(days=1)
        return d
