"""FX currency-pair conventions (port of Java
``com.quantfinlib.fx.CurrencyPair``).

Quotation precision, pip size, spot lag, and settlement-date
arithmetic against BOTH currencies' holiday calendars. Conventions
encoded:

- pip size / precision — JPY-quoted pairs quote to 3 decimals
  (pip = 0.01), everything else to 5 (pip = 0.0001);
- spot lag — T+2 for most pairs; T+1 for USDCAD/USDTRY/USDRUB/USDPHP;
- joint calendar — a settlement date must be a business day in both
  currencies' centers (intermediate days treated the same way, the
  documented simplification);
- forward tenors — months/years roll modified-following with the
  end-end rule.
"""

from __future__ import annotations

import calendar as _cal
import datetime as dt

from quantfinlib.fx.business_calendar import BusinessCalendar, Roll

_T_PLUS_ONE = {"USDCAD", "USDTRY", "USDRUB", "USDPHP"}


def _plus_months(date: dt.date, months: int) -> dt.date:
    """Java ``LocalDate.plusMonths``: day-of-month clamped to month end."""
    month_index = date.year * 12 + (date.month - 1) + months
    year, month = divmod(month_index, 12)
    month += 1
    day = min(date.day, _cal.monthrange(year, month)[1])
    return dt.date(year, month, day)


def _end_of_month(date: dt.date) -> dt.date:
    return dt.date(date.year, date.month, _cal.monthrange(date.year, date.month)[1])


class CurrencyPair:
    """Market conventions for an FX pair; construct via :meth:`of` or
    :meth:`custom`."""

    def __init__(self, base: str, quote: str, pip_size: float,
                 price_precision: int, spot_lag_days: int,
                 base_calendar: BusinessCalendar,
                 quote_calendar: BusinessCalendar):
        self._base = base
        self._quote = quote
        self._pip_size = pip_size
        self._price_precision = price_precision
        self._spot_lag_days = spot_lag_days
        self._base_calendar = base_calendar
        self._quote_calendar = quote_calendar
        # Both centers as ONE calendar (holiday union).
        self._joint_calendar = base_calendar.union(quote_calendar)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def of(pair_or_base: str, quote: str | None = None) -> "CurrencyPair":
        """Standard conventions for "EURUSD" (or explicit base/quote),
        weekends-only calendars; attach real calendars via
        :meth:`with_calendars`."""
        if quote is None:
            pair = pair_or_base
            if pair is None or len(pair) != 6:
                raise ValueError(f"pair must be 6 letters, e.g. EURUSD: {pair}")
            base, quote = pair[:3], pair[3:]
        else:
            base = pair_or_base
        b = base.upper()
        q = quote.upper()
        # JPY-quoted pairs trade to 3 decimals; everything else to 5.
        pip = 0.01 if q == "JPY" else 0.0001
        precision = 3 if q == "JPY" else 5
        lag = 1 if b + q in _T_PLUS_ONE else 2
        weekends = BusinessCalendar.weekends_only()
        return CurrencyPair(b, q, pip, precision, lag, weekends, weekends)

    @staticmethod
    def custom(base: str, quote: str, pip_size: float, price_precision: int,
               spot_lag_days: int, base_calendar: BusinessCalendar,
               quote_calendar: BusinessCalendar) -> "CurrencyPair":
        """Fully custom conventions (exotic pairs, onshore fixings, tests)."""
        if pip_size <= 0 or spot_lag_days < 0:
            raise ValueError("pip_size must be > 0 and spot_lag_days >= 0")
        return CurrencyPair(base.upper(), quote.upper(), pip_size,
                            price_precision, spot_lag_days,
                            base_calendar, quote_calendar)

    def with_calendars(self, base_calendar: BusinessCalendar,
                       quote_calendar: BusinessCalendar) -> "CurrencyPair":
        """Same conventions with real holiday calendars per center."""
        return CurrencyPair(self._base, self._quote, self._pip_size,
                            self._price_precision, self._spot_lag_days,
                            base_calendar, quote_calendar)

    # ------------------------------------------------------------------
    # Static conventions
    # ------------------------------------------------------------------

    def base(self) -> str:
        return self._base

    def quote(self) -> str:
        return self._quote

    def symbol(self) -> str:
        """"EURUSD" style symbol, the natural key for the tick bus."""
        return self._base + self._quote

    def pip_size(self) -> float:
        """One pip in price terms (0.0001, or 0.01 for JPY quotes)."""
        return self._pip_size

    def price_precision(self) -> int:
        """Quoted decimal places (5, or 3 for JPY quotes)."""
        return self._price_precision

    def spot_lag_days(self) -> int:
        """Spot settlement lag in business days (T+2, or T+1 exceptions)."""
        return self._spot_lag_days

    def pips(self, price_difference: float) -> float:
        """Price difference to pips (0.00013 -> 1.3 pips on EURUSD)."""
        return price_difference / self._pip_size

    def price_from_pips(self, pips: float) -> float:
        """Pips to a price difference (the inverse of :meth:`pips`)."""
        return pips * self._pip_size

    def round(self, price: float) -> float:
        """Rounds a raw price to the pair's quoted precision (half-up)."""
        scale = 10.0 ** self._price_precision
        return int(price * scale + 0.5) / scale  # Math.round semantics

    # ------------------------------------------------------------------
    # Settlement-date arithmetic
    # ------------------------------------------------------------------

    def base_calendar(self) -> BusinessCalendar:
        """The base currency's own holiday calendar."""
        return self._base_calendar

    def quote_calendar(self) -> BusinessCalendar:
        """The quote currency's own calendar — the restricted currency's
        local calendar that NDF fixing conventions count in."""
        return self._quote_calendar

    def is_joint_business_day(self, date: dt.date) -> bool:
        """Business day in BOTH currencies' calendars."""
        return self._joint_calendar.is_business_day(date)

    def spot_date(self, trade_date: dt.date) -> dt.date:
        """Spot settlement: ``spot_lag_days`` joint business days after
        the trade date."""
        return self.add_joint_business_days(trade_date, self._spot_lag_days)

    def trade_date_for_spot(self, spot_date: dt.date) -> dt.date:
        """The inverse of :meth:`spot_date` — the trade date whose spot
        is ``spot_date`` (which must be a joint business day)."""
        return self._joint_calendar.subtract_business_days(
            spot_date, self._spot_lag_days)

    def tenor_date(self, trade_date: dt.date, tenor: str) -> dt.date:
        """Forward settlement date for a market tenor: ON/TN/SN,
        ``<n>D``/``<n>W`` rolled following, ``<n>M``/``<n>Y``
        modified-following with the end-end rule."""
        t = tenor.upper().strip()
        spot = self.spot_date(trade_date)
        if t == "ON":
            return self.add_joint_business_days(trade_date, 1)
        if t == "TN":
            return self.add_joint_business_days(trade_date, 2)
        if t == "SN":
            return self.add_joint_business_days(spot, 1)
        try:
            n = int(t[:-1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"unsupported tenor: {tenor}") from exc
        unit = t[-1]
        # Short dates roll forward only: crossing month-end is expected.
        if unit == "D":
            return self._roll_following(spot + dt.timedelta(days=n))
        if unit == "W":
            return self._roll_following(spot + dt.timedelta(weeks=n))
        if unit == "M":
            return self._month_tenor(spot, n)
        if unit == "Y":
            return self._month_tenor(spot, n * 12)
        raise ValueError(f"unsupported tenor: {tenor}")

    def _month_tenor(self, spot: dt.date, months: int) -> dt.date:
        """Month/year tenor: end-end rule first, else modified-following."""
        target = _plus_months(spot, months)
        if spot == self._last_joint_business_day_of(spot):
            return self._last_joint_business_day_of(target)
        return self._joint_calendar.roll(target, Roll.MODIFIED_FOLLOWING)

    def add_joint_business_days(self, date: dt.date, n: int) -> dt.date:
        """Adds ``n`` joint business days (n >= 0)."""
        return self._joint_calendar.add_business_days(date, n)

    def _roll_following(self, date: dt.date) -> dt.date:
        return self._joint_calendar.roll(date, Roll.FOLLOWING)

    def _last_joint_business_day_of(self, any_day_in_month: dt.date) -> dt.date:
        return self._joint_calendar.roll(_end_of_month(any_day_in_month),
                                         Roll.PRECEDING)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CurrencyPair):
            return NotImplemented
        return (self._base == other._base and self._quote == other._quote
                and self._pip_size == other._pip_size
                and self._spot_lag_days == other._spot_lag_days)

    def __hash__(self) -> int:
        return hash((self._base, self._quote))

    def __repr__(self) -> str:
        return self.symbol()
