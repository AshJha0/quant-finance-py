"""Discrete (cash) dividends for equity derivatives (port of Java
``com.quantfinlib.pricing.DividendSchedule``).

A continuous dividend yield misprices single stocks: dividends arrive as
dated cash amounts, and an option spanning an ex-date is worth
measurably less (calls) or more (puts) than the yield approximation
says. This class implements the **escrowed dividend** model: the PV of
all dividends with ex-dates before expiry is stripped from spot, and the
remainder diffuses lognormally::

    S* = S - sum d_i * e^{-r t_i}   (t_i <= T),   F = S* * e^{(r - borrow) T}

Borrow cost enters exactly like a continuous yield (``carry`` in
``BlackScholes``). Instances are immutable.
"""

from __future__ import annotations

import math

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType


class DividendSchedule:
    """Immutable dividend schedule; build with ``DividendSchedule.of``."""

    NONE: "DividendSchedule"  # set after the class body

    def __init__(self, ex_times: tuple[float, ...], amounts: tuple[float, ...]) -> None:
        # Private in spirit: use of() which validates.
        self._ex_times = ex_times
        self._amounts = amounts

    @staticmethod
    def of(ex_times_years, amounts) -> "DividendSchedule":
        """Builds a schedule from aligned ascending ex-times and cash amounts.

        Args:
            ex_times_years: ex-dividend times in years from valuation, ascending.
            amounts: cash amounts per share, aligned with the times.
        """
        ex_times_years = [float(t) for t in ex_times_years]
        amounts = [float(a) for a in amounts]
        if len(ex_times_years) != len(amounts):
            raise ValueError("times and amounts must align")
        for i in range(len(ex_times_years)):
            if ex_times_years[i] <= 0 or amounts[i] < 0:
                raise ValueError("ex-times must be > 0 and amounts >= 0")
            if i > 0 and ex_times_years[i] <= ex_times_years[i - 1]:
                raise ValueError("ex-times must be strictly ascending")
        return DividendSchedule(tuple(ex_times_years), tuple(amounts))

    def present_value(self, rate: float, horizon_years: float) -> float:
        """Present value of all dividends with ex-dates on or before the horizon."""
        pv = 0.0
        for t, a in zip(self._ex_times, self._amounts):
            if t > horizon_years:
                break
            pv += a * math.exp(-rate * t)
        return pv

    def adjusted_spot(self, spot: float, rate: float, horizon_years: float) -> float:
        """Escrowed spot: what actually diffuses once dividend PV is stripped."""
        adjusted = spot - self.present_value(rate, horizon_years)
        if adjusted <= 0:
            raise ValueError("dividend PV exceeds spot — check amounts/horizon")
        return adjusted

    def forward(self, spot: float, rate: float, borrow: float, horizon_years: float) -> float:
        """Equity forward with discrete dividends and a continuous borrow fee."""
        return (self.adjusted_spot(spot, rate, horizon_years)
                * math.exp((rate - borrow) * horizon_years))

    def european_price(self, option_type: OptionType, spot: float, strike: float,
                       rate: float, borrow: float, vol: float, time_years: float) -> float:
        """European price under the escrowed model: BS on the adjusted spot.

        With no dividends and no borrow this is exactly the plain
        Black-Scholes price.
        """
        return BlackScholes.price(option_type, self.adjusted_spot(spot, rate, time_years),
                                  strike, rate, borrow, vol, time_years)

    def count(self) -> int:
        return len(self._ex_times)


#: Empty schedule (no dividends): forwards collapse to the yield-free case.
DividendSchedule.NONE = DividendSchedule((), ())
