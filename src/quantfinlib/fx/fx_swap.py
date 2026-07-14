"""FX swap (port of Java ``com.quantfinlib.fx.FxSwap``).

Two offsetting FX exchanges — buy (sell) base on the near date, sell
(buy) it back on the far date. An at-market swap has zero value at
inception; value appears as the points move. Sign convention:
``base_notional > 0`` means the near leg BUYS base (far leg sells it
back). All results are in quote currency.

Aged swaps: a leg whose settlement date lies before the marking
curve's spot has already settled — realized cash, not MTM — so it
contributes zero.
"""

from __future__ import annotations

import datetime as dt

from quantfinlib.fx.currency_pair import CurrencyPair
from quantfinlib.fx.swap_points_curve import SwapPointsCurve
from quantfinlib.rates.yield_curve import YieldCurve


class FxSwap:

    def __init__(self, pair: CurrencyPair, base_notional: float,
                 near_date: dt.date, near_rate: float,
                 far_date: dt.date, far_rate: float):
        if far_date <= near_date:
            raise ValueError("far date must be after near date")
        if near_rate <= 0 or far_rate <= 0 or base_notional == 0:
            raise ValueError("rates must be > 0 and notional non-zero")
        self._pair = pair
        self._base_notional = base_notional
        self._near_date = near_date
        self._near_rate = near_rate
        self._far_date = far_date
        self._far_rate = far_rate

    @staticmethod
    def of(pair: CurrencyPair, base_notional: float,
           near_date: dt.date, near_rate: float,
           far_date: dt.date, far_rate: float) -> "FxSwap":
        """Explicit legs (off-market swaps, historical bookings)."""
        return FxSwap(pair, base_notional, near_date, near_rate,
                      far_date, far_rate)

    @staticmethod
    def at_market(curve: SwapPointsCurve, near_tenor: str, far_tenor: str,
                  base_notional: float) -> "FxSwap":
        """At-market swap struck off a points curve: both legs at the
        curve's outrights, so inception value is zero by construction.
        ``"SPOT"`` is accepted as the near tenor."""
        if near_tenor.upper() == "SPOT":
            near = curve.spot_date()
            near_rate = curve.spot_rate()
        else:
            near = _date_of(curve, near_tenor)
            near_rate = curve.outright(near)
        far = _date_of(curve, far_tenor)
        return FxSwap(curve.pair(), base_notional, near, near_rate,
                      far, curve.outright(far))

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def mark_to_market(self, current: SwapPointsCurve,
                       quote_discount: YieldCurve | None = None) -> float:
        """Mark-to-market in quote currency against a current curve:
        each leg's (current forward - traded rate) x signed notional.
        Settled legs (before the curve's spot) contribute zero. With
        ``quote_discount``, live-leg P&Ls are discounted off a
        quote-currency zero curve (ACT/365 from the curve's spot)."""
        if quote_discount is None:
            return (self._leg_pnl(self._near_date, self._near_rate, +1, current)
                    + self._leg_pnl(self._far_date, self._far_rate, -1, current))
        mtm = 0.0
        for leg_date, leg_rate, sign in (
                (self._near_date, self._near_rate, +1),
                (self._far_date, self._far_rate, -1)):
            if leg_date >= current.spot_date():
                t = (leg_date - current.spot_date()).days / 365.0
                df = quote_discount.discount_factor(t) if t > 0 else 1.0
                mtm += self._leg_pnl(leg_date, leg_rate, sign, current) * df
        return mtm

    def _leg_pnl(self, leg_date: dt.date, leg_rate: float, sign: int,
                 current: SwapPointsCurve) -> float:
        """One leg's undiscounted P&L: zero once settled, spot at spot,
        else the outright."""
        if leg_date < current.spot_date():
            return 0.0  # settled: realized cash, not MTM
        forward = (current.spot_rate() if leg_date == current.spot_date()
                   else current.outright(leg_date))
        return sign * self._base_notional * (forward - leg_rate)

    def swap_points_pips(self) -> float:
        """The traded points differential (far - near) in pips."""
        return self._pair.pips(self._far_rate - self._near_rate)

    @staticmethod
    def roll_cost(pair: CurrencyPair, base_notional: float,
                  tom_next_pips: float) -> float:
        """Quote-currency cost of rolling a base position one day at a
        quoted tom-next rate."""
        return base_notional * pair.price_from_pips(tom_next_pips)

    def pair(self) -> CurrencyPair:
        return self._pair

    def base_notional(self) -> float:
        return self._base_notional

    def near_date(self) -> dt.date:
        return self._near_date

    def near_rate(self) -> float:
        return self._near_rate

    def far_date(self) -> dt.date:
        return self._far_date

    def far_rate(self) -> float:
        return self._far_rate

    def __repr__(self) -> str:
        return (f"{self._pair.symbol()} swap {self._base_notional} "
                f"{self._pair.base()} {self._near_date}@{self._near_rate} "
                f"/ {self._far_date}@{self._far_rate}")


def _date_of(curve: SwapPointsCurve, tenor: str) -> dt.date:
    """Anchor tenor dates at the curve's own spot via spot inversion."""
    pair = curve.pair()
    return pair.tenor_date(pair.trade_date_for_spot(curve.spot_date()), tenor)
