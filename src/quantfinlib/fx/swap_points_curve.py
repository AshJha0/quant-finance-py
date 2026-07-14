"""FX forward (swap-points) curve (port of Java
``com.quantfinlib.fx.SwapPointsCurve``).

Stores pillar dates resolved through ``CurrencyPair.tenor_date`` and
interpolates points linearly in actual days between pillars — the
interbank convention for broken dates. Beyond the last pillar it
extrapolates the final segment's slope; before the first pillar it
interpolates from zero points at spot. ``implied_carry`` inverts
covered interest parity (``ln(F/S)/tau``, ACT/365).
"""

from __future__ import annotations

import datetime as dt
import math

from quantfinlib.fx.currency_pair import CurrencyPair


class SwapPointsCurveBuilder:
    """Accumulates tenor/points quotes, then freezes them into a curve."""

    def __init__(self, pair: CurrencyPair, trade_date: dt.date,
                 spot_rate: float):
        if spot_rate <= 0:
            raise ValueError(f"spot_rate must be > 0: {spot_rate}")
        self._pair = pair
        self._trade_date = trade_date
        self._spot_rate = spot_rate
        self._tenors: list[str] = []
        self._dates: list[dt.date] = []
        self._points: list[float] = []

    def add(self, tenor: str, pips: float) -> "SwapPointsCurveBuilder":
        """Adds a pillar quoted in PIPS (market form: "1M EURUSD +12.6"),
        scaled by pip size internally. Negative points are normal when
        the base currency yields more than the quote currency."""
        date = self._pair.tenor_date(self._trade_date, tenor)
        if date <= self._pair.spot_date(self._trade_date):
            raise ValueError(
                f"tenor {tenor} does not settle after spot; pre-spot legs "
                "(ON/TN) belong to the roll, not the forward curve")
        self._tenors.append(tenor)
        self._dates.append(date)
        self._points.append(pips * self._pair.pip_size())
        return self

    def build(self) -> "SwapPointsCurve":
        if not self._dates:
            raise RuntimeError("at least one pillar required")
        spot = self._pair.spot_date(self._trade_date)
        # Sort pillars by date (quotes may arrive in any order).
        order = sorted(range(len(self._dates)), key=lambda i: self._dates[i])
        days = [(self._dates[i] - spot).days for i in order]
        pts = [self._points[i] for i in order]
        tns = [self._tenors[i] for i in order]
        for i in range(1, len(days)):
            if days[i] == days[i - 1]:
                raise ValueError(f"duplicate pillar date at {tns[i]}")
        return SwapPointsCurve(self._pair, spot, self._spot_rate,
                               days, pts, tns)


class SwapPointsCurve:
    """Immutable after ``build``; construct via :meth:`builder`."""

    def __init__(self, pair: CurrencyPair, spot_date: dt.date,
                 spot_rate: float, pillar_days: list[int],
                 pillar_points: list[float], tenors: list[str]):
        self._pair = pair
        self._spot_date = spot_date
        self._spot_rate = spot_rate
        self._pillar_days = pillar_days
        self._pillar_points = pillar_points
        self._tenors = tenors

    @staticmethod
    def builder(pair: CurrencyPair, trade_date: dt.date,
                spot_rate: float) -> SwapPointsCurveBuilder:
        return SwapPointsCurveBuilder(pair, trade_date, spot_rate)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def forward_points(self, value_date: dt.date) -> float:
        """Interpolated forward points (price terms) for a settlement
        date: linear in actual days, anchored at zero on spot."""
        d = (value_date - self._spot_date).days
        if d < 0:
            raise ValueError(
                f"value_date {value_date} is before spot {self._spot_date}")
        if d == 0:
            return 0.0
        days = self._pillar_days
        pts = self._pillar_points
        n = len(days)
        # Before the first pillar: interpolate from (0 days, 0 points).
        if d <= days[0]:
            return pts[0] * (d / days[0])
        # Beyond the last pillar: extend the final segment's slope.
        if d >= days[n - 1]:
            if n == 1:
                return pts[0] * (d / days[0])
            slope = (pts[n - 1] - pts[n - 2]) / (days[n - 1] - days[n - 2])
            return pts[n - 1] + slope * (d - days[n - 1])
        # Binary search for the bracketing pillars.
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if days[mid] <= d:
                lo = mid
            else:
                hi = mid
        w = (d - days[lo]) / (days[hi] - days[lo])
        return pts[lo] + w * (pts[hi] - pts[lo])

    def outright(self, value_date_or_tenor) -> float:
        """Outright forward: spot plus interpolated points. Accepts a
        settlement date or a market tenor string."""
        if isinstance(value_date_or_tenor, str):
            return self.outright(self._tenor_date_from_spot(value_date_or_tenor))
        return self._spot_rate + self.forward_points(value_date_or_tenor)

    def implied_carry(self, value_date: dt.date) -> float:
        """Continuously compounded rate differential (quote minus base)
        implied by covered interest parity: ``ln(F/S)/tau``, ACT/365.
        Positive when the quote currency yields more."""
        d = (value_date - self._spot_date).days
        if d <= 0:
            raise ValueError("value_date must be after spot")
        tau = d / 365.0
        return math.log(self.outright(value_date) / self._spot_rate) / tau

    def spot_date(self) -> dt.date:
        """The spot settlement date all pillar offsets are measured from."""
        return self._spot_date

    def spot_rate(self) -> float:
        return self._spot_rate

    def pair(self) -> CurrencyPair:
        return self._pair

    def pillar_tenors(self) -> list[str]:
        """Pillar tenors in date order (diagnostics/reporting)."""
        return list(self._tenors)

    def _tenor_date_from_spot(self, tenor: str) -> dt.date:
        """Resolves a tenor to a date using this curve's spot as anchor."""
        return self._pair.tenor_date(
            self._pair.trade_date_for_spot(self._spot_date), tenor)
