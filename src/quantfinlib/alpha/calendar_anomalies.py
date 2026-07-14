"""Calendar anomaly profiles (port of Java ``alpha.CalendarAnomalies``)
-- day-of-week and turn-of-month seasonality with the t-statistics
that keep them honest. The honest part is the POINT: most published
calendar anomalies (the weekend effect, the January effect) decayed or
died after publication, and the difference between a tradable seasonal
and a data-mined ghost is a t-stat that survives out of sample. This
class hands you the profile AND the significance; treat ``|t| < 2`` as
decoration, and re-test out of sample before believing anything (see
:mod:`quantfinlib.alpha.alpha_validation`).

Turn-of-month windows use CALENDAR days of month (last
``days_before_month_end`` calendar days + first
``days_after_month_start``), not trading days -- stated, not hidden;
with daily equity data the difference is a day around holidays.
Timestamps are interpreted in UTC. Static, deterministic, research
lane.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from quantfinlib.util import math_utils


@dataclass(frozen=True)
class DayOfWeekProfile:
    """Per-day-of-week profile, indexed Monday = 0 ... Sunday = 6.
    Days with no observations report NaN mean and t."""

    mean_return: np.ndarray
    t_stat: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class TurnOfMonth:
    """The turn-of-month split, with a Welch t-stat on the
    difference."""

    inside_mean: float
    outside_mean: float
    t_stat: float
    inside_count: int
    outside_count: int


def _to_date(epoch_millis: int):
    return datetime.fromtimestamp(epoch_millis / 1000.0, tz=timezone.utc).date()


def _signed_infinity_or_ratio(numerator: float, se: float) -> float:
    """Java idiom ``se > 0 ? numerator/se : signum(numerator)*INFINITY``;
    IEEE ``0 * Infinity`` is NaN regardless of the zero's sign, so a
    dead-flat difference (numerator == 0, se == 0) is NaN, not 0."""
    if se > 0:
        return numerator / se
    if numerator == 0:
        return math.nan
    return math.copysign(math.inf, numerator)


class CalendarAnomalies:
    """Static entry points; see the module docstring."""

    @staticmethod
    def day_of_week(returns, epoch_millis) -> DayOfWeekProfile:
        """``returns``: per-period returns aligned with
        ``epoch_millis``."""
        r = np.asarray(returns, dtype=float)
        t = np.asarray(epoch_millis, dtype=np.int64)
        _require_aligned(r, t)
        by_day = [[] for _ in range(7)]
        for i in range(r.shape[0]):
            d = _to_date(int(t[i])).weekday()  # Monday = 0 ... Sunday = 6
            by_day[d].append(r[i])
        mean = np.zeros(7)
        tstat = np.zeros(7)
        counts = np.zeros(7, dtype=int)
        for d in range(7):
            vals = np.array(by_day[d], dtype=float)
            counts[d] = vals.shape[0]
            if counts[d] < 2:
                mean[d] = vals[0] if counts[d] == 1 else math.nan
                tstat[d] = math.nan
                continue
            m = math_utils.mean(vals)
            se = math_utils.std_dev_sample(vals, 0, vals.shape[0]) / math.sqrt(counts[d])
            mean[d] = m
            tstat[d] = _signed_infinity_or_ratio(m, se)
        return DayOfWeekProfile(mean, tstat, counts)

    @staticmethod
    def turn_of_month(returns, epoch_millis, days_before_month_end: int,
                      days_after_month_start: int) -> TurnOfMonth:
        """
        Args:
            days_before_month_end: calendar days at month end in the
                window, >= 0.
            days_after_month_start: calendar days at month start in
                the window, >= 0 (at least one of the two > 0).
        """
        r = np.asarray(returns, dtype=float)
        t = np.asarray(epoch_millis, dtype=np.int64)
        _require_aligned(r, t)
        if (days_before_month_end < 0 or days_after_month_start < 0
                or days_before_month_end + days_after_month_start == 0):
            raise ValueError("the window must cover at least one day")
        inside = []
        outside = []
        for i in range(r.shape[0]):
            d = _to_date(int(t[i]))
            days_in_month = calendar.monthrange(d.year, d.month)[1]
            is_in = (d.day <= days_after_month_start
                     or d.day > days_in_month - days_before_month_end)
            (inside if is_in else outside).append(r[i])
        n_in = len(inside)
        n_out = len(outside)
        if n_in < 2 or n_out < 2:
            raise ValueError(
                "both windows need >= 2 observations (inside "
                f"{n_in}, outside {n_out})")
        inside_arr = np.array(inside, dtype=float)
        outside_arr = np.array(outside, dtype=float)
        m_in = math_utils.mean(inside_arr)
        m_out = math_utils.mean(outside_arr)
        v_in = math_utils.std_dev_sample(inside_arr, 0, n_in) ** 2
        v_out = math_utils.std_dev_sample(outside_arr, 0, n_out) ** 2
        se = math.sqrt(v_in / n_in + v_out / n_out)
        t_stat = _signed_infinity_or_ratio(m_in - m_out, se)
        return TurnOfMonth(m_in, m_out, t_stat, n_in, n_out)


def _require_aligned(returns: np.ndarray, epoch_millis: np.ndarray) -> None:
    if returns.shape[0] != epoch_millis.shape[0] or returns.shape[0] < 30:
        raise ValueError("need >= 30 aligned observations")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must be finite")
