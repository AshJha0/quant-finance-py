"""Pins for calendar seasonality profiles.

Java source: AlphaResearchRoundTest.java (calendarProfilesRecoverPlantedSeasonalityExactly).
"""

import calendar
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from quantfinlib.alpha.calendar_anomalies import CalendarAnomalies


def _epoch_millis(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def test_calendar_profiles_recover_planted_seasonality_exactly():
    # 400 weekdays from Monday 2024-01-01: Mondays planted at -0.2%, the
    # rest at +0.05%, with an alternating +/-1bp wiggle so group stdevs
    # are nonzero and group means stay EXACT (even counts).
    n = 400
    returns = np.zeros(n)
    stamps = np.zeros(n, dtype=np.int64)
    d = date(2024, 1, 1)
    per_day = [0] * 7
    i = 0
    while i < n:
        if d.weekday() < 5:  # Mon-Fri
            dow = d.weekday()
            base = -0.002 if dow == 0 else 0.0005
            returns[i] = base + (1e-4 if per_day[dow] % 2 == 0 else -1e-4)
            per_day[dow] += 1
            stamps[i] = _epoch_millis(d)
            i += 1
        d += timedelta(days=1)

    profile = CalendarAnomalies.day_of_week(returns, stamps)
    assert profile.count[0] == 80          # 80 full weeks of Mondays
    assert profile.mean_return[0] == pytest.approx(-0.002, abs=1e-12)
    assert profile.t_stat[0] < -5
    assert profile.mean_return[2] == pytest.approx(0.0005, abs=1e-12)
    assert profile.count[5] == 0            # no Saturdays in the data
    assert math.isnan(profile.mean_return[6])

    # Turn of month: plant +0.3% inside the (1 before, 3 after) window,
    # +0.01% outside -- the split and the Welch t find it.
    tom = np.zeros(n)
    parity = [0, 0]
    dates = [datetime.fromtimestamp(int(s) / 1000.0, tz=timezone.utc).date()
             for s in stamps]
    for i, dt in enumerate(dates):
        days_in_month = calendar.monthrange(dt.year, dt.month)[1]
        inside = dt.day <= 3 or dt.day > days_in_month - 1
        base = 0.003 if inside else 0.0001
        idx = 0 if inside else 1
        tom[i] = base + (1e-4 if parity[idx] % 2 == 0 else -1e-4)
        parity[idx] += 1

    split = CalendarAnomalies.turn_of_month(tom, stamps, 1, 3)
    assert split.inside_mean == pytest.approx(0.003, abs=1.1e-4)
    assert split.outside_mean == pytest.approx(0.0001, abs=1.1e-4)
    assert split.t_stat > 10
    assert split.inside_count + split.outside_count == n

    with pytest.raises(ValueError):
        CalendarAnomalies.turn_of_month(tom, stamps, 0, 0)
