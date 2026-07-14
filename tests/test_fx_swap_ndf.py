"""FX swap and NDF bookings, ported from Java FxSwapNdfTest.

At-market swaps value to zero on their own curve, MTM tracks points
moves with the right sign, aged legs settle to zero, NDF fixing dates
walk back by the restricted currency's lag, and the USD settlement
formula divides by the fixing.
"""

import datetime as dt
import math

import pytest

from quantfinlib.fx import BusinessCalendar, CurrencyPair, FxSwap, Ndf, SwapPointsCurve
from quantfinlib.rates import YieldCurve

EURUSD = CurrencyPair.of("EURUSD")
TRADE = dt.date(2026, 1, 7)  # Wed, spot Fri 01-09


def _curve(spot, one_month_pips, three_month_pips):
    return (SwapPointsCurve.builder(EURUSD, TRADE, spot)
            .add("1M", one_month_pips)
            .add("3M", three_month_pips)
            .build())


def test_at_market_swap_values_to_zero_on_its_own_curve():
    c = _curve(1.0850, 12.6, 38.4)
    swap = FxSwap.at_market(c, "SPOT", "3M", 10_000_000)
    assert swap.mark_to_market(c) == pytest.approx(0, abs=1e-9)
    assert swap.swap_points_pips() == pytest.approx(38.4, abs=1e-9)
    assert swap.near_date() == c.spot_date()
    assert "EURUSD" in repr(swap)


def test_swap_mtm_tracks_points_moves_with_the_right_sign():
    struck = _curve(1.0850, 12.6, 38.4)
    swap = FxSwap.at_market(struck, "SPOT", "3M", 10_000_000)
    # Points widen 38.4 -> 45.0: the far leg (short base forward) loses.
    wider = _curve(1.0850, 12.6, 45.0)
    mtm = swap.mark_to_market(wider)
    assert mtm == pytest.approx(-10_000_000 * 6.6 * 0.0001, abs=1.0)
    # Discounting shrinks the magnitude but keeps the sign.
    usd = YieldCurve.of_zero_rates([0.25, 1], [0.05, 0.05])
    discounted = swap.mark_to_market(wider, usd)
    assert discounted < 0 and abs(discounted) < abs(mtm)


def test_mismatched_swap_carries_forward_exposure():
    struck = _curve(1.0850, 12.6, 38.4)
    ff = FxSwap.at_market(struck, "1M", "3M", 5_000_000)
    # Parallel spot move with unchanged points: legs cancel -> MTM ~ 0.
    spot_up = _curve(1.0950, 12.6, 38.4)
    assert ff.mark_to_market(spot_up) == pytest.approx(0, abs=1e-9)


def test_ndf_booking_walks_fixing_back_by_the_currency_lag():
    usdinr = CurrencyPair.of("USDINR")
    ndf = Ndf.of_tenor(usdinr, TRADE, "1M", 84.50, 1_000_000)
    assert Ndf.fixing_lag_days("INR") == 2
    assert Ndf.fixing_lag_days("BRL") == 1
    assert Ndf.fixing_lag_days("XXX") == 2  # unlisted default
    # Settlement Mon 2026-02-09 (1M from spot Fri 01-09, rolled following);
    # fixing two business days earlier.
    assert ndf.settlement_date() == dt.date(2026, 2, 9)
    assert ndf.fixing_date() == dt.date(2026, 2, 5)
    assert ndf.fixing_date() < ndf.settlement_date()


def test_aged_swap_marks_against_a_later_curve_without_throwing():
    struck = _curve(1.0850, 12.6, 38.4)
    swap = FxSwap.at_market(struck, "SPOT", "3M", 10_000_000)
    later_trade = TRADE + dt.timedelta(weeks=2)
    later = (SwapPointsCurve.builder(EURUSD, later_trade, 1.0900)
             .add("1M", 11.0)
             .add("3M", 33.0)
             .build())
    # Near date (old spot) is before the new curve's spot: settled leg.
    assert swap.near_date() < later.spot_date()
    mtm = swap.mark_to_market(later)
    far_only = -10_000_000 * (later.outright(swap.far_date()) - swap.far_rate())
    assert mtm == pytest.approx(far_only, abs=1e-9)
    usd = YieldCurve.of_zero_rates([1], [0.05])
    assert math.copysign(1, far_only) == math.copysign(1, swap.mark_to_market(later, usd))


def test_ndf_fixing_counts_local_not_joint_business_days():
    # USDINR settling Wednesday with the preceding Monday a US-only
    # holiday: the RBI fixing counts INDIAN business days, so Monday
    # still counts and the fixing is Monday.
    us_holiday = BusinessCalendar.with_holidays(dt.date(2026, 2, 9))
    usdinr = CurrencyPair.of("USDINR").with_calendars(us_holiday, BusinessCalendar.weekends_only())
    ndf = Ndf.of(usdinr, 1_000_000, 84.50, dt.date(2026, 2, 9), dt.date(2026, 2, 11))
    assert ndf.fixing_date() == dt.date(2026, 2, 9)
    # The tenor-booking path: fixing walked back on the QUOTE (INR)
    # calendar ignores the US holiday.
    booked = Ndf.of_tenor(usdinr, dt.date(2026, 1, 7), "1M", 84.50, 1_000_000)
    expected = booked.settlement_date()
    for _ in range(2):  # walk back 2 INR business days
        expected = expected - dt.timedelta(days=1)
        while not usdinr.quote_calendar().is_business_day(expected):
            expected = expected - dt.timedelta(days=1)
    assert booked.fixing_date() == expected


def test_ndf_settlement_divides_by_the_fixing():
    usdinr = CurrencyPair.of("USDINR")
    ndf = Ndf.of_tenor(usdinr, TRADE, "1M", 84.50, 1_000_000)
    # Fixing above contract: base buyer (long USD) gains, settled in USD.
    assert ndf.settlement_amount(85.00) == pytest.approx(
        1_000_000 * (85.00 - 84.50) / 85.00, abs=1e-9)
    # Fixing below: buyer pays.
    assert ndf.settlement_amount(84.00) < 0
    with pytest.raises(ValueError):
        ndf.settlement_amount(0)


def test_ndf_marks_against_the_forward_to_the_fixing_date():
    usdinr = CurrencyPair.of("USDINR")
    c = (SwapPointsCurve.builder(usdinr, TRADE, 84.20)
         .add("1M", 3000)   # 30 paise of points (pip = 0.0001 -> 0.30)
         .add("3M", 9000)
         .build())
    ndf = Ndf.of_tenor(usdinr, TRADE, "1M", 84.20, 1_000_000)
    fwd_at_fixing = c.outright(ndf.fixing_date())
    assert ndf.mark_to_market(c) == pytest.approx(
        1_000_000 * (fwd_at_fixing - 84.20) / fwd_at_fixing, abs=1e-9)
    usd = YieldCurve.of_zero_rates([1], [0.05])
    assert 0 < ndf.mark_to_market(c, usd) < ndf.mark_to_market(c)


def test_roll_cost_and_validation():
    # Paying 0.15 pips tom-next on 10m base.
    assert FxSwap.roll_cost(EURUSD, 10_000_000, 0.15) == pytest.approx(
        10_000_000 * 0.15 * 0.0001, abs=1e-9)
    with pytest.raises(ValueError):
        FxSwap.of(EURUSD, 1_000_000, TRADE + dt.timedelta(days=30), 1.09,
                 TRADE + dt.timedelta(days=10), 1.10)
    with pytest.raises(ValueError):
        Ndf.of(CurrencyPair.of("USDINR"), 1_000_000, 0,
              TRADE + dt.timedelta(days=28), TRADE + dt.timedelta(days=30))
