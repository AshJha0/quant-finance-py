"""FX pair conventions, ported from Java CurrencyPairTest.

Pip/precision tables, T+1/T+2 spot lags across dual holiday calendars,
and forward tenor rolls (following, modified-following, end-end).
"""

import datetime as dt

import pytest

from quantfinlib.fx import BusinessCalendar, CurrencyPair


def test_convention_table_covers_majors_and_jpy_quotes():
    eurusd = CurrencyPair.of("EURUSD")
    assert eurusd.pip_size() == 0.0001
    assert eurusd.price_precision() == 5
    assert eurusd.spot_lag_days() == 2

    usdjpy = CurrencyPair.of("USDJPY")
    assert usdjpy.pip_size() == 0.01
    assert usdjpy.price_precision() == 3

    # The market's T+1 exceptions.
    assert CurrencyPair.of("USDCAD").spot_lag_days() == 1
    assert CurrencyPair.of("USDTRY").spot_lag_days() == 1


def test_pip_conversions_round_trip():
    eurusd = CurrencyPair.of("EURUSD")
    assert eurusd.pips(0.00013) == pytest.approx(1.3, abs=1e-9)
    assert eurusd.price_from_pips(1.3) == pytest.approx(0.00013, abs=1e-12)
    assert eurusd.round(1.0853499999) == pytest.approx(1.08535, abs=1e-9)

    usdjpy = CurrencyPair.of("USDJPY")
    assert usdjpy.pips(0.025) == pytest.approx(2.5, abs=1e-9)


def test_spot_date_skips_weekends():
    eurusd = CurrencyPair.of("EURUSD")
    # Friday 2026-01-02 + 2 business days = Tuesday 2026-01-06.
    assert eurusd.spot_date(dt.date(2026, 1, 2)) == dt.date(2026, 1, 6)
    # T+1 pair: Friday -> Monday.
    assert CurrencyPair.of("USDCAD").spot_date(dt.date(2026, 1, 2)) == dt.date(2026, 1, 5)


def test_spot_date_honours_both_currencies_holidays():
    # Monday 2026-01-05 is a base-side holiday: Fri trade must skip it.
    eur_hols = BusinessCalendar.with_holidays(dt.date(2026, 1, 5))
    pair = CurrencyPair.of("EURUSD").with_calendars(eur_hols, BusinessCalendar.weekends_only())
    # Fri 01-02 + 2 joint business days: Tue 01-06, Wed 01-07.
    assert pair.spot_date(dt.date(2026, 1, 2)) == dt.date(2026, 1, 7)

    # The same holiday on the quote side must block equally.
    mirrored = CurrencyPair.of("EURUSD").with_calendars(BusinessCalendar.weekends_only(), eur_hols)
    assert mirrored.spot_date(dt.date(2026, 1, 2)) == dt.date(2026, 1, 7)


def test_week_tenors_roll_forward():
    eurusd = CurrencyPair.of("EURUSD")
    # Trade Wed 2026-01-07 -> spot Fri 01-09 -> 1W = Fri 01-16.
    assert eurusd.tenor_date(dt.date(2026, 1, 7), "1W") == dt.date(2026, 1, 16)
    # Spot Mon 2026-01-12 (trade Thu 01-08): +2D lands Wed.
    assert eurusd.tenor_date(dt.date(2026, 1, 8), "2D") == dt.date(2026, 1, 14)


def test_month_tenor_uses_modified_following():
    eurusd = CurrencyPair.of("EURUSD")
    # Trade Mon 2026-04-27 -> spot Wed 04-29 -> 1M = Fri 05-29 (no roll needed).
    assert eurusd.tenor_date(dt.date(2026, 4, 27), "1M") == dt.date(2026, 5, 29)
    # Trade Wed 2026-01-28 -> spot Fri 01-30 -> 1M unadjusted Sat 02-28 ->
    # following would cross into March, so modified-following rolls BACK
    # to Fri 02-27.
    assert eurusd.tenor_date(dt.date(2026, 1, 28), "1M") == dt.date(2026, 2, 27)


def test_end_end_rule_pins_month_ends():
    eurusd = CurrencyPair.of("EURUSD")
    # Spot Fri 2026-02-27 (trade Wed 02-25) is the last business day of
    # February; 1M forward must pin to the last business day of March
    # (Tue 03-31), not 03-27.
    trade = dt.date(2026, 2, 25)
    assert eurusd.spot_date(trade) == dt.date(2026, 2, 27)
    assert eurusd.tenor_date(trade, "1M") == dt.date(2026, 3, 31)


def test_pre_spot_tenors_and_validation():
    eurusd = CurrencyPair.of("EURUSD")
    trade = dt.date(2026, 1, 7)   # Wednesday
    assert eurusd.tenor_date(trade, "ON") == dt.date(2026, 1, 8)
    assert eurusd.tenor_date(trade, "TN") == dt.date(2026, 1, 9)   # = spot for T+2
    assert eurusd.spot_date(trade) == eurusd.tenor_date(trade, "TN")
    assert eurusd.tenor_date(trade, "SN") == dt.date(2026, 1, 12)  # spot Fri + 1 = Mon

    with pytest.raises(ValueError):
        CurrencyPair.of("EUR")
    with pytest.raises(ValueError):
        eurusd.tenor_date(trade, "1Q")
    assert eurusd.symbol() == "EURUSD"
