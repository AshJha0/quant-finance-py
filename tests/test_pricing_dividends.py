"""Pins for quantfinlib.pricing.dividend_schedule, ported from
DividendScheduleTest.java.
"""

import math

import pytest

from quantfinlib.pricing import BlackScholes, DividendSchedule, OptionType

S = 100.0
R = 0.04
VOL = 0.25
T = 1.0


@pytest.fixture()
def divs():
    # Quarterly $1 dividends.
    return DividendSchedule.of([0.25, 0.50, 0.75, 1.00], [1, 1, 1, 1])


def test_present_value_discounts_and_filters_by_horizon(divs):
    expected = math.exp(-R * 0.25) + math.exp(-R * 0.50)
    # Only ex-dates on/before the horizon count.
    assert divs.present_value(R, 0.6) == pytest.approx(expected, abs=1e-12)
    assert divs.present_value(R, 0.1) == pytest.approx(0, abs=1e-12)
    assert divs.count() == 4
    assert DividendSchedule.NONE.present_value(R, 10) == pytest.approx(0, abs=1e-12)


def test_forward_drops_by_dividends_and_borrow(divs):
    no_div_forward = S * math.exp(R * T)
    with_divs = divs.forward(S, R, 0, T)
    assert with_divs < no_div_forward
    assert with_divs == pytest.approx(divs.adjusted_spot(S, R, T) * math.exp(R * T), abs=1e-12)
    # Borrow fee acts like extra yield: forward drops further.
    assert divs.forward(S, R, 0.02, T) < with_divs
    # No dividends, no borrow: the plain cost-of-carry forward.
    assert DividendSchedule.NONE.forward(S, R, 0, T) == pytest.approx(no_div_forward, abs=1e-12)


def test_option_prices_move_in_the_dividend_direction(divs):
    plain_call = BlackScholes.price(OptionType.CALL, S, 100, R, 0, VOL, T)
    plain_put = BlackScholes.price(OptionType.PUT, S, 100, R, 0, VOL, T)
    div_call = divs.european_price(OptionType.CALL, S, 100, R, 0, VOL, T)
    div_put = divs.european_price(OptionType.PUT, S, 100, R, 0, VOL, T)
    # Dividends leak value out of the diffusion: calls down, puts up.
    assert div_call < plain_call
    assert div_put > plain_put
    # Empty schedule reproduces Black-Scholes exactly.
    assert DividendSchedule.NONE.european_price(
        OptionType.CALL, S, 100, R, 0, VOL, T) == pytest.approx(plain_call, abs=1e-12)
    # Put-call parity on the ADJUSTED spot (the escrowed model's parity).
    adjusted = divs.adjusted_spot(S, R, T)
    assert div_call - div_put == pytest.approx(adjusted - 100 * math.exp(-R * T), abs=1e-10)


def test_validation_rejects_bad_schedules():
    with pytest.raises(ValueError):
        DividendSchedule.of([1], [1, 2])
    with pytest.raises(ValueError):
        DividendSchedule.of([0], [1])
    with pytest.raises(ValueError):
        DividendSchedule.of([0.5, 0.25], [1, 1])
    with pytest.raises(ValueError):
        DividendSchedule.of([0.5], [-1])
    # Dividend PV swamping spot must be reported, not priced.
    huge = DividendSchedule.of([0.5], [200])
    with pytest.raises(ValueError):
        huge.adjusted_spot(S, R, T)
