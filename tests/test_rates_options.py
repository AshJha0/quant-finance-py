"""RatesOptions pins, ported from Java RatesOptionsTest.

Swaptions and caps priced off the curve, pinned by the two parities
that any correct implementation must satisfy EXACTLY, whatever the vol:
payer - receiver = annuity*(F - K), and cap - floor = the swap PV.
"""

import math

import pytest

from quantfinlib.rates import RatesOptions, YieldCurve


def flat5() -> YieldCurve:
    tenors = [i + 1 for i in range(10)]
    return YieldCurve.of_zero_rates(tenors, [0.05] * 10)


def test_flat_curve_forward_swap_rate_is_the_simple_forward():
    # Flat cc curve: every annual simple forward is e^0.05 - 1, and a
    # par swap rate over identical forwards equals that forward.
    c = flat5()
    expected = math.exp(0.05) - 1
    assert RatesOptions.forward_swap_rate(c, 1, 5) == pytest.approx(expected, abs=1e-12)
    assert RatesOptions.forward_swap_rate(c, 0, 10) == pytest.approx(expected, abs=1e-12)


def test_swaption_put_call_parity_is_exact():
    c = flat5()
    strike = 0.04
    payer = RatesOptions.swaption(c, 1, 5, strike, 0.25, True)
    receiver = RatesOptions.swaption(c, 1, 5, strike, 0.25, False)
    annuity = RatesOptions.annuity(c, 1, 5)
    fsr = RatesOptions.forward_swap_rate(c, 1, 5)
    assert payer - receiver == pytest.approx(annuity * (fsr - strike), abs=1e-12), \
        "payer - receiver must equal the forward swap PV, any vol"
    assert payer > 0 and receiver > 0

    # At-the-money forward: payer and receiver are worth the same.
    atm_payer = RatesOptions.swaption(c, 1, 5, fsr, 0.25, True)
    atm_receiver = RatesOptions.swaption(c, 1, 5, fsr, 0.25, False)
    assert atm_payer == pytest.approx(atm_receiver, abs=1e-12)


def test_cap_minus_floor_is_the_swap_pv():
    c = flat5()
    strike = 0.04
    cap = RatesOptions.cap(c, 5, strike, 0.30)
    floor = RatesOptions.floor(c, 5, strike, 0.30)
    swap_pv = 0.0
    for i in range(1, 6):
        df_pay = c.discount_factor(i)
        fwd = c.discount_factor(i - 1) / df_pay - 1
        swap_pv += df_pay * (fwd - strike)
    assert cap - floor == pytest.approx(swap_pv, abs=1e-12), "cap - floor = swap, any vol"
    # Longer caps contain more caplets: strictly more valuable.
    assert RatesOptions.cap(c, 7, strike, 0.30) > cap


def test_rates_options_gates():
    c = flat5()
    with pytest.raises(ValueError):
        RatesOptions.annuity(c, 1, 0)
    with pytest.raises(ValueError):
        RatesOptions.annuity(c, -1, 5)
    with pytest.raises(ValueError):
        RatesOptions.swaption(c, 1, 5, 0, 0.25, True)
    with pytest.raises(ValueError):
        RatesOptions.swaption(c, 1, 5, 0.04, 0, True)
    with pytest.raises(ValueError):
        RatesOptions.cap(c, 0, 0.04, 0.30)
