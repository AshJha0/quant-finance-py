"""Pins for Margrabe, Kirk, quanto and the vol-swap strike, ported from
StructuredClosedFormsTest.java: each formula must collapse onto an
already-tested pricer where the theory says it must — the strongest
cheap correctness check there is.
"""

import math

import pytest

from quantfinlib.pricing import (Black76, BlackScholes, ExchangeOption,
                                 OptionType, QuantoOption, VarianceSwap)

# ------------------------------------------------------------------ Margrabe


def test_margrabe_with_constant_second_asset_is_a_black_scholes_call():
    # sigma2 = 0, q1 = 0: asset 2 is a bond worth S2 at expiry — the
    # exchange option IS a vanilla call struck at S2 with rate q2.
    m = ExchangeOption.margrabe(100, 95, 0.0, 0.03, 0.25, 0.0, 0.5, 1.0)
    bs = BlackScholes.price(OptionType.CALL, 100, 95, 0.03, 0.0, 0.25, 1.0)
    assert m == pytest.approx(bs, abs=1e-12)


def test_perfectly_correlated_equal_vol_assets_pay_forward_intrinsic():
    # rho = 1, sigma1 = sigma2: the ratio cannot move — pure forward.
    m = ExchangeOption.margrabe(100, 90, 0.01, 0.02, 0.30, 0.30, 1.0, 2.0)
    assert m == pytest.approx(100 * math.exp(-0.02) - 90 * math.exp(-0.04), abs=1e-12)
    # And it is never negative even when the forward is under water.
    assert ExchangeOption.margrabe(90, 100, 0, 0, 0.30, 0.30, 1.0, 2.0) == 0


def test_margrabe_expiry_and_correlation_behavior():
    assert ExchangeOption.margrabe(100, 90, 0, 0, 0.3, 0.2, 0.5, 0.0) == 10
    # Lower correlation -> higher ratio vol -> dearer option.
    low_rho = ExchangeOption.margrabe(100, 100, 0, 0, 0.2, 0.2, -0.5, 1)
    high_rho = ExchangeOption.margrabe(100, 100, 0, 0, 0.2, 0.2, 0.8, 1)
    assert low_rho > high_rho


# ---------------------------------------------------------------------- Kirk


def test_kirk_collapses_to_margrabe_at_zero_strike():
    kirk = ExchangeOption.kirk_spread_call(100, 90, 0, 0.05, 0.3, 0.2, 0.4, 1.5)
    margrabe_fwd = (math.exp(-0.05 * 1.5)
                    * ExchangeOption.margrabe(100, 90, 0, 0, 0.3, 0.2, 0.4, 1.5))
    assert kirk == pytest.approx(margrabe_fwd, abs=1e-12)


def test_kirk_collapses_to_black76_with_no_second_leg():
    # f2 = 0: the "spread" is just an option on F1 struck at K, and
    # sigma2/rho must drop out entirely.
    kirk = ExchangeOption.kirk_spread_call(100, 0, 95, 0.03, 0.2, 0.7, -0.9, 1.0)
    b76 = Black76.price(OptionType.CALL, 100, 95, 0.03, 0.2, 1.0)
    assert kirk == pytest.approx(b76, abs=1e-12)


def test_kirk_gates():
    with pytest.raises(ValueError):
        ExchangeOption.kirk_spread_call(100, -1, 95, 0, 0.2, 0.2, 0, 1)
    with pytest.raises(ValueError):
        ExchangeOption.kirk_spread_call(100, 0, 0, 0, 0.2, 0.2, 0, 1)  # f2+K=0
    with pytest.raises(ValueError):
        ExchangeOption.margrabe(100, 90, 0, 0, 0.2, 0.2, 1.5, 1)       # rho


# -------------------------------------------------------------------- quanto


def test_zero_correlation_quanto_is_the_plain_vanilla():
    q = QuantoOption.price(OptionType.CALL, 100, 100, 0.03, 0.01, 0.2, 0.1, 0.0, 1)
    bs = BlackScholes.price(OptionType.CALL, 100, 100, 0.03, 0.01, 0.2, 1)
    assert q == pytest.approx(bs, abs=1e-12)
    assert QuantoOption.quanto_forward(100, 0.03, 0.01, 0.2, 0.1, 0.0, 1) == pytest.approx(
        100 * math.exp(0.02), abs=1e-12)


def test_positive_correlation_lowers_the_quanto_forward_and_call():
    # Asset up when foreign ccy strengthens: the hedger's drag, priced.
    f0 = QuantoOption.quanto_forward(100, 0.03, 0.01, 0.2, 0.1, 0.0, 1)
    f_pos = QuantoOption.quanto_forward(100, 0.03, 0.01, 0.2, 0.1, 0.6, 1)
    assert f_pos < f0
    # Exact drift: rho*sigmaS*sigmaFX = 0.012 off the carry.
    assert f_pos == pytest.approx(100 * math.exp(0.02 - 0.012), abs=1e-12)
    call_pos = QuantoOption.price(OptionType.CALL, 100, 100, 0.03, 0.01, 0.2, 0.1, 0.6, 1)
    call_0 = QuantoOption.price(OptionType.CALL, 100, 100, 0.03, 0.01, 0.2, 0.1, 0.0, 1)
    assert call_pos < call_0  # a lower forward makes the call cheaper


# ------------------------------------------------------------ vol swap strike


def test_vol_swap_strike_is_sqrt_minus_convexity_correction():
    # Zero vol-of-vol: exactly sqrt(K_var).
    assert VarianceSwap.vol_swap_strike(0.04, 0.0) == 0.2
    # Brockhaus-Long at Var(V)=0.0008: 0.2 - 0.0008/(8*0.008) = 0.1875.
    assert VarianceSwap.vol_swap_strike(0.04, 0.0008) == pytest.approx(0.1875, abs=1e-15)
    # Jensen: the vol strike sits below sqrt of the variance strike.
    assert VarianceSwap.vol_swap_strike(0.04, 0.0004) < 0.2
    with pytest.raises(ValueError):
        VarianceSwap.vol_swap_strike(0, 0.0004)
    with pytest.raises(ValueError):
        VarianceSwap.vol_swap_strike(0.04, -1e-9)
