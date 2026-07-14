"""Pins for quantfinlib.risk.risk_metrics.

Java sources: FormulaPinsTest.sharpeAndSortinoPinnedIncludingTheDownsideDenominator
plus hand-derived pins in the same house style (every value derived in
a comment, Java-matching tolerances).
"""

import math

import pytest

from quantfinlib.risk import risk_metrics as rm


def test_volatility_is_sample_std():
    # [1,2,3,4]%: sample variance 5/3 * 1e-4 -> vol = sqrt(5/3)*1e-2.
    assert rm.volatility([0.01, 0.02, 0.03, 0.04]) == pytest.approx(
        math.sqrt(5.0 / 3.0) * 1e-2, abs=1e-15)


def test_annualized_volatility_scales_by_sqrt_time():
    assert rm.annualized_volatility([0.01, 0.02, 0.03, 0.04], 252) == pytest.approx(
        math.sqrt(5.0 / 3.0) * 1e-2 * math.sqrt(252), abs=1e-12)


def test_sharpe_pinned_from_java_formula_pins():
    # {0.01, 0.03}: mean 0.02*252 = 5.04; sample std sqrt(2e-4)*sqrt(252).
    assert rm.sharpe_ratio([0.01, 0.03], 0, 252) == pytest.approx(
        5.04 / (math.sqrt(2e-4) * math.sqrt(252)), abs=1e-9), \
        "annualize the mean by 252, the vol by sqrt(252)"


def test_sortino_pinned_downside_denominator():
    # {0.02,-0.01,0.03,-0.02}: downside dev sqrt((1e-4+4e-4)/4) = sqrt(1.25e-4);
    # sortino = (0.005*252)/(sqrt(1.25e-4)*sqrt(252)) — using the FULL stdev
    # by mistake yields 3.63 and fails.
    assert rm.sortino_ratio([0.02, -0.01, 0.03, -0.02], 0, 252) == pytest.approx(
        0.005 * 252 / (math.sqrt(1.25e-4) * math.sqrt(252)), abs=1e-9), \
        "the denominator is DOWNSIDE deviation, not stdev"


def test_downside_deviation_pin():
    # MAR 0: only -0.01 and -0.02 contribute: sqrt((1e-4 + 4e-4)/4).
    assert rm.downside_deviation([0.02, -0.01, 0.03, -0.02], 0) == pytest.approx(
        math.sqrt(1.25e-4), abs=1e-15)


def test_zero_vol_ratios_are_zero_not_nan():
    assert rm.sharpe_ratio([0.01, 0.01], 0, 252) == 0.0
    assert rm.sortino_ratio([0.01, 0.02], 0, 252) == 0.0  # no downside => dd 0


def test_historical_var_pin():
    # sorted [-0.05,-0.02,0.01,0.03], p=0.05 -> idx 0.15:
    # -0.05*0.85 + -0.02*0.15 = -0.0455 -> VaR 0.0455.
    assert rm.historical_var([0.01, -0.05, 0.03, -0.02], 0.95) == pytest.approx(
        0.0455, abs=1e-15)
    # A quantile that is a gain floors at 0.
    assert rm.historical_var([0.01, 0.02, 0.03, 0.04], 0.95) == 0.0


def test_parametric_var_pin():
    # mean -0.0075, sample std of [0.01,-0.05,0.03,-0.02]:
    # devs {0.0175,-0.0425,0.0375,-0.0125}; ss = (3.0625 + 18.0625 +
    # 14.0625 + 1.5625)e-4 = 3.675e-3; var = ss/3.
    returns = [0.01, -0.05, 0.03, -0.02]
    sigma = math.sqrt(3.675e-3 / 3)
    from quantfinlib.util import math_utils as mu
    expected = -(-0.0075 + mu.norm_inv(0.05) * sigma)
    assert rm.parametric_var(returns, 0.95) == pytest.approx(expected, abs=1e-12)


def test_conditional_var_pin():
    # threshold = 5th percentile = -0.0455 (see historical pin); only
    # -0.05 <= threshold -> CVaR = 0.05.
    assert rm.conditional_var([0.01, -0.05, 0.03, -0.02], 0.95) == pytest.approx(
        0.05, abs=1e-15)
    assert rm.expected_shortfall([0.01, -0.05, 0.03, -0.02], 0.95) == pytest.approx(
        0.05, abs=1e-15)


def test_max_drawdown_pin():
    # Peak 120, trough 90: (120-90)/120 = 0.25; later recovery is ignored.
    assert rm.max_drawdown([100.0, 120.0, 90.0, 105.0]) == pytest.approx(
        0.25, abs=1e-15)
    # Monotone equity has zero drawdown.
    assert rm.max_drawdown([100.0, 110.0, 120.0]) == 0.0


def test_beta_pin():
    # b = [0.01,-0.01,0.02,-0.02]: mean 0, var = 1e-3/3... sample:
    # ss = 1e-4+1e-4+4e-4+4e-4 = 1e-3, var = 1e-3/3.
    # a = 2*b exactly -> cov = 2*var -> beta = 2.
    b = [0.01, -0.01, 0.02, -0.02]
    a = [0.02, -0.02, 0.04, -0.04]
    assert rm.beta(a, b) == pytest.approx(2.0, abs=1e-12)
    # A flat benchmark yields beta 0, not a division blowup.
    assert rm.beta(a, [0.01, 0.01, 0.01, 0.01]) == 0.0


def test_correlation_delegates():
    a = [1.0, 2.0, 3.0]
    assert rm.correlation(a, [2.0, 4.0, 6.0]) == pytest.approx(1.0, abs=1e-12)
