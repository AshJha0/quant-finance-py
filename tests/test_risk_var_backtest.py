"""Pins for quantfinlib.risk.var_backtest, ported from Java VarBacktestTest.

Gaussian return generation uses numpy's default_rng instead of Java's
SplittableRandom; every assertion is a statistical pass/fail verdict
with wide Java tolerances (e.g. exceptions within +/-30 of 100, ~3
sigma), so the stream change cannot flip them. Module-qualified calls
(vb.test) keep pytest from collecting the ported Java method name."""

import math

import numpy as np
import pytest

from quantfinlib.risk import var_backtest as vb


def _gaussian_returns(n, vol, seed):
    return vol * np.random.default_rng(seed).standard_normal(n)


def test_well_calibrated_var_passes_all_tests():
    returns = _gaussian_returns(2_000, 0.01, 42)
    # True 95% VaR of N(0, 1%) is 1.645%.
    r = vb.test(returns, 0.01645, 0.95)
    assert r.observations == 2_000
    assert r.expected_exceptions == pytest.approx(100, abs=1e-9)
    assert abs(r.exceptions - 100) < 30, f"exceptions={r.exceptions}"
    assert r.calibrated(0.05), f"kupiec p={r.kupiec_p_value}"
    assert r.independent(0.05), f"independence p={r.independence_p_value}"
    assert r.passes(0.05), f"cc p={r.conditional_coverage_p_value}"
    # Exact chi-square(2) survival for the joint statistic.
    assert r.conditional_coverage_p_value == pytest.approx(
        math.exp(-r.conditional_coverage_statistic / 2), abs=1e-12)


def test_underestimated_var_is_rejected():
    returns = _gaussian_returns(2_000, 0.01, 7)
    # Claiming 95% coverage with roughly a 68% quantile: far too many exceptions.
    r = vb.test(returns, 0.010, 0.95)
    assert r.exceptions > 200
    assert not r.calibrated(0.01), f"kupiec p={r.kupiec_p_value}"
    assert not r.passes(0.01)


def test_overestimated_var_is_also_rejected():
    # Kupiec is two-sided: a VaR so conservative it never breaks also fails.
    returns = _gaussian_returns(2_000, 0.01, 9)
    r = vb.test(returns, 0.05, 0.95)
    assert r.exceptions == 0
    assert not r.calibrated(0.01), f"kupiec p={r.kupiec_p_value}"


def test_clustered_exceptions_fail_independence():
    # Two series with identical exception COUNTS (25/500) but different
    # timing. Deterministic — ported exactly from Java.
    n = 500
    clustered = np.zeros(n)
    scattered = np.zeros(n)
    for i in range(25):
        clustered[100 + i] = -0.05        # one 25-day crisis
        scattered[i * 20] = -0.05         # evenly spread
    bad = vb.test(clustered, 0.02, 0.95)
    good = vb.test(scattered, 0.02, 0.95)
    assert bad.exceptions == good.exceptions
    assert bad.kupiec_statistic == pytest.approx(
        good.kupiec_statistic, abs=1e-9)   # same rate
    assert not bad.independent(0.01), \
        f"clustered independence p={bad.independence_p_value}"
    assert good.independent(0.05), \
        f"scattered independence p={good.independence_p_value}"
    assert not bad.passes(0.01)   # conditional coverage catches the clustering


def test_per_period_forecasts_are_supported():
    # GARCH-style varying forecasts: scale VaR with a known vol path.
    rng = np.random.default_rng(11)
    n = 1_000
    vols = np.where(np.arange(n) < 500, 0.005, 0.02)   # regime shift
    returns = vols * rng.standard_normal(n)
    var = 1.645 * vols                                  # correctly scaled
    adaptive = vb.test(returns, var, 0.95)
    assert adaptive.passes(0.05), \
        f"adaptive cc p={adaptive.conditional_coverage_p_value}"
    # A constant VaR calibrated to the calm regime fails on the same data.
    constant = vb.test(returns, 1.645 * 0.005, 0.95)
    assert not constant.passes(0.01)


def test_kupiec_statistic_hand_pin():
    # 20 observations, 0 exceptions at 95%: LR = 2*(0 - 20*ln(0.95))
    # = -40*ln(0.95) (the alternative likelihood is 20*ln(1) = 0).
    returns = np.full(20, 0.01)
    r = vb.test(returns, 0.05, 0.95)
    assert r.exceptions == 0
    assert r.kupiec_statistic == pytest.approx(-40 * math.log(0.95), abs=1e-12)
    # No exceptions -> the independence LR is degenerate and reports 0/p=1.
    assert r.independence_statistic == 0.0
    assert r.independence_p_value == 1.0


def test_gates():
    with pytest.raises(ValueError):
        vb.test(np.zeros(19), 0.01, 0.95)                # too short
    with pytest.raises(ValueError):
        vb.test(np.zeros(30), np.zeros(29), 0.95)        # misaligned
    with pytest.raises(ValueError):
        vb.test(np.zeros(30), 0.01, 1.0)                 # confidence gate
