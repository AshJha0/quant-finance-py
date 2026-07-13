"""Pins for quantfinlib.backtest.benchmark_comparison.

Java source: ValidationRobustnessTest.java (BenchmarkComparison
section) — the hand-computed half-beta case, the identical-series case,
the NaN-not-zero capture semantics and the input gates.
"""

import math

import pytest

from quantfinlib.backtest import BenchmarkComparison


def test_benchmark_comparison_matches_hand_computed_half_beta_case():
    # Strategy = 0.5 * benchmark + 10bp: beta is exactly 0.5, alpha is
    # exactly 10bp * 252, and the capture ratios are exact fractions.
    rb = [0.01, -0.01, 0.02, -0.02]
    rs = [0.5 * b + 0.001 for b in rb]
    r = BenchmarkComparison.compare(rs, rb, 252)

    assert r.beta == pytest.approx(0.5, abs=1e-12)
    assert r.alpha == pytest.approx(0.252, abs=1e-12)          # 0.001 * 252
    assert r.active_return == pytest.approx(0.252, abs=1e-12)
    # active = 0.001 - 0.5*rb -> sample var = 2.5e-4/3.
    te = math.sqrt(2.5e-4 / 3) * math.sqrt(252)
    assert r.tracking_error == pytest.approx(te, abs=1e-12)
    assert r.information_ratio == pytest.approx(0.252 / te, abs=1e-12)
    # Up periods: rs mean 0.0085, rb mean 0.015 -> 17/30.
    assert r.up_capture == pytest.approx(17.0 / 30.0, abs=1e-12)
    # Down periods: rs mean -0.0065, rb mean -0.015 -> 13/30.
    assert r.down_capture == pytest.approx(13.0 / 30.0, abs=1e-12)


def test_identical_series_has_unit_beta_zero_tracking_error():
    r = [0.01, -0.02, 0.015, 0.003, -0.007]
    res = BenchmarkComparison.compare(r, r, 252)
    assert res.beta == pytest.approx(1.0, abs=1e-12)
    assert res.alpha == pytest.approx(0.0, abs=1e-12)
    assert res.tracking_error == pytest.approx(0.0, abs=1e-12)
    assert res.information_ratio == 0.0    # 0/0 defined as 0, not NaN


def test_capture_is_nan_not_zero_when_benchmark_never_fell():
    rb = [0.01, 0.02, 0.01, 0.03]
    rs = [0.005, 0.01, 0.02, 0.01]
    r = BenchmarkComparison.compare(rs, rb, 252)
    assert math.isnan(r.down_capture)      # no evidence, not zero
    assert math.isfinite(r.up_capture)


def test_benchmark_comparison_refuses_bad_input():
    four = [0.01, -0.01, 0.02, -0.02]
    with pytest.raises(ValueError):
        BenchmarkComparison.compare([0.1, 0.2], four, 252)        # mismatch
    with pytest.raises(ValueError):
        BenchmarkComparison.compare(four, [0.01, 0.01, 0.01, 0.01], 252)  # flat
    with pytest.raises(ValueError):
        BenchmarkComparison.compare([0.01, math.nan, 0.02, -0.02], four, 252)
    with pytest.raises(ValueError):
        BenchmarkComparison.compare(four, four, 0)                # bad periods
    with pytest.raises(ValueError):
        BenchmarkComparison.compare(four[:2], four[:2], 252)      # too short
