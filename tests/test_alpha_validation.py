"""Pins for alpha factor validation (walk-forward, blocked
cross-validation, Monte Carlo permutation robustness, parameter
sensitivity).

Java source: AlphaValidation.java / AlphaValidationTest.java. This
module had no Python test coverage before this pin — it is exercised
only incidentally (``AlphaValidation.mean_ic``) by test_alpha_report.py.
"""

import math

import numpy as np
import pytest

from quantfinlib.alpha.alpha_context import AlphaContext
from quantfinlib.alpha.alpha_validation import AlphaValidation
from quantfinlib.alpha.factors import Factors
from quantfinlib.data.bar_series import BarSeries

BARS = 260
# Persistent per-symbol geometric drifts: momentum ranks every symbol
# identically at every bar (the drift ratio over any fixed lookback is
# constant), so any lookback should score essentially perfectly in and
# out of sample.
DRIFTS = (0.004, 0.002, 0.001, -0.001, -0.002, -0.004)


def _drift_panel() -> AlphaContext:
    data = {}
    for s, drift in enumerate(DRIFTS):
        b = BarSeries.builder(f"S{s}")
        close = 100.0
        for i in range(BARS):
            open_ = close
            close = 100 * (1 + drift) ** (i + 1)
            b.add(i, open_, max(open_, close), min(open_, close), close, 500_000)
        data[f"S{s}"] = b.build()
    return AlphaContext.of(data)


def _zigzag_panel() -> AlphaContext:
    """Synchronized zigzags with distinct amplitudes and shuffled tiny
    drifts (same construction as test_alpha_report._zigzag_panel):
    mean-reversion at horizon 1 predicts the flip almost exactly, so
    scores vary bar-to-bar in lockstep with the next return -- a
    permutation test should find this "trivially" significant."""
    amps = (0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01)
    drift_ranks = (4, 8, 2, 6, 1, 5, 3, 7)
    drifts = [(r - 4.5) * 1e-4 for r in drift_ranks]
    data = {}
    for s, amp in enumerate(amps):
        b = BarSeries.builder(f"S{s}")
        for i in range(120):
            close = (100 * (1 + drifts[s]) ** i
                    * (1 + amp * (1 if i % 2 == 0 else -1)))
            b.add(i, close, close * 1.0001, close * 0.9999, close, 1_000)
        data[f"S{s}"] = b.build()
    return AlphaContext.of(data)


# ------------------------------------------------------------------
# Walk-forward
# ------------------------------------------------------------------

def test_walk_forward_persistent_signal_transfers_is_to_oos():
    ctx = _drift_panel()
    candidates = [Factors.momentum(lb, 0) for lb in (5, 20, 60)]
    result = AlphaValidation.walk_forward(ctx, candidates, horizon=5,
                                          start_index=65, train_bars=100,
                                          test_bars=50)
    assert len(result.folds) > 0
    for fold in result.folds:
        # Model selection is strictly in-sample: chosen factor's train
        # window ends exactly where the test window starts.
        assert fold.test_start == fold.train_start + 100
        assert fold.test_end == fold.test_start + 50
        assert fold.chosen_factor in {"MOMENTUM(5-0)", "MOMENTUM(20-0)",
                                     "MOMENTUM(60-0)"}
    # A perfectly persistent cross-sectional ranking: both IS and OOS
    # rank correlation should be (near) perfect, so efficiency ~ 1.
    assert result.mean_in_sample_ic == pytest.approx(1.0, abs=1e-6)
    assert result.mean_out_of_sample_ic == pytest.approx(1.0, abs=1e-6)
    assert result.efficiency() == pytest.approx(1.0, abs=1e-6)


def test_walk_forward_rejects_bad_arguments():
    ctx = _drift_panel()
    mom = Factors.momentum(20, 0)
    with pytest.raises(ValueError):
        AlphaValidation.walk_forward(ctx, [], horizon=5, start_index=65,
                                     train_bars=100, test_bars=50)
    with pytest.raises(ValueError):
        # train_bars <= horizon
        AlphaValidation.walk_forward(ctx, [mom], horizon=10, start_index=65,
                                     train_bars=10, test_bars=50)
    with pytest.raises(ValueError):
        # test_bars <= horizon
        AlphaValidation.walk_forward(ctx, [mom], horizon=10, start_index=65,
                                     train_bars=100, test_bars=5)
    with pytest.raises(ValueError):
        # sample too short for even one fold
        AlphaValidation.walk_forward(ctx, [mom], horizon=5, start_index=65,
                                     train_bars=1000, test_bars=1000)


# ------------------------------------------------------------------
# Blocked cross-validation
# ------------------------------------------------------------------

def test_cross_validate_blocks_and_sign_consistency():
    ctx = _drift_panel()
    mom = Factors.momentum(20, 0)
    result = AlphaValidation.cross_validate(ctx, mom, horizon=5,
                                            start_index=65, k=4)
    assert result.block_ics.shape[0] == 4
    # Persistent ranking: every block agrees with the overall sign.
    assert result.sign_consistency() == pytest.approx(1.0)
    assert result.mean_ic == pytest.approx(1.0, abs=1e-6)
    assert result.ic_std == pytest.approx(0.0, abs=1e-6)


def test_cross_validate_rejects_short_blocks():
    ctx = _drift_panel()
    mom = Factors.momentum(20, 0)
    with pytest.raises(ValueError):
        AlphaValidation.cross_validate(ctx, mom, horizon=5, start_index=65, k=1)
    with pytest.raises(ValueError):
        # k too large: span // k <= horizon
        AlphaValidation.cross_validate(ctx, mom, horizon=5, start_index=65,
                                       k=1000)


# ------------------------------------------------------------------
# Monte Carlo robustness
# ------------------------------------------------------------------

def test_monte_carlo_robustness_flags_time_varying_signal_as_significant():
    ctx = _zigzag_panel()
    mean_rev = Factors.mean_reversion(2)
    result = AlphaValidation.monte_carlo_robustness(
        ctx, mean_rev, horizon=1, start_index=10, trials=200, seed=7)
    assert result.observed_mean_ic > 0.9
    # Time-varying signal aligned with returns: permutation destroys the
    # link almost every trial, so the observed IC is far in the tail.
    assert result.p_value < 0.05
    assert result.trials == 200
    # Add-one smoothing: never exactly 0.
    assert result.p_value > 0.0


def test_monte_carlo_robustness_static_ranking_is_never_significant():
    # Deliberate conservatism (see module docstring): a persistent,
    # time-invariant cross-sectional ranking is invariant under date
    # permutation, so it must NOT read as significant however strong
    # its raw IC is.
    ctx = _drift_panel()
    mom = Factors.momentum(20, 0)
    result = AlphaValidation.monte_carlo_robustness(
        ctx, mom, horizon=5, start_index=65, trials=200, seed=11)
    assert result.observed_mean_ic == pytest.approx(1.0, abs=1e-6)
    assert result.p_value > 0.5


def test_monte_carlo_robustness_rejects_too_few_trials():
    ctx = _drift_panel()
    mom = Factors.momentum(20, 0)
    with pytest.raises(ValueError):
        AlphaValidation.monte_carlo_robustness(ctx, mom, horizon=5,
                                               start_index=65, trials=5,
                                               seed=1)


# ------------------------------------------------------------------
# Parameter sensitivity
# ------------------------------------------------------------------

def test_parameter_sensitivity_reports_best_and_worst_drop():
    ctx = _drift_panel()
    sweep = [Factors.momentum(lb, 0) for lb in (5, 10, 20, 40, 60)]
    result = AlphaValidation.parameter_sensitivity(ctx, sweep, horizon=5,
                                                   start_index=65)
    assert result.names == ("MOMENTUM(5-0)", "MOMENTUM(10-0)",
                           "MOMENTUM(20-0)", "MOMENTUM(40-0)",
                           "MOMENTUM(60-0)")
    assert len(result.mean_ics) == 5
    # Persistent ranking: every lookback scores near-perfectly, so the
    # sweep is a plateau -- worst drop should be tiny.
    assert result.worst_neighbor_drop < 1e-6
    assert result.best() in result.names


def test_parameter_sensitivity_rejects_short_sweep():
    ctx = _drift_panel()
    with pytest.raises(ValueError):
        AlphaValidation.parameter_sensitivity(
            ctx, [Factors.momentum(20, 0)], horizon=5, start_index=65)


# ------------------------------------------------------------------
# Shared helper
# ------------------------------------------------------------------

def test_mean_ic_matches_across_equivalent_windows():
    ctx = _drift_panel()
    mom = Factors.momentum(20, 0)
    ic_full = AlphaValidation.mean_ic(ctx, mom, 65, ctx.bars(), 5)
    assert ic_full == pytest.approx(1.0, abs=1e-6)
    assert not math.isnan(ic_full)
