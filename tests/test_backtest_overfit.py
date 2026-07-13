"""Pins for quantfinlib.backtest.validation.overfit_probability.

Java source: ValidationRobustnessTest.java (OverfitProbability section)
— genuine skill scores PBO 0 with every logit ln 2, regime-flipping
noise scores PBO 1, the Sharpe objective's flat-series handling and the
C(8,4) = 70 combination count, plus the input gates.
"""

import math

import numpy as np
import pytest

from quantfinlib.backtest.validation import OverfitProbability
from quantfinlib.util import math_utils as mu


def test_genuine_skill_scores_pbo_zero():
    # Variant 0 beats variant 1 in EVERY period: whatever half is
    # in-sample, the winner is 0 and it ranks 2 of 2 out of sample.
    # omega = 2/3 on every one of C(4,2)=6 splits -> lambda = ln 2 > 0.
    r = np.tile([0.01, -0.01], (16, 1))
    res = OverfitProbability.cscv(r, 4, mu.mean)
    assert res.combinations == 6
    assert res.pbo == 0.0
    for lam in res.logits:
        assert lam == pytest.approx(math.log(2), abs=1e-12)


def test_regime_flipping_noise_scores_pbo_one():
    # Variant 0 wins the first half of time, variant 1 (its mirror) wins
    # the second half. Whatever blocks you train on, the winner does no
    # better than tie out of sample: every logit <= 0, PBO = 1.
    # +/-1.0 (not 0.01) so mixed-block means cancel EXACTLY in floating
    # point and ties are true ties, not 1e-18 rank noise.
    r = np.empty((16, 2))
    r[:8, 0] = 1.0
    r[8:, 0] = -1.0
    r[:, 1] = -r[:, 0]
    res = OverfitProbability.cscv(r, 4, mu.mean)
    assert res.combinations == 6
    assert res.pbo == 1.0


def test_sharpe_objective_handles_flat_series_and_counts_combinations():
    # 3 variants, 8 blocks: C(8,4) = 70 combinations. Variant 2 is flat
    # (zero variance) — the Sharpe objective scores it 0 rather than
    # dividing by zero.
    r = np.empty((32, 3))
    for t in range(32):
        r[t, 0] = ((t * 7 + 3) % 5 - 2) / 100.0
        r[t, 1] = ((t * 11 + 1) % 7 - 3) / 100.0
        r[t, 2] = 0.0
    res = OverfitProbability.cscv_sharpe(r, 8)
    assert res.combinations == 70
    assert 0 <= res.pbo <= 1


def test_non_finite_objective_is_rejected():
    r = np.tile([0.01, -0.01], (16, 1))
    with pytest.raises(ValueError):
        OverfitProbability.cscv(r, 4, lambda s: math.inf)


def test_cscv_validates_its_inputs():
    ok = np.zeros((16, 2))
    with pytest.raises(ValueError):
        OverfitProbability.cscv_sharpe(ok, 5)                  # odd
    with pytest.raises(ValueError):
        OverfitProbability.cscv_sharpe(ok, 18)                 # > cap
    with pytest.raises(ValueError):
        OverfitProbability.cscv_sharpe(np.zeros((16, 1)), 4)   # one variant
    with pytest.raises(ValueError):
        OverfitProbability.cscv_sharpe(np.zeros((6, 2)), 4)    # too short
    with pytest.raises(ValueError):
        OverfitProbability.cscv_sharpe([[0.1, 0.2], [0.1]], 4)  # ragged
    nan = np.zeros((16, 2))
    nan[3, 1] = math.nan
    with pytest.raises(ValueError):
        OverfitProbability.cscv_sharpe(nan, 4)                 # non-finite
