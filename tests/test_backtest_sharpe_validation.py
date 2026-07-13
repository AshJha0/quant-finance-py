"""Pins for quantfinlib.backtest.validation.sharpe_validation.

Java sources: ValidationTest.java (probabilistic / deflated Sharpe
behavior) and AlphaResearchRoundTest.java (minimum track record length
round trip). Trial Sharpes for the deflation test come from
np.random.default_rng noise (the Java SplittableRandom stream is not
reproduced; the pinned property is the haircut ordering, not the exact
value).
"""

import math

import numpy as np
import pytest

from quantfinlib.backtest.validation import SharpeValidation


def test_probabilistic_sharpe_behaves_sanely():
    # Strong Sharpe over a long track vs zero benchmark: near certainty.
    high = SharpeValidation.probabilistic_sharpe(0.15, 0, 1_000, 0, 3)
    assert high > 0.99
    # Same Sharpe over a tiny track: much less confident.
    low = SharpeValidation.probabilistic_sharpe(0.15, 0, 20, 0, 3)
    assert low < high
    # Negative skew and fat tails reduce confidence.
    ugly_tails = SharpeValidation.probabilistic_sharpe(0.15, 0, 1_000, -1.5, 8)
    assert ugly_tails < high
    with pytest.raises(ValueError):
        SharpeValidation.probabilistic_sharpe(0.15, 0, 1, 0, 3)  # n < 2


def test_deflated_sharpe_applies_multiple_testing_haircut():
    # 100 trials whose Sharpes are pure noise around zero.
    rng = np.random.default_rng(9)
    trials = 0.05 * rng.standard_normal(100)
    observed = 0.10   # the "winner"
    psr = SharpeValidation.probabilistic_sharpe(observed, 0, 500, 0, 3)
    dsr = SharpeValidation.deflated_sharpe(observed, trials, 500, 0, 3)
    # Deflation must cost confidence versus the naive zero benchmark.
    assert dsr < psr
    assert SharpeValidation.expected_max_sharpe(100, 0.0025) > 0
    with pytest.raises(ValueError):
        SharpeValidation.expected_max_sharpe(1, 0.0025)  # < 2 trials


def test_min_track_record_length_round_trips_through_probabilistic_sharpe():
    sr = 0.1                      # per period
    n = SharpeValidation.min_track_record_length(sr, 0, -0.5, 4, 0.95)
    assert 100 < n < 1_000        # a plausible track record
    # The round trip: AT the computed length, PSR clears the bar...
    assert SharpeValidation.probabilistic_sharpe(
        sr, 0, math.ceil(n), -0.5, 4) >= 0.95 - 1e-3
    # ...and meaningfully short of it, it does not.
    assert SharpeValidation.probabilistic_sharpe(
        sr, 0, int(n * 0.7), -0.5, 4) < 0.95
    # More confidence demands a longer record, always.
    assert SharpeValidation.min_track_record_length(sr, 0, -0.5, 4, 0.99) > n
    # No record length proves an edge the record does not show.
    assert SharpeValidation.min_track_record_length(0.0, 0.1, 0, 3, 0.95) == math.inf
    with pytest.raises(ValueError):
        SharpeValidation.min_track_record_length(0.1, 0, 0, 3, 1.0)
    with pytest.raises(ValueError):
        SharpeValidation.min_track_record_length(math.nan, 0, 0, 3, 0.95)
