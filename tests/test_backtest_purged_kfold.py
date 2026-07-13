"""Pins for quantfinlib.backtest.validation.purged_kfold.

Java source: ValidationRobustnessTest.java (PurgedKFold section) — the
hand-derived index arithmetic, the plain K-fold reduction, and the
degenerate-setup gates. Every expected value is derivable on paper from
train = [0, t0-h) U [t1+h+embargo, n).
"""

import numpy as np
import pytest

from quantfinlib.backtest.validation import PurgedKFold


def test_purged_splits_match_hand_derived_index_arithmetic():
    # n=20, k=4, labelHorizon=2, embargo=1. Fold 1 tests [5,10):
    # head = [0, 5-2) = {0,1,2}; tail = [10+2+1, 20) = {13..19}.
    splits = PurgedKFold.splits(20, 4, 2, 1)
    assert len(splits) == 4

    f1 = splits[1]
    assert f1.test_from == 5
    assert f1.test_to == 10
    assert np.array_equal(f1.train_indices,
                          [0, 1, 2, 13, 14, 15, 16, 17, 18, 19])

    # Fold 0 has no head at all: train = [0+5+2+1, 20) = {8..19}.
    f0 = splits[0]
    assert np.array_equal(f0.train_indices,
                          [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

    # Test ranges partition [0, 20) with no gaps.
    covered = sum(s.test_to - s.test_from for s in splits)
    assert covered == 20


def test_zero_horizon_zero_embargo_reduces_to_plain_kfold():
    # With nothing to purge, train must be the exact complement of test.
    for s in PurgedKFold.splits(12, 3, 0, 0):
        assert len(s.train_indices) == 12 - (s.test_to - s.test_from)
        for i in s.train_indices:
            assert i < s.test_from or i >= s.test_to


def test_purged_kfold_refuses_degenerate_setups():
    with pytest.raises(ValueError):
        PurgedKFold.splits(20, 1, 0, 0)     # k < 2
    with pytest.raises(ValueError):
        PurgedKFold.splits(5, 3, 0, 0)      # n < 2k
    with pytest.raises(ValueError):
        PurgedKFold.splits(20, 4, -1, 0)    # negative horizon
    # Horizon so wide the purge eats ALL training data must throw, not
    # silently return an empty training set.
    with pytest.raises(ValueError):
        PurgedKFold.splits(8, 2, 10, 0)
