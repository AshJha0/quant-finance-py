"""Purged K-fold splits with an embargo (port of Java ``validation.PurgedKFold``).

The fix for the quiet leak that ordinary K-fold has on financial data
(Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7).

The leak: a sample's LABEL is usually computed from bars that come after
it (e.g. "the 5-bar forward return"). Ordinary K-fold happily puts bar
99 in the training set and bar 100 in the test set — but bar 99's label
was computed from bars 100-104, so the model has already seen the test
answer. The backtest looks skillful; the skill is leakage.

Two defenses, both index arithmetic:

* **Purging** removes every training sample whose label window
  ``[i, i + label_horizon]`` overlaps any test label window. For a
  contiguous test fold ``[t0, t1)`` that means dropping training indices
  in ``[t0 - label_horizon, t0)`` (labels reach INTO the fold) and
  ``[t1, t1 + label_horizon)`` (labels reach OUT of it).
* **Embargo** drops a further ``embargo`` samples after the purge zone
  that follows the test fold. Serial correlation means features just
  after the test window still echo test-period information even when
  the label windows don't overlap; the embargo is the buffer for that
  echo. A common choice is ~1% of n.

So the training set for test fold ``[t0, t1)`` is exactly
``[0, t0 - label_horizon) U [t1 + label_horizon + embargo, n)`` —
hand-checkable, and the tests do. Every fold's training set must be
non-empty or the split refuses: silently training on nothing is how
"great" fold scores happen.

Static, deterministic, research lane. Pair with
:mod:`~quantfinlib.backtest.validation.overfit_probability` (is the
SELECTION process overfit?).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PurgedKFold:
    """Static split construction; see the module docstring."""

    @dataclass(frozen=True, eq=False)
    class Split:
        """One fold: test on ``[test_from, test_to)``, train on
        ``train_indices`` (ascending, purged and embargoed)."""

        fold: int
        test_from: int
        test_to: int
        train_indices: np.ndarray

    @staticmethod
    def splits(n: int, k: int, label_horizon: int,
               embargo: int) -> list["PurgedKFold.Split"]:
        """Builds the k purged/embargoed splits over ``n`` samples.

        Args:
            n: Number of samples (bars/observations), >= 2k.
            k: Number of folds, >= 2.
            label_horizon: Bars each label looks ahead (0 = label known
                at the sample's own bar; 5 = 5-bar forward return).
            embargo: Extra bars dropped after the post-test purge zone.

        Raises:
            ValueError: on a degenerate setup or a fold whose training
                set would be empty.
        """
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        if n < 2 * k:
            raise ValueError(f"need n >= 2k samples: n={n} k={k}")
        if label_horizon < 0 or embargo < 0:
            raise ValueError("labelHorizon and embargo must be >= 0, got "
                             f"{label_horizon} and {embargo}")
        out: list[PurgedKFold.Split] = []
        for f in range(k):
            t0 = f * n // k
            t1 = (f + 1) * n // k
            head_end = max(0, t0 - label_horizon)              # [0, head_end)
            tail_start = min(n, t1 + label_horizon + embargo)  # [tail_start, n)
            size = head_end + (n - tail_start)
            if size == 0:
                raise ValueError(f"fold {f} leaves no training data: n={n} k={k} "
                                 f"labelHorizon={label_horizon} embargo={embargo}")
            train = np.concatenate([np.arange(head_end),
                                    np.arange(tail_start, n)])
            out.append(PurgedKFold.Split(f, t0, t1, train))
        return out
