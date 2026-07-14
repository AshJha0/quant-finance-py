"""Gradient-boosted regression over decision stumps (port of Java
``ml.GradientBoostedRegressor``).

XGBoost-style additive boosting with squared-error loss, implemented
in pure Python/NumPy with no ML dependencies. Suited to small/medium
tabular problems such as risk forecasting features.

Deterministic by construction: unlike many GBDT implementations, this
one has no subsampling and no randomized feature selection -- each
round's stump is the exact SSE-optimal single split found by a full
sorted-prefix-sum sweep over every feature. There is therefore no RNG
seeding to reproduce from the Java reference; the same ``(x, y)``
input always yields the same trained stumps in both ports, bit for
bit, modulo floating-point summation order (this port sums with
``numpy`` in the same left-to-right order as the Java loop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class _Stump:
    feature: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, x: np.ndarray) -> float:
        return self.left_value if x[self.feature] <= self.threshold else self.right_value


class GradientBoostedRegressor:
    """Gradient-boosted regression over decision stumps."""

    def __init__(self, rounds: int, learning_rate: float) -> None:
        self._rounds = rounds
        self._learning_rate = learning_rate
        self._stumps: List[_Stump] = []
        self._baseline = 0.0
        self._fitted = False

    @staticmethod
    def with_defaults() -> "GradientBoostedRegressor":
        return GradientBoostedRegressor(200, 0.1)

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> "GradientBoostedRegressor":
        """Fits the model on ``x[sample][feature]`` / ``y[sample]``."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = y.shape[0]
        if n == 0 or x.shape[0] != n:
            raise ValueError("x/y size mismatch or empty")
        self._stumps = []
        self._baseline = float(np.mean(y))
        pred = np.full(n, self._baseline)

        for _ in range(self._rounds):
            residual = y - pred
            best = _best_stump(x, residual)
            if best is None:
                break
            self._stumps.append(best)
            for i in range(n):
                pred[i] += self._learning_rate * best.predict(x[i])
        self._fitted = True
        return self

    def predict(self, x: Sequence[float]) -> float:
        if not self._fitted:
            raise RuntimeError("model not fitted")
        x = np.asarray(x, dtype=float)
        p = self._baseline
        for s in self._stumps:
            p += self._learning_rate * s.predict(x)
        return p

    def predict_all(self, x: Sequence[Sequence[float]]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.array([self.predict(row) for row in x])

    def rmse(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> float:
        """Root mean squared error on a labeled set."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        s = 0.0
        for i in range(y.shape[0]):
            d = self.predict(x[i]) - y[i]
            s += d * d
        return math.sqrt(s / y.shape[0])


def _best_stump(x: np.ndarray, residual: np.ndarray) -> Optional[_Stump]:
    """Finds the SSE-optimal stump over all features via sorted prefix sums."""
    n = residual.shape[0]
    features = x.shape[1]
    total_sum = float(np.sum(residual))

    best_gain = 1e-12
    best: Optional[_Stump] = None

    for f in range(features):
        order = np.argsort(x[:, f], kind="stable")
        left_sum = 0.0
        for k in range(n - 1):
            left_sum += residual[order[k]]
            # Only split between distinct feature values.
            if x[order[k], f] == x[order[k + 1], f]:
                continue
            left_n = k + 1
            right_n = n - left_n
            right_sum = total_sum - left_sum
            # Variance-reduction gain of the split (up to constants).
            gain = left_sum * left_sum / left_n + right_sum * right_sum / right_n - total_sum * total_sum / n
            if gain > best_gain:
                best_gain = gain
                threshold = (x[order[k], f] + x[order[k + 1], f]) / 2
                best = _Stump(f, threshold, left_sum / left_n, right_sum / right_n)
    return best
