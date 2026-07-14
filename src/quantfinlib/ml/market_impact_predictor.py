"""ML market impact prediction (port of Java ``ml.MarketImpactPredictor``).

Learns realized impact (bps) from order and book features using
gradient-boosted trees, and estimates the probability a marketable
order sweeps through the visible top of book.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from quantfinlib.ml.gradient_boosted_regressor import GradientBoostedRegressor


class MarketImpactPredictor:
    def __init__(self) -> None:
        self._model = GradientBoostedRegressor(150, 0.1)
        self._fitted = False

    @staticmethod
    def features(size_vs_adv: float, spread_bps: float, book_imbalance: float, volatility: float) -> np.ndarray:
        """Standard feature vector.

        :param size_vs_adv: order size / average daily volume
        :param spread_bps: quoted spread in bps at arrival
        :param book_imbalance: depth imbalance in [-1, 1] (signed toward the order side)
        :param volatility: recent per-period volatility
        """
        return np.array([size_vs_adv, spread_bps, book_imbalance, volatility])

    def fit(self, x: Sequence[Sequence[float]], realized_impact_bps: Sequence[float]) -> "MarketImpactPredictor":
        """Trains on historical (features, realized impact bps) observations."""
        self._model.fit(x, realized_impact_bps)
        self._fitted = True
        return self

    def predict_impact_bps(self, features: Sequence[float]) -> float:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        return self._model.predict(features)

    @staticmethod
    def sweep_probability(order_qty: int, visible_contra_depth: int) -> float:
        """Probability a marketable order of ``order_qty`` sweeps beyond
        the visible contra depth at the touch: logistic in the
        size/depth ratio, 0.5 exactly when the order equals the visible
        depth."""
        if visible_contra_depth <= 0:
            return 1.0
        ratio = order_qty / visible_contra_depth
        return 1.0 / (1 + math.exp(-4 * (ratio - 1)))
