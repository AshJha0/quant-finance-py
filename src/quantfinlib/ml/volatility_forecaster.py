"""Machine Learning Risk Forecasting (port of Java ``ml.VolatilityForecaster``).

Predicts forward realized volatility from a return series using
gradient-boosted trees over engineered features (multi-horizon
realized vol, momentum, and shock magnitude), and maps the forecast to
an intuitive 0-100 risk score.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from quantfinlib.ml.gradient_boosted_regressor import GradientBoostedRegressor
from quantfinlib.util import mean, std_dev_sample

_LOOKBACK = 21


class VolatilityForecaster:
    def __init__(self, horizon: int) -> None:
        """:param horizon: forward window (in periods) whose realized vol is predicted"""
        self._horizon = horizon
        self._model = GradientBoostedRegressor(150, 0.08)
        self._training_targets: np.ndarray = np.zeros(0)
        self._fitted = False

    @staticmethod
    def weekly() -> "VolatilityForecaster":
        return VolatilityForecaster(5)

    def fit(self, returns: Sequence[float]) -> "VolatilityForecaster":
        """Trains on a historical return series (needs at least ~3 months of data)."""
        returns = np.asarray(returns, dtype=float)
        xs: List[np.ndarray] = []
        ys: List[float] = []
        t = _LOOKBACK
        while t + self._horizon <= returns.shape[0]:
            xs.append(_features(returns, t))
            ys.append(std_dev_sample(returns, t, t + self._horizon))
            t += 1
        if len(xs) < 30:
            raise ValueError(
                f"insufficient history: {returns.shape[0]} returns for horizon {self._horizon}"
            )
        x = np.array(xs)
        y = np.array(ys)
        self._model.fit(x, y)
        self._training_targets = y
        self._fitted = True
        return self

    def forecast(self, returns: Sequence[float]) -> float:
        """Forecast of next-``horizon``-period volatility (per-period units)."""
        self._require_fitted()
        returns = np.asarray(returns, dtype=float)
        if returns.shape[0] < _LOOKBACK:
            raise ValueError(f"need at least {_LOOKBACK} returns")
        return max(0.0, self._model.predict(_features(returns, returns.shape[0])))

    def risk_score(self, returns: Sequence[float]) -> float:
        """Intelligent risk score in [0, 100]: the percentile of the
        forecast within the distribution of historically realized
        volatilities."""
        f = self.forecast(returns)
        below = int(np.count_nonzero(self._training_targets <= f))
        return 100.0 * below / self._training_targets.shape[0]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() first")


def _features(returns: np.ndarray, t: int) -> np.ndarray:
    """Feature vector computed from returns strictly before index t."""
    vol5 = std_dev_sample(returns, t - 5, t)
    vol10 = std_dev_sample(returns, t - 10, t)
    vol21 = std_dev_sample(returns, t - _LOOKBACK, t)
    mom5 = mean(returns, t - 5, t)
    mom21 = mean(returns, t - _LOOKBACK, t)
    last_abs = abs(returns[t - 1])
    max_abs5 = 0.0
    for i in range(t - 5, t):
        max_abs5 = max(max_abs5, abs(returns[i]))
    return np.array([vol5, vol10, vol21, mom5, mom21, last_abs, max_abs5])
