"""Analytics over Monte Carlo terminal portfolio values (port of Java
``simulation.SimulationResult``).

Probabilities, VaR/CVaR, confidence intervals and scenario extremes.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from quantfinlib.util import mean, percentile_sorted


class SimulationResult:
    def __init__(self, initial_value: float, final_values: Sequence[float]) -> None:
        self._initial_value = initial_value
        self._sorted_finals = np.sort(np.asarray(final_values, dtype=float))

    def initial_value(self) -> float:
        return self._initial_value

    def simulations(self) -> int:
        return self._sorted_finals.shape[0]

    def probability_of_profit(self) -> float:
        wins = int(np.count_nonzero(self._sorted_finals > self._initial_value))
        return wins / self._sorted_finals.shape[0]

    def probability_of_loss(self) -> float:
        return 1 - self.probability_of_profit()

    def value_at_risk(self, confidence: float) -> float:
        """VaR at the given confidence as a positive loss fraction of the
        initial value."""
        q = percentile_sorted(self._sorted_finals, 1 - confidence)
        return max(0.0, (self._initial_value - q) / self._initial_value)

    def conditional_value_at_risk(self, confidence: float) -> float:
        """CVaR: average loss fraction in the tail beyond the VaR quantile."""
        n = self._sorted_finals.shape[0]
        tail = max(1, int(np.floor((1 - confidence) * n)))
        avg_tail = float(np.sum(self._sorted_finals[:tail])) / tail
        return max(0.0, (self._initial_value - avg_tail) / self._initial_value)

    def confidence_interval(self, level: float) -> Tuple[float, float]:
        """Two-sided confidence interval of terminal value, e.g. level =
        0.90 -> [p5, p95]."""
        alpha = (1 - level) / 2
        return (
            percentile_sorted(self._sorted_finals, alpha),
            percentile_sorted(self._sorted_finals, 1 - alpha),
        )

    def best_case(self) -> float:
        return float(self._sorted_finals[-1])

    def worst_case(self) -> float:
        return float(self._sorted_finals[0])

    def expected_value(self) -> float:
        return mean(self._sorted_finals)

    def median_value(self) -> float:
        return percentile_sorted(self._sorted_finals, 0.5)

    def __str__(self) -> str:
        return (
            f"MonteCarlo[{self.simulations()} sims]: expected={self.expected_value():.2f}, "
            f"median={self.median_value():.2f}, pProfit={self.probability_of_profit() * 100:.1f}%, "
            f"VaR95={self.value_at_risk(0.95) * 100:.2f}%, "
            f"CVaR95={self.conditional_value_at_risk(0.95) * 100:.2f}%, "
            f"best={self.best_case():.2f}, worst={self.worst_case():.2f}"
        )
