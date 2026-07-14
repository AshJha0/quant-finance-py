"""Monte Carlo Portfolio Simulation (port of Java ``simulation.MonteCarloSimulator``).

Runs tens of thousands to hundreds of thousands of GBM scenarios;
results are deterministic for a given seed regardless of execution
order -- each path gets its own :class:`~quantfinlib.sim._java_random.SplittableRandom64`
seeded from ``mix(seed, path_index)``, a bit-exact reproduction of the
Java reference's ``java.util.SplittableRandom`` per-path seeding (see
``_java_random.py`` for the verification against real JVM output).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from quantfinlib.sim._java_random import SplittableRandom64, mix_seed
from quantfinlib.sim.simulation_result import SimulationResult
from quantfinlib.util import cholesky


class MonteCarloSimulator:
    def __init__(self, seed: int = 7) -> None:
        self._seed = seed

    def simulate(
        self,
        initial_value: float,
        annual_return: float,
        annual_vol: float,
        horizon_days: int,
        simulations: int,
    ) -> SimulationResult:
        """Single-asset / whole-portfolio GBM simulation with daily steps.

        :param initial_value: starting portfolio value
        :param annual_return: annualized drift (e.g. 0.08)
        :param annual_vol: annualized volatility (e.g. 0.15)
        :param horizon_days: trading days to simulate
        :param simulations: number of scenarios (10_000+ recommended)
        """
        dt = 1.0 / 252
        drift = (annual_return - 0.5 * annual_vol * annual_vol) * dt
        diffusion = annual_vol * math.sqrt(dt)

        finals = np.zeros(simulations)
        for s in range(simulations):
            rnd = SplittableRandom64(mix_seed(self._seed, s))
            log_v = math.log(initial_value)
            for _ in range(horizon_days):
                log_v += drift + diffusion * _gaussian(rnd)
            finals[s] = math.exp(log_v)
        return SimulationResult(initial_value, finals)

    def simulate_portfolio(
        self,
        initial_value: float,
        weights: Sequence[float],
        daily_mean_returns: Sequence[float],
        daily_covariance,
        horizon_days: int,
        simulations: int,
    ) -> SimulationResult:
        """Correlated multi-asset portfolio simulation using daily mean
        returns and daily covariance (e.g. estimated from historical
        returns)."""
        weights = np.asarray(weights, dtype=float)
        daily_mean_returns = np.asarray(daily_mean_returns, dtype=float)
        k = weights.shape[0]
        chol = cholesky(np.asarray(daily_covariance, dtype=float))

        finals = np.zeros(simulations)
        for s in range(simulations):
            rnd = SplittableRandom64(mix_seed(self._seed, s))
            value = initial_value
            z = np.zeros(k)
            for _ in range(horizon_days):
                for i in range(k):
                    z[i] = _gaussian(rnd)
                port_return = 0.0
                for i in range(k):
                    r = daily_mean_returns[i]
                    for j in range(i + 1):
                        r += chol[i][j] * z[j]
                    port_return += weights[i] * r
                value *= 1 + port_return
                if value <= 0:
                    value = 0.0
                    break
            finals[s] = value
        return SimulationResult(initial_value, finals)


def _gaussian(rnd: SplittableRandom64) -> float:
    """Marsaglia polar method -- ~2x faster than Box-Muller with trig calls."""
    while True:
        u = 2 * rnd.next_double() - 1
        v = 2 * rnd.next_double() - 1
        s = u * u + v * v
        if 0 < s < 1:
            return u * math.sqrt(-2 * math.log(s) / s)
