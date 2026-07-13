"""Portfolio optimization engine (port of Java ``optimization.PortfolioOptimizer``).

Long-only, fully invested. Supports maximum-Sharpe and minimum-volatility
portfolios plus efficient frontier construction. The optimizer uses
stochastic search over the simplex (Dirichlet sampling) followed by
deterministic pairwise-transfer refinement, which is robust for
non-smooth objectives and needs no external solver. Results are
deterministic for a given seed via ``np.random.default_rng`` (port note:
the Java reference samples from ``SplittableRandom``; the RNG stream is
not reproduced across ports — the pinned checks are the analytic
solutions the search must find within tolerance).

Expected returns and covariance must share the same periodicity (e.g.
both annualized, or both daily).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, eq=False)
class Allocation:
    """An optimized allocation with its risk/return profile
    (same periodicity as the inputs)."""

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float


class PortfolioOptimizer:
    """Long-only fully-invested optimizer; see the module docstring."""

    SAMPLES = 20_000
    REFINE_SWEEPS = 60

    def __init__(self, expected_returns, covariance, seed: int = 42):
        """Configures the optimizer.

        Raises:
            ValueError: on a returns/covariance dimension mismatch.
        """
        mu = np.array(expected_returns, dtype=float)
        cov = np.asarray(covariance, dtype=float)
        if mu.shape[0] != cov.shape[0]:
            raise ValueError("returns/covariance dimension mismatch")
        self._mu = mu
        self._cov = cov
        self._seed = seed

    def max_sharpe(self, risk_free_rate: float) -> Allocation:
        """Maximum Sharpe ratio portfolio. ``risk_free_rate`` in the same
        periodicity as the inputs."""
        def objective(w: np.ndarray) -> np.ndarray:
            vols = self._vols(w)
            rets = w @ self._mu
            safe = np.where(vols > 0, vols, 1.0)
            return np.where(vols > 0, (rets - risk_free_rate) / safe, -np.inf)
        return self._to_allocation(self._optimize(objective), risk_free_rate)

    def min_volatility(self) -> Allocation:
        """Minimum volatility portfolio."""
        return self._to_allocation(self._optimize(lambda w: -self._vols(w)), 0.0)

    def efficient_frontier(self, points: int) -> list[Allocation]:
        """Efficient frontier: minimum-volatility portfolios across a grid
        of target returns between the min and max asset expected returns."""
        lo = float(np.min(self._mu))
        hi = float(np.max(self._mu))
        frontier: list[Allocation] = []
        for p in range(points):
            target = lo + (hi - lo) * p / max(1, points - 1)

            # Penalized objective: minimize vol subject to hitting the target.
            def objective(w: np.ndarray, target: float = target) -> np.ndarray:
                shortfall = target - w @ self._mu
                penalty = np.where(shortfall > 0, shortfall * 100, 0.0)
                return -(self._vols(w) + penalty)
            frontier.append(self._to_allocation(self._optimize(objective), 0.0))
        return frontier

    @staticmethod
    def rebalance(current_weights, target_weights) -> np.ndarray:
        """Rebalancing deltas: target minus current weights, per asset."""
        return (np.asarray(target_weights, dtype=float)
                - np.asarray(current_weights, dtype=float))

    # ------------------------------------------------------------------

    def _optimize(self, objective: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Dirichlet sampling then pairwise-transfer hill climbing.

        ``objective`` is vectorized: it maps an (m, n) weight matrix to m
        scores (higher is better).
        """
        n = self._mu.shape[0]
        rng = np.random.default_rng(self._seed)

        # Phase 1: Dirichlet(1,...,1) sampling over the simplex
        # (normalized exponentials, as in the Java -log(U) draws).
        best = np.full(n, 1.0 / n)
        best_score = float(objective(best[None, :])[0])
        cand = rng.standard_exponential((self.SAMPLES, n))
        cand /= cand.sum(axis=1, keepdims=True)
        scores = objective(cand)
        k = int(np.argmax(scores))
        if scores[k] > best_score:
            best_score = float(scores[k])
            best = cand[k].copy()

        # Phase 2: deterministic pairwise-transfer hill climbing with
        # shrinking step.
        step = 0.10
        for _ in range(self.REFINE_SWEEPS):
            improved = False
            for frm in range(n):
                if best[frm] < 1e-12:
                    continue
                for to in range(n):
                    if to == frm:
                        continue
                    move = min(step, best[frm])
                    trial = best.copy()
                    trial[frm] -= move
                    trial[to] += move
                    score = float(objective(trial[None, :])[0])
                    if score > best_score:
                        best_score = score
                        best = trial
                        improved = True
            if not improved:
                step /= 2
                if step < 1e-6:
                    break
        return best

    def _vols(self, w: np.ndarray) -> np.ndarray:
        q = np.einsum("ij,jk,ik->i", w, self._cov, w)
        return np.sqrt(np.maximum(q, 0.0))

    def _to_allocation(self, w: np.ndarray, risk_free_rate: float) -> Allocation:
        ret = float(w @ self._mu)
        vol = float(self._vols(w[None, :])[0])
        sharpe = 0.0 if vol == 0 else (ret - risk_free_rate) / vol
        return Allocation(w, ret, vol, sharpe)
