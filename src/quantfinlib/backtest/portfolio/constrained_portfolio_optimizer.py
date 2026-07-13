"""Constrained long-only optimizer (port of Java
``optimization.ConstrainedPortfolioOptimizer``).

Per-asset weight bounds (position caps / floors) and an optional
turnover penalty against current holdings —
``adjusted return = mu . w - penalty * sum |w - w_current|`` — so the
optimizer trades expected gain against the real cost of getting there.
Same derivative-free search as
:class:`~quantfinlib.backtest.portfolio.portfolio_optimizer.PortfolioOptimizer`
with feasibility projection. Deterministic per seed via
``np.random.default_rng`` (the Java ``SplittableRandom`` stream is not
reproduced across ports).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from quantfinlib.backtest.portfolio.portfolio_optimizer import Allocation


class ConstrainedPortfolioOptimizer:
    """Bounded long-only optimizer; see the module docstring."""

    _SAMPLES = 20_000
    _REFINE_SWEEPS = 80

    def __init__(self, expected_returns, covariance, seed: int = 42):
        self._mu = np.array(expected_returns, dtype=float)
        self._cov = np.asarray(covariance, dtype=float)
        self._seed = seed
        n = self._mu.shape[0]
        self._min = np.zeros(n)
        self._max = np.ones(n)
        self._current: np.ndarray | None = None
        self._turnover_penalty = 0.0

    def with_bounds(self, min_weights, max_weights) -> "ConstrainedPortfolioOptimizer":
        """Per-asset weight bounds; must admit a fully-invested portfolio.

        Raises:
            ValueError: on a bad per-asset bound or bounds that admit no
                fully-invested portfolio.
        """
        lo = np.asarray(min_weights, dtype=float)
        hi = np.asarray(max_weights, dtype=float)
        for i in range(lo.shape[0]):
            if lo[i] < 0 or hi[i] > 1 or lo[i] > hi[i]:
                raise ValueError(f"bad bounds for asset {i}")
        if float(np.sum(lo)) > 1 + 1e-12 or float(np.sum(hi)) < 1 - 1e-12:
            raise ValueError("bounds admit no fully-invested portfolio")
        self._min = lo.copy()
        self._max = hi.copy()
        return self

    def with_turnover_penalty(self, current_weights,
                              penalty_per_unit_turnover: float
                              ) -> "ConstrainedPortfolioOptimizer":
        """Charges ``penalty_per_unit_turnover`` expected-return units per
        unit of one-way turnover (e.g. transaction cost)."""
        self._current = np.array(current_weights, dtype=float)
        self._turnover_penalty = penalty_per_unit_turnover
        return self

    def max_sharpe(self, risk_free_rate: float) -> Allocation:
        """Maximum Sharpe on the turnover-adjusted return."""
        def objective(w: np.ndarray) -> np.ndarray:
            vols = self._vols(w)
            adj = w @ self._mu - self._turnover_cost(w)
            safe = np.where(vols > 0, vols, 1.0)
            return np.where(vols > 0, (adj - risk_free_rate) / safe, -np.inf)
        return self._to_allocation(self._optimize(objective), risk_free_rate)

    def min_volatility(self) -> Allocation:
        """Minimum volatility (plus turnover cost, if configured)."""
        def objective(w: np.ndarray) -> np.ndarray:
            return -(self._vols(w) + self._turnover_cost(w))
        return self._to_allocation(self._optimize(objective), 0.0)

    # ------------------------------------------------------------------

    def _turnover_cost(self, w: np.ndarray) -> np.ndarray:
        if self._current is None or self._turnover_penalty == 0:
            return np.zeros(w.shape[0])
        return self._turnover_penalty * np.sum(
            np.abs(w - self._current[None, :]), axis=1)

    def _vols(self, w: np.ndarray) -> np.ndarray:
        q = np.einsum("ij,jk,ik->i", w, self._cov, w)
        return np.sqrt(np.maximum(q, 0.0))

    def _optimize(self, objective: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        n = self._mu.shape[0]
        rng = np.random.default_rng(self._seed)

        best = self._project(np.full((1, n), 1.0 / n))[0]
        best_score = float(objective(best[None, :])[0])
        cand = rng.standard_exponential((self._SAMPLES, n))
        cand /= cand.sum(axis=1, keepdims=True)
        feasible = self._project(cand)
        scores = objective(feasible)
        k = int(np.argmax(scores))
        if scores[k] > best_score:
            best_score = float(scores[k])
            best = feasible[k].copy()

        # Pairwise-transfer refinement within bounds.
        step = 0.05
        for _ in range(self._REFINE_SWEEPS):
            improved = False
            for frm in range(n):
                for to in range(n):
                    if to == frm:
                        continue
                    move = min(step, best[frm] - self._min[frm],
                               self._max[to] - best[to])
                    if move <= 0:
                        continue
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

    def _project(self, w: np.ndarray) -> np.ndarray:
        """Projects rows onto the box-constrained simplex (clip, then
        redistribute the deficit proportionally to the available headroom)."""
        w = w.copy()
        lo = self._min[None, :]
        hi = self._max[None, :]
        for _ in range(100):
            np.clip(w, lo, hi, out=w)
            deficit = 1.0 - w.sum(axis=1)
            active = np.abs(deficit) >= 1e-12
            if not np.any(active):
                return w
            pos = deficit > 0
            room = np.where(pos[:, None], hi - w, w - lo)
            capacity = room.sum(axis=1)
            # Rows with no capacity are left as-is (bounds validated at
            # configuration; defensive, as in Java).
            ok = active & (capacity > 0)
            if not np.any(ok):
                return w
            w[ok] += (deficit[ok] / capacity[ok])[:, None] * room[ok]
        return w

    def _to_allocation(self, w: np.ndarray, risk_free_rate: float) -> Allocation:
        ret = float(w @ self._mu)
        vol = float(self._vols(w[None, :])[0])
        sharpe = 0.0 if vol == 0 else (ret - risk_free_rate) / vol
        return Allocation(w, ret, vol, sharpe)
