"""UCB1 multi-armed bandit (port of Java ``execution.Ucb1Selector``).

Principled selection among venues, LPs or algo variants when the
scorecards are still THIN. The exploration problem is real: always
using the best-so-far venue means never learning whether another
improved; rotating uniformly wastes flow on known-bad ones. UCB1
(Auer et al. 2002) picks the arm maximizing::

    mean_reward + sqrt(2 * ln(N) / n_i)

-- the optimism bonus shrinks as an arm is tried, so exploration decays
exactly as fast as the evidence accumulates, with logarithmic regret
guaranteed. Rewards must be in [0, 1] (the theory's scale -- map fill
quality, negated cost bps, or markout onto it; the gate enforces it
because a mis-scaled reward silently breaks the exploration balance).

Where this sits vs the scorecards: a venue scorecard is the RIGHT tool
once hundreds of fills per venue exist -- it models fill rate, latency
and markout separately. UCB1 is for the cold start and for A/B-ing ALGO
variants (is the new schedule actually better?), where a single scalar
reward and a regret guarantee beat a half-warmed-up model.
Deterministic (ties break to the lowest index), O(arms) per selection.
"""

from __future__ import annotations

import math

import numpy as np


class Ucb1Selector:
    """UCB1 bandit selector; see the module docstring."""

    def __init__(self, arms: int) -> None:
        """
        Args:
            arms: number of venues/variants, >= 2.
        """
        if arms < 2:
            raise ValueError("a one-armed bandit is not a decision")
        self._reward_sums = np.zeros(arms)
        self._pulls = np.zeros(arms, dtype=np.int64)
        self._total_pulls = 0

    def select(self) -> int:
        """The arm to use next: each arm once first (in index order),
        then highest upper confidence bound, ties to the lowest index."""
        for i in range(self._pulls.shape[0]):
            if self._pulls[i] == 0:
                return i                    # every arm earns one look
        best = 0
        best_ucb = -math.inf
        log_n = math.log(self._total_pulls)
        for i in range(self._pulls.shape[0]):
            ucb = (self._reward_sums[i] / self._pulls[i]
                  + math.sqrt(2 * log_n / self._pulls[i]))
            if ucb > best_ucb:
                best_ucb = ucb
                best = i
        return best

    def record(self, arm: int, reward: float) -> None:
        """Records the observed reward for an arm.

        Args:
            arm: the arm that was used.
            reward: in [0, 1] -- the UCB1 theory's scale; rescale
                upstream, never here.
        """
        if arm < 0 or arm >= self._pulls.shape[0]:
            raise ValueError(f"arm {arm} of {self._pulls.shape[0]}")
        if not (reward >= 0 and reward <= 1):
            raise ValueError(
                f"reward must be in [0, 1], got {reward} -- a mis-scaled "
                "reward silently breaks the exploration balance")
        self._reward_sums[arm] += reward
        self._pulls[arm] += 1
        self._total_pulls += 1

    def pulls(self, arm: int) -> int:
        """Times an arm has been used."""
        return int(self._pulls[arm])

    def mean_reward(self, arm: int) -> float:
        """The arm's observed mean reward (NaN before its first pull)."""
        return (math.nan if self._pulls[arm] == 0
               else float(self._reward_sums[arm] / self._pulls[arm]))

    def total_pulls(self) -> int:
        return self._total_pulls
