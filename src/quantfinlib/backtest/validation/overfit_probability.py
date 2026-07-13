"""Probability of backtest overfitting via CSCV (port of Java
``validation.OverfitProbability``).

Combinatorially symmetric cross-validation — CSCV (Bailey, Borwein,
Lopez de Prado & Zhu 2015, "The probability of backtest overfitting").

``SharpeValidation`` asks whether ONE track record is luck. This class
asks the prior question: is the SELECTION PROCESS itself broken? When a
desk tries N parameter sets and reports the best, the reported Sharpe is
a maximum of N draws — and the right diagnostic is: how often does the
in-sample winner turn out to be a BELOW-MEDIAN performer out of sample?

The construction: take the T x N matrix of per-period returns (one
column per strategy variant), slice time into S equal blocks, and form
every one of the C(S, S/2) ways to pick half the blocks as in-sample
(IS) and the complementary half as out-of-sample (OOS) — symmetric by
construction, so IS and OOS have identical length and no arrow-of-time
bias. For each combination:

1. concatenate the IS blocks and pick the variant with the best IS
   objective (ties break to the first column — stated, and the tie
   ranks below make that conservative);
2. rank that winner's OOS objective among all N variants:
   ``rank = 1 + #(strictly worse)``, relative rank
   ``w = rank / (N + 1)`` (never exactly 0 or 1);
3. record the logit ``lambda = ln(w / (1 - w))`` — positive means the
   IS winner was above the OOS median, negative below.

**PBO = the fraction of combinations with lambda <= 0**: the probability
that the config you would have picked is an out-of-sample loser. Rules
of thumb: PBO < 0.1 — selection is finding something real; PBO >= 0.5 —
the selection is pure noise-mining and the "best" backtest is
meaningless regardless of how good it looks.

Trailing periods that don't fill a whole block are dropped (stated: with
T = 1007 and S = 8, each block is 125 periods and the last 7 are
unused). S is capped at 16 — C(16, 8) = 12,870 combinations is already a
full re-scoring of every variant 12,870 times; beyond that the cost
explodes for no statistical gain. Deterministic (no RNG), research lane.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from quantfinlib.util import math_utils as mu


class OverfitProbability:
    """Static CSCV analysis; see the module docstring."""

    #: Hard cap on blocks: C(16, 8) = 12,870 combinations.
    MAX_BLOCKS = 16

    @dataclass(frozen=True, eq=False)
    class Result:
        """CSCV output.

        Attributes:
            pbo: Fraction of IS/OOS splits whose in-sample winner ranked
                at or below the out-of-sample median.
            combinations: Number of symmetric splits evaluated, C(S, S/2).
            logits: One lambda per combination (enumeration order).
        """

        pbo: float
        combinations: int
        logits: np.ndarray

    @staticmethod
    def cscv(returns, blocks: int,
             objective: Callable[[np.ndarray], float]) -> "OverfitProbability.Result":
        """CSCV with the caller's objective (higher is better).

        Args:
            returns: T x N rectangular matrix, ``returns[t][j]`` =
                period-t return of strategy variant j; all finite, N >= 2.
            blocks: S: even, 4 <= S <= 16; each block needs >= 2 periods.
            objective: Score for a variant's concatenated return
                sub-series; higher is better, must be finite.

        Raises:
            ValueError: on invalid shape, block count, non-finite input
                or a non-finite objective value.
        """
        r = _validated(returns, blocks)
        n_variants = r.shape[1]
        block_len = r.shape[0] // blocks
        half = blocks // 2

        # All C(blocks, half) ascending index combinations, lexicographic.
        combos = list(itertools.combinations(range(blocks), half))
        all_blocks = frozenset(range(blocks))

        logits = np.empty(len(combos))
        below = 0
        for c, combo in enumerate(combos):
            oos_blocks = sorted(all_blocks - set(combo))
            is_rows = _rows(combo, block_len)
            oos_rows = _rows(oos_blocks, block_len)
            # Score every variant IS and OOS on this split.
            best_is = -math.inf
            winner = -1
            oos_scores = np.empty(n_variants)
            for j in range(n_variants):
                is_score = _score(r[is_rows, j], objective, j)
                oos_scores[j] = _score(r[oos_rows, j], objective, j)
                if is_score > best_is:
                    best_is = is_score
                    winner = j
            rank = 1 + int(np.sum(oos_scores < oos_scores[winner]))
            w = rank / (n_variants + 1)
            lam = math.log(w / (1 - w))
            logits[c] = lam
            if lam <= 0:
                below += 1
        return OverfitProbability.Result(below / len(combos), len(combos), logits)

    @staticmethod
    def cscv_sharpe(returns, blocks: int) -> "OverfitProbability.Result":
        """CSCV with the per-period Sharpe objective ``mean / std_dev``
        (sample standard deviation; a zero-variance sub-series scores 0 —
        a flat line has no risk-adjusted evidence either way)."""
        def sharpe(r: np.ndarray) -> float:
            sd = mu.std_dev(r)
            return mu.mean(r) / sd if sd > 0 else 0.0
        return OverfitProbability.cscv(returns, blocks, sharpe)


def _rows(blocks_picked, block_len: int) -> np.ndarray:
    """Row indices of the picked blocks, concatenated in ascending order."""
    return np.concatenate([np.arange(b * block_len, (b + 1) * block_len)
                           for b in blocks_picked])


def _score(series: np.ndarray, objective, variant: int) -> float:
    s = float(objective(series))
    # Finite required, not merely non-NaN: an all -inf in-sample column
    # would leave the argmax with no winner at all.
    if not math.isfinite(s):
        raise ValueError(f"objective returned non-finite {s} for variant {variant}")
    return s


def _validated(returns, blocks: int) -> np.ndarray:
    if blocks < 4 or blocks % 2 != 0 or blocks > OverfitProbability.MAX_BLOCKS:
        raise ValueError(
            f"blocks must be even, in [4, {OverfitProbability.MAX_BLOCKS}], got {blocks}")
    r = np.asarray(returns, dtype=float)  # ragged input raises ValueError here
    if r.ndim != 2:
        raise ValueError(f"returns must be a T x N matrix, got ndim={r.ndim}")
    if r.shape[0] < 2 * blocks:
        raise ValueError(f"need >= 2 periods per block: "
                         f"{r.shape[0]} periods for {blocks} blocks")
    if r.shape[1] < 2:
        raise ValueError(f"need >= 2 strategy variants, got {r.shape[1]}")
    if not np.all(np.isfinite(r)):
        raise ValueError("non-finite return in matrix")
    return r
