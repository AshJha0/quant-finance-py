"""Monte Carlo trade reshuffling (port of Java ``validation.MonteCarloTradeShuffle``).

The answer to "was my equity curve's SHAPE luck?". A backtest reports
one path: this strategy's trades in the order they happened, producing
one max drawdown and one terminal equity. But the ORDER is itself a
sample. Reshuffle the same trade P&Ls into thousands of random sequences
and you get the distribution of drawdowns and terminal wealth the
strategy's OWN trades imply — and the honest question stops being "the
backtest drew down 18%" and becomes "the 95th-percentile drawdown of my
trade set is 31%, so plan for 31, not 18".

What reshuffling holds fixed and what it breaks: it preserves the
MULTISET of trade outcomes (win rate, average win/loss, profit factor
are invariant) and destroys their ORDER — so it isolates
"path/sequencing risk" from "edge". Its blind spot, stated plainly: it
assumes trades are exchangeable, so it UNDERSTATES risk for a strategy
whose losses cluster (serial correlation, regime dependence) — a
martingale that wins small and loses catastrophically looks tamer
reshuffled than it is. For serially-correlated PATHS use
:mod:`~quantfinlib.backtest.validation.block_bootstrap` on the return
series; this class is the per-trade complement, and the two disagreeing
is itself the signal that your trades are not independent.

Drawdown is computed on the cumulative P&L path (additive, so no
starting-capital assumption). Deterministic given the seed via
``np.random.default_rng`` (port note: the Java reference draws its
Fisher-Yates swaps from ``SplittableRandom``; the RNG stream is not
reproduced across ports — the pinned properties are order-invariance of
terminal P&L, percentile ordering, determinism-per-seed and the
worst-case-ordering tail rank). Research lane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantfinlib.backtest.trade import Trade
from quantfinlib.util import math_utils as mu


class MonteCarloTradeShuffle:
    """Static reshuffling analysis; see the module docstring."""

    @dataclass(frozen=True)
    class Result:
        """Shuffle-distribution statistics (P&L units).

        Attributes:
            median_max_drawdown: 50th-percentile max drawdown across shuffles.
            p95_max_drawdown: 95th-percentile max drawdown — the plan-for number.
            p99_max_drawdown: 99th-percentile max drawdown — the tail.
            median_terminal_pnl: 50th-percentile terminal cumulative P&L.
            prob_loss: Fraction of shuffles ending below zero.
            actual_max_drawdown: The observed (unshuffled) path's max drawdown.
            actual_drawdown_pct: Where the observed drawdown sits in the
                shuffle distribution (0..1); high = the real order was
                UNUSUALLY painful, a hint of loss clustering.
        """

        median_max_drawdown: float
        p95_max_drawdown: float
        p99_max_drawdown: float
        median_terminal_pnl: float
        prob_loss: float
        actual_max_drawdown: float
        actual_drawdown_pct: float

    @staticmethod
    def analyze(trades: Sequence[Trade], shuffles: int,
                seed: int) -> "MonteCarloTradeShuffle.Result":
        """Reshuffles >= 2 trades over >= 100 random orderings.

        Raises:
            ValueError: on fewer than 2 trades, fewer than 100 shuffles,
                or a non-finite pnl.
        """
        n = len(trades)
        if n < 2:
            raise ValueError(f"need >= 2 trades, got {n}")
        if shuffles < 100:
            raise ValueError(f"need >= 100 shuffles, got {shuffles}")
        pnls = np.empty(n)
        for i, t in enumerate(trades):
            if not math.isfinite(t.pnl):
                raise ValueError(f"non-finite trade pnl at {i}")
            pnls[i] = t.pnl

        actual_dd = _max_drawdown(pnls)
        dd = np.empty(shuffles)
        terminal = np.empty(shuffles)
        losses = 0
        actual_at_or_below = 0
        rng = np.random.default_rng(seed)
        for s in range(shuffles):
            # rng.permutation is a Fisher-Yates shuffle of a fresh copy.
            work = rng.permutation(pnls)
            dd[s] = _max_drawdown(work)
            total = float(np.sum(work))
            terminal[s] = total
            if total < 0:
                losses += 1
            if dd[s] <= actual_dd:
                actual_at_or_below += 1
        return MonteCarloTradeShuffle.Result(
            mu.percentile(dd, 0.50),
            mu.percentile(dd, 0.95),
            mu.percentile(dd, 0.99),
            mu.percentile(terminal, 0.50),
            losses / shuffles,
            actual_dd,
            actual_at_or_below / shuffles)


def _max_drawdown(pnls: np.ndarray) -> float:
    """Max peak-to-trough drop on the cumulative-P&L path (>= 0).

    The running peak starts at 0 (the pre-trade cumulative P&L), exactly
    as in the Java walk.
    """
    cum = np.cumsum(pnls)
    peak = np.maximum(np.maximum.accumulate(cum), 0.0)
    return float(np.max(peak - cum))
