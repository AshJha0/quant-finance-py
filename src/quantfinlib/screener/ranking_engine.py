"""Ranking engine (port of Java ``screener.RankingEngine``).

Scores stocks by a weighted blend of min-max-normalized criteria and
sorts them best-first.

Why normalize before blending: raw metrics live on wildly different
scales (ROE ~0.15, market cap ~1e11), so a weighted sum of raw values
is just "whichever metric has the biggest units wins". Min-max
normalization maps each criterion to [0,1] ACROSS THE CANDIDATE SET
first; the weights then express genuine relative importance. Two
consequences to design around: scores are relative to this run's
universe (the same stock scores differently in a different candidate
list -- fine for "pick the best 20 today", wrong for tracking one name
through time), and min-max is outlier-sensitive (one absurd P/E
compresses everyone else's spread; screen out garbage with
``fundamental_filters`` BEFORE ranking, which is the intended pipeline
order). Negative weights invert a criterion -- lower P/E ranks higher
-- without a separate "ascending" flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence

from quantfinlib.screener.stock_snapshot import StockSnapshot

_Extractor = Callable[[StockSnapshot], float]


@dataclass(frozen=True, slots=True)
class ScoredStock:
    stock: StockSnapshot
    score: float


@dataclass(frozen=True, slots=True)
class _Criterion:
    name: str
    weight: float
    extractor: _Extractor


class RankingEngine:
    """Scores stocks by a weighted blend of min-max-normalized criteria."""

    def __init__(self) -> None:
        self._criteria: List[_Criterion] = []

    def add_criterion(self, name: str, weight: float, extractor: _Extractor) -> "RankingEngine":
        """
        :param weight: relative importance; use a negative weight to prefer
            smaller values (e.g. lower P/E ranks higher)
        :param extractor: metric to score, e.g. ``lambda s: s.fundamentals.roe``
        """
        self._criteria.append(_Criterion(name, weight, extractor))
        return self

    def rank(self, stocks: Sequence[StockSnapshot]) -> List[ScoredStock]:
        if not self._criteria:
            raise RuntimeError("no ranking criteria configured")
        n = len(stocks)
        scores = [0.0] * n
        total_abs_weight = sum(abs(c.weight) for c in self._criteria)

        for c in self._criteria:
            values = [c.extractor(stocks[i]) for i in range(n)]
            valid = [v for v in values if not math.isnan(v)]
            lo = min(valid) if valid else math.inf
            hi = max(valid) if valid else -math.inf
            for i in range(n):
                v = values[i]
                if math.isnan(v) or hi == lo:
                    norm = 0.5
                else:
                    norm = (v - lo) / (hi - lo)
                if c.weight < 0:
                    norm = 1 - norm
                scores[i] += abs(c.weight) / total_abs_weight * norm

        out = [ScoredStock(stocks[i], scores[i]) for i in range(n)]
        out.sort(key=lambda ss: ss.score, reverse=True)
        return out
