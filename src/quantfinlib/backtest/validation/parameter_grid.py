"""Named parameter grid (port of Java ``backtest.validation.ParameterGrid``).

``combinations()`` enumerates the cartesian product in deterministic
(insertion) order. The Java ``StrategyFactory`` functional interface
maps to a plain callable ``dict[str, float] -> TradingStrategy``.
"""

from __future__ import annotations

from typing import Dict, List


class ParameterGrid:
    """A named parameter grid for strategy optimization."""

    def __init__(self) -> None:
        self._params: Dict[str, List[float]] = {}

    def add(self, name: str, *values: float) -> "ParameterGrid":
        if len(values) == 0:
            raise ValueError(f"no values for parameter {name}")
        self._params[name] = [float(v) for v in values]
        return self

    def size(self) -> int:
        if not self._params:
            return 0
        n = 1
        for v in self._params.values():
            n *= len(v)
        return n

    def combinations(self) -> List[Dict[str, float]]:
        """All parameter combinations, insertion-ordered and
        deterministic. Empty for an empty grid, consistent with
        :meth:`size` — never a single empty dict, which a strategy
        factory would crash on."""
        if not self._params:
            return []
        out: List[Dict[str, float]] = [{}]
        for name, values in self._params.items():
            nxt: List[Dict[str, float]] = []
            for base in out:
                for v in values:
                    combo = dict(base)
                    combo[name] = v
                    nxt.append(combo)
            out = nxt
        return out
