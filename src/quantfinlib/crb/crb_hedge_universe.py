"""The hedge-instrument universe (port of Java
``com.quantfinlib.crb.CrbHedgeUniverse``).

Aligned to a book's factor registry -- because hand-assembling
``loadings[factor][instrument]`` is the most error-prone step in the
whole hedging workflow (one transposed index and the optimizer
confidently hedges the wrong thing).

Each instrument declares what ONE UNIT of its notional does to the
factor space, in the same conventions ``CentralRiskBook`` books with:

- :meth:`add_single_factor` -- an instrument that is 1-for-1 one
  factor: an index future onto ``EQ:<index>``, a variance swap onto
  ``EQVEGA:<sym>``, an FX vol trade onto ``FXVEGA:<pair>``;
- :meth:`add_fx_forward` -- one unit of base notional loads
  ``CCY:<base>`` +1 and ``CCY:<quote>`` -rate, exactly like a booked
  spot/forward;
- :meth:`add` -- anything else, factor names and per-unit loadings side
  by side.

Factors named here are REGISTERED on the shared registry if new (a
hedge-only factor simply has zero book exposure), and :meth:`loadings`
materializes the matrix at the registry's CURRENT size -- build it
after all booking and adding is done, and feed it straight to
``HedgeOptimizer``/``CrbAutoHedger`` with :meth:`costs`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantfinlib.crb.factor_registry import FactorRegistry


@dataclass(frozen=True)
class _Instrument:
    name: str
    cost_per_unit: float
    factor_ids: list[int]
    per_unit: list[float]


class CrbHedgeUniverse:

    def __init__(self, registry: FactorRegistry):
        """``registry``: the book's registry -- ``book.factors()``."""
        self._registry = registry
        self._instruments: list[_Instrument] = []

    def add_single_factor(self, name: str, factor: str,
                          cost_per_unit: float) -> "CrbHedgeUniverse":
        """An instrument that is one unit of exactly one factor."""
        return self.add(name, cost_per_unit, [factor], [1.0])

    def add_fx_forward(self, name: str, pair: str, rate: float,
                       cost_per_unit: float) -> "CrbHedgeUniverse":
        """An FX forward/spot hedge on ``pair``: one unit of base
        notional loads the two currency legs exactly as a booked trade
        would."""
        if pair is None or len(pair) != 6:
            raise ValueError(f"pair must be 6 chars like EURUSD: {pair}")
        if not (rate > 0) or rate == math.inf:
            raise ValueError("rate must be positive and finite")
        return self.add(name, cost_per_unit,
                        [f"CCY:{pair[0:3]}", f"CCY:{pair[3:6]}"], [1.0, -rate])

    def add(self, name: str, cost_per_unit: float, factors: list[str],
           per_unit: list[float]) -> "CrbHedgeUniverse":
        """A general instrument: per-unit loadings onto named
        factors."""
        if name is None or name.strip() == "":
            raise ValueError("instrument must be named")
        if not (cost_per_unit >= 0) or cost_per_unit == math.inf:
            raise ValueError("cost_per_unit must be >= 0 and finite")
        if len(factors) != len(per_unit) or len(factors) == 0:
            raise ValueError("need aligned, non-empty factor/loading arrays")
        ids = []
        for i, f in enumerate(factors):
            if not math.isfinite(per_unit[i]):
                raise ValueError("loadings must be finite")
            ids.append(self._registry.id(f))       # registers hedge-only factors
        self._instruments.append(_Instrument(name, cost_per_unit, ids, list(per_unit)))
        return self

    def loadings(self) -> list[list[float]]:
        """The loadings matrix [factor][instrument] at the registry's
        CURRENT size -- call after all booking/adding, alongside
        :meth:`costs`."""
        n = self._registry.size()
        m = len(self._instruments)
        out = [[0.0] * m for _ in range(n)]
        for i, inst in enumerate(self._instruments):
            for k, fid in enumerate(inst.factor_ids):
                out[fid][i] += inst.per_unit[k]
        return out

    def costs(self) -> list[float]:
        """Per-unit costs aligned with :meth:`loadings` columns."""
        return [inst.cost_per_unit for inst in self._instruments]

    def name(self, instrument: int) -> str:
        """Instrument name for a ``CrbAutoHedger.HedgeOrder.instrument``
        index."""
        return self._instruments[instrument].name

    def size(self) -> int:
        return len(self._instruments)
