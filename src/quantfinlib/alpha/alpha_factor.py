"""Cross-sectional alpha factor interface (port of Java
``alpha.AlphaFactor``)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from quantfinlib.alpha.alpha_context import AlphaContext


class AlphaFactor(ABC):
    """A cross-sectional alpha factor: at a bar index, one raw score per
    symbol, where **higher = more attractive to own** (buy high scores,
    sell low).

    Scores are *raw*, in whatever natural unit the factor has (return
    spread, z-score, yield). Normalization into portfolio weights is
    deliberately a separate step (:mod:`portfolio_construction`) so the
    same factor can be evaluated rank-wise (rank IC is scale-invariant)
    and constructed under different schemes without re-implementing the
    signal.

    Contract:

    * the returned array aligns with :meth:`AlphaContext.symbols`;
    * NaN means "no score" (insufficient history, missing fundamentals)
      — every downstream step skips NaN entries;
    * implementations must only read bars ``<= index``: a factor that
      peeks forward invalidates every evaluation built on it. The
      validation suite cannot detect look-ahead mechanically — this
      contract is the guard;
    * custom factors should honor :meth:`AlphaContext.is_active`
      (return NaN for inactive names) the way the built-ins do.
    """

    @abstractmethod
    def scores(self, ctx: AlphaContext, index: int) -> np.ndarray:
        """Raw scores at ``index``, aligned with the context's symbols."""

    def name(self) -> str:
        """Human-readable name used in reports; override for real
        factors."""
        return type(self).__name__
