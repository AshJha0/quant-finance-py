"""Portfolio construction (port of Java ``alpha.PortfolioConstruction``):
turns raw factor scores into tradeable weight vectors — deliberately a
chain of small, composable, pure functions so a construction pipeline
reads as what it does::

    w = PortfolioConstruction.z_score_weights(scores, 1.0, 0.05)
    w = PortfolioConstruction.sector_neutralize(w, sectors)
    w = PortfolioConstruction.beta_neutralize(w, betas)
    w = PortfolioConstruction.inverse_vol_budget(w, vols, 1.0)

All functions take and return weight arrays aligned with the
:class:`AlphaContext` symbol order; NaN scores become zero weight.
Inputs are never mutated. Weights are fractions of equity (0.05 = 5%),
signed (short = negative), and each step documents what it preserves
and what it re-normalizes — neutralization steps change gross exposure,
so gross targeting is a *final* step or a re-application.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

from quantfinlib.alpha.alpha_context import AlphaContext
from quantfinlib.risk import risk_metrics
from quantfinlib.util import math_utils

#: Sector label prefix for symbols missing from a sector map: each such
#: symbol gets its own singleton sector (``UNKNOWN_SECTOR_PREFIX + sym``)
#: so it demeans against itself (to zero) instead of polluting a real
#: sector's offset. The Java reference used a ``"\\0UNKNOWN:"`` NUL-byte
#: sentinel to avoid colliding with real labels; Python strings make the
#: NUL trick unnecessary and unreadable, so this port uses a documented
#: module constant instead. Collides only if a caller names a real
#: sector with this exact prefix — don't.
UNKNOWN_SECTOR_PREFIX = "__UNKNOWN__:"


class PortfolioConstruction:
    """Static construction pipeline steps; see the module docstring."""

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    @staticmethod
    def z_score_weights(scores, gross_target: float,
                        max_weight: float = math.inf) -> np.ndarray:
        """Z-score sizing, the workhorse: demean scores
        cross-sectionally, scale by their dispersion, clamp at +/-3
        sigma (a single outlier must not own the book), then normalize
        to ``sum(|w|) = gross_target`` and cap per-name weight at
        ``max_weight``.

        Demeaning makes the book dollar-neutral by construction whenever
        scores are symmetric; the clamp bounds concentration before the
        cap even applies.
        """
        if gross_target <= 0 or max_weight <= 0:
            raise ValueError("grossTarget and maxWeight must be > 0")
        s = np.asarray(scores, dtype=float)
        n = s.shape[0]
        scored = ~np.isnan(s)
        if int(np.sum(scored)) < 2:
            return np.zeros(n)  # nothing to rank against: hold cash
        mean = float(np.mean(s[scored]))
        std = math.sqrt(float(np.mean((s[scored] - mean) ** 2)))
        w = np.zeros(n)
        if std == 0:
            return w  # all scores identical: no cross-sectional information
        # Winsorized z-score: the raw signal in dispersion units.
        w[scored] = np.clip((s[scored] - mean) / std, -3, 3)
        PortfolioConstruction._normalize_gross(w, gross_target)
        return PortfolioConstruction.cap_weights(w, max_weight, gross_target)

    @staticmethod
    def cap_weights(weights, max_weight: float,
                    gross_target: float) -> np.ndarray:
        """Caps each ``|weight|`` at ``max_weight``, then re-normalizes
        the rest toward ``gross_target`` without breaching the cap
        (single pass of redistribution; residual gross shortfall stays
        in cash — honest, rather than looping until the cap itself
        binds everywhere)."""
        w = np.array(weights, dtype=float)
        over = np.abs(w) > max_weight
        if not np.any(over):
            return w
        w[over] = np.sign(w[over]) * max_weight
        gross = PortfolioConstruction._gross(w)
        if gross == 0:
            return w
        scale = gross_target / gross
        return np.clip(w * scale, -max_weight, max_weight)

    # ------------------------------------------------------------------
    # Risk budgeting
    # ------------------------------------------------------------------

    @staticmethod
    def inverse_vol_budget(weights, vols,
                           gross_target: float) -> np.ndarray:
        """Inverse-volatility risk budgeting: rescales each position by
        ``1/sigma_i`` (keeping its sign and relative signal strength),
        so every name contributes comparably to portfolio risk instead
        of the volatile names dominating — the first-order version of
        equal risk contribution, exact when correlations are equal.
        Re-normalized to ``gross_target``.

        Throws on unusable vols for held positions — a flat (sigma = 0)
        name inside a signal-weighted book is a data problem to
        surface, not to paper over.
        """
        w_in = np.asarray(weights, dtype=float)
        v = np.asarray(vols, dtype=float)
        if w_in.shape[0] != v.shape[0]:
            raise ValueError("weights and vols must align")
        w = np.zeros(w_in.shape[0])
        for i in range(w_in.shape[0]):
            if w_in[i] != 0:
                if math.isnan(v[i]) or v[i] <= 0:
                    raise ValueError(
                        f"position {i} has no usable vol ({v[i]})")
                w[i] = w_in[i] / v[i]
        PortfolioConstruction._normalize_gross(w, gross_target)
        return w

    @staticmethod
    def trailing_vols(ctx: AlphaContext, index: int,
                      lookback: int) -> np.ndarray:
        """Trailing return volatilities per symbol at ``index`` — the
        standard input to :meth:`inverse_vol_budget` (per-bar sigma;
        the common scale cancels in the renormalization)."""
        if index < lookback:
            raise ValueError(f"index {index} < lookback {lookback}")
        vols = np.zeros(ctx.symbol_count())
        for i in range(ctx.symbol_count()):
            closes = ctx.series(i).closes()
            r = (closes[index - lookback + 1:index + 1]
                 / closes[index - lookback:index] - 1)
            vols[i] = math_utils.std_dev(r)
        return vols

    # ------------------------------------------------------------------
    # Neutralization
    # ------------------------------------------------------------------

    @staticmethod
    def sector_neutralize_by_symbol(ctx: AlphaContext, weights,
                                    sector_by_symbol: Dict[str, str]
                                    ) -> np.ndarray:
        """:meth:`sector_neutralize` with alignment by construction:
        sector labels come as a dict keyed by symbol and are resolved
        against the context's frozen (sorted!) symbol order —
        :meth:`AlphaContext.of` re-sorts symbols, so a hand-built array
        in the caller's insertion order would silently demean against
        permuted labels. Symbols missing from the dict keep their own
        singleton sector (i.e. they demean to zero) — see
        :data:`UNKNOWN_SECTOR_PREFIX`."""
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != ctx.symbol_count():
            raise ValueError("weights must align with the context panel")
        sectors = [
            sector_by_symbol.get(sym, UNKNOWN_SECTOR_PREFIX + sym)
            for sym in ctx.symbols()
        ]
        return PortfolioConstruction.sector_neutralize(w, sectors)

    @staticmethod
    def sector_neutralize(weights, sectors: Sequence[str]) -> np.ndarray:
        """Sector neutrality: demeans weights within each sector, so
        every sector's net weight is exactly zero and the book carries
        stock selection, not sector bets. Names with weight 0 stay 0
        (they are not dragged in to fund their sector's offset). Gross
        exposure changes — re-target gross afterwards if it matters.

        ``sectors`` must align with the weights — which follow
        :meth:`AlphaContext.symbols` order (SORTED, not your input
        map's order); prefer :meth:`sector_neutralize_by_symbol`, which
        cannot misalign.
        """
        w = np.array(weights, dtype=float)
        if w.shape[0] != len(sectors):
            raise ValueError("weights and sectors must align")
        sums: Dict[str, list] = {}
        for i in range(w.shape[0]):
            if w[i] != 0:
                entry = sums.setdefault(sectors[i], [0.0, 0])
                entry[0] += w[i]
                entry[1] += 1
        for i in range(w.shape[0]):
            if w[i] != 0:
                total, count = sums[sectors[i]]
                w[i] -= total / count
        return w

    @staticmethod
    def beta_neutralize(weights, betas) -> np.ndarray:
        """Beta neutrality: removes the market-beta component by
        projecting the weight vector orthogonal to the beta vector —
        ``w - beta * (w.beta)/(beta.beta)`` — so ``sum(w_i * beta_i) = 0``
        exactly and the book's P&L stops being a leveraged market bet.
        Gross changes; re-target afterwards if needed."""
        w_in = np.asarray(weights, dtype=float)
        b = np.asarray(betas, dtype=float)
        if w_in.shape[0] != b.shape[0]:
            raise ValueError("weights and betas must align")
        wb = 0.0
        bb = 0.0
        for i in range(w_in.shape[0]):
            if w_in[i] != 0:
                if math.isnan(b[i]):
                    raise ValueError(f"position {i} has NaN beta")
                wb += w_in[i] * b[i]
                bb += b[i] * b[i]
        w = w_in.copy()
        if bb == 0:
            return w  # no active betas: nothing to project out
        lam = wb / bb
        active = w != 0
        w[active] -= lam * b[active]
        return w

    @staticmethod
    def trailing_betas(ctx: AlphaContext, index: int,
                       lookback: int) -> np.ndarray:
        """Trailing OLS betas of each symbol against the equal-weight
        universe return — the in-panel market proxy when no index
        series is supplied."""
        if index < lookback:
            raise ValueError(f"index {index} < lookback {lookback}")
        n = ctx.symbol_count()
        returns = np.zeros((n, lookback))
        for i in range(n):
            closes = ctx.series(i).closes()
            returns[i] = (closes[index - lookback + 1:index + 1]
                          / closes[index - lookback:index] - 1)
        market = np.mean(returns, axis=0)
        # One beta definition library-wide: risk.risk_metrics owns it
        # (including the zero-variance-benchmark policy).
        return np.array([risk_metrics.beta(returns[i], market)
                         for i in range(n)])

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    @staticmethod
    def mean_variance_tilt(alphas, covariance,
                           gross_target: float) -> np.ndarray:
        """Unconstrained mean-variance tilt: ``w ~ inv(Sigma) alpha``
        (the Markowitz solution up to scale), solved via Gaussian
        elimination and normalized to ``gross_target``. Unlike z-score
        sizing this *uses* the correlation structure: two highly
        correlated names with the same alpha share one bet instead of
        doubling it. Feed a shrunk/regularized covariance — the raw
        sample matrix of a near-singular universe inverts into garbage,
        which is a data problem no solver fixes.
        """
        a_in = np.asarray(alphas, dtype=float)
        cov_in = np.asarray(covariance, dtype=float)
        n = a_in.shape[0]
        if cov_in.shape[0] != n:
            raise ValueError("alphas and covariance must align")
        # Solve only over scored names; excluded names keep zero weight.
        idx = np.where(~np.isnan(a_in))[0]
        if idx.shape[0] == 0:
            return np.zeros(n)
        cov = cov_in[np.ix_(idx, idx)]
        a = a_in[idx]
        try:
            solved = math_utils.solve_linear(cov, a)
        except ValueError as e:
            # The solver reports compacted-matrix columns, which mislead
            # when NaN alphas were squeezed out — re-raise in the
            # caller's terms.
            raise ValueError(
                "covariance is singular (a flat/duplicated/forward-filled "
                "series has zero or dependent variance) — shrink or "
                f"regularize it, e.g. add lambda*I ({e})") from e
        w = np.zeros(n)
        w[idx] = solved
        PortfolioConstruction._normalize_gross(w, gross_target)
        return w

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_gross(w: np.ndarray, gross_target: float) -> None:
        """Scales weights in place so ``sum(|w|) = gross_target``
        (no-op on a flat book)."""
        gross = PortfolioConstruction._gross(w)
        if gross == 0:
            return
        w *= gross_target / gross

    @staticmethod
    def _gross(w: np.ndarray) -> float:
        return float(np.sum(np.abs(w)))
