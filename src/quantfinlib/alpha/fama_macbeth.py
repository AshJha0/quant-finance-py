"""Fama-MacBeth cross-sectional regression (port of Java
``alpha.FamaMacBeth``) -- the standard answer to the question the IC
cannot answer: what is a factor exposure WORTH, per period, in return
space? Two passes:

1. each period, regress the cross-section of forward returns on that
   period's factor exposures (with intercept) -- one premium estimate
   ``lambda_k`` per factor per period;
2. the factor premium is the time-series MEAN of the ``lambda_k``'s,
   and its t-statistic uses the time-series standard error -- which is
   the method's entire point: cross-sectional correlation between
   assets (the thing that wrecks a naive pooled regression's standard
   errors) is absorbed, because each period contributes exactly one
   observation per factor.

Reading the output: a premium with ``|t| > 2`` is priced; the
INTERCEPT should be near zero with ``|t| < 2`` -- a significant
intercept says returns exist that your factors do not explain. NaN
entries (asset not in the cross-section that period -- the
:class:`~quantfinlib.alpha.alpha_context.AlphaContext` convention) are
skipped per period; periods with fewer assets than factors + 2 are
skipped entirely and counted. Plain time-series t-stats (no
Newey-West correction -- stated, not hidden; premia autocorrelation
inflates them). Static, deterministic, research lane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.util import math_utils


@dataclass(frozen=True)
class FamaMacBethResult:
    """Fama-MacBeth fit result.

    Attributes:
        premia: mean per-period premium per factor.
        t_stats: time-series t-stat per factor.
        intercept_mean: mean per-period intercept (should be ~0).
        intercept_t_stat: its t-stat (significant = unexplained
            returns).
        periods_used: cross-sections that had enough assets.
    """

    premia: np.ndarray
    t_stats: np.ndarray
    intercept_mean: float
    intercept_t_stat: float
    periods_used: int


class FamaMacBeth:
    """Static entry point; see the module docstring."""

    @staticmethod
    def fit(exposures, forward_returns) -> FamaMacBethResult:
        """
        Args:
            exposures: ``exposures[t][asset][factor]`` -- the factor
                loadings KNOWN at t. Ragged per period (asset counts
                may vary).
            forward_returns: ``forward_returns[t][asset]`` -- the
                return realized AFTER t (lookahead is the caller's sin
                to avoid; align like
                :mod:`quantfinlib.alpha.signal_evaluator`).
        """
        periods = len(exposures)
        if len(forward_returns) != periods or periods < 12:
            raise ValueError(f"need >= 12 aligned periods, got {periods}")
        factors = len(exposures[0][0]) if len(exposures[0]) > 0 else 0
        if factors < 1:
            raise ValueError("need >= 1 factor")
        dim = factors + 1                      # intercept first
        lambdas = []
        for t in range(periods):
            assets = len(exposures[t])
            if len(forward_returns[t]) != assets:
                raise ValueError(f"period {t} misaligned")
            xtx = np.zeros((dim, dim))
            xty = np.zeros(dim)
            rows = 0
            for a in range(assets):
                exp_ta = exposures[t][a]
                if len(exp_ta) != factors:
                    raise ValueError(
                        f"period {t} asset {a} has {len(exp_ta)} "
                        f"factors, expected {factors}")
                y = forward_returns[t][a]
                exp_arr = np.asarray(exp_ta, dtype=float)
                # NaN = not in this cross-section (the AlphaContext
                # convention): skip. INFINITY = a data error (a broken
                # adjustment upstream): fail fast, never silent
                # garbage.
                if math.isnan(y) or np.any(np.isnan(exp_arr)):
                    continue
                if math.isinf(y) or np.any(np.isinf(exp_arr)):
                    raise ValueError(
                        f"infinite value at period {t} asset {a} -- a "
                        "data error, not a missing name")
                row = np.empty(dim)
                row[0] = 1.0
                row[1:] = exp_arr
                xtx += np.outer(row, row)
                xty += row * y
                rows += 1
            if rows < factors + 2:
                continue                        # too thin: skip, counted
            try:
                lam = math_utils.solve_linear(xtx, xty)
                lambdas.append(lam)
            except ValueError:
                # A collinear cross-section (a factor constant across
                # the surviving assets -- sector dummies in filtered
                # universes do this) prices nothing THAT period. Skip
                # it like a thin period; aborting 59 good periods for
                # one bad one would be the wrong trade.
                continue
        used = len(lambdas)
        if used < 12:
            raise ValueError(
                f"only {used} usable cross-sections -- premia need a "
                "time series")
        lambda_mat = np.array(lambdas)          # [used, dim]
        premia = np.zeros(factors)
        t_stats = np.zeros(factors)
        intercept_mean = 0.0
        intercept_t = 0.0
        for k in range(dim):
            series = lambda_mat[:, k]
            mean = math_utils.mean(series)
            se = math_utils.std_dev(series) / math.sqrt(used)
            # Java: se > 0 ? mean/se : signum(mean)*INFINITY -- and IEEE
            # 0 * Infinity is NaN regardless of the zero's sign, so a
            # dead-flat series (mean == 0, se == 0) is NaN, not 0.
            t_stat = (mean / se if se > 0
                      else math.nan if mean == 0
                      else math.copysign(math.inf, mean))
            if k == 0:
                intercept_mean = mean
                intercept_t = t_stat
            else:
                premia[k - 1] = mean
                t_stats[k - 1] = t_stat
        return FamaMacBethResult(premia, t_stats, intercept_mean, intercept_t, used)
