"""Alpha reporting (port of Java ``alpha.AlphaReport``) -- the
diagnostics that explain a factor's P&L rather than just totalling it:

* **Alpha decay** -- mean IC as a function of the forward horizon. A
  signal predictive at 1 day but dead at 5 needs fast, expensive
  trading; the ``half_life`` estimate says how long the edge
  survives, which bounds the viable rebalance cadence.
* **Factor attribution** -- OLS of portfolio returns on factor return
  streams: how much of the "alpha" is just repackaged momentum/value
  beta, and what residual (true) alpha remains.
* **Curves and ratios** -- cumulative return, drawdown series, and the
  full ratio set (Sharpe, Sortino, Calmar, CAGR, max drawdown) reused
  verbatim from :mod:`quantfinlib.backtest.performance_analytics`, so
  alpha reports and strategy backtests can never disagree on a
  definition.
* **Rolling metrics** -- the windowed Sharpe that shows whether
  performance is steady or one lucky year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from quantfinlib.alpha.alpha_context import AlphaContext
from quantfinlib.alpha.alpha_factor import AlphaFactor
from quantfinlib.alpha.alpha_validation import AlphaValidation
from quantfinlib.backtest.performance_analytics import (PerformanceAnalytics,
                                                        PerformanceMetrics)
from quantfinlib.util import math_utils


@dataclass(frozen=True)
class Decay:
    """IC per horizon plus the interpolated half-life of the
    shortest-horizon IC."""

    horizons: np.ndarray
    mean_ics: np.ndarray
    half_life_bars: float

    def format(self) -> str:
        parts = [f" h={h}:{ic:.4f}" for h, ic in zip(self.horizons, self.mean_ics)]
        return ("alpha decay:" + "".join(parts)
                + f" | half-life={self.half_life_bars:.1f} bars")


@dataclass(frozen=True)
class Attribution:
    """OLS attribution: per-bar residual alpha, factor betas, and fit
    quality."""

    alpha_per_bar: float
    betas: np.ndarray
    factor_names: Tuple[str, ...]
    r_squared: float

    def format(self) -> str:
        parts = [f" {name}={b:.3f}"
                 for name, b in zip(self.factor_names, self.betas)]
        return (f"attribution: alpha={self.alpha_per_bar:.6f}/bar"
                + "".join(parts) + f" R2={self.r_squared:.3f}")


class AlphaReport:
    """Static reporting entry points; see the module docstring."""

    # ------------------------------------------------------------------
    # Alpha decay
    # ------------------------------------------------------------------

    @staticmethod
    def decay_profile(ctx: AlphaContext, factor: AlphaFactor,
                      start_index: int, horizons: Sequence[int]) -> Decay:
        """Evaluates the factor's mean IC at each horizon. The
        half-life is the first horizon (linearly interpolated) where
        the IC falls below half of its shortest-horizon value;
        ``+inf`` when it never does within the tested range -- the
        honest answer for slow signals."""
        if len(horizons) < 2:
            raise ValueError("need at least 2 horizons")
        ics = np.zeros(len(horizons))
        for i, h in enumerate(horizons):
            if i > 0 and horizons[i] <= horizons[i - 1]:
                raise ValueError("horizons must be ascending")
            ics[i] = AlphaValidation.mean_ic(ctx, factor, start_index,
                                             ctx.bars(), h)
        target = ics[0] / 2
        half_life = math.inf
        # First crossing below half the base IC, linearly interpolated
        # -- only meaningful when the base IC is positive to begin
        # with.
        if ics[0] > 0:
            for i in range(1, ics.shape[0]):
                if ics[i] < target:
                    w = (ics[i - 1] - target) / (ics[i - 1] - ics[i])
                    half_life = horizons[i - 1] + w * (horizons[i] - horizons[i - 1])
                    break
        return Decay(np.array(horizons), ics, half_life)

    # ------------------------------------------------------------------
    # Factor attribution
    # ------------------------------------------------------------------

    @staticmethod
    def attribute(portfolio_returns, factor_returns,
                 factor_names: Sequence[str]) -> Attribution:
        """Regresses portfolio returns on factor return streams (with
        an intercept) via the normal equations:
        ``r_p = alpha + sum(beta_i * f_i) + eps``. The intercept is
        the residual alpha -- what survives after the known factors
        take their share. Keep the factor count far below the bar
        count; the normal equations of collinear factors are a data
        problem, not a solver problem."""
        y = np.asarray(portfolio_returns, dtype=float)
        n = y.shape[0]
        factors = [np.asarray(f, dtype=float) for f in factor_returns]
        k = len(factors)
        if k != len(factor_names) or k == 0:
            raise ValueError("factor streams and names must align")
        for f in factors:
            if f.shape[0] != n:
                raise ValueError("factor stream length mismatch")
        if n <= k + 1:
            raise ValueError("more factors than observations")
        # NaN = missing everywhere in this package; one NaN bar would
        # poison the normal equations into all-NaN betas silently.
        # Fail with the index instead: trim factor warm-up bars before
        # attributing.
        for t in range(n):
            if math.isnan(y[t]):
                raise ValueError(f"NaN portfolio return at index {t}")
            for j in range(k):
                if math.isnan(factors[j][t]):
                    raise ValueError(
                        f"NaN in factor stream '{factor_names[j]}' at "
                        f"index {t} -- trim warm-up bars before "
                        "attribution")
        # Design matrix X = [1 | factors]; solve (X'X) b = X'y.
        p = k + 1
        x = np.empty((n, p))
        x[:, 0] = 1.0
        for j in range(k):
            x[:, j + 1] = factors[j]
        xtx = x.T @ x
        xty = x.T @ y
        beta = math_utils.solve_linear(xtx, xty)
        # R^2 from the residuals of the fitted model.
        mean_y = math_utils.mean(y)
        fit = x @ beta
        ss_res = float(np.sum((y - fit) ** 2))
        ss_tot = float(np.sum((y - mean_y) ** 2))
        betas = beta[1:].copy()
        r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
        return Attribution(float(beta[0]), betas, tuple(factor_names), r2)

    # ------------------------------------------------------------------
    # Curves and rolling metrics
    # ------------------------------------------------------------------

    @staticmethod
    def returns_of(equity) -> np.ndarray:
        """Per-bar simple returns of an equity curve -- the input to
        attribution/rolling."""
        e = np.asarray(equity, dtype=float)
        if e.shape[0] < 2:
            raise ValueError("need at least 2 equity points")
        return e[1:] / e[:-1] - 1

    @staticmethod
    def drawdown_curve(equity) -> np.ndarray:
        """Drawdown series: fraction below the running peak (0 at new
        highs). Guards ``peak > 0`` exactly like the risk module's
        max-drawdown so ``min(drawdown_curve)`` and the headline
        max-drawdown metric can never disagree on a curve that
        touches zero."""
        e = np.asarray(equity, dtype=float)
        dd = np.zeros(e.shape[0])
        peak = e[0]
        for i in range(e.shape[0]):
            peak = max(peak, e[i])
            dd[i] = e[i] / peak - 1 if peak > 0 else 0.0
        return dd

    @staticmethod
    def rolling_sharpe(returns, window: int,
                       periods_per_year: int) -> np.ndarray:
        """Rolling annualized Sharpe over a trailing window of per-bar
        returns; NaN until the window fills. The steadiness plot: a
        flat positive line is a strategy, a single spike is an
        anecdote.

        Uses the SAMPLE standard deviation (n-1) -- the same
        definition the headline Sharpe uses -- so a full-sample
        rolling window reproduces the headline Sharpe exactly rather
        than differing by ``sqrt(n/(n-1))``.
        """
        r = np.asarray(returns, dtype=float)
        if window < 2 or window > r.shape[0]:
            raise ValueError("window must be in [2, returns.length]")
        out = np.full(r.shape[0], math.nan)
        for i in range(window - 1, r.shape[0]):
            mean = math_utils.mean(r, i - window + 1, i + 1)
            sd = math_utils.std_dev_sample(r, i - window + 1, i + 1)
            out[i] = 0.0 if sd == 0 else mean / sd * math.sqrt(periods_per_year)
        return out

    @staticmethod
    def summarize(equity, periods_per_year: int) -> PerformanceMetrics:
        """The full ratio set on an equity curve -- Sharpe, Sortino,
        Calmar, CAGR, max drawdown -- computed by the same engine the
        backtesters use, so definitions never fork between research
        and backtest reports."""
        return PerformanceAnalytics.compute(equity, (), periods_per_year)
