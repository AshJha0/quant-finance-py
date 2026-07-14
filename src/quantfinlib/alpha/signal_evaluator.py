"""Signal evaluation (port of Java ``alpha.SignalEvaluator``) — the
metrics that decide whether a factor is worth constructing a portfolio
from, computed *before* any backtest so weak signals die cheaply:

* **IC** (information coefficient) — Spearman rank correlation between
  scores at ``t`` and forward returns over ``(t, t+horizon]``, per
  evaluation date. Rank (not Pearson) because factor scores have
  arbitrary units and fat tails; rank IC is invariant to any monotone
  transform of the signal.
* **IR** — ``mean(IC) / std(IC)``: signal strength per unit of signal
  inconsistency, the standard Grinold-Kahn quality number (0.05 mean IC
  with steady sign beats 0.10 that flips).
* **t-stat** — ``mean(IC) / (std(IC)/sqrt(n))``: is the mean IC
  distinguishable from zero given how many dates we observed.
* **Hit rate** — fraction of (symbol, date) pairs where the score sign
  called the forward-return sign.
* **Turnover** — half the L1 change in normalized weights between
  consecutive evaluation dates: how much trading the signal demands,
  the denominator of "does the alpha survive costs".
* **Factor exposure** — mean cross-sectional rank correlation against
  another factor: a "new" signal that is 0.9 rank-correlated with
  momentum is momentum.

Evaluation dates step by ``horizon`` so forward-return windows don't
overlap — overlapping windows inflate the t-stat through serial
correlation, the classic way factor research fools itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np

from quantfinlib.alpha.alpha_context import AlphaContext
from quantfinlib.alpha.alpha_factor import AlphaFactor
from quantfinlib.alpha.portfolio_construction import PortfolioConstruction
from quantfinlib.util import math_utils


@dataclass(frozen=True)
class Report:
    """The evaluation scorecard; :meth:`format` renders it for humans."""

    factor_name: str
    mean_ic: float
    ic_std: float
    ir: float
    t_stat: float
    hit_rate: float
    mean_turnover: float
    observations: int
    ic_series: np.ndarray

    def format(self) -> str:
        """One-line human summary in fixed order for report diffs."""
        return (f"{self.factor_name}: IC={self.mean_ic:.4f} "
                f"(t={self.t_stat:.2f}, n={self.observations}) "
                f"IR={self.ir:.2f} hit={self.hit_rate * 100:.1f}% "
                f"turnover={self.mean_turnover * 100:.1f}%")


@dataclass(frozen=True)
class QuantileReport:
    """Mean forward return per score quantile — the picture behind the
    IC: ``mean_returns[0]`` is the average forward return of the
    lowest-scored names, the last entry of the highest-scored, and
    :meth:`spread` is the top-minus-bottom long/short return per period.
    A real factor shows MONOTONE quantile returns; a factor whose spread
    lives entirely in one extreme quantile is a tail bet wearing a
    factor costume — the IC alone cannot tell the difference, which is
    why desks always plot both."""

    factor_name: str
    mean_returns: np.ndarray
    counts: np.ndarray
    quantiles: int
    dates: int

    def spread(self) -> float:
        """Top-quantile minus bottom-quantile mean forward return."""
        return float(self.mean_returns[-1] - self.mean_returns[0])


class SignalEvaluator:
    """Static evaluation entry points; see the module docstring."""

    @staticmethod
    def evaluate(ctx: AlphaContext, factor: AlphaFactor,
                 start_index: int, horizon: int) -> Report:
        """Evaluates a factor over ``[start_index, ctx.bars() - horizon)``,
        stepping by ``horizon`` (non-overlapping forward windows)."""
        if horizon <= 0 or start_index < 0:
            raise ValueError("need horizon > 0 and startIndex >= 0")
        ics: List[float] = []
        hits = 0.0
        hit_pairs = 0.0
        turnover_sum = 0.0
        turnover_count = 0
        prev_weights = None

        for t in range(start_index, ctx.bars() - horizon, horizon):
            scores = factor.scores(ctx, t)
            fwd = SignalEvaluator.forward_returns(ctx, t, horizon)

            # Pairwise-complete: a NaN on either side drops the pair.
            ic = SignalEvaluator.spearman(scores, fwd)
            if math.isnan(ic):
                # Warm-up / unscored date: it is in NO metric's
                # denominator. Counting its all-zero weight vector as a
                # zero-turnover observation would understate turnover
                # against the same dates the IC series excludes.
                continue
            ics.append(ic)
            valid = (~np.isnan(scores)) & (scores != 0) & (~np.isnan(fwd))
            hit_pairs += int(np.sum(valid))
            hits += int(np.sum(np.sign(scores[valid])
                               == np.sign(fwd[valid])))
            # Turnover on the normalized (gross = 1) weights the scores
            # imply, between consecutive SCORED dates — one denominator
            # shared with the IC series.
            w = PortfolioConstruction.z_score_weights(scores, 1.0)
            if prev_weights is not None:
                l1 = float(np.sum(np.abs(w - prev_weights)))
                turnover_sum += l1 / 2  # buys and sells each counted once
                turnover_count += 1
            prev_weights = w

        n = len(ics)
        if n < 2:
            raise ValueError("fewer than 2 IC observations — extend the "
                             "sample or shrink the horizon")
        ic_arr = np.array(ics)
        mean = math_utils.mean(ic_arr)
        std = math_utils.std_dev(ic_arr)
        ir = 0.0 if std == 0 else mean / std
        return Report(
            factor.name(), mean, std, ir,
            0.0 if std == 0 else mean / (std / math.sqrt(n)),
            0.0 if hit_pairs == 0 else hits / hit_pairs,
            0.0 if turnover_count == 0 else turnover_sum / turnover_count,
            n, ic_arr)

    @staticmethod
    def quantile_returns(ctx: AlphaContext, factor: AlphaFactor,
                         start_index: int, horizon: int,
                         quantiles: int) -> QuantileReport:
        """Buckets each evaluation date's cross-section into
        ``quantiles`` score-ranked groups and averages the forward
        returns per group, over the same non-overlapping date grid as
        :meth:`evaluate`: dates step by ``horizon`` from
        ``start_index``, a NaN score or NaN forward return drops that
        (symbol, date) pair, and a date with fewer complete pairs than
        ``quantiles`` contributes to no bucket at all. Bucketing is by
        ascending score rank (``rank * quantiles // n``), so groups are
        as equal-sized as the cross-section allows; ties are split by
        input order at the boundary — quantile membership, unlike the
        rank IC, is not fully tie-invariant, stated."""
        if horizon <= 0 or start_index < 0:
            raise ValueError("need horizon > 0 and startIndex >= 0")
        if quantiles < 2:
            raise ValueError(f"need quantiles >= 2, got {quantiles}")
        sums = np.zeros(quantiles)
        counts = np.zeros(quantiles, dtype=int)
        dates = 0
        for t in range(start_index, ctx.bars() - horizon, horizon):
            scores = factor.scores(ctx, t)
            fwd = SignalEvaluator.forward_returns(ctx, t, horizon)
            # Pairwise-complete entries only, exactly as the IC computes.
            complete = (~np.isnan(scores)) & (~np.isnan(fwd))
            n = int(np.sum(complete))
            if n < quantiles:
                continue  # cannot form the buckets: date excluded entirely
            s = scores[complete]
            f = fwd[complete]
            # Sort by score ascending (stable, so boundary ties split by
            # input order), carrying the forward returns along.
            order = np.argsort(s, kind="stable")
            for rank in range(n):
                bucket = rank * quantiles // n
                sums[bucket] += f[order[rank]]
                counts[bucket] += 1
            dates += 1
        if dates == 0:
            raise ValueError(f"no date had >= {quantiles} scored names — "
                             "shrink quantiles or the NaNs")
        # dates >= 1 and n >= quantiles per date guarantee counts >= 1.
        means = sums / counts
        return QuantileReport(factor.name(), means, counts, quantiles, dates)

    @staticmethod
    def factor_exposure(ctx: AlphaContext, a: AlphaFactor, b: AlphaFactor,
                        start_index: int, step: int) -> float:
        """Mean cross-sectional rank correlation between two factors'
        scores — how much of factor B is already inside factor A. Above
        ~0.7 the "new" factor adds little beyond the old one."""
        total = 0.0
        n = 0
        for t in range(start_index, ctx.bars(), step):
            rho = SignalEvaluator.spearman(a.scores(ctx, t),
                                           b.scores(ctx, t))
            if not math.isnan(rho):
                total += rho
                n += 1
        if n == 0:
            raise ValueError("no overlapping scored dates")
        return total / n

    @staticmethod
    def forward_returns(ctx: AlphaContext, t: int,
                        horizon: int) -> np.ndarray:
        """Forward simple returns over ``(t, t+h]``, aligned with
        symbols."""
        return np.array([ctx.return_over(i, t, t + horizon)
                         for i in range(ctx.symbol_count())])

    @staticmethod
    def spearman(x, y) -> float:
        """Spearman rank correlation over pairwise-complete entries:
        rank both sides (midranks for ties), then Pearson on the ranks.
        NaN when fewer than 3 complete pairs — a correlation of 2
        points is noise."""
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        complete = (~np.isnan(xa)) & (~np.isnan(ya))
        if int(np.sum(complete)) < 3:
            return math.nan
        return math_utils.correlation(SignalEvaluator._ranks(xa[complete]),
                                      SignalEvaluator._ranks(ya[complete]))

    @staticmethod
    def _ranks(v: np.ndarray) -> np.ndarray:
        """Midrank transform (average rank for ties), 1-based — values
        only feed Pearson. Each value's midrank is the average of its
        first/last positions in the sorted array via binary search."""
        sorted_v = np.sort(v)
        lo = np.searchsorted(sorted_v, v, side="left")
        hi = np.searchsorted(sorted_v, v, side="right")
        return (lo + hi - 1) / 2.0 + 1
