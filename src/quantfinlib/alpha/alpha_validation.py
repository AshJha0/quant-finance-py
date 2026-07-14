"""Validation for alpha factors (port of Java ``alpha.AlphaValidation``)
— the overfitting defense, run before any capital-weighted conclusion is
drawn:

* **Walk-forward** — pick the best factor variant on a training window
  by in-sample IC, measure it on the following unseen window, roll
  forward. The IS->OOS gap is the overfitting, measured.
* **K-fold (blocked) cross-validation** — the IC recomputed on k
  contiguous time blocks. Time-series data forbids shuffled folds (they
  leak adjacent bars across the train/test line), so blocks it is; a
  factor that only works in one block is a regime story, not a signal.
* **Monte Carlo robustness** — a permutation test: re-pair score dates
  with return dates at random to build the null distribution of mean
  IC, and report where the observed value falls. This asks the right
  question ("could this IC arise from no relationship?") without any
  normality assumption.
* **Parameter sensitivity** — mean IC across a parameter sweep, plus
  the worst drop between adjacent parameters. A real effect degrades
  smoothly as parameters move; a spike at exactly one value is the
  signature of a lucky backtest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from quantfinlib.alpha.alpha_context import AlphaContext
from quantfinlib.alpha.alpha_factor import AlphaFactor
from quantfinlib.alpha.signal_evaluator import SignalEvaluator
from quantfinlib.util import math_utils


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: what was chosen, and how it did out of
    sample."""

    train_start: int
    test_start: int
    test_end: int
    chosen_factor: str
    in_sample_ic: float
    out_of_sample_ic: float


@dataclass(frozen=True)
class WalkForwardResult:
    """All folds plus the aggregate in-sample vs out-of-sample
    comparison."""

    folds: Tuple[Fold, ...]
    mean_in_sample_ic: float
    mean_out_of_sample_ic: float

    def efficiency(self) -> float:
        """OOS/IS efficiency: below ~0.5 the selection is mostly
        fitting noise."""
        return (0.0 if self.mean_in_sample_ic == 0
                else self.mean_out_of_sample_ic / self.mean_in_sample_ic)


@dataclass(frozen=True)
class CrossValidationResult:
    """Per-block ICs with their dispersion — consistency across
    regimes."""

    block_ics: np.ndarray
    mean_ic: float
    ic_std: float

    def sign_consistency(self) -> float:
        """Fraction of blocks where the IC kept the overall sign."""
        if self.block_ics.shape[0] == 0:
            return 0.0
        sign = np.sign(self.mean_ic)
        return float(np.mean(np.sign(self.block_ics) == sign))


@dataclass(frozen=True)
class RobustnessResult:
    """Observed mean IC against its permutation null distribution."""

    observed_mean_ic: float
    p_value: float
    null_mean: float
    null_std: float
    trials: int


@dataclass(frozen=True)
class SensitivityResult:
    """IC across the sweep plus the worst adjacent-parameter drop."""

    names: Tuple[str, ...]
    mean_ics: np.ndarray
    worst_neighbor_drop: float

    def best(self) -> str:
        """The candidate with the best mean IC."""
        return self.names[int(np.argmax(self.mean_ics))]


class AlphaValidation:
    """Static validation entry points; see the module docstring."""

    # ------------------------------------------------------------------
    # Walk-forward
    # ------------------------------------------------------------------

    @staticmethod
    def walk_forward(ctx: AlphaContext,
                     candidates: Sequence[AlphaFactor],
                     horizon: int, start_index: int,
                     train_bars: int, test_bars: int) -> WalkForwardResult:
        """Rolls a train/test split across the sample: each fold picks
        the candidate with the best training-window mean IC and scores
        it on the next ``test_bars`` unseen bars.

        Evaluation dates lie on ONE global grid (``start_index``,
        stepping by the horizon) shared by every fold: consecutive
        folds' training windows overlap by ``train_bars - test_bars``,
        so scoring per fold would recompute the same (candidate, date)
        work up to ``train_bars/test_bars`` times — instead the whole
        IC matrix is computed once (forward returns shared across
        candidates, too) and folds average slices of it. Window
        containment still holds: a date contributes to a window only
        when its ENTIRE forward window fits inside it.
        """
        if (not candidates or train_bars <= horizon
                or test_bars <= horizon):
            raise ValueError("need candidates and train/test windows "
                             "longer than the horizon")
        dates = AlphaValidation._grid(ctx, start_index, horizon)
        ic = AlphaValidation._ic_matrix(ctx, candidates, dates, horizon)

        folds: List[Fold] = []
        is_sum = 0.0
        oos_sum = 0.0
        train_start = start_index
        while train_start + train_bars + test_bars <= ctx.bars():
            test_start = train_start + train_bars
            test_end = test_start + test_bars
            # Model selection happens STRICTLY inside the training
            # window: only dates whose forward window fits before
            # test_start count.
            best = -1
            best_ic = -math.inf
            for c in range(len(candidates)):
                mean = AlphaValidation._window_mean(
                    ic[c], dates, train_start, test_start, horizon)
                # NaN (factor entirely in warm-up over this window)
                # never wins.
                if not math.isnan(mean) and mean > best_ic:
                    best_ic = mean
                    best = c
            if best < 0:
                raise ValueError(
                    "no candidate produced a training IC in fold starting "
                    f"at {train_start} — factor warm-up likely exceeds "
                    "the training window")
            oos = AlphaValidation._window_mean(
                ic[best], dates, test_start, test_end, horizon)
            folds.append(Fold(train_start, test_start, test_end,
                              candidates[best].name(), best_ic, oos))
            is_sum += best_ic
            oos_sum += oos
            train_start += test_bars
        if not folds:
            raise ValueError("sample too short for one train+test fold")
        return WalkForwardResult(tuple(folds), is_sum / len(folds),
                                 oos_sum / len(folds))

    @staticmethod
    def _grid(ctx: AlphaContext, start_index: int,
              horizon: int) -> np.ndarray:
        """Evaluation dates: start_index, stepping by horizon, forward
        window in-sample."""
        return np.arange(start_index, ctx.bars() - horizon, horizon)

    @staticmethod
    def _ic_matrix(ctx: AlphaContext, candidates: Sequence[AlphaFactor],
                   dates: np.ndarray, horizon: int) -> np.ndarray:
        """Per-candidate IC at every grid date, computed ONCE: forward
        returns are candidate-independent and shared across the whole
        sweep."""
        ic = np.zeros((len(candidates), dates.shape[0]))
        for d in range(dates.shape[0]):
            t = int(dates[d])
            fwd = SignalEvaluator.forward_returns(ctx, t, horizon)
            for c in range(len(candidates)):
                ic[c, d] = SignalEvaluator.spearman(
                    candidates[c].scores(ctx, t), fwd)
        return ic

    @staticmethod
    def _window_mean(ic: np.ndarray, dates: np.ndarray, from_: int,
                     to: int, horizon: int) -> float:
        """Mean IC over grid dates whose whole forward window fits in
        ``[from_, to)``."""
        mask = ((dates >= from_) & (dates + horizon < to)
                & (~np.isnan(ic)))
        n = int(np.sum(mask))
        return math.nan if n == 0 else float(np.mean(ic[mask]))

    # ------------------------------------------------------------------
    # Blocked cross-validation
    # ------------------------------------------------------------------

    @staticmethod
    def cross_validate(ctx: AlphaContext, factor: AlphaFactor,
                       horizon: int, start_index: int,
                       k: int) -> CrossValidationResult:
        """Splits the evaluation range into ``k`` contiguous blocks and
        recomputes the mean IC inside each. (Stateless factors have
        nothing to fit, so this is a pure consistency check — the
        honest reading of "cross-validation" for unfitted signals.)"""
        span = ctx.bars() - start_index
        if k < 2 or span // k <= horizon:
            raise ValueError("blocks must be longer than the horizon")
        ics = np.zeros(k)
        block_len = span // k
        for b in range(k):
            from_ = start_index + b * block_len
            to = ctx.bars() if b == k - 1 else from_ + block_len
            ics[b] = AlphaValidation.mean_ic(ctx, factor, from_, to, horizon)
        return CrossValidationResult(ics, math_utils.mean(ics),
                                     math_utils.std_dev(ics))

    # ------------------------------------------------------------------
    # Monte Carlo robustness (permutation test)
    # ------------------------------------------------------------------

    @staticmethod
    def monte_carlo_robustness(ctx: AlphaContext, factor: AlphaFactor,
                               horizon: int, start_index: int,
                               trials: int, seed: int) -> RobustnessResult:
        """Permutation test on the score/return pairing: per trial,
        scores from date ``t_i`` are paired with forward returns from a
        shuffled date ``t_j``, destroying any true predictive link
        while preserving both marginal distributions. The p-value is
        the fraction of trials whose ``|mean IC|`` reaches the observed
        ``|mean IC|`` (two-sided, add-one smoothed so p is never
        exactly 0).

        **Deliberate conservatism**: a signal whose scores never change
        over time (a static ranking) is invariant under date
        permutation, so it earns p ~ 1 regardless of its in-sample IC —
        correctly so, because a time-invariant cross-section against
        persistent drifts is one effective observation, however many
        dates it is sampled on. Only signals whose time variation
        aligns with return variation can earn a small p here.
        """
        if trials < 10:
            raise ValueError("need at least 10 trials")
        # Precompute per-date scores and forward returns once.
        scores: List[np.ndarray] = []
        forwards: List[np.ndarray] = []
        for t in range(start_index, ctx.bars() - horizon, horizon):
            scores.append(factor.scores(ctx, t))
            forwards.append(SignalEvaluator.forward_returns(ctx, t, horizon))
        dates = len(scores)
        if dates < 3:
            raise ValueError("too few evaluation dates for a "
                             "permutation test")
        identity = np.arange(dates)
        observed = AlphaValidation._mean_ic_of(scores, forwards, identity)

        rng = np.random.default_rng(seed)
        null_sum = 0.0
        null_sq = 0.0
        as_extreme = 0
        for _ in range(trials):
            perm = rng.permutation(dates)
            ic_null = AlphaValidation._mean_ic_of(scores, forwards, perm)
            null_sum += ic_null
            null_sq += ic_null * ic_null
            if abs(ic_null) >= abs(observed):
                as_extreme += 1
        null_mean = null_sum / trials
        null_var = null_sq / trials - null_mean * null_mean
        # Add-one smoothing: with T trials the resolution is 1/(T+1).
        p = (as_extreme + 1.0) / (trials + 1.0)
        return RobustnessResult(observed, p, null_mean,
                                math.sqrt(max(0.0, null_var)), trials)

    # ------------------------------------------------------------------
    # Parameter sensitivity
    # ------------------------------------------------------------------

    @staticmethod
    def parameter_sensitivity(ctx: AlphaContext,
                              sweep: Sequence[AlphaFactor],
                              horizon: int,
                              start_index: int) -> SensitivityResult:
        """Evaluates each candidate (an ORDERED parameter sweep —
        neighbors in the list must be neighbors in parameter space) and
        reports the worst IC drop between adjacent candidates. Small
        drop = plateau = robust; large drop = the chosen parameter is a
        lucky spike."""
        if len(sweep) < 2:
            raise ValueError("a sweep needs at least 2 candidates")
        # Shared grid + shared forward returns across the sweep — the
        # same caching walk_forward uses: forward returns are
        # candidate-free.
        dates = AlphaValidation._grid(ctx, start_index, horizon)
        ic_by_candidate = AlphaValidation._ic_matrix(ctx, sweep, dates,
                                                     horizon)
        names = tuple(f.name() for f in sweep)
        ics = np.array([
            AlphaValidation._window_mean(ic_by_candidate[i], dates,
                                         start_index, ctx.bars(), horizon)
            for i in range(len(sweep))
        ])
        worst_drop = float(np.max(np.abs(np.diff(ics)))) if len(sweep) > 1 else 0.0
        return SensitivityResult(names, ics, worst_drop)

    # ------------------------------------------------------------------
    # Shared IC arithmetic
    # ------------------------------------------------------------------

    @staticmethod
    def mean_ic(ctx: AlphaContext, factor: AlphaFactor, from_: int,
                to_exclusive: int, horizon: int) -> float:
        """Mean rank IC over ``[from_, to_exclusive)``, stepping by the
        horizon. The ENTIRE forward window ``(t, t+horizon]`` must fit
        inside the range: letting it spill past ``to_exclusive`` would
        leak test-window returns into training-window ICs — the
        walk-forward selection would peek at exactly the data it claims
        not to see."""
        total = 0.0
        n = 0
        end = min(to_exclusive, ctx.bars())
        t = from_
        while t + horizon < end:
            ic = SignalEvaluator.spearman(
                factor.scores(ctx, t),
                SignalEvaluator.forward_returns(ctx, t, horizon))
            if not math.isnan(ic):
                total += ic
                n += 1
            t += horizon
        return math.nan if n == 0 else total / n

    @staticmethod
    def _mean_ic_of(scores: List[np.ndarray], forwards: List[np.ndarray],
                    pairing: np.ndarray) -> float:
        total = 0.0
        n = 0
        for i in range(pairing.shape[0]):
            ic = SignalEvaluator.spearman(scores[i],
                                          forwards[int(pairing[i])])
            if not math.isnan(ic):
                total += ic
                n += 1
        return math.nan if n == 0 else total / n
