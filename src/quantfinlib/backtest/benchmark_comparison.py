"""Benchmark-relative performance (port of Java ``backtest.BenchmarkComparison``).

The numbers an allocator actually asks for. A standalone Sharpe answers
"was this good?"; these answer "was this good *compared to just buying
the index*?", which is the question every active strategy must survive.

* **beta** — ``Cov(r_s, r_b) / Var(r_b)``: how much of the strategy is
  just the benchmark in disguise;
* **alpha** — annualized Jensen intercept
  ``(mean(r_s) - beta * mean(r_b)) * P`` at zero risk-free rate: the
  return left over after the benchmark exposure is paid for;
* **tracking error** — annualized stdev of active returns
  ``a_t = r_s,t - r_b,t``: how far the strategy strays;
* **information ratio** — annualized ``mean(a) / TE``: alpha per unit of
  straying. The active-management analogue of Sharpe; sustained IR > 0.5
  is good, > 1 is elite;
* **up/down capture** — mean strategy return over periods when the
  benchmark rose (fell), divided by the benchmark's own mean in those
  periods. The dream profile is up > 1, down < 1. Arithmetic means of
  per-period returns, not compounded — stated, and the right choice at
  daily granularity where cross-terms are negligible. ``NaN`` when the
  benchmark had no up (down) periods: no evidence, not zero.

Both series must be the same length and aligned period-by-period — this
class cannot detect a one-day offset, and an offset silently destroys
beta (it becomes a lead-lag estimate). Align first, then compare.
Requires the benchmark to actually vary; comparing against a constant
series is refused rather than returning a 0/0 beta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.backtest import _risk
from quantfinlib.util import math_utils as mu


class BenchmarkComparison:
    """Static benchmark-relative analytics; see the module docstring."""

    @dataclass(frozen=True)
    class Result:
        """Benchmark-relative statistics.

        Attributes:
            alpha: Annualized Jensen alpha (risk-free = 0).
            beta: Regression beta vs the benchmark.
            tracking_error: Annualized stdev of active returns.
            information_ratio: Annualized active return / tracking error;
                0 when TE is 0 (identical series).
            up_capture: Capture ratio over benchmark-up periods (NaN if none).
            down_capture: Capture ratio over benchmark-down periods (NaN if none).
            active_return: Annualized mean of (strategy - benchmark).
        """

        alpha: float
        beta: float
        tracking_error: float
        information_ratio: float
        up_capture: float
        down_capture: float
        active_return: float

    @staticmethod
    def compare(strategy, benchmark,
                periods_per_year: int) -> "BenchmarkComparison.Result":
        """Compares aligned per-period return series (>= 3 periods, finite).

        Raises:
            ValueError: on length mismatch, fewer than 3 periods,
                non-positive ``periods_per_year``, non-finite returns, or
                a benchmark that carries no variance.
        """
        rs = np.asarray(strategy, dtype=float)
        rb = np.asarray(benchmark, dtype=float)
        if rs.shape[0] != rb.shape[0]:
            raise ValueError(
                f"length mismatch: strategy={rs.shape[0]} benchmark={rb.shape[0]}")
        n = rs.shape[0]
        if n < 3:
            raise ValueError(f"need >= 3 aligned periods, got {n}")
        if periods_per_year <= 0:
            raise ValueError(f"periodsPerYear must be > 0, got {periods_per_year}")
        if not (np.all(np.isfinite(rs)) and np.all(np.isfinite(rb))):
            raise ValueError("non-finite return in input series")
        var_b = mu.variance(rb)
        if not (var_b > 0):
            raise ValueError("benchmark returns carry no variance")

        beta = _risk.beta(rs, rb)
        mean_s = mu.mean(rs)
        mean_b = mu.mean(rb)
        alpha = (mean_s - beta * mean_b) * periods_per_year

        active = rs - rb
        active_ann = mu.mean(active) * periods_per_year
        te = mu.std_dev(active) * math.sqrt(periods_per_year)
        ir = active_ann / te if te > 0 else 0.0

        up = rb > 0
        down = rb < 0
        ups = int(np.sum(up))
        downs = int(np.sum(down))
        up_capture = math.nan if ups == 0 else (
            float(np.sum(rs[up])) / ups) / (float(np.sum(rb[up])) / ups)
        down_capture = math.nan if downs == 0 else (
            float(np.sum(rs[down])) / downs) / (float(np.sum(rb[down])) / downs)

        return BenchmarkComparison.Result(alpha, beta, te, ir,
                                          up_capture, down_capture, active_ann)
