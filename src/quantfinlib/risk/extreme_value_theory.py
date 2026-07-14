"""Extreme value theory via peaks-over-threshold.

Port of Java ``com.quantfinlib.risk.ExtremeValueTheory``. Fits the
Generalized Pareto Distribution to the exceedances over a high
threshold, then extrapolates along the fitted tail::

    VaR_p = u + (beta/xi) * [((n/N_u)(1-p))^{-xi} - 1]

The shape xi is the number to stare at: xi ~ 0 is an exponential tail,
xi > 0 a power-law tail (equity returns typically xi ~ 0.2-0.4), and
xi >= 1 means the tail mean does not exist
(:meth:`GpdFit.expected_shortfall` refuses rather than returning a
finite lie). Fitting uses probability-weighted moments — closed-form,
no optimizer, well-behaved for xi < 0.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GpdFit:
    """A fitted POT tail model.

    Attributes:
        threshold: u — losses above this were fitted.
        shape: xi — the tail index (fat when positive).
        scale: beta — the GPD scale.
        exceedances: how many losses exceeded u.
        sample_size: total losses the tail sits in.
    """

    threshold: float
    shape: float
    scale: float
    exceedances: int
    sample_size: int

    def var(self, p: float) -> float:
        """Tail VaR at one-sided confidence ``p`` (must lie in the tail)."""
        tail_prob = self.exceedances / self.sample_size
        if not (p > 1 - tail_prob) or not (p < 1):
            raise ValueError(
                f"p = {p} must lie in ({1 - tail_prob}, 1) — below that use "
                "plain historical VaR; 99.9% is 0.999, not 99.9")
        ratio = (1 - p) / tail_prob
        if abs(self.shape) < 1e-9:
            return self.threshold - self.scale * math.log(ratio)
        return self.threshold + self.scale / self.shape * (ratio ** -self.shape - 1)

    def expected_shortfall(self, p: float) -> float:
        """Tail expected shortfall at ``p``.

        Raises:
            RuntimeError: when xi >= 1 — the fitted tail's mean is
                infinite, and a finite number here would be the exact
                lie EVT exists to prevent.
        """
        if self.shape >= 1:
            raise RuntimeError(
                f"shape {self.shape} >= 1: the fitted tail has no finite mean")
        v = self.var(p)
        return (v + self.scale - self.shape * self.threshold) / (1 - self.shape)


def fit_pot(losses, threshold_quantile: float) -> GpdFit:
    """Fits a GPD to the losses exceeding the ``threshold_quantile`` of the
    sample (e.g. 0.90), via probability-weighted moments. Losses are
    positive numbers (feed ``-returns`` or a loss series directly)."""
    losses = np.asarray(losses, dtype=float)
    if losses.shape[0] < 50:
        raise ValueError("need >= 50 losses for a tail fit")
    if not (0.5 <= threshold_quantile < 1):
        raise ValueError("thresholdQuantile must be in [0.5, 1)")
    sorted_losses = np.sort(losses)
    n = sorted_losses.shape[0]
    # NaN/Infinity sort into the tail — exactly where they would poison
    # the PWMs with a misleading "degenerate tail" message.
    if not math.isfinite(sorted_losses[0]) or not math.isfinite(sorted_losses[n - 1]):
        raise ValueError("losses must be finite")
    threshold_index = math.floor(threshold_quantile * n) - 1
    u = float(sorted_losses[max(0, threshold_index)])

    # Exceedances y_i = loss - u, ascending — STRICTLY above u: on
    # discretized data (P&L snapped to ticks) ties equal to u would
    # otherwise enter as y = 0 exceedances, deflating both PWMs and
    # biasing the shape estimate.
    start = threshold_index + 1
    while start < n and sorted_losses[start] <= u:
        start += 1
    m = n - start
    if m < 10:
        raise ValueError(
            f"only {m} exceedances — lower the threshold or bring more data")
    # Probability-weighted moments (Hosking & Wallis 1987):
    # b0 = E[Y] = mean(y);  a1 = E[Y*(1-F(Y))], estimated over the
    # ASCENDING order statistics with weight (m-1-i)/(m-1). Then
    # xi = 2 - b0/(b0 - 2 a1),  beta = 2 b0 a1/(b0 - 2 a1) — for
    # GPD(xi, beta): b0 = beta/(1-xi), a1 = beta/(2(2-xi)), and the
    # algebra inverts exactly.
    y = sorted_losses[start:] - u
    i = np.arange(m, dtype=float)
    b0 = float(np.sum(y)) / m
    a1 = float(np.sum(y * (m - 1 - i) / (m - 1))) / m
    denom = b0 - 2 * a1
    # Java divides through to Infinity and fails the finite gate below;
    # Python floats would raise ZeroDivisionError first, so gate denom = 0
    # explicitly with the same message.
    if denom == 0:
        raise ValueError("degenerate tail (all exceedances equal?) — PWM fit failed")
    shape = 2 - b0 / denom
    scale = 2 * b0 * a1 / denom
    if not (scale > 0) or not math.isfinite(shape):
        raise ValueError("degenerate tail (all exceedances equal?) — PWM fit failed")
    return GpdFit(u, shape, scale, m, n)
