"""Ornstein-Uhlenbeck estimation (port of Java
``microstructure.OrnsteinUhlenbeck``).

The mean-reversion engine under every pairs trade and basis position:
``dx = kappa*(theta - x)*dt + sigma*dW``. Fit from a sampled series by
exact AR(1) mapping (``x_{t+1} = a + b*x_t + eps`` with
``b = exp(-kappa*dt)``), giving the three numbers a spread trader
actually uses:

* **half-life** ``ln2/kappa`` — how long the spread takes to close half
  its gap: the holding-period estimate, and the first filter (a 200-day
  half-life is not a trade);
* **z-score** ``(x - theta)/sigma_stat`` with the STATIONARY stdev
  ``sigma/sqrt(2*kappa)`` — entry/exit in units the strategy can
  threshold;
* **the refusal**: a fitted ``b >= 1`` means the series shows NO mean
  reversion in-sample — the fit raises rather than reporting an
  infinite half-life as a tradable number, because fitting OU to a
  random walk is how pairs desks die.

Small-sample honesty: the OLS AR(1) slope is DOWNWARD-biased in finite
samples (Kendall: ``E[b_hat - b] ~ -(1+3b)/n``), so near the minimum
n=30 the fitted kappa runs high and the half-life SHORT. Treat
short-sample half-lives as optimistic lower bounds. Stated, not
corrected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class OrnsteinUhlenbeck:
    """Static OU fitter; see the module docstring."""

    @dataclass(frozen=True)
    class Params:
        """Fitted OU parameters.

        Attributes:
            kappa: Mean-reversion speed, per unit of ``dt``'s time.
            theta: Long-run mean.
            sigma: Diffusion volatility (same time units as kappa).
            half_life: ``ln2/kappa``, in ``dt`` time units.
        """

        kappa: float
        theta: float
        sigma: float
        half_life: float

        def stationary_stdev(self) -> float:
            """Stationary standard deviation ``sigma/sqrt(2*kappa)`` —
            the z-score's yardstick."""
            return self.sigma / math.sqrt(2 * self.kappa)

        def z_score(self, x: float) -> float:
            """``(x - theta) /`` stationary stdev: the entry/exit signal."""
            if not math.isfinite(x):
                raise ValueError("x must be finite")
            return (x - self.theta) / self.stationary_stdev()

    @staticmethod
    def fit(series, dt: float) -> "OrnsteinUhlenbeck.Params":
        """Fits OU to a series sampled every ``dt`` time units (e.g.
        dt = 1/252 for daily samples in years). Raises when the series
        shows no mean reversion — see the module docstring.

        Args:
            series: >= 30 finite observations.
            dt: Sampling interval, > 0.
        """
        x = np.asarray(series, dtype=float)
        if x.shape[0] < 30:
            raise ValueError(f"need >= 30 observations, got {x.shape[0]}")
        if not (dt > 0) or dt == math.inf:
            raise ValueError("dt must be positive and finite")
        if not np.all(np.isfinite(x)):
            raise ValueError("series must be finite")
        # AR(1) OLS: x_{t+1} = a + b x_t + e.
        n = x.shape[0] - 1
        xs = x[:-1]
        ys = x[1:]
        sx = float(np.sum(xs))
        sy = float(np.sum(ys))
        sxx = float(np.sum(xs * xs))
        sxy = float(np.sum(xs * ys))
        denom = n * sxx - sx * sx
        if denom <= 0:
            raise ValueError("degenerate series (constant?)")
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        if not (0 < b < 1):
            raise ValueError(
                f"fitted AR coefficient {b} is outside (0, 1): the series "
                "shows no mean reversion in-sample — an OU fit here would "
                "be a random walk in costume")
        kappa = -math.log(b) / dt
        theta = a / (1 - b)
        # Residual variance -> diffusion sigma via the exact
        # discretization: Var(e) = sigma^2 (1 - b^2) / (2 kappa).
        e = ys - a - b * xs
        sse = float(np.sum(e * e))
        var_e = sse / (n - 2)
        sigma = math.sqrt(var_e * 2 * kappa / (1 - b * b))
        return OrnsteinUhlenbeck.Params(kappa, theta, sigma,
                                        math.log(2) / kappa)

    @staticmethod
    def last_z_score(series, dt: float) -> float:
        """Convenience: the fitted z-score of the LAST observation."""
        p = OrnsteinUhlenbeck.fit(series, dt)
        return p.z_score(float(np.asarray(series, dtype=float)[-1]))
