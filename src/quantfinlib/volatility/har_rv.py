"""Port of Java ``com.quantfinlib.volatility.HarRv``.

HAR-RV (Corsi's Heterogeneous AutoRegressive realized-volatility model)
— the forecasting benchmark GARCH papers have to beat, and it is three
regressors and an intercept:

    RV_{t+1} = c + bd * RV_t + bw * RVbar_t^(5) + bm * RVbar_t^(22) + eps

daily, weekly-average and monthly-average realized variance — the
"heterogeneous" traders operating at three horizons. Fits by plain OLS
on the normal equations (no optimizer), forecasts one step ahead, and
floors the forecast at zero (a negative variance forecast is an
extrapolation artifact, not a market view).

Feed it realized DAILY variance — squared returns summed intraday, or a
jump-robust bipower variance when jumps should not contaminate the
forecast (the standard pairing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from quantfinlib.util import math_utils as mu

_WEEK = 5
_MONTH = 22


class HarRv:
    """Static HAR-RV OLS fit and one-step forecast (Java parity)."""

    @dataclass(frozen=True)
    class Params:
        """Fitted coefficients: ``rv+ = c + bd*d + bw*w + bm*m``."""
        intercept: float
        beta_daily: float
        beta_weekly: float
        beta_monthly: float

    @staticmethod
    def fit(realized_variance) -> "HarRv.Params":
        """Fits by OLS. Needs enough history for the monthly window plus a
        meaningful regression sample.

        Args:
            realized_variance: daily RV series, >= 60 finite non-negative
                observations.

        Raises:
            ValueError: on short history or negative/NaN/inf observations
                (the ``not (rv >= 0)`` gate is NaN-rejecting).
        """
        rv = np.asarray(realized_variance, dtype=float)
        n = rv.shape[0]
        if n < 60:
            raise ValueError(f"need >= 60 daily observations, got {n}")
        if bool(np.any(~(rv >= 0))) or bool(np.any(rv == np.inf)):
            raise ValueError("realized variance must be >= 0 and finite")
        # Rows t = MONTH-1 .. n-2 predict rv[t+1].
        rows = n - _MONTH
        if rows < 30:
            raise ValueError(f"only {rows} regression rows")
        t = np.arange(_MONTH - 1, n - 1)
        weekly = sliding_window_view(rv, _WEEK).mean(axis=1)   # mean over [i, i+5)
        monthly = sliding_window_view(rv, _MONTH).mean(axis=1)  # mean over [i, i+22)
        x = np.column_stack([
            np.ones(t.shape[0]),
            rv[t],
            weekly[t - _WEEK + 1],    # mean over [t-4, t+1)
            monthly[t - _MONTH + 1],  # mean over [t-21, t+1)
        ])
        y = rv[t + 1]
        beta = mu.solve_linear(x.T @ x, x.T @ y)
        return HarRv.Params(float(beta[0]), float(beta[1]),
                            float(beta[2]), float(beta[3]))

    @staticmethod
    def forecast(realized_variance, p: "HarRv.Params") -> float:
        """One-step-ahead RV forecast from the series' most recent day/
        week/month, floored at zero.

        Raises:
            ValueError: with fewer than 22 observations.
        """
        rv = np.asarray(realized_variance, dtype=float)
        n = rv.shape[0]
        if n < _MONTH:
            raise ValueError(f"need >= {_MONTH} observations to forecast")
        d = rv[n - 1]
        w = mu.mean(rv, n - _WEEK, n)
        m = mu.mean(rv, n - _MONTH, n)
        return max(0.0, p.intercept
                   + p.beta_daily * d + p.beta_weekly * w + p.beta_monthly * m)
