"""Intraday volatility seasonality (port of Java
``microstructure.VolatilityCurve``) -- the third leg of the
seasonality trio beside :class:`~quantfinlib.microstructure.volume_curve.VolumeCurve`
and :class:`~quantfinlib.microstructure.spread_forecaster.SpreadForecaster`:
volatility is U-shaped through an equity day (wild open, quiet lunch,
busy close) and session-humped through an FX day (London open, NY
overlap), so "is the market volatile right now?" is meaningless
without "...for this time of day."

Each session accumulates a per-bucket mean of the observed volatility;
:meth:`roll_day` folds it into a per-bucket baseline with the
day-over-day EWMA, the first session seeding directly.

:meth:`regime` is the point of the class: the normalized
volatility-regime signal -- current vol against the time-of-day
baseline, mapped to ~0 (calm for this hour) ... 1 (extreme), so "the
open is always wild" doesn't read as an urgency signal but a genuinely
wild lunchtime does. Before any baseline is learned the regime is 0
(neutral) -- the honest default. Cross-asset, single writer.

Note: no ``persist.Checkpoint`` lane in this port (see
:mod:`quantfinlib.microstructure.kyles_lambda`).
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils


class VolatilityCurve:
    """Learned time-of-day volatility baseline with a normalized
    regime signal; see the module docstring."""

    __slots__ = ("_buckets", "_day_alpha", "_baseline_ewma", "_today_sum",
                 "_today_count", "_days_learned")

    def __init__(self, buckets: int = 78, day_alpha: float = 0.1) -> None:
        """
        Args:
            buckets: time buckets per session (78 equities, 288 for a
                24h FX day).
            day_alpha: baseline EWMA weight across days, e.g. 0.1.
        """
        if buckets < 1 or day_alpha <= 0 or day_alpha > 1:
            raise ValueError("need buckets >= 1, dayAlpha in (0,1]")
        self._buckets = buckets
        self._day_alpha = day_alpha
        self._baseline_ewma = np.zeros(buckets)
        self._today_sum = np.zeros(buckets)
        self._today_count = np.zeros(buckets, dtype=np.int64)
        self._days_learned = 0

    def seed_baseline(self, vol_per_bucket) -> "VolatilityCurve":
        """Seeds the baseline from a known shape (same units you will
        feed) -- optional."""
        v = np.asarray(vol_per_bucket, dtype=float)
        if v.shape[0] != self._buckets:
            raise ValueError("baseline length must equal bucket count")
        self._baseline_ewma[:] = v
        self._days_learned = 1
        return self

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def on_vol(self, bucket: int, vol_per_sqrt_second: float) -> None:
        """An observed volatility reading for ``bucket`` (e.g.
        return-per-sqrt-second, polled per interval). Non-finite or
        negative readings are ignored."""
        if (not (vol_per_sqrt_second >= 0)
                or vol_per_sqrt_second == math.inf):
            return                          # !(x >= 0) also catches NaN
        self._today_sum[bucket] += vol_per_sqrt_second
        self._today_count[bucket] += 1

    def roll_day(self) -> None:
        """Closes the session: folds today's per-bucket mean vol into
        the baseline (buckets without observations keep their learned
        value). Seeding is PER BUCKET -- a bucket first observed on
        day 5 (feed started mid-session on day 1, a half day skipped
        the afternoon) seeds from its own first observation rather
        than EWMA-ramping from 0, which would leave :meth:`regime`
        falsely reading "extreme" at that hour for weeks."""
        for b in range(self._buckets):
            if self._today_count[b] > 0:
                today_mean = self._today_sum[b] / self._today_count[b]
                self._baseline_ewma[b] = (
                    today_mean if self._baseline_ewma[b] == 0
                    else self._baseline_ewma[b]
                    + self._day_alpha * (today_mean - self._baseline_ewma[b]))
            self._today_sum[b] = 0.0
            self._today_count[b] = 0
        self._days_learned += 1

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def baseline(self, bucket: int) -> float:
        """The learned time-of-day baseline vol for a bucket (0 until
        learned)."""
        return float(self._baseline_ewma[bucket])

    def regime(self, bucket: int, current_vol_per_sqrt_second: float) -> float:
        """The normalized volatility-regime signal: how elevated the
        current vol is against this hour's baseline,
        ``clamp(current/baseline - 1, 0, 1)``. 0 when
        calm-for-the-hour, unlearned, or fed a non-finite reading -- a
        bad input reads as neutral, never as urgency."""
        base = self._baseline_ewma[bucket]
        if (base <= 0 or not (current_vol_per_sqrt_second >= 0)
                or current_vol_per_sqrt_second == math.inf):
            return 0.0
        return math_utils.clamp(current_vol_per_sqrt_second / base - 1, 0, 1)

    def buckets(self) -> int:
        return self._buckets

    def days_learned(self) -> int:
        return self._days_learned
