"""Short-term spread prediction (port of Java
``microstructure.SpreadForecaster``). The bid/ask spread an execution
algo will pay in a few seconds is well modelled by two components a
live feed gives you for free:

1. **A time-of-day baseline** -- spreads are wide at the open, tight
   midday, wide into the close. Each session accumulates a per-bucket
   mean; :meth:`roll_day` folds it into the baseline with the
   ``day_alpha`` day-over-day EWMA (the first session seeds it
   directly) -- the spread analogue of
   :class:`~quantfinlib.microstructure.volume_curve.VolumeCurve`;
2. **A fast mean-reverting deviation** -- the current spread relative
   to its time-of-day baseline, blended per observation with
   :data:`DEVIATION_ALPHA` and decayed toward 0 with the configured
   half-life. Spreads spike on events and revert; blending the live
   deviation with the baseline forecasts the near-term spread better
   than either alone.

:meth:`forecast` returns the predicted spread over the next moment.
Before the first :meth:`roll_day` there is no learned baseline yet, so
the forecast degrades honestly to the last observed spread.
Cross-asset, single writer.

Note: no ``persist.Checkpoint`` lane in this port (see
:mod:`quantfinlib.microstructure.kyles_lambda`).
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils

#: Per-observation blend weight of the live deviation (distinct from
#: ``day_alpha``, which is the DAY-over-day baseline weight -- the two
#: timescales must not share a knob).
DEVIATION_ALPHA = 0.25


class SpreadForecaster:
    """Time-of-day baseline plus a mean-reverting live deviation; see
    the module docstring."""

    __slots__ = ("_buckets", "_day_alpha", "_deviation_half_life_nanos",
                 "_baseline_ewma", "_today_sum", "_today_count",
                 "_deviation", "_deviation_time", "_has_deviation",
                 "_last_spread", "_days_learned")

    def __init__(self, buckets: int = 78, day_alpha: float = 0.1,
                 deviation_half_life_nanos: int = 5_000_000_000) -> None:
        """
        Args:
            buckets: time buckets per session (e.g. 78 for equities,
                288 for a 24h FX day).
            day_alpha: baseline EWMA weight across days, e.g. 0.1.
            deviation_half_life_nanos: how fast a spread shock reverts
                to baseline.
        """
        if (buckets < 1 or day_alpha <= 0 or day_alpha > 1
                or deviation_half_life_nanos <= 0):
            raise ValueError(
                "need buckets >= 1, dayAlpha in (0,1], half-life > 0")
        self._buckets = buckets
        self._day_alpha = day_alpha
        self._deviation_half_life_nanos = deviation_half_life_nanos
        self._baseline_ewma = np.zeros(buckets)
        self._today_sum = np.zeros(buckets)
        self._today_count = np.zeros(buckets, dtype=np.int64)
        self._deviation = 0.0
        self._deviation_time = 0
        self._has_deviation = False
        self._last_spread = math.nan
        self._days_learned = 0

    def seed_baseline(self, spread_per_bucket) -> "SpreadForecaster":
        """Seeds the time-of-day baseline from a known shape --
        optional."""
        v = np.asarray(spread_per_bucket, dtype=float)
        if v.shape[0] != self._buckets:
            raise ValueError("baseline length must equal bucket count")
        self._baseline_ewma[:] = v
        self._days_learned = 1
        return self

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def on_spread(self, bucket: int, spread: float,
                  timestamp_nanos: int) -> None:
        """Observed spread at ``bucket``. Accumulates today's
        per-bucket mean (folded into the baseline at :meth:`roll_day`)
        and updates the mean-reverting deviation from the learned
        baseline. Non-finite or negative spreads are ignored -- one
        +inf must not poison a bucket forever."""
        if not (spread >= 0) or spread == math.inf:
            return                          # !(x >= 0) also catches NaN
        self._last_spread = spread
        self._today_sum[bucket] += spread
        self._today_count[bucket] += 1
        if self._days_learned == 0 or self._baseline_ewma[bucket] == 0:
            # No baseline yet -- for the session OR for this bucket
            # (never observed on a prior day): deviating from 0 would
            # inject the whole spread as a "shock" into the shared
            # deviation and contaminate forecasts at every bucket.
            return
        dev = spread - self._baseline_ewma[bucket]
        prior = self._decayed_deviation(timestamp_nanos)
        self._deviation = (prior + DEVIATION_ALPHA * (dev - prior)
                           if self._has_deviation else dev)
        self._has_deviation = True
        self._deviation_time = timestamp_nanos

    def roll_day(self) -> None:
        """Closes the session: folds today's per-bucket mean spreads
        into the baseline with the day-over-day EWMA (buckets with no
        observations keep their learned value) and resets the
        intraday state."""
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
        self._deviation = 0.0
        self._has_deviation = False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def forecast(self, bucket: int, now_nanos: int) -> float:
        """Forecast spread at ``bucket`` as of ``now_nanos``: the
        learned time-of-day baseline plus the mean-reverting live
        deviation. Before the first :meth:`roll_day`/seed there is no
        baseline, so it returns the last observed spread (the honest
        live estimate), or NaN before any observation at all."""
        if self._days_learned == 0:
            return self._last_spread
        return max(0.0, self._baseline_ewma[bucket]
                   + self._decayed_deviation(now_nanos))

    def baseline(self, bucket: int) -> float:
        """The learned time-of-day baseline spread for a bucket (0
        until learned)."""
        return float(self._baseline_ewma[bucket])

    def current_deviation(self, now_nanos: int) -> float:
        """Current deviation from baseline (decayed to now)."""
        return self._decayed_deviation(now_nanos)

    def _decayed_deviation(self, now: int) -> float:
        if not self._has_deviation:
            return 0.0
        return self._deviation * math_utils.decay_factor(
            now - self._deviation_time, self._deviation_half_life_nanos)

    def buckets(self) -> int:
        return self._buckets

    def days_learned(self) -> int:
        return self._days_learned
