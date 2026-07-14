"""Dynamic intraday volume prediction (port of Java
``microstructure.VolumeCurve``) -- the model that makes a VWAP
schedule live instead of historical. Two parts:

1. **The learned profile** -- per-bucket volume EWMA across days (feed
   each session via :meth:`on_volume` and close it with
   :meth:`roll_day`), giving the classic U-shaped expected curve
   without any external data;
2. **The intraday rescale** -- today rarely trades the average day's
   volume. The projection scales the remaining curve by today's
   realized-vs-expected ratio, shrunk toward 1 early in the day when
   the ratio is mostly noise: ``scale = 1 + w*(ratio - 1)`` with ``w``
   = the fraction of the expected day already elapsed. A 2x morning
   turns into a confident 2x afternoon only as evidence accumulates.

:meth:`expected_fraction_elapsed` is the live VWAP curve input.
Cross-asset (volumes are just sums), single writer.

:meth:`VolumeCurve.write_state`/:meth:`VolumeCurve.read_state` persist
the learned profile (cross-day state) via ``persist.Checkpoint``;
intraday state resets on restore.
"""

from __future__ import annotations

import numpy as np

from quantfinlib.persist import Checkpoint


class VolumeCurve:
    """Learned intraday volume profile with live rescaling; see the
    module docstring."""

    __slots__ = ("_buckets", "_alpha", "_profile_ewma", "_cum_profile",
                 "_today", "_today_total", "_profile_total",
                 "_days_learned")

    def __init__(self, buckets: int = 78, alpha: float = 0.1) -> None:
        """
        Args:
            buckets: buckets per session (e.g. 78 five-minute buckets
                for a 6.5h equity day; 288 for a 24h FX day).
            alpha: day-over-day EWMA weight, e.g. 0.1.
        """
        if buckets < 1 or alpha <= 0 or alpha > 1:
            raise ValueError("need buckets >= 1, alpha in (0,1]")
        self._buckets = buckets
        self._alpha = alpha
        self._profile_ewma = np.zeros(buckets)
        self._cum_profile = np.zeros(buckets)
        self._today = np.zeros(buckets)
        self._today_total = 0.0
        self._profile_total = 0.0
        self._days_learned = 0

    def seed_profile(self, volumes_per_bucket) -> "VolumeCurve":
        """Seeds the profile from a known shape (any positive scale)
        -- optional."""
        v = np.asarray(volumes_per_bucket, dtype=float)
        if v.shape[0] != self._buckets:
            raise ValueError("profile length must equal bucket count")
        if np.any(v < 0):
            raise ValueError("negative volume in profile")
        self._profile_ewma[:] = v
        self._rebuild_prefix_sums()
        self._days_learned = 1
        return self

    def _rebuild_prefix_sums(self) -> None:
        cum = 0.0
        for b in range(self._buckets):
            cum += self._profile_ewma[b]
            self._cum_profile[b] = cum
        self._profile_total = cum

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def on_volume(self, bucket: int, qty: float) -> None:
        """Market volume observed in ``bucket`` (call as prints
        arrive)."""
        if qty > 0:
            self._today[bucket] += qty
            self._today_total += qty

    def roll_day(self) -> None:
        """Closes the session: folds today into the learned profile
        and resets the intraday state. Call once per trading day.
        Unlike the vol/spread curves, a zero bucket IS a real
        observation here (no prints = no volume), so partial-coverage
        sessions (feed started mid-day) bias the profile -- exclude
        them from ``roll_day`` or seed the shape via
        :meth:`seed_profile` instead."""
        for b in range(self._buckets):
            self._profile_ewma[b] = (
                self._today[b] if self._days_learned == 0
                else self._profile_ewma[b]
                + self._alpha * (self._today[b] - self._profile_ewma[b]))
            self._today[b] = 0.0
        self._rebuild_prefix_sums()
        self._today_total = 0.0
        self._days_learned += 1

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def expected_fraction_elapsed(self, bucket: int,
                                  frac_within_bucket: float) -> float:
        """Expected fraction of TODAY's total volume already traded,
        at ``frac_within_bucket`` through ``bucket``. Falls back to
        linear time when no profile is learned yet (VWAP degrades to
        TWAP, the honest default)."""
        f = max(0.0, min(1.0, frac_within_bucket))
        if self._profile_total <= 0:
            return (bucket + f) / self._buckets
        cum = ((self._cum_profile[bucket - 1] if bucket > 0 else 0.0)
               + f * self._profile_ewma[bucket])
        return min(1.0, cum / self._profile_total)

    def projected_day_volume(self, bucket: int,
                             frac_within_bucket: float) -> float:
        """Projected total volume for today: the learned day total
        scaled by today's realized-vs-expected ratio, shrunk toward 1
        by how much of the expected day has elapsed. Returns the
        learned total before any intraday evidence, 0 when nothing is
        learned or realized."""
        if self._profile_total <= 0:
            return self.realized_today()
        expected_so_far = (self.expected_fraction_elapsed(bucket, frac_within_bucket)
                          * self._profile_total)
        realized = self.realized_today()
        if expected_so_far <= 0:
            return self._profile_total
        ratio = realized / expected_so_far
        w = min(1.0, expected_so_far / self._profile_total)
        scale = 1 + w * (ratio - 1)
        return self._profile_total * max(0.0, scale)

    def expected_volume_remaining(self, bucket: int,
                                  frac_within_bucket: float) -> float:
        """Volume still expected between now and the close, under the
        projection."""
        return max(0.0, self.projected_day_volume(bucket, frac_within_bucket)
                   - self.realized_today())

    def realized_today(self) -> float:
        """Today's realized volume so far (O(1) running total)."""
        return self._today_total

    def profile_volume(self, bucket: int) -> float:
        """The learned average volume for one bucket."""
        return float(self._profile_ewma[bucket])

    def buckets(self) -> int:
        return self._buckets

    def days_learned(self) -> int:
        return self._days_learned

    # ------------------------------------------------------------------
    # Persistence (persist.Checkpoint)
    # ------------------------------------------------------------------

    def write_state(self, out) -> None:
        """Persists the learned profile (cross-day state) -- see
        :mod:`quantfinlib.persist`."""
        out.write_byte(1)
        out.write_int(self._days_learned)
        Checkpoint.write_doubles(out, self._profile_ewma)

    def read_state(self, inp) -> None:
        """Restores the learned profile; intraday state resets
        (restore at session start). Raises if the checkpoint was
        written with a different bucket count or an unknown state
        version."""
        Checkpoint.require_version(inp, 1, "VolumeCurve")
        days = inp.read_int()
        Checkpoint.read_doubles_into(inp, self._profile_ewma)
        self._days_learned = days
        self._rebuild_prefix_sums()
        self._today[:] = 0.0
        self._today_total = 0.0
