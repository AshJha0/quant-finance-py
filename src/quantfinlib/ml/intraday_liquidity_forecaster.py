"""Intraday liquidity forecasting (port of Java ``ml.IntradayLiquidityForecaster``).

Accumulates per-bucket volumes across days into a seasonal profile
(e.g. 24 hourly buckets) to predict when liquidity peaks -- London
open, the London/New York overlap, etc. Feed the profile to a VWAP
scheduler to align execution with expected volume.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class IntradayLiquidityForecaster:
    def __init__(self, buckets: int) -> None:
        """:param buckets: buckets per day (24 = hourly, 48 = half-hourly)"""
        self._buckets = buckets
        self._sums = np.zeros(buckets)
        self._days = 0

    def add_day(self, volumes_per_bucket: Sequence[float]) -> "IntradayLiquidityForecaster":
        """Adds one day's observed volume per bucket."""
        if len(volumes_per_bucket) != self._buckets:
            raise ValueError(f"expected {self._buckets} buckets")
        for i in range(self._buckets):
            self._sums[i] += volumes_per_bucket[i]
        self._days += 1
        return self

    def forecast_volume(self, bucket: int) -> float:
        """Expected volume in a bucket (historical mean)."""
        return 0.0 if self._days == 0 else float(self._sums[bucket]) / self._days

    def profile(self) -> np.ndarray:
        """Normalized profile summing to 1 -- directly usable as a VWAP
        weight curve."""
        total = float(np.sum(self._sums))
        if total == 0:
            return np.full(self._buckets, 1.0 / self._buckets)
        return self._sums / total

    def peak_bucket(self) -> int:
        """Bucket with the highest expected liquidity."""
        return int(np.argmax(self._sums))

    def session_share(self, from_bucket: int, to_bucket: int) -> float:
        """Share of daily liquidity expected within [from_bucket, to_bucket)."""
        p = self.profile()
        return float(np.sum(p[from_bucket:to_bucket]))

    @staticmethod
    def fx_session(hour_utc: int) -> str:
        """FX session label for an hour of day in UTC."""
        if hour_utc >= 22:
            return "SYDNEY"
        if hour_utc < 7:
            return "TOKYO"
        if hour_utc < 12:
            return "LONDON"
        if hour_utc < 17:
            return "LONDON_NY_OVERLAP"
        return "NEW_YORK"
