"""VPIN -- Volume-synchronized Probability of INformed trading (port
of Java ``microstructure.Vpin``; Easley, Lopez de Prado & O'Hara): the
flow-toxicity gauge a market maker watches to decide when quoting is
no longer a business. Trades fill fixed-VOLUME buckets (volume time,
not clock time -- informed trading compresses clock time but not
volume time); each full bucket scores its absolute buy/sell imbalance;
VPIN is the average over the last ``window`` buckets::

    VPIN = (1/n) * sum(|buyVol - sellVol| / bucketVolume)

Balanced two-way flow reads near 0; one-sided (informed) flow reads
toward 1 -- famously elevated in the hour before the 2010 flash crash.
Feed it :class:`~quantfinlib.microstructure.trade_classifier.TradeClassifier`'s
aggressor sides when the venue does not disclose them. A trade larger
than the bucket's remaining capacity SPLITS across buckets, as the
original construction requires. Deterministic, O(1) amortized per
trade (overflow handled arithmetically, never looped per bucket).
"""

from __future__ import annotations

import math

import numpy as np


class Vpin:
    """Volume-bucketed order-flow toxicity estimator; see the module
    docstring."""

    __slots__ = ("_bucket_volume", "_imbalances", "_filled", "_head",
                 "_bucket_buy", "_bucket_sell")

    def __init__(self, bucket_volume: float, window: int) -> None:
        """
        Args:
            bucket_volume: shares/contracts per volume bucket, > 0
                (the classic choice: ~1/50th of average daily volume).
            window: completed buckets averaged, >= 1 (classic: 50).
        """
        if bucket_volume <= 0:
            raise ValueError("bucketVolume must be > 0")
        if window < 1:
            raise ValueError("window must be >= 1")
        self._bucket_volume = bucket_volume
        self._imbalances = np.zeros(window)     # ring of completed buckets
        self._filled = 0
        self._head = 0
        self._bucket_buy = 0.0
        self._bucket_sell = 0.0

    def on_trade(self, quantity: float, buy_aggressor: bool) -> None:
        """One classified trade. Splits across bucket boundaries as
        needed.

        Args:
            quantity: traded volume, > 0.
            buy_aggressor: true if the buyer was the aggressor.
        """
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        remaining = quantity
        window = self._imbalances.shape[0]
        # Finish the open bucket first.
        capacity = self._bucket_volume - (self._bucket_buy + self._bucket_sell)
        take = min(remaining, capacity)
        if buy_aggressor:
            self._bucket_buy += take
        else:
            self._bucket_sell += take
        remaining -= take
        if self._bucket_buy + self._bucket_sell == self._bucket_volume:
            self._complete_bucket()
        if remaining == 0:
            return
        # Whole one-sided buckets, handled ARITHMETICALLY: a block of
        # any size is O(window), never O(quantity/bucketVolume) --
        # buckets older than the window would be evicted anyway, so
        # only the last min(full, window) matter (each has imbalance
        # exactly 1).
        full = int(remaining // self._bucket_volume)
        to_record = min(full, window)
        for _ in range(to_record):
            self._imbalances[self._head] = 1.0
            self._head = (self._head + 1) % window
        # Clamp full BEFORE adding: a corrupt huge trade must not wrap
        # filled negative.
        self._filled = min(window, self._filled + min(full, window))
        remaining -= full * self._bucket_volume
        if remaining > 0:
            if buy_aggressor:
                self._bucket_buy = remaining
            else:
                self._bucket_sell = remaining

    def _complete_bucket(self) -> None:
        window = self._imbalances.shape[0]
        self._imbalances[self._head] = (
            abs(self._bucket_buy - self._bucket_sell) / self._bucket_volume)
        self._head = (self._head + 1) % window
        if self._filled < window:
            self._filled += 1
        self._bucket_buy = 0.0
        self._bucket_sell = 0.0

    def vpin(self) -> float:
        """The toxicity estimate over completed buckets; NaN until the
        first bucket completes (an empty average pretending to be
        calm would be exactly the wrong default for a risk signal)."""
        if self._filled == 0:
            return math.nan
        return float(np.sum(self._imbalances[:self._filled])) / self._filled

    def ready(self) -> bool:
        """True once the full window of buckets has completed."""
        return self._filled == self._imbalances.shape[0]

    def buckets_completed(self) -> int:
        return self._filled
