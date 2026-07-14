"""Corporate action adjustment (port of Java ``data.CorporateActions``).

Back-adjusts a raw price series for splits and cash dividends
(CRSP-style multiplicative factors), so returns computed across
ex-dates reflect economics rather than mechanical price drops -- the
difference between toy and usable equity backtests.

* Split r-for-1: bars before the ex-date get prices / r and volume * r.
* Cash dividend d: bars before the ex-date get prices *
  ``(prev_close - d) / prev_close``, where ``prev_close`` is the last
  close before the ex-date.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

from quantfinlib.data.bar_series import BarSeries


class ActionType(Enum):
    SPLIT = "SPLIT"
    CASH_DIVIDEND = "CASH_DIVIDEND"


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """
    :param ex_timestamp: first bar timestamp trading on the adjusted basis
    :param type: split or cash dividend
    :param value: split ratio (2 = 2-for-1) or cash dividend per share
    """

    ex_timestamp: int
    type: ActionType
    value: float

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("action value must be positive")


def adjust(series: BarSeries, actions: Sequence[CorporateAction]) -> BarSeries:
    """Returns a new back-adjusted series; the input is untouched."""
    n = series.size()
    price_factor = [1.0] * n
    volume_factor = [1.0] * n

    sorted_actions = sorted(actions, key=lambda a: a.ex_timestamp)

    for action in sorted_actions:
        ex_index = _first_index_at_or_after(series, action.ex_timestamp)
        if ex_index <= 0:
            continue  # no bars before the ex-date: nothing to adjust
        vol_factor = 1.0
        if action.type == ActionType.SPLIT:
            factor = 1.0 / action.value
            vol_factor = action.value
        else:
            prev_close = series.close(ex_index - 1)
            if action.value >= prev_close:
                raise ValueError(f"dividend {action.value} >= prior close {prev_close}")
            factor = (prev_close - action.value) / prev_close
        for i in range(ex_index):
            price_factor[i] *= factor
            volume_factor[i] *= vol_factor

    builder = BarSeries.builder(series.symbol())
    for i in range(n):
        bar = series.bar(i)
        builder.add(
            bar.timestamp,
            bar.open * price_factor[i],
            bar.high * price_factor[i],
            bar.low * price_factor[i],
            bar.close * price_factor[i],
            bar.volume * volume_factor[i],
        )
    return builder.build()


def _first_index_at_or_after(series: BarSeries, timestamp: int) -> int:
    for i in range(series.size()):
        if series.timestamp(i) >= timestamp:
            return i
    return series.size()
