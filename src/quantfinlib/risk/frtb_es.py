"""FRTB Internal Models Approach expected shortfall.

Port of Java ``com.quantfinlib.risk.FrtbEs``: ES at 97.5% on a base
10-day horizon, scaled up across LIQUIDITY HORIZONS and anchored to a
STRESSED period::

    ES = sqrt( sum_j [ ES_j * sqrt((LH_j - LH_{j-1})/10) ]^2 )   (the LH cascade)
    IMCC = ES_current * (ES_stressed,reduced / ES_current,reduced)

Styled after BCBS MAR33, not certified: the FORMULAS are the
regulation's, the tests pin their arithmetic, but regulatory capital
additionally requires desk approvals, the P&L attribution program
(:mod:`pnl_attribution`), NMRF capital and the standardized-approach
floor — deliberately out of scope.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from quantfinlib.risk import var_engine

# The five regulatory liquidity horizons, in days.
LH_10 = 10
LH_20 = 20
LH_40 = 40
LH_60 = 60
LH_120 = 120


def es975(losses) -> float:
    """ES at 97.5% of a loss sample (positive losses), the FRTB tail measure."""
    return var_engine.tail(losses, 0.975).expected_shortfall


def liquidity_horizon_es(es_by_horizon, horizons) -> float:
    """The liquidity-horizon cascade (MAR33.5).

    Args:
        es_by_horizon: ``es_by_horizon[j]`` = 10-day ES with only the
            factors of liquidity horizon >= ``horizons[j]`` shocked;
            index 0 is the full set at LH 10.
        horizons: ascending, starting at 10 (e.g. [10, 20, 60]).
    """
    es_by_horizon = np.asarray(es_by_horizon, dtype=float)
    horizons = list(horizons)
    if (es_by_horizon.shape[0] != len(horizons) or len(horizons) < 1
            or horizons[0] != 10):
        raise ValueError("need aligned arrays with horizons starting at 10")
    sum_sq = 0.0
    for j, h in enumerate(horizons):
        if j > 0 and h <= horizons[j - 1]:
            raise ValueError("horizons must ascend")
        term = float(es_by_horizon[j])
        if not (term >= 0) or term == math.inf:
            raise ValueError("ES terms must be >= 0 and finite")
        prev = 0 if j == 0 else horizons[j - 1]
        scaled = term * math.sqrt((h - prev) / 10.0)
        sum_sq += scaled * scaled
    return math.sqrt(sum_sq)


def stress_calibrated_es(es_current_full: float, es_stressed_reduced: float,
                         es_current_reduced: float) -> float:
    """The stressed-calibration multiplier (MAR33.6): current full-factor
    ES scaled by the reduced-factor-set ratio between the stressed period
    and today. The regulatory floor: the ratio is at least 1 — a
    calmer-than-today stressed period must not DISCOUNT capital."""
    if (not (es_current_full >= 0) or es_current_full == math.inf
            or not (es_stressed_reduced >= 0) or es_stressed_reduced == math.inf
            or not (es_current_reduced > 0) or es_current_reduced == math.inf):
        raise ValueError(
            "ES inputs must be non-negative and finite (current reduced > 0)")
    ratio = max(1.0, es_stressed_reduced / es_current_reduced)
    return es_current_full * ratio


class TrafficLight(Enum):
    """Basel backtesting traffic light over 250 days of 99% VaR exceptions:
    GREEN <= 4 (model fine), AMBER 5-9 (capital multiplier rises),
    RED >= 10 (model presumed wrong)."""

    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"

    @classmethod
    def of(cls, exceptions_250d: int) -> "TrafficLight":
        if exceptions_250d < 0:
            raise ValueError("exceptions must be >= 0")
        if exceptions_250d <= 4:
            return cls.GREEN
        return cls.AMBER if exceptions_250d <= 9 else cls.RED
