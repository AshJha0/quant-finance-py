"""Surveillance anomaly detection (port of Java ``ml.AnomalyDetector``).

Interval-aggregated market activity:

* Quote stuffing -- message-rate spikes (robust z-score) combined with
  an abnormal order-to-trade ratio: lots of quoting, little trading.
* Price spikes -- interval returns far outside their recent
  distribution.

Scores are ROBUST z-scores -- ``(x - median) / (1.4826 * MAD)`` -- not
mean/stdev: an anomaly detector whose baseline includes the anomalies
inflates its own scale and misses exactly the events it hunts (a storm
of stuffing intervals raises the stdev until nothing clears the
threshold). Median/MAD ignores up to half the sample being
contaminated; 1.4826 rescales MAD to stdev units under normality so
thresholds keep their familiar sigma meaning. When MAD is 0 (more than
half the intervals identical) the detector falls back to mean/stdev,
and gives up only when that is 0 too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from quantfinlib.util import std_dev

QUOTE_STUFFING = "QUOTE_STUFFING"
PRICE_SPIKE = "PRICE_SPIKE"


@dataclass(frozen=True, slots=True)
class Anomaly:
    interval_index: int
    type: str
    score: float


def detect_quote_stuffing(
    messages_per_interval: Sequence[int],
    trades_per_interval: Sequence[int],
    z_threshold: float,
    min_order_to_trade_ratio: float,
) -> List[Anomaly]:
    """Flags intervals where the message count is a ``z_threshold``-sigma
    outlier AND the order-to-trade ratio exceeds ``min_order_to_trade_ratio``.

    :param messages_per_interval: order/cancel/replace message counts per interval
    :param trades_per_interval: trade counts per interval (aligned)
    """
    if len(messages_per_interval) != len(trades_per_interval):
        raise ValueError("series must align")
    m = np.asarray(messages_per_interval, dtype=float)
    center = _median(m)
    scale = _robust_scale(m, center)
    out: List[Anomaly] = []
    if scale == 0:
        return out
    for i in range(m.shape[0]):
        z = (m[i] - center) / scale
        otr = messages_per_interval[i] / max(1, trades_per_interval[i])
        if z >= z_threshold and otr >= min_order_to_trade_ratio:
            out.append(Anomaly(i, QUOTE_STUFFING, z))
    return out


def detect_price_spikes(mids: Sequence[float], z_threshold: float) -> List[Anomaly]:
    """Flags intervals whose return is a ``z_threshold``-sigma outlier."""
    mids = np.asarray(mids, dtype=float)
    if mids.shape[0] < 3:
        return []
    rets = mids[1:] / mids[:-1] - 1
    center = _median(rets)
    scale = _robust_scale(rets, center)
    out: List[Anomaly] = []
    if scale == 0:
        return out
    for i in range(rets.shape[0]):
        z = abs(rets[i] - center) / scale
        if z >= z_threshold:
            out.append(Anomaly(i + 1, PRICE_SPIKE, z))
    return out


def _median(v: np.ndarray) -> float:
    sorted_v = np.sort(v)
    n = sorted_v.shape[0]
    if n % 2 == 1:
        return float(sorted_v[n // 2])
    return 0.5 * (float(sorted_v[n // 2 - 1]) + float(sorted_v[n // 2]))


def _robust_scale(v: np.ndarray, center: float) -> float:
    """1.4826 * MAD, falling back to stdev when MAD is degenerate."""
    dev = np.abs(v - center)
    mad = _median(dev)
    return 1.4826 * mad if mad > 0 else std_dev(v)
