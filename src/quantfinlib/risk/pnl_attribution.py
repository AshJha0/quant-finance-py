"""FRTB P&L attribution test (PLAT).

Port of Java ``com.quantfinlib.risk.PnlAttribution``: does the risk
engine's theoretical P&L (RTPL) actually track the desk's hypothetical
P&L (HPL)? Two statistics, per MAR32:

* Spearman correlation between daily HPL and RTPL — do they RANK days
  the same way;
* Kolmogorov-Smirnov statistic between their empirical distributions —
  do they have the same SHAPE.

Zones per the regulation: GREEN (corr > 0.80 and KS < 0.09), RED
(corr < 0.70 or KS > 0.12), AMBER between. Styled after BCBS MAR32,
not certified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from quantfinlib.risk import dependence


class Zone(Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


@dataclass(frozen=True)
class Result:
    """The PLAT verdict for one desk over one window."""

    spearman_correlation: float
    ks_statistic: float
    zone: Zone


def test(hypothetical_pnl, risk_theoretical_pnl) -> Result:
    """Runs the PLAT over aligned daily P&L series (250 days is the
    regulatory window; anything >= 20 computes).

    Args:
        hypothetical_pnl: HPL — actual book, actual prices.
        risk_theoretical_pnl: RTPL — the risk model's factors + pricers.
    """
    hpl = np.asarray(hypothetical_pnl, dtype=float)
    rtpl = np.asarray(risk_theoretical_pnl, dtype=float)
    if hpl.shape[0] != rtpl.shape[0] or hpl.shape[0] < 20:
        raise ValueError("need aligned series of >= 20 days")
    corr = dependence.spearman(hpl, rtpl)
    ks = ks_statistic(hpl, rtpl)
    if corr < 0.70 or ks > 0.12:
        zone = Zone.RED
    elif corr > 0.80 and ks < 0.09:
        zone = Zone.GREEN
    else:
        zone = Zone.AMBER
    return Result(corr, ks, zone)


def ks_statistic(a, b) -> float:
    """The two-sample KS statistic: max gap between the empirical CDFs,
    evaluated after BOTH samples have consumed each distinct value —
    ties must not register a transient gap (identical series score
    exactly 0)."""
    sa = np.sort(np.asarray(a, dtype=float))
    sb = np.sort(np.asarray(b, dtype=float))
    if sa.shape[0] == 0 or sb.shape[0] == 0:
        raise ValueError("need non-empty samples")
    # A NaN would freeze the tie-consuming advance below (NaN == NaN is
    # false — neither index ever moves), turning one missing P&L day
    # into an INFINITE LOOP. Sorted arrays put -Inf first and NaN/+Inf
    # last, so checking the ends rejects every non-finite.
    if (not math.isfinite(sa[0]) or not math.isfinite(sa[-1])
            or not math.isfinite(sb[0]) or not math.isfinite(sb[-1])):
        raise ValueError("P&L series must be finite")
    na, nb = sa.shape[0], sb.shape[0]
    i = 0
    j = 0
    max_gap = 0.0
    while i < na and j < nb:
        v = min(sa[i], sb[j])
        while i < na and sa[i] == v:
            i += 1
        while j < nb and sb[j] == v:
            j += 1
        gap = abs(i / na - j / nb)
        if gap > max_gap:
            max_gap = gap
    return max_gap
