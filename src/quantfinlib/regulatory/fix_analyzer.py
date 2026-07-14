"""WM/Reuters-style 4pm fix analysis (port of Java ``regulatory.FixAnalyzer``).

Computes the fix rate from mid samples inside the fixing window
(median, per WM/R methodology) and screens a participant's flow for
the classic "banging the close" signature -- a large share of window
volume, a price run-up aligned with the participant's net flow into
the fix, and reversion afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class FixImpactReport:
    fix_rate: float
    run_up_bps: float  # pre-window mid -> fix
    reversion_bps: float  # fix -> post-window mid
    participation_share: float  # participant volume / market volume in the window
    net_flow: int  # participant buys - sells
    flagged: bool


def calculate_fix(mid_samples_in_window: Sequence[float]) -> float:
    """Fix rate = median of the mid samples captured inside the fixing window."""
    if len(mid_samples_in_window) == 0:
        raise ValueError("no samples")
    sorted_v = np.sort(np.asarray(mid_samples_in_window, dtype=float))
    n = sorted_v.shape[0]
    if n % 2 == 1:
        return float(sorted_v[n // 2])
    return float(sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2


def analyze(
    mid_samples_in_window: Sequence[float],
    pre_window_mid: float,
    post_window_mid: float,
    participant_buy_qty: int,
    participant_sell_qty: int,
    market_volume: int,
    share_threshold: float,
) -> FixImpactReport:
    """Screens one participant's fixing-window activity.

    Flags when all three hold: participation share >= threshold, the
    run-up into the fix is aligned with the participant's net flow, and
    the price reverts against that flow after the window (impact that
    decays is the footprint of pressure, not information).

    :param mid_samples_in_window: mid samples inside the fixing window
    :param pre_window_mid: mid just before the window opens
    :param post_window_mid: mid after the window closes
    :param participant_buy_qty: participant buy volume in the window
    :param participant_sell_qty: participant sell volume in the window
    :param market_volume: total market volume in the window
    :param share_threshold: participation share that triggers scrutiny (e.g. 0.25)
    """
    fix = calculate_fix(mid_samples_in_window)
    run_up_bps = (fix - pre_window_mid) / pre_window_mid * 1e4
    reversion_bps = (post_window_mid - fix) / fix * 1e4
    net_flow = participant_buy_qty - participant_sell_qty
    share = (
        0.0
        if market_volume <= 0
        else (participant_buy_qty + participant_sell_qty) / market_volume
    )

    flow_sign = (net_flow > 0) - (net_flow < 0)
    aligned_run_up = flow_sign != 0 and flow_sign * run_up_bps > 0
    reverts_after = flow_sign != 0 and flow_sign * reversion_bps < 0
    flagged = share >= share_threshold and aligned_run_up and reverts_after

    return FixImpactReport(fix, run_up_bps, reversion_bps, share, net_flow, flagged)
