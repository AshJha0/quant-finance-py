"""Concentration risk metrics over exposures (by asset, counterparty,
sector, currency, ...).

Port of Java ``com.quantfinlib.risk.ConcentrationRisk``:
Herfindahl-Hirschman index, effective number of positions, top-N share,
and single-name limit breaches. Java's ``LinkedHashMap`` grouping maps
to plain ``dict`` (insertion-ordered in Python).
"""

from __future__ import annotations

import numpy as np


def herfindahl_index(exposures) -> float:
    """Herfindahl-Hirschman index of |exposure| shares;
    1/N (diversified) .. 1 (single name)."""
    a = np.abs(np.asarray(exposures, dtype=float))
    total = float(np.sum(a))
    if total == 0:
        return 0.0
    shares_ = a / total
    return float(np.sum(shares_ * shares_))


def effective_positions(exposures) -> float:
    """Effective number of equally-weighted positions: 1 / HHI."""
    hhi = herfindahl_index(exposures)
    return 0.0 if hhi == 0 else 1 / hhi


def top_n_share(exposures, n: int) -> float:
    """Combined |exposure| share of the largest ``n`` positions."""
    a = np.abs(np.asarray(exposures, dtype=float))
    total = float(np.sum(a))
    if total == 0:
        return 0.0
    a = np.sort(a)
    top = float(np.sum(a[max(0, a.shape[0] - n):]))
    return top / total


def shares(exposure_by_group: dict[str, float]) -> dict[str, float]:
    """|Exposure| share per group key (insertion order preserved)."""
    total = sum(abs(e) for e in exposure_by_group.values())
    return {k: (0.0 if total == 0 else abs(v) / total)
            for k, v in exposure_by_group.items()}


def limit_breaches(exposure_by_group: dict[str, float],
                   max_share: float) -> list[str]:
    """Group keys whose share exceeds the single-name concentration limit."""
    return [k for k, share in shares(exposure_by_group).items()
            if share > max_share]
