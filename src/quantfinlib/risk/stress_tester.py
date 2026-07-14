"""Stress testing and scenario analysis.

Port of Java ``com.quantfinlib.risk.StressTester``. Three modes over
one book representation (factor exposures d, optional gammas G):

* Named scenarios — a vector of factor shocks (fractions: -0.20 = down
  20%) applied through the delta-gamma P&L ``d'Dx + 0.5 Dx'GDx``.
  Ship-your-own scenarios, plus :func:`black_monday_1987`,
  :func:`lehman_2008` and :func:`covid_march_2020` as STARTING
  TEMPLATES for a [equity, rates(bp/1e4), FX-USD, commodity,
  vol-points/1e2] factor ordering.
* Sensitivity ladders — one factor swept over a shock range,
  everything else flat.
* Reverse stress — for a linear book under covariance S, the
  most-probable shock producing a target loss L has the closed form
  ``Dx* = -(L/(d'Sd)) * Sd``; its Mahalanobis distance ``L/sqrt(d'Sd)``
  says how implausible the breaking move is (in "sigmas").

Losses are returned as negative P&L (a scenario that makes money
reports positive). The Java overloads collapse into optional ``gamma``
arguments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.risk import var_engine


@dataclass(frozen=True)
class ReverseStress:
    """The most-probable shock vector and its Mahalanobis distance."""

    shocks: np.ndarray
    mahalanobis_sigmas: float


def scenario_pnl(exposures, shocks, gamma=None) -> float:
    """Scenario P&L: delta-only ``d'Dx``, or delta-gamma
    ``d'Dx + 0.5 Dx'GDx`` when ``gamma`` is given."""
    e = np.asarray(exposures, dtype=float)
    s = np.asarray(shocks, dtype=float)
    if e.shape[0] != s.shape[0] or e.shape[0] < 1:
        raise ValueError("exposures and shocks must align")
    _require_finite(e, "exposures")
    _require_finite(s, "shocks")
    pnl = float(e @ s)
    if gamma is not None:
        g = np.asarray(gamma, dtype=float)
        if g.shape[0] != e.shape[0]:
            raise ValueError("gamma must match exposures")
        _require_finite(g, "gamma")
        pnl += 0.5 * float(s @ g @ s)
    return pnl


def sensitivity_ladder(exposures, factor: int, shock_range: float, steps: int,
                       gamma=None) -> np.ndarray:
    """One factor swept over ``[-shock_range, +shock_range]`` in ``steps``
    increments, everything else flat — the sensitivity ladder. Returns
    P&L per rung, ascending shock.

    Without ``gamma`` the ladder is DELTA-ONLY: curvature is ignored —
    a book carrying gamma needs the delta-gamma form, or the down rungs
    will look symmetric when they are not. With ``gamma`` the swept
    factor's own curvature ``0.5 * G_ff * shock^2`` is included
    (cross-gammas stay out: the other factors are flat by construction).
    """
    e = np.asarray(exposures, dtype=float)
    if (factor < 0 or factor >= e.shape[0] or steps < 2
            or not (shock_range > 0) or shock_range == math.inf):
        raise ValueError("invalid ladder spec")
    _require_finite(e, "exposures")
    g_ff = 0.0
    if gamma is not None:
        g = np.asarray(gamma, dtype=float)
        if g.shape[0] != e.shape[0]:
            raise ValueError("gamma must match exposures")
        _require_finite(g, "gamma")
        g_ff = float(g[factor, factor])
    pnl = np.empty(steps + 1)
    for s in range(steps + 1):
        shock = -shock_range + 2 * shock_range * s / steps
        pnl[s] = e[factor] * shock + 0.5 * g_ff * shock * shock
    return pnl


def reverse_stress(exposures, covariance, target_loss: float) -> ReverseStress:
    """The most-probable factor move (under Gaussian factors with
    covariance S) that loses exactly ``target_loss`` on a linear book —
    closed form, no search. The returned Mahalanobis distance is the
    plausibility verdict: how many "joint sigmas" away the breaking
    scenario sits.

    Args:
        target_loss: positive loss to reverse-engineer, currency units.
    """
    if not (target_loss > 0) or target_loss == math.inf:
        raise ValueError("targetLoss must be positive and finite")
    sigma_p = var_engine.portfolio_stdev(exposures, covariance)
    if not (sigma_p > 0):
        raise ValueError(
            f"the book carries no factor risk — no finite move loses {target_loss}")
    e = np.asarray(exposures, dtype=float)
    c = np.asarray(covariance, dtype=float)
    sigma_delta = c @ e
    scale = -target_loss / (sigma_p * sigma_p)   # d'Sd = sigma_P^2
    return ReverseStress(scale * sigma_delta, target_loss / sigma_p)


# ---------------------------------------------------------------------------
# Historical templates — STARTING POINTS, not certified replays.
# Factor order: [equity, rates(dr as a fraction, +50bp = +0.005),
# FX (USD strength), commodity, vol (d vol points as a fraction)].
# ---------------------------------------------------------------------------

def black_monday_1987() -> np.ndarray:
    """1987-10-19 stylized: equities -20%, flight-to-quality rates, vol explosion."""
    return np.array([-0.20, -0.0050, 0.02, -0.05, 0.20])


def lehman_2008() -> np.ndarray:
    """2008-09-15 (Lehman week) stylized: -9% equities, -40bp, USD bid, oil down, vol +16pts."""
    return np.array([-0.09, -0.0040, 0.04, -0.07, 0.16])


def covid_march_2020() -> np.ndarray:
    """2020-03-16 stylized: -12% equities, -30bp, USD squeeze, oil collapse, VIX ATH."""
    return np.array([-0.12, -0.0030, 0.05, -0.15, 0.25])


def _require_finite(a: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite (one NaN exposure would "
                         "print NaN for every scenario in the report)")
