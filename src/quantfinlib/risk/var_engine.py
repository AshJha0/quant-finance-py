"""Portfolio Value-at-Risk, all four classic flavors over one input shape.

Port of Java ``com.quantfinlib.risk.VarEngine``: factor EXPOSURES
(currency P&L per unit factor return — a delta vector) against a factor
covariance matrix or a factor-return history.

* Delta-normal — sigma_P = sqrt(d'Sd), VaR = z*sigma_P. Instant, and
  exactly wrong for optionality.
* Monte Carlo — Cholesky-correlated Gaussian factor draws through the
  linear map; converges to delta-normal for a linear book.
* Delta-gamma (Cornish-Fisher) — second-order P&L ``d'Dx + 0.5 Dx'GDx``
  whose skew tilts the quantile: a short-gamma book's VaR is WORSE than
  delta-normal says.
* Historical — replay actual factor-return rows through the exposures.

Conventions: VaR and ES are POSITIVE losses in currency units;
confidence is the one-sided level (0.99 = 99%); factor returns and the
covariance are per-horizon. Deviation from Java: :func:`monte_carlo_var`
draws from ``numpy.random.default_rng(seed)`` instead of
``java.util.Random`` — deterministic per seed, but a different stream,
so cross-port pins on MC output are agreement/identity pins, not
stream-exact values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from quantfinlib.risk import gaussian_copula
from quantfinlib.util import math_utils as mu


@dataclass(frozen=True)
class VarResult:
    """VaR (quantile) and ES (tail mean) of a loss sample, positive = loss."""

    var: float
    expected_shortfall: float


# ---------------------------------------------------------------------------
# Delta-normal
# ---------------------------------------------------------------------------

def portfolio_stdev(exposures, covariance) -> float:
    """Portfolio stdev sqrt(d'Sd) in currency units."""
    e, c, _ = _require_square(exposures, covariance)
    variance = float(e @ c @ e)
    return math.sqrt(max(variance, 0.0))


def delta_normal_var(exposures, covariance, confidence: float) -> float:
    """Delta-normal VaR: z-quantile of the Gaussian portfolio P&L."""
    return _z_of(confidence) * portfolio_stdev(exposures, covariance)


def delta_normal_es(exposures, covariance, confidence: float) -> float:
    """Delta-normal ES: the Gaussian tail mean, sigma*phi(z)/(1-c)."""
    z = _z_of(confidence)
    return portfolio_stdev(exposures, covariance) * mu.norm_pdf(z) / (1 - confidence)


# ---------------------------------------------------------------------------
# Monte Carlo (linear book)
# ---------------------------------------------------------------------------

def monte_carlo_var(exposures, covariance, confidence: float,
                    scenarios: int, seed: int) -> VarResult:
    """VaR and ES from Gaussian Monte Carlo factor scenarios."""
    e, c, n = _require_square(exposures, covariance)
    if scenarios < 100:
        raise ValueError("need >= 100 scenarios")
    chol = gaussian_copula.cholesky(c)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((scenarios, n))
    shocks = z @ chol.T           # row s: correlated factor moves
    losses = -(shocks @ e)
    return tail(losses, confidence)


# ---------------------------------------------------------------------------
# Delta-gamma (Cornish-Fisher)
# ---------------------------------------------------------------------------

def delta_gamma_var(exposures, gamma, covariance, confidence: float) -> float:
    """Second-order VaR via the Cornish-Fisher quantile of the delta-gamma P&L.

    Moments of ``d'Dx + 0.5 Dx'GDx`` under Gaussian factors: mean
    ``0.5 tr(GS)``, variance ``d'Sd + 0.5 tr((GS)^2)``, and the skew
    that moves the quantile. Accurate for MODERATE gamma — the expansion
    degrades when the quadratic term dominates (skew beyond ~1).
    """
    mean, variance, skew = _delta_gamma_cumulants(exposures, gamma, covariance)
    if variance <= 0:
        return 0.0
    z = _z_of(confidence)
    # Cornish-Fisher: the LOSS quantile uses -z (left tail of P&L).
    z_cf = -z + (z * z - 1) * skew / 6
    pnl_quantile = mean + z_cf * math.sqrt(variance)
    return max(0.0, -pnl_quantile)


def delta_gamma_es(exposures, gamma, covariance, confidence: float) -> float:
    """Second-order ES: the Cornish-Fisher tail mean, in CLOSED FORM.

    With loss quantile ``q(p) = -mu + sigma*(z_p + (z_p^2 - 1)*s/6)``
    (s = loss skew), the identities ``E[Z*1{Z>z}] = phi(z)`` and
    ``E[(Z^2-1)*1{Z>z}] = z*phi(z)`` give
    ``ES = -mu + sigma*phi(z)/(1-c)*(1 + z*s/6)`` — no numerical
    integration, reducing EXACTLY to :func:`delta_normal_es` when G = 0.
    """
    mean, variance, skew = _delta_gamma_cumulants(exposures, gamma, covariance)
    if variance <= 0:
        return 0.0
    z = _z_of(confidence)
    loss_skew = -skew
    es = -mean + math.sqrt(variance) * mu.norm_pdf(z) / (1 - confidence) \
        * (1 + z * loss_skew / 6)
    return max(0.0, es)


def _delta_gamma_cumulants(exposures, gamma, covariance):
    """(mean, variance, skew) of ``d'Dx + 0.5 Dx'GDx`` under Gaussian factors."""
    e, c, n = _require_square(exposures, covariance)
    g = np.asarray(gamma, dtype=float)
    if g.shape[0] != n:
        raise ValueError("gamma must match exposures")
    gs = g @ c
    mean = 0.5 * float(np.trace(gs))
    lin_var = float(e @ c @ e)
    gs2 = gs @ gs
    variance = lin_var + 0.5 * float(np.trace(gs2))
    if variance <= 0:
        return mean, variance, 0.0
    # Third cumulant: 3 d'S G S d + tr((GS)^3).
    sd = c @ e
    dgd = float(sd @ g @ sd)
    kappa3 = 3 * dgd + float(np.trace(gs2 @ gs))
    return mean, variance, kappa3 / variance ** 1.5


# ---------------------------------------------------------------------------
# Historical
# ---------------------------------------------------------------------------

def historical_var(exposures, factor_returns, confidence: float) -> VarResult:
    """Historical simulation: each row of ``factor_returns`` is one scenario
    replayed through the exposures."""
    e = np.asarray(exposures, dtype=float)
    fr = np.asarray(factor_returns, dtype=float)
    if fr.shape[0] < 20:
        raise ValueError("need >= 20 scenarios")
    if fr.ndim != 2 or fr.shape[1] != e.shape[0]:
        raise ValueError("scenario width mismatch")
    losses = -(fr @ e)
    return tail(losses, confidence)


# ---------------------------------------------------------------------------
# Full revaluation
# ---------------------------------------------------------------------------

def full_revaluation_var(scenarios, pricer: Callable[[np.ndarray], float],
                         confidence: float) -> VarResult:
    """Full-revaluation VaR: every scenario repriced through the CALLER'S
    pricer (a plain callable ``factor_moves -> P&L``).

    A pricer returning NaN/Infinity raises: a scenario your pricer
    cannot price is a modelling problem, not a quantile.
    """
    scen = np.asarray(scenarios, dtype=float)
    if scen.shape[0] < 20:
        raise ValueError("need >= 20 scenarios")
    losses = np.empty(scen.shape[0])
    for s in range(scen.shape[0]):
        pnl = float(pricer(scen[s]))
        if not math.isfinite(pnl):
            raise ValueError(
                f"pricer returned {pnl} for scenario {s} — a scenario the "
                "pricer cannot price is a modelling problem, not a quantile")
        losses[s] = -pnl
    return tail(losses, confidence)


# ---------------------------------------------------------------------------
# Shared tail arithmetic
# ---------------------------------------------------------------------------

def tail(losses, confidence: float) -> VarResult:
    """VaR (quantile) and ES (tail mean) of a loss sample, positive = loss."""
    if not (0.5 < confidence < 1):
        raise ValueError("confidence must be in (0.5, 1)")
    sorted_losses = np.sort(np.asarray(losses, dtype=float))
    n = sorted_losses.shape[0]
    index = min(n - 1, math.ceil(confidence * n) - 1)
    var = max(0.0, float(sorted_losses[index]))
    es = float(np.sum(sorted_losses[index:])) / (n - index)
    return VarResult(var, max(0.0, es))


def _z_of(confidence: float) -> float:
    if not (0.5 < confidence < 1):
        raise ValueError("confidence must be in (0.5, 1)")
    return mu.norm_inv(confidence)


def _require_square(exposures, covariance):
    e = np.asarray(exposures, dtype=float)
    c = np.asarray(covariance, dtype=float)
    n = e.shape[0]
    if n < 1 or c.shape[0] != n:
        raise ValueError("covariance must match exposures")
    return e, c, n
