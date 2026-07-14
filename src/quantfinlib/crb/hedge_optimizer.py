"""Cost-aware minimum-variance hedging (port of Java
``com.quantfinlib.crb.HedgeOptimizer``).

The question is never "how do we flatten this" (sell everything) but
"what is the CHEAPEST basket of liquid instruments that takes the risk
below the limit". Minimizes::

    (e + L*h)' Sigma (e + L*h)  +  lambda * sum_i c_i*|h_i|

over hedge notionals ``h``: ``e`` the factor exposures, ``L`` each
instrument's factor loadings, ``Sigma`` the factor covariance, ``c_i``
the instrument's all-in cost per unit notional (spread + expected
impact -- a ``KylesLambda`` estimate slots in directly), ``lambda`` the
risk/cost trade-off.

Solved by cyclic coordinate descent with the exact soft-threshold
update -- deterministic, no external optimizer, and the L1 term does
what a hedging desk actually wants: instruments whose marginal risk
reduction is worth less than their cost get EXACTLY zero, not a dusty
small position. ``lambda = 0`` recovers the closed-form minimum-
variance hedge.
"""

from __future__ import annotations

import math

from quantfinlib.util import math_utils as mu

# Generous: near-collinear instruments (0.999-correlated futures)
# contract slowly under Gauss-Seidel, and a silent max-iteration exit
# would return a plausible-looking, grossly unconverged hedge.
_MAX_ITERATIONS = 20_000
_RELATIVE_TOLERANCE = 1e-10


class HedgeOptimizer:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def hedge(exposures: list[float], covariance: list[list[float]],
             loadings: list[list[float]], cost_per_unit: list[float],
             cost_weight: float) -> list[float]:
        """
        exposures: factor exposures e (length n)
        covariance: n x n factor covariance Sigma
        loadings: loadings[f][i] -- factor f exposure created by one
            unit of instrument i (n x m)
        cost_per_unit: c_i >= 0 per unit |notional| (length m)
        cost_weight: lambda >= 0 -- 0 is pure minimum variance
        Returns hedge notionals h (length m), signed.
        """
        n = len(exposures)
        m = len(cost_per_unit)
        _require_finite(exposures, "exposures")
        _require_finite(cost_per_unit, "costPerUnit")
        if not (cost_weight >= 0) or cost_weight == math.inf:
            raise ValueError("costWeight must be >= 0 and finite")
        if len(covariance) != n or len(loadings) != n:
            raise ValueError(f"covariance and loadings must have {n} factor rows")
        for row in covariance:
            if len(row) != n:
                raise ValueError("covariance must be square")
            # NaN here would slip past every comparison below and come
            # back as a silent all-zero "hedge" for a live breach.
            _require_finite(row, "covariance")
        for row in loadings:
            if len(row) != m:
                raise ValueError(f"each loadings row needs {m} columns")
            _require_finite(row, "loadings")
        for c in cost_per_unit:
            if c < 0:
                raise ValueError("costs must be >= 0")

        # Precompute per-instrument quadratic terms: a_i = L_i' Sigma L_i
        # and the cross terms G[i][j] = L_i' Sigma L_j, plus d_i = L_i' Sigma e.
        sigma_l = [[0.0] * m for _ in range(n)]
        for f in range(n):
            for i in range(m):
                s = 0.0
                for g in range(n):
                    s += covariance[f][g] * loadings[g][i]
                sigma_l[f][i] = s
        gram = [[0.0] * m for _ in range(m)]
        d = [0.0] * m
        for i in range(m):
            for j in range(m):
                s = 0.0
                for f in range(n):
                    s += loadings[f][i] * sigma_l[f][j]
                gram[i][j] = s
            s = 0.0
            for f in range(n):
                s += loadings[f][i] * _mat_vec_row(covariance, exposures, f)
            d[i] = s

        h = [0.0] * m
        converged = False
        for _ in range(_MAX_ITERATIONS):
            if converged:
                break
            max_delta = 0.0
            max_h = 1.0
            for i in range(m):
                a = gram[i][i]
                if a < 0:
                    raise ValueError(
                        f"covariance is not PSD: instrument {i} has L'ΣL = {a}")
                if a == 0:
                    continue          # instrument carries no risk: leave at 0
                # b = L_i' Sigma (e + sum_{j!=i} L_j h_j)
                b = d[i]
                for j in range(m):
                    if j != i:
                        b += gram[i][j] * h[j]
                # minimize a*h^2 + 2b*h + lambda*c*|h| -> soft threshold at lambda*c/2.
                threshold = cost_weight * cost_per_unit[i] / 2
                if b > threshold:
                    next_h = -(b - threshold) / a
                elif b < -threshold:
                    next_h = -(b + threshold) / a
                else:
                    next_h = 0.0
                delta = abs(next_h - h[i])
                if delta > max_delta:
                    max_delta = delta
                h[i] = next_h
                mag = abs(next_h)
                if mag > max_h:
                    max_h = mag
            # RELATIVE tolerance: notionals run in the millions, and an
            # absolute epsilon there is either never met or means nothing.
            converged = max_delta < _RELATIVE_TOLERANCE * max_h
        if not converged:
            # Near-collinear instruments can defeat Gauss-Seidel -- an
            # unconverged hedge that LOOKS like an answer is worse than
            # an exception (drop the redundant twin instrument).
            raise RuntimeError(
                f"coordinate descent failed to converge in {_MAX_ITERATIONS} "
                "iterations — hedge instruments may be near-collinear")
        return h

    @staticmethod
    def residual(exposures: list[float], loadings: list[list[float]],
                h: list[float]) -> list[float]:
        """Post-hedge factor exposures e + L*h."""
        n = len(exposures)
        out = [0.0] * n
        for f in range(n):
            s = exposures[f]
            for i in range(len(h)):
                s += loadings[f][i] * h[i]
            out[f] = s
        return out

    @staticmethod
    def risk(exposures: list[float], covariance: list[list[float]]) -> float:
        """Portfolio stdev of an exposure vector under Sigma -- the risk
        being cut."""
        return math.sqrt(max(0.0, mu.quadratic_form(exposures, covariance)))


def _mat_vec_row(cov: list[list[float]], e: list[float], row: int) -> float:
    s = 0.0
    for g in range(len(e)):
        s += cov[row][g] * e[g]
    return s


def _require_finite(a: list[float], name: str) -> None:
    for x in a:
        if not math.isfinite(x):
            raise ValueError(f"{name} must be finite")
