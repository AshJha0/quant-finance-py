"""Nelson-Siegel yield-curve fit (port of Java ``com.quantfinlib.rates.NelsonSiegel``).

The parametric answer to "what SHAPE is the curve", where ``YieldCurve``
is the exact-repricing answer to "what IS the curve". Central banks
(ECB, Fed) publish curves in exactly this form because four numbers
carry the whole story:

    z(t) = b0 + b1 * (1-e^{-t/l})/(t/l) + b2 * [(1-e^{-t/l})/(t/l) - e^{-t/l}]

* b0 — the LEVEL: z(inf), where the long end settles;
* b1 — the SLOPE: z(0) = b0 + b1, so b1 < 0 is an upward curve and
  b1 > 0 is INVERSION — the recession-signal number;
* b2 — the CURVATURE: the mid-curve hump, peaking near t ~ lambda;
* lambda — WHERE the hump sits (years).

Fitting exploits the model's one great convenience: for FIXED lambda
the model is LINEAR in (b0, b1, b2) — an exact 3x3 least-squares solve.
So the fit is a log-spaced grid search over lambda with an OLS solve at
each node, keeping the whole thing deterministic and free of
local-minimum roulette (the classic failure of fitting all four
jointly). Betas are NOT constrained to "sensible" signs: an inverted
curve is data, not an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantfinlib.util import math_utils


class NelsonSiegel:
    """Deterministic grid-plus-OLS Nelson-Siegel fitter."""

    @dataclass(frozen=True)
    class Fit:
        """Fitted parameters plus the fit's root-mean-square error."""

        b0: float
        b1: float
        b2: float
        lambda_: float
        rmse: float

        def zero_rate(self, t: float) -> float:
            """Model zero rate at tenor ``t`` years, > 0."""
            if not (t > 0) or t == math.inf:
                raise ValueError(f"tenor must be positive and finite, got {t}")
            x = t / self.lambda_
            slope_loading = (1 - math.exp(-x)) / x
            return (self.b0 + self.b1 * slope_loading
                    + self.b2 * (slope_loading - math.exp(-x)))

        def short_rate(self) -> float:
            """The short-rate limit z(0+) = b0 + b1 (exact in the model)."""
            return self.b0 + self.b1

        def long_rate(self) -> float:
            """The long-end asymptote z(inf) = b0."""
            return self.b0

    @staticmethod
    def fit(tenor_years, zero_rates) -> "NelsonSiegel.Fit":
        """Fits by log-spaced lambda grid (0.1y-10y, 80 nodes) + exact OLS per node.

        Args:
            tenor_years: observation tenors, >= 4 distinct, all > 0.
            zero_rates: observed zero rates (continuously compounded).
        """
        n = len(tenor_years)
        if n < 4 or len(zero_rates) != n:
            raise ValueError(
                f"need >= 4 aligned tenor/rate observations, got {n}/{len(zero_rates)}")
        for i in range(n):
            if not (tenor_years[i] > 0) or tenor_years[i] == math.inf:
                raise ValueError(f"tenor must be positive and finite: {tenor_years[i]}")
            if not math.isfinite(zero_rates[i]):
                raise ValueError(f"zero rate must be finite: {zero_rates[i]}")
        best_sse = math.inf
        best: NelsonSiegel.Fit | None = None
        nodes = 80
        lo, hi = math.log(0.1), math.log(10.0)
        for g in range(nodes):
            lam = math.exp(lo + (hi - lo) * g / (nodes - 1))
            # OLS in (b0, b1, b2) via 3x3 normal equations.
            xtx = [[0.0] * 3 for _ in range(3)]
            xty = [0.0] * 3
            for i in range(n):
                x = tenor_years[i] / lam
                f1 = (1 - math.exp(-x)) / x
                row = (1.0, f1, f1 - math.exp(-x))
                for a in range(3):
                    xty[a] += row[a] * zero_rates[i]
                    for b in range(3):
                        xtx[a][b] += row[a] * row[b]
            try:
                beta = math_utils.solve_linear(xtx, xty)
            except ValueError:
                continue    # degenerate node (e.g. all tenors << lambda); skip
            candidate = NelsonSiegel.Fit(beta[0], beta[1], beta[2], lam, 0.0)
            sse = 0.0
            for i in range(n):
                e = candidate.zero_rate(tenor_years[i]) - zero_rates[i]
                sse += e * e
            if sse < best_sse:
                best_sse = sse
                best = NelsonSiegel.Fit(beta[0], beta[1], beta[2], lam,
                                        math.sqrt(sse / n))
        if best is None:
            raise ValueError("no lambda node produced a solvable fit")
        return best
