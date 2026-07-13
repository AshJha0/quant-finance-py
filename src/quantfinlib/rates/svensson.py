"""Svensson yield-curve fit (port of Java ``com.quantfinlib.rates.Svensson``).

Nelson-Siegel-Svensson: ``NelsonSiegel`` with a SECOND curvature hump,
the form most central banks actually publish (the ECB's daily curve is
exactly this):

    z(t) = b0 + b1 * f1(t/l1) + b2 * f2(t/l1) + b3 * f2(t/l2)
    f1(x) = (1 - e^-x)/x        f2(x) = (1 - e^-x)/x - e^-x

* b0 — the LEVEL: z(infinity);
* b1 — the SLOPE: z(0) = b0 + b1 (b1 > 0 is inversion);
* b2, lambda1 — the FIRST hump and where it sits;
* b3, lambda2 — the SECOND hump: the long-end flex a single hump cannot
  bend into — real curves routinely show a short-end bump (policy
  expectations) AND a 10y+ dip (convexity demand), and plain
  Nelson-Siegel must split the difference.

Fitting mirrors ``NelsonSiegel`` exactly: for FIXED (l1, l2) the model
is LINEAR in (b0, b1, b2, b3) — an exact 4-regressor OLS solve — so the
fit is a 2-D log-spaced grid over the lambdas with an OLS solve per
node, deterministic and free of local-minimum roulette. Nodes with
``lambda2 <= lambda1`` are skipped: the two f2 regressors collide as
the lambdas meet (exact collinearity at equality), and the ordering
makes the parameterization identifiable — hump one is always the
shorter-dated one. Betas are NOT sign-constrained: an inverted or
double-dipped curve is data, not an error.

With b3 = 0 the model IS Nelson-Siegel, so with the lambdas free
Svensson can always match NS in-sample; the two FITTERS search
different lambda grids (NS: 80 nodes 1-D; here: 50 nodes 2-D with
lambda1 < 10), so on data whose best single lambda falls between this
grid's nodes NS can win by a grid-granularity sliver (rmse differences
at the 1e-8 level, tested to agree within tolerance). The price of the
extra hump is two more parameters — on sparse or single-hump curves
prefer NS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantfinlib.util import math_utils


class Svensson:
    """Deterministic 2-D grid-plus-OLS Svensson fitter."""

    @dataclass(frozen=True)
    class Fit:
        """Fitted parameters plus the fit's root-mean-square error."""

        b0: float
        b1: float
        b2: float
        b3: float
        lambda1: float
        lambda2: float
        rmse: float

        def zero_rate(self, t: float) -> float:
            """Model zero rate at tenor ``t`` years, > 0."""
            if not (t > 0) or t == math.inf:
                raise ValueError(f"tenor must be positive and finite, got {t}")
            x1 = t / self.lambda1
            x2 = t / self.lambda2
            f1 = (1 - math.exp(-x1)) / x1
            g1 = f1 - math.exp(-x1)
            g2 = (1 - math.exp(-x2)) / x2 - math.exp(-x2)
            return self.b0 + self.b1 * f1 + self.b2 * g1 + self.b3 * g2

        def short_rate(self) -> float:
            """The short-rate limit z(0+) = b0 + b1 (exact in the model)."""
            return self.b0 + self.b1

        def long_rate(self) -> float:
            """The long-end asymptote z(infinity) = b0."""
            return self.b0

    @staticmethod
    def fit(tenor_years, zero_rates) -> "Svensson.Fit":
        """Fits by 2-D log-spaced lambda grid (0.1y-10y, 50 nodes per axis,
        ``lambda2 > lambda1`` only) + exact 4-regressor OLS per node.

        Args:
            tenor_years: observation tenors, >= 6 distinct, all > 0.
            zero_rates: observed zero rates (continuously compounded).
        """
        n = len(tenor_years)
        if n < 6 or len(zero_rates) != n:
            raise ValueError(
                f"need >= 6 aligned tenor/rate observations, got {n}/{len(zero_rates)}")
        for i in range(n):
            if not (tenor_years[i] > 0) or tenor_years[i] == math.inf:
                raise ValueError(f"tenor must be positive and finite: {tenor_years[i]}")
            if not math.isfinite(zero_rates[i]):
                raise ValueError(f"zero rate must be finite: {zero_rates[i]}")
        best_sse = math.inf
        best: Svensson.Fit | None = None
        nodes = 50
        lo, hi = math.log(0.1), math.log(10.0)
        for g1 in range(nodes):
            lambda1 = math.exp(lo + (hi - lo) * g1 / (nodes - 1))
            for g2 in range(g1 + 1, nodes):
                lambda2 = math.exp(lo + (hi - lo) * g2 / (nodes - 1))
                # OLS in (b0, b1, b2, b3) via 4x4 normal equations.
                xtx = [[0.0] * 4 for _ in range(4)]
                xty = [0.0] * 4
                for i in range(n):
                    x1 = tenor_years[i] / lambda1
                    x2 = tenor_years[i] / lambda2
                    f1 = (1 - math.exp(-x1)) / x1
                    row = (1.0, f1, f1 - math.exp(-x1),
                           (1 - math.exp(-x2)) / x2 - math.exp(-x2))
                    for a in range(4):
                        xty[a] += row[a] * zero_rates[i]
                        for b in range(4):
                            xtx[a][b] += row[a] * row[b]
                try:
                    beta = math_utils.solve_linear(xtx, xty)
                except ValueError:
                    continue    # degenerate node (near-collinear regressors); skip
                candidate = Svensson.Fit(beta[0], beta[1], beta[2], beta[3],
                                         lambda1, lambda2, 0.0)
                sse = 0.0
                for i in range(n):
                    e = candidate.zero_rate(tenor_years[i]) - zero_rates[i]
                    sse += e * e
                if sse < best_sse:
                    best_sse = sse
                    best = Svensson.Fit(beta[0], beta[1], beta[2], beta[3],
                                        lambda1, lambda2, math.sqrt(sse / n))
        if best is None:
            raise ValueError("no lambda node produced a solvable fit")
        return best
