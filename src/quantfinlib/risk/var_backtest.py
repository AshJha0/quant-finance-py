"""VaR model validation: do exceptions occur at the promised rate, independently?

Port of Java ``com.quantfinlib.risk.VarBacktest``:

* Kupiec POF — likelihood-ratio test that the exception frequency
  matches ``1 - confidence`` (two-sided: both too many and too few
  exceptions reject). chi2(1).
* Christoffersen independence — LR test that exceptions do not cluster
  (a model right on average but wrong in crises fails here). chi2(1).
* Conditional coverage — the joint test (POF + independence), chi2(2),
  with the exact p-value ``exp(-LR/2)``.

A p-value below the chosen significance (e.g. 0.05) rejects the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantfinlib.util import math_utils as mu


@dataclass(frozen=True)
class VarBacktestResult:
    observations: int
    exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec_statistic: float
    kupiec_p_value: float
    independence_statistic: float
    independence_p_value: float
    conditional_coverage_statistic: float
    conditional_coverage_p_value: float

    def calibrated(self, significance: float) -> bool:
        """Exception frequency consistent with the confidence level?"""
        return self.kupiec_p_value > significance

    def independent(self, significance: float) -> bool:
        """Exceptions arrive independently (no crisis clustering)?"""
        return self.independence_p_value > significance

    def passes(self, significance: float) -> bool:
        """Joint test: right rate AND independent."""
        return self.conditional_coverage_p_value > significance


def test(returns, var_forecasts, confidence: float) -> VarBacktestResult:
    """Backtests VaR forecasts against realized returns.

    Args:
        returns: realized periodic returns.
        var_forecasts: VaR forecast per period as a positive loss
            fraction, aligned with ``returns`` — or a scalar for a
            constant VaR (the Java constant-VaR overload).
        confidence: the VaR confidence level (e.g. 0.95, 0.99).
    """
    r = np.asarray(returns, dtype=float)
    if np.ndim(var_forecasts) == 0:
        v = np.full(r.shape[0], float(var_forecasts))
    else:
        v = np.asarray(var_forecasts, dtype=float)
    if r.shape[0] != v.shape[0] or r.shape[0] < 20:
        raise ValueError("need >= 20 aligned returns/forecasts")
    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be in (0,1)")
    n = r.shape[0]
    exception = r < -v
    x = int(np.count_nonzero(exception))
    p = 1 - confidence

    # Kupiec proportion-of-failures LR.
    observed_rate = x / n
    ll_null = _x_ln_p(n - x, 1 - p) + _x_ln_p(x, p)
    ll_alt = _x_ln_p(n - x, 1 - observed_rate) + _x_ln_p(x, observed_rate)
    kupiec = max(0.0, 2 * (ll_alt - ll_null))
    kupiec_p = _chi_square_1_p_value(kupiec)

    # Christoffersen independence LR over exception transitions.
    prev = exception[:-1]
    cur = exception[1:]
    n00 = int(np.count_nonzero(~prev & ~cur))
    n01 = int(np.count_nonzero(~prev & cur))
    n10 = int(np.count_nonzero(prev & ~cur))
    n11 = int(np.count_nonzero(prev & cur))
    independence = 0.0
    transitions = n00 + n01 + n10 + n11
    exceptions_after = n01 + n11
    if transitions > 0 and 0 < exceptions_after < transitions:
        pi = exceptions_after / transitions
        pi01 = 0.0 if n00 + n01 == 0 else n01 / (n00 + n01)
        pi11 = 0.0 if n10 + n11 == 0 else n11 / (n10 + n11)
        ll_pooled = _x_ln_p(n00 + n10, 1 - pi) + _x_ln_p(n01 + n11, pi)
        ll_markov = (_x_ln_p(n00, 1 - pi01) + _x_ln_p(n01, pi01)
                     + _x_ln_p(n10, 1 - pi11) + _x_ln_p(n11, pi11))
        independence = max(0.0, 2 * (ll_markov - ll_pooled))
    independence_p = _chi_square_1_p_value(independence)

    cc = kupiec + independence
    cc_p = math.exp(-cc / 2)   # exact chi-square(2) survival

    return VarBacktestResult(n, x, p * n, observed_rate,
                             kupiec, kupiec_p, independence, independence_p,
                             cc, cc_p)


def _x_ln_p(count: int, probability: float) -> float:
    """``count * ln(probability)`` with the 0*ln(0) = 0 convention."""
    if count == 0:
        return 0.0
    if probability <= 0:
        return -math.inf
    return count * math.log(probability)


def _chi_square_1_p_value(statistic: float) -> float:
    """chi2(1) survival function via the normal CDF."""
    if statistic <= 0:
        return 1.0
    return 2 * (1 - mu.norm_cdf(math.sqrt(statistic)))
