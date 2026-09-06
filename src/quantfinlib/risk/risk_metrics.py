"""Core quantitative risk metrics.

Port of Java ``com.quantfinlib.risk.RiskMetrics``. Return-based metrics
take simple periodic returns (e.g. daily); VaR/CVaR are reported as
positive loss fractions.

Input contract (the NaN gate): every function here requires a
NON-EMPTY, FINITE series and raises ``ValueError`` otherwise. Python's
``max(0.0, x)`` returns ``0.0`` when ``x`` is NaN (the comparison is
False and the first argument wins), so without the gate a single
missing print in a return series would report ZERO risk. The gate
runs before any clamp, so NaN can never reach one.

Convention note: :func:`conditional_var` averages the returns at or
below the INTERPOLATED ``(1 - confidence)`` percentile (the Java
``RiskMetrics`` rule), whereas :func:`quantfinlib.risk.var_engine.tail`
uses the ``ceil(c * n) - 1`` order statistic. For small samples the two
differ by an order statistic; both are documented, neither is "wrong".
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils as mu

TRADING_DAYS_PER_YEAR = 252


def _require_series(values, name: str = "returns") -> np.ndarray:
    """Non-empty, all-finite 1-D float array or ``ValueError``."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1-D series")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (NaN/Inf found)")
    return arr


def _require_confidence(confidence: float) -> None:
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def _clamp_loss(x: float) -> float:
    """Non-negative loss; only ever called on a finite number."""
    return 0.0 if x < 0 else float(x)


def volatility(returns) -> float:
    """Per-period sample volatility."""
    return mu.std_dev(_require_series(returns))


def annualized_volatility(returns, periods_per_year: int) -> float:
    """Sample volatility scaled by the square root of time."""
    return volatility(returns) * math.sqrt(periods_per_year)


def historical_var(returns, confidence: float) -> float:
    """Historical Value at Risk at the given confidence level (e.g. 0.95).

    Returned as a positive loss fraction; 0 if the quantile is a gain.

    Raises:
        ValueError: on an empty or non-finite series, or confidence
            outside (0, 1).
    """
    returns = _require_series(returns)
    _require_confidence(confidence)
    q = mu.percentile(returns, 1 - confidence)
    return _clamp_loss(-q)


def parametric_var(returns, confidence: float) -> float:
    """Parametric (Gaussian) VaR at the given confidence level.

    Raises:
        ValueError: on an empty or non-finite series, or confidence
            outside (0, 1).
    """
    returns = _require_series(returns)
    _require_confidence(confidence)
    m = mu.mean(returns)
    sigma = mu.std_dev(returns)
    z = mu.norm_inv(1 - confidence)
    return _clamp_loss(-(m + z * sigma))


def conditional_var(returns, confidence: float) -> float:
    """Conditional VaR / Expected Shortfall: mean loss beyond the VaR threshold.

    Raises:
        ValueError: on an empty or non-finite series, or confidence
            outside (0, 1).
    """
    returns = _require_series(returns)
    _require_confidence(confidence)
    threshold = mu.percentile(returns, 1 - confidence)
    tail = returns[returns <= threshold]
    if tail.shape[0] == 0:
        return 0.0
    return _clamp_loss(-float(np.sum(tail)) / tail.shape[0])


def expected_shortfall(returns, confidence: float) -> float:
    """Alias for :func:`conditional_var`."""
    return conditional_var(returns, confidence)


def sharpe_ratio(returns, risk_free_rate: float, periods_per_year: int) -> float:
    """Annualized Sharpe ratio; ``risk_free_rate`` is annual."""
    returns = _require_series(returns)
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    annual_return = mu.mean(returns) * periods_per_year
    return (annual_return - risk_free_rate) / vol


def sortino_ratio(returns, risk_free_rate: float, periods_per_year: int) -> float:
    """Annualized Sortino ratio using downside deviation below the periodic MAR."""
    returns = _require_series(returns)
    mar = risk_free_rate / periods_per_year
    dd = downside_deviation(returns, mar) * math.sqrt(periods_per_year)
    if dd == 0:
        return 0.0
    annual_return = mu.mean(returns) * periods_per_year
    return (annual_return - risk_free_rate) / dd


def downside_deviation(returns, mar: float) -> float:
    """Per-period downside deviation below the minimum acceptable return."""
    returns = _require_series(returns)
    d = np.minimum(0.0, returns - mar)
    return math.sqrt(float(np.sum(d * d)) / returns.shape[0])


def max_drawdown(equity) -> float:
    """Maximum peak-to-trough drawdown of an equity curve, as a positive fraction.

    Raises:
        ValueError: on an empty or non-finite equity curve.
    """
    equity = _require_series(equity, "equity")
    peaks = np.maximum.accumulate(equity)
    positive = peaks > 0  # a non-positive peak contributes no drawdown (Java gate)
    dd = np.where(positive, (peaks - equity) / np.where(positive, peaks, 1.0), 0.0)
    return _clamp_loss(float(np.max(dd)))


def beta(asset_returns, benchmark_returns) -> float:
    """Beta of an asset versus a benchmark (equal-length return series)."""
    asset_returns = _require_series(asset_returns, "asset_returns")
    benchmark_returns = _require_series(benchmark_returns, "benchmark_returns")
    if asset_returns.shape[0] != benchmark_returns.shape[0]:
        raise ValueError("asset and benchmark series must have equal length")
    var_b = mu.variance(benchmark_returns)
    if var_b == 0:
        return 0.0
    return mu.covariance(asset_returns, benchmark_returns) / var_b


def correlation(a, b) -> float:
    """Pearson correlation (delegates to math_utils)."""
    return mu.correlation(a, b)
