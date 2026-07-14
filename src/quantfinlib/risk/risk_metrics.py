"""Core quantitative risk metrics.

Port of Java ``com.quantfinlib.risk.RiskMetrics``. Return-based metrics
take simple periodic returns (e.g. daily); VaR/CVaR are reported as
positive loss fractions.
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils as mu

TRADING_DAYS_PER_YEAR = 252


def volatility(returns) -> float:
    """Per-period sample volatility."""
    return mu.std_dev(returns)


def annualized_volatility(returns, periods_per_year: int) -> float:
    """Sample volatility scaled by the square root of time."""
    return volatility(returns) * math.sqrt(periods_per_year)


def historical_var(returns, confidence: float) -> float:
    """Historical Value at Risk at the given confidence level (e.g. 0.95).

    Returned as a positive loss fraction; 0 if the quantile is a gain.
    """
    q = mu.percentile(returns, 1 - confidence)
    return max(0.0, -q)


def parametric_var(returns, confidence: float) -> float:
    """Parametric (Gaussian) VaR at the given confidence level."""
    m = mu.mean(returns)
    sigma = mu.std_dev(returns)
    z = mu.norm_inv(1 - confidence)
    return max(0.0, -(m + z * sigma))


def conditional_var(returns, confidence: float) -> float:
    """Conditional VaR / Expected Shortfall: mean loss beyond the VaR threshold."""
    returns = np.asarray(returns, dtype=float)
    threshold = mu.percentile(returns, 1 - confidence)
    tail = returns[returns <= threshold]
    if tail.shape[0] == 0:
        return 0.0
    return max(0.0, -float(np.sum(tail)) / tail.shape[0])


def expected_shortfall(returns, confidence: float) -> float:
    """Alias for :func:`conditional_var`."""
    return conditional_var(returns, confidence)


def sharpe_ratio(returns, risk_free_rate: float, periods_per_year: int) -> float:
    """Annualized Sharpe ratio; ``risk_free_rate`` is annual."""
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    annual_return = mu.mean(returns) * periods_per_year
    return (annual_return - risk_free_rate) / vol


def sortino_ratio(returns, risk_free_rate: float, periods_per_year: int) -> float:
    """Annualized Sortino ratio using downside deviation below the periodic MAR."""
    mar = risk_free_rate / periods_per_year
    dd = downside_deviation(returns, mar) * math.sqrt(periods_per_year)
    if dd == 0:
        return 0.0
    annual_return = mu.mean(returns) * periods_per_year
    return (annual_return - risk_free_rate) / dd


def downside_deviation(returns, mar: float) -> float:
    """Per-period downside deviation below the minimum acceptable return."""
    returns = np.asarray(returns, dtype=float)
    d = np.minimum(0.0, returns - mar)
    return math.sqrt(float(np.sum(d * d)) / returns.shape[0])


def max_drawdown(equity) -> float:
    """Maximum peak-to-trough drawdown of an equity curve, as a positive fraction."""
    equity = np.asarray(equity, dtype=float)
    peaks = np.maximum.accumulate(equity)
    positive = peaks > 0  # a non-positive peak contributes no drawdown (Java gate)
    dd = np.where(positive, (peaks - equity) / np.where(positive, peaks, 1.0), 0.0)
    return max(0.0, float(np.max(dd)))


def beta(asset_returns, benchmark_returns) -> float:
    """Beta of an asset versus a benchmark (equal-length return series)."""
    var_b = mu.variance(benchmark_returns)
    if var_b == 0:
        return 0.0
    return mu.covariance(asset_returns, benchmark_returns) / var_b


def correlation(a, b) -> float:
    """Pearson correlation (delegates to math_utils)."""
    return mu.correlation(a, b)
