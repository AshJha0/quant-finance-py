"""Risk-metric helpers used by the backtest analytics lane.

Private port of the slice of Java ``com.quantfinlib.risk.RiskMetrics``
that ``PerformanceAnalytics`` and ``validation.BlockBootstrap`` consume
(annualized volatility, Sharpe, Sortino, max drawdown, beta). Kept
inside the ``backtest`` package so the analytics port is self-contained;
formulas are transcriptions of the Java arithmetic.
"""

from __future__ import annotations

import math

import numpy as np

from quantfinlib.util import math_utils as mu


def volatility(returns) -> float:
    """Per-period sample volatility."""
    return mu.std_dev(returns)


def annualized_volatility(returns, periods_per_year: int) -> float:
    """Sample volatility scaled by sqrt(periods per year)."""
    return volatility(returns) * math.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free_rate: float, periods_per_year: int) -> float:
    """Annualized Sharpe ratio; 0 when the volatility is 0."""
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
    r = np.asarray(returns, dtype=float)
    d = np.minimum(0.0, r - mar)
    return math.sqrt(float(np.sum(d * d)) / r.shape[0])


def max_drawdown(equity) -> float:
    """Maximum peak-to-trough drawdown of an equity curve, as a positive fraction."""
    e = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(e)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - e) / peak, 0.0)
    return float(np.max(dd)) if dd.size else 0.0


def beta(asset_returns, benchmark_returns) -> float:
    """Beta of an asset versus a benchmark; 0 when the benchmark is constant."""
    var_b = mu.variance(benchmark_returns)
    if var_b == 0:
        return 0.0
    return mu.covariance(asset_returns, benchmark_returns) / var_b
