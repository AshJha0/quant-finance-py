"""Equity-curve performance metrics (port of Java ``backtest.PerformanceAnalytics``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantfinlib.backtest import _risk
from quantfinlib.backtest.trade import Trade
from quantfinlib.util import math_utils as mu


@dataclass(frozen=True)
class PerformanceMetrics:
    """Strategy performance analytics.

    Returns/drawdown are fractions; ratios are annualized.
    """

    total_return: float
    cagr: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    trade_count: int
    final_equity: float


class PerformanceAnalytics:
    """Computes :class:`PerformanceMetrics` from an equity curve and trades."""

    @staticmethod
    def compute(equity, trades: Sequence[Trade],
                periods_per_year: int) -> PerformanceMetrics:
        """Computes the metric set on ``equity`` (>= 1 point) and ``trades``."""
        e = np.asarray(equity, dtype=float)
        n = e.shape[0]
        start = float(e[0])
        end = float(e[-1])
        total_return = end / start - 1

        cagr = 0.0
        if n > 1 and end > 0 and start > 0:
            cagr = (end / start) ** (periods_per_year / (n - 1)) - 1

        rets = e[1:] / e[:-1] - 1 if n > 1 else np.empty(0)

        if rets.shape[0] == 0:
            ann_return = ann_vol = sharpe = sortino = 0.0
        else:
            ann_return = mu.mean(rets) * periods_per_year
            ann_vol = _risk.annualized_volatility(rets, periods_per_year)
            sharpe = _risk.sharpe_ratio(rets, 0, periods_per_year)
            sortino = _risk.sortino_ratio(rets, 0, periods_per_year)
        max_dd = _risk.max_drawdown(e)
        calmar = 0.0 if max_dd == 0 else cagr / max_dd

        gross_profit = 0.0
        gross_loss = 0.0
        wins = 0
        for t in trades:
            if t.pnl >= 0:
                gross_profit += t.pnl
                if t.pnl > 0:
                    wins += 1
            else:
                gross_loss -= t.pnl
        if gross_loss == 0:
            profit_factor = math.inf if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss
        win_rate = 0.0 if len(trades) == 0 else wins / len(trades)

        return PerformanceMetrics(total_return, cagr, ann_return, ann_vol,
                                  sharpe, sortino, calmar, max_dd,
                                  profit_factor, win_rate, len(trades), end)
