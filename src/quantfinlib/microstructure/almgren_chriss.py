"""Almgren-Chriss (2000) optimal execution (port of Java
``microstructure.AlmgrenChriss``).

The trading trajectory that minimizes ``E[cost] + lambda * Var[cost]``
when liquidating X shares over a horizon, trading off temporary impact
(fast execution is expensive) against price risk (slow execution is
risky). Closed-form discrete solution: holdings decay as
``x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)``, with the
urgency parameter kappa growing with risk aversion. lambda -> 0
recovers TWAP.

Uses the same impact parameterization as
:class:`~quantfinlib.microstructure.market_impact_model.MarketImpactModel`:
temporary impact eta (price concession per unit trade rate) and
permanent impact gamma (per share). All quantities share one time unit
(e.g. days): sigma is price volatility per sqrt(time), T the horizon in
those units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Sequence

import numpy as np


class AlmgrenChriss:
    """Static optimal-execution solver; see the module docstring."""

    @dataclass(frozen=True)
    class Params:
        """Almgren-Chriss problem parameters.

        Attributes:
            total_shares: X, the parent order size.
            horizon: T, execution horizon (in the chosen time unit).
            intervals: N trading intervals.
            volatility: sigma, price volatility per sqrt(time-unit).
            temporary_impact: eta, price concession per unit of trade rate.
            permanent_impact: gamma, permanent price move per share traded.
            risk_aversion: lambda >= 0, variance penalty (0 = TWAP).
        """

        total_shares: float
        horizon: float
        intervals: int
        volatility: float
        temporary_impact: float
        permanent_impact: float
        risk_aversion: float

        def __post_init__(self) -> None:
            if (self.total_shares <= 0 or self.horizon <= 0
                    or self.intervals < 1):
                raise ValueError("need positive size, horizon, intervals")
            if (self.temporary_impact <= 0 or self.risk_aversion < 0
                    or self.volatility < 0):
                raise ValueError("need eta > 0, lambda >= 0, sigma >= 0")

        def with_risk_aversion(self, lam: float) -> "AlmgrenChriss.Params":
            return replace(self, risk_aversion=lam)

    @dataclass(frozen=True)
    class Trajectory:
        """The optimal schedule: ``holdings[j]`` is the position after
        interval j (holdings[0] = X, holdings[N] = 0); ``trades[j]`` is
        sold in interval j+1. Costs are in price*shares units versus the
        arrival price.
        """

        holdings: np.ndarray
        trades: np.ndarray
        kappa: float
        expected_cost: float
        cost_variance: float

    @staticmethod
    def optimal_trajectory(p: "AlmgrenChriss.Params") -> "AlmgrenChriss.Trajectory":
        """Solves for the optimal liquidation trajectory.

        Raises:
            ValueError: when ``eta - gamma*tau/2 <= 0`` (permanent impact
                too large for the interval size).
        """
        n = p.intervals
        tau = p.horizon / n
        eta_tilde = p.temporary_impact - 0.5 * p.permanent_impact * tau
        if eta_tilde <= 0:
            raise ValueError("eta - gamma*tau/2 must be positive; "
                             "shorten intervals or check impacts")
        kappa = 0.0
        lambda_sigma2 = p.risk_aversion * p.volatility * p.volatility
        if lambda_sigma2 > 0:
            cosh_kappa_tau = 1 + lambda_sigma2 * tau * tau / (2 * eta_tilde)
            kappa = math.acosh(cosh_kappa_tau) / tau

        holdings = np.zeros(n + 1)
        x = p.total_shares
        if kappa * p.horizon < 1e-9:
            # Risk-neutral limit: linear (TWAP) trajectory.
            for j in range(n + 1):
                holdings[j] = x * (1.0 - j / n)
        else:
            denom = math.sinh(kappa * p.horizon)
            for j in range(n + 1):
                holdings[j] = x * math.sinh(kappa * (p.horizon - j * tau)) / denom
        holdings[n] = 0.0

        trades = np.zeros(n)
        expected_cost = 0.5 * p.permanent_impact * x * x
        variance = 0.0
        for j in range(n):
            trades[j] = holdings[j] - holdings[j + 1]
            expected_cost += eta_tilde / tau * trades[j] * trades[j]
            variance += (p.volatility * p.volatility * tau
                         * holdings[j + 1] * holdings[j + 1])
        return AlmgrenChriss.Trajectory(holdings, trades, kappa,
                                        expected_cost, variance)

    @staticmethod
    def twap(p: "AlmgrenChriss.Params") -> "AlmgrenChriss.Trajectory":
        """The risk-neutral (lambda = 0) linear schedule, for comparison."""
        return AlmgrenChriss.optimal_trajectory(p.with_risk_aversion(0))

    @staticmethod
    def efficient_frontier(base: "AlmgrenChriss.Params",
                           risk_aversions: Sequence[float]
                           ) -> List["AlmgrenChriss.Trajectory"]:
        """Cost/risk frontier across risk aversions (for choosing urgency)."""
        return [AlmgrenChriss.optimal_trajectory(base.with_risk_aversion(lam))
                for lam in risk_aversions]
