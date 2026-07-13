"""Position sizing rules (port of Java ``backtest.portfolio.PositionSizing``).

Kelly, fixed-fractional risk, inverse-volatility weighting, and
volatility targeting — the building blocks for portfolio weight
construction.
"""

from __future__ import annotations

import numpy as np


class PositionSizing:
    """Static position-sizing rules; see the module docstring."""

    @staticmethod
    def kelly_fraction(mean_return: float, variance: float) -> float:
        """Full Kelly fraction for a return stream: ``f* = mu / sigma^2``."""
        return 0.0 if variance == 0 else mean_return / variance

    @staticmethod
    def half_kelly(mean_return: float, variance: float) -> float:
        """Half-Kelly — the practitioner's standard, trading growth for drawdown."""
        return PositionSizing.kelly_fraction(mean_return, variance) / 2

    @staticmethod
    def fixed_fractional_quantity(equity: float, risk_fraction: float,
                                  entry_price: float, stop_price: float) -> float:
        """Fixed-fractional sizing: shares such that hitting the stop loses
        exactly ``risk_fraction`` of equity.

        Raises:
            ValueError: when entry equals stop (undefined risk).
        """
        per_share_risk = abs(entry_price - stop_price)
        if per_share_risk == 0:
            raise ValueError("entry equals stop: undefined risk")
        return equity * risk_fraction / per_share_risk

    @staticmethod
    def inverse_volatility_weights(vols) -> np.ndarray:
        """Normalized inverse-volatility weights (equal weight for any zero vols)."""
        v = np.asarray(vols, dtype=float)
        if np.any(v <= 0):
            # Degenerate input: fall back to equal weight.
            return np.full(v.shape[0], 1.0 / v.shape[0])
        inv = 1.0 / v
        return inv / float(np.sum(inv))

    @staticmethod
    def volatility_target_leverage(current_annual_vol: float,
                                   target_annual_vol: float) -> float:
        """Leverage multiplier that scales current volatility to the target."""
        if current_annual_vol <= 0:
            return 0.0
        return target_annual_vol / current_annual_vol
