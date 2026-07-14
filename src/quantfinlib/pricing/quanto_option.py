"""Quanto adjustment (port of Java ``com.quantfinlib.pricing.QuantoOption``).

For payoffs on a foreign asset settled in domestic currency at a FIXED
conversion rate (a Nikkei option paying in USD at 1:1). The buyer bears
no FX risk, but the HEDGER does: the delta hedge lives in the asset's
own currency, so the hedge P&L converts at a floating rate that is
CORRELATED with the asset. That correlation has a price, and it shows up
as a drift correction::

    F_quanto = S * e^{(r_dom - q - rho * vol_S * vol_FX) * T}

``rho`` is the correlation between the asset and the FX rate quoted as
DOMESTIC PER FOREIGN. Positive rho lowers the quanto forward. The vol of
the quanto payoff stays the asset's own vol_S: only the drift moves.
Pricing delegates to ``BlackScholes`` with the carry bumped by
``rho * vol_S * vol_FX``. Research lane.
"""

from __future__ import annotations

import math

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType


class QuantoOption:
    """Static namespace, mirroring the Java final class."""

    @staticmethod
    def quanto_forward(spot: float, domestic_rate: float, asset_yield: float,
                       asset_vol: float, fx_vol: float, rho: float,
                       time_years: float) -> float:
        """The quanto-adjusted forward (domestic-settled, fixed conversion)."""
        _validate(spot, asset_vol, fx_vol, rho, domestic_rate, asset_yield)
        if not (time_years >= 0) or time_years == math.inf:
            raise ValueError("timeYears must be >= 0 and finite")
        return spot * math.exp(
            (domestic_rate - asset_yield - rho * asset_vol * fx_vol) * time_years)

    @staticmethod
    def price(option_type: OptionType, spot: float, strike: float,
              domestic_rate: float, asset_yield: float,
              asset_vol: float, fx_vol: float, rho: float, time_years: float) -> float:
        """Quanto vanilla priced in domestic currency per unit of the fixed
        conversion rate: Black-Scholes with the carry shifted by
        ``rho * vol_S * vol_FX``."""
        _validate(spot, asset_vol, fx_vol, rho, domestic_rate, asset_yield)
        if not (strike > 0) or strike == math.inf:
            raise ValueError(f"strike must be positive and finite, got {strike}")
        return BlackScholes.price(option_type, spot, strike, domestic_rate,
                                  asset_yield + rho * asset_vol * fx_vol,
                                  asset_vol, time_years)


def _validate(spot: float, asset_vol: float, fx_vol: float, rho: float,
              domestic_rate: float, asset_yield: float) -> None:
    if not (spot > 0) or spot == math.inf:
        raise ValueError(f"spot must be positive and finite, got {spot}")
    if (not (asset_vol >= 0) or asset_vol == math.inf
            or not (fx_vol >= 0) or fx_vol == math.inf):
        raise ValueError("vols must be >= 0 and finite")
    if not (rho >= -1) or not (rho <= 1):
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
    if not (math.isfinite(domestic_rate) and math.isfinite(asset_yield)):
        raise ValueError("rates must be finite")
