"""Temporary / permanent market impact models (port of Java
``microstructure.MarketImpactModel``).

Parameterized by average daily volume (ADV) and daily volatility:

* Square-root law (empirical standard):
  ``impact = Y * sigma_daily * sqrt(Q / ADV)``.
* Almgren-Chriss style decomposition: linear temporary impact in
  participation rate and linear permanent impact in size, with the
  expected cost of an execution schedule
  ``E[cost] = permanent/2 + temporary``.

All results in basis points of the arrival price. Coefficients default
to commonly cited magnitudes; calibrate per market with
:meth:`MarketImpactModel.with_coefficients`.
"""

from __future__ import annotations

import math
from typing import Optional

from quantfinlib.data.bar_series import BarSeries


class MarketImpactModel:
    """ADV/vol-parameterized impact model; see the module docstring."""

    __slots__ = ("_adv", "_daily_volatility", "_y", "_eta_bps", "_gamma")

    def __init__(self, adv: float, daily_volatility: float,
                 y: float = 1.0, eta_bps: float = 20.0,
                 gamma: float = 0.5) -> None:
        if adv <= 0:
            raise ValueError("adv must be positive")
        self._adv = adv
        self._daily_volatility = daily_volatility
        self._y = y                # square-root law coefficient (~1)
        self._eta_bps = eta_bps    # temporary impact bps at 100% participation
        self._gamma = gamma        # permanent impact coefficient

    def with_coefficients(self, y: float, eta_bps: float,
                          gamma: float) -> "MarketImpactModel":
        """Copy with calibrated coefficients (sqrt Y, temporary eta bps,
        permanent gamma)."""
        return MarketImpactModel(self._adv, self._daily_volatility,
                                 y, eta_bps, gamma)

    @staticmethod
    def estimate(series: BarSeries, index: int,
                 window: int) -> Optional["MarketImpactModel"]:
        """Estimates a model from a series' trailing window: ADV as the
        mean volume, daily vol as the stdev of close-to-close returns —
        the one canonical bar-data -> impact-model bridge, shared by the
        backtesters so their impact numbers can never diverge. Returns
        ``None`` when the window has no volume (impact needs ADV; absent
        volume is a data gap, not free liquidity — callers charge their
        flat costs and skip impact).
        """
        if index < window:
            raise ValueError(f"index {index} < estimation window {window}")
        adv_sum = 0.0
        mean = 0.0
        for j in range(window):
            adv_sum += series.volume(index - window + j + 1)
            mean += (series.close(index - window + j + 1)
                     / series.close(index - window + j) - 1)
        adv = adv_sum / window
        if adv <= 0:
            return None
        mean /= window
        var = 0.0
        for j in range(window):
            r = (series.close(index - window + j + 1)
                 / series.close(index - window + j) - 1)
            var += (r - mean) * (r - mean)
        return MarketImpactModel(adv, math.sqrt(var / window))

    def square_root_impact_bps(self, quantity: float) -> float:
        """Square-root-law total impact for an order of ``quantity``."""
        return self._y * self._daily_volatility * math.sqrt(
            quantity / self._adv) * 1e4

    def temporary_impact_bps(self, participation_rate: float) -> float:
        """Temporary (execution-rate) impact at a participation in [0, 1]."""
        return self._eta_bps * participation_rate

    def permanent_impact_bps(self, quantity: float) -> float:
        """Permanent (information) impact, linear in size relative to ADV."""
        return (self._gamma * self._daily_volatility
                * (quantity / self._adv) * 1e4)

    def expected_cost_bps(self, quantity: float,
                          participation_rate: float) -> float:
        """Expected implementation cost of executing ``quantity`` at the
        given participation rate: half the permanent impact (average price
        concession over the schedule) plus the full temporary impact."""
        return (self.permanent_impact_bps(quantity) / 2
                + self.temporary_impact_bps(participation_rate))
