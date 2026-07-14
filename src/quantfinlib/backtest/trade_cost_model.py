"""Pluggable per-trade cost model (port of Java ``backtest.TradeCostModel``).

The ONE definition of "what a trade costs" shared by the backtest
engines, so an execution-aware number and a survivorship-aware number
can come out of the *same* run:

* :meth:`TradeCostModel.flat` — a fixed all-in bps (the classic
  commission assumption, and the exact equivalent of the legacy
  ``commission_rate`` configs);
* :meth:`TradeCostModel.institutional` — commission + half-spread +
  slippage + square-root market impact, with per-symbol ADV/vol
  estimated from the trailing bars via
  :meth:`~quantfinlib.microstructure.market_impact_model.MarketImpactModel.estimate`.
  The impact term is what makes cost grow with book size, i.e. what
  turns "capacity" into a number.

The contract prices ONE side of a trade (a buy or a sell), all-in, in
basis points of traded notional. Implementations must be pure functions
of their arguments — engines may call them at any bar in any order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quantfinlib.data.bar_series import BarSeries
from quantfinlib.microstructure.market_impact_model import MarketImpactModel


class TradeCostModel(ABC):
    """One-way, all-in trade cost in bps of traded notional."""

    @abstractmethod
    def cost_bps(self, series: BarSeries, index: int,
                 notional: float) -> float:
        """All-in one-way cost, in bps of traded notional, for trading
        ``notional`` (currency units, always positive) of ``series`` at
        bar ``index``."""

    @staticmethod
    def flat(bps: float) -> "TradeCostModel":
        """Fixed all-in bps per trade — commission-only, size-independent."""
        if bps < 0:
            raise ValueError("bps must be >= 0")
        return _Flat(bps)

    @staticmethod
    def institutional(commission_bps: float, half_spread_bps: float,
                      slippage_bps: float,
                      impact_window: int) -> "TradeCostModel":
        """The institutional four-component model. Impact needs trailing
        ADV/vol: bars before ``impact_window`` charge the flat components
        only (rather than reading before bar 0), as do series without
        volume data — documented degradation, never a crash.

        Args:
            commission_bps: Commission per side.
            half_spread_bps: Half the quoted spread, paid on every trade.
            slippage_bps: Fixed implementation noise.
            impact_window: Trailing bars for ADV/vol estimation (>= 2).
        """
        if (commission_bps < 0 or half_spread_bps < 0 or slippage_bps < 0
                or impact_window < 2):
            raise ValueError(
                "cost components must be >= 0 and impact_window >= 2")
        return _Institutional(
            commission_bps + half_spread_bps + slippage_bps, impact_window)


class _Flat(TradeCostModel):

    __slots__ = ("_bps",)

    def __init__(self, bps: float) -> None:
        self._bps = bps

    def cost_bps(self, series: BarSeries, index: int,
                 notional: float) -> float:
        return self._bps


class _Institutional(TradeCostModel):

    __slots__ = ("_flat", "_impact_window")

    def __init__(self, flat: float, impact_window: int) -> None:
        self._flat = flat
        self._impact_window = impact_window

    def cost_bps(self, series: BarSeries, index: int,
                 notional: float) -> float:
        if index < self._impact_window:
            return self._flat
        impact = MarketImpactModel.estimate(series, index,
                                            self._impact_window)
        if impact is None:
            return self._flat  # no volume data: impact unknowable
        shares = notional / series.close(index)
        return self._flat + impact.square_root_impact_bps(shares)
