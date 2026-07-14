"""Market quality indices (port of Java ``regulatory.MarketQualityMetrics``).

Used in execution-quality and venue-quality reporting: quoted /
effective / realized spread, price impact, and order-to-trade ratio.
All spreads in basis points; signs follow the convention that positive
= cost to the liquidity taker.
"""

from __future__ import annotations

import math

from quantfinlib.microstructure.execution import Side


def quoted_spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2
    return math.nan if mid == 0 else (ask - bid) / mid * 1e4


def effective_spread_bps(taker_side: Side, price: float, mid_at_execution: float) -> float:
    """Effective spread: ``2 * sign * (price - mid) / mid`` -- what the
    taker actually paid."""
    return 2.0 * taker_side.sign() * (price - mid_at_execution) / mid_at_execution * 1e4


def realized_spread_bps(taker_side: Side, price: float, mid_after_horizon: float) -> float:
    """Realized spread: effective spread measured against the mid some
    horizon after the trade -- the part of the spread the liquidity
    provider kept after adverse selection."""
    return 2.0 * taker_side.sign() * (price - mid_after_horizon) / mid_after_horizon * 1e4


def price_impact_bps(taker_side: Side, mid_at_execution: float, mid_after_horizon: float) -> float:
    """Price impact: how far the mid moved in the taker's direction after
    the trade (``2 * sign * (mid_after - mid_at_exec) / mid_at_exec``);
    effective spread ~= realized spread + price impact."""
    return 2.0 * taker_side.sign() * (mid_after_horizon - mid_at_execution) / mid_at_execution * 1e4


def order_to_trade_ratio(messages: int, trades: int) -> float:
    """Messages (orders + cancels + replaces) per executed trade."""
    return math.inf if trades == 0 else messages / trades
