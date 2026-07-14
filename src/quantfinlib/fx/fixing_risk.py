"""Benchmark-fixing exposure analytics (port of Java
``com.quantfinlib.fx.FixingRisk``).

A book that must trade AT the fix (hedging an NDF settlement, matching
a benchmarked mandate) works the order through the calculation window
and receives roughly the window TWAP/VWAP while its liability
references the fix print. These helpers quantify both sides.
"""

from __future__ import annotations

import math

from quantfinlib.fx.currency_pair import CurrencyPair


class FixingRisk:
    """Static analytics namespace, mirroring the Java final class."""

    @staticmethod
    def window_twap(prices) -> float:
        """Time-weighted average of window prices (equally spaced)."""
        if len(prices) == 0:
            raise ValueError("empty window")
        return sum(prices) / len(prices)

    @staticmethod
    def window_vwap(prices, sizes) -> float:
        """Volume-weighted average of window prices."""
        if len(prices) == 0 or len(prices) != len(sizes):
            raise ValueError("prices/sizes must be non-empty and aligned")
        v = sum(sizes)
        if v <= 0:
            raise ValueError("window volume must be > 0")
        return sum(p * s for p, s in zip(prices, sizes)) / v

    @staticmethod
    def slippage_vs_fix(pair: CurrencyPair, achieved_price: float,
                        fix_price: float) -> float:
        """Realized slippage vs the fix print, in pips (signed, buy side)."""
        return pair.pips(achieved_price - fix_price)

    @staticmethod
    def tracking_error_std(vol_per_sqrt_minute: float,
                           window_minutes: float) -> float:
        """Ex-ante 1-sigma tracking error (price terms) between a fix
        print and a uniform execution across the window: for arithmetic
        Brownian motion, var(fix - TWAP) = sigma^2 T / 3 — the classic
        TWAP-vs-close result."""
        if vol_per_sqrt_minute < 0 or window_minutes <= 0:
            raise ValueError("vol must be >= 0 and window > 0")
        return vol_per_sqrt_minute * math.sqrt(window_minutes / 3.0)

    @staticmethod
    def participation_rate(order_qty: float,
                           expected_window_volume: float) -> float:
        """Order size as a fraction of expected window volume — above
        ~20% the order moves the fix it is trying to match."""
        if expected_window_volume <= 0:
            raise ValueError("expected window volume must be > 0")
        return abs(order_qty) / expected_window_volume
