"""Tick-frequency Greek estimation without tick-frequency repricing (port
of Java ``com.quantfinlib.pricing.IncrementalGreeks``).

A full Black-Scholes evaluation anchors the position, and every tick
updates price/delta by the delta-gamma Taylor expansion — a handful of
multiplies — while the anchor is refreshed off the hot path::

    price(S) ~ price0 + delta0 (S - S0) + 0.5 gamma0 (S - S0)^2
    delta(S) ~ delta0 + gamma0 (S - S0)

``needs_reprice`` tells the slow path when spot has drifted far enough
to re-anchor. Single-threaded by design — one instance per position per
risk thread.
"""

from __future__ import annotations

import math

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType


class IncrementalGreeks:
    """Delta-gamma anchored tick estimator."""

    def __init__(self) -> None:
        # Anchor state from the last full reprice.
        self._base_spot = math.nan
        self._base_price = 0.0
        self._base_delta = 0.0
        self._base_gamma = 0.0
        self._base_vega = 0.0
        self._base_theta = 0.0
        # Tick-fresh estimates.
        self._last_spot = math.nan
        self._estimated_price = 0.0
        self._estimated_delta = 0.0

    def reprice(self, option_type: OptionType, spot: float, strike: float, rate: float,
                carry: float, vol: float, time_years: float) -> None:
        """Full reprice: re-anchors the expansion. Call from the pricing/slow
        thread — at start-up, on ``needs_reprice``, on vol or rate marks."""
        g = BlackScholes.greeks(option_type, spot, strike, rate, carry, vol, time_years)
        self._base_spot = spot
        self._base_price = g.price
        self._base_delta = g.delta
        self._base_gamma = g.gamma
        self._base_vega = g.vega
        self._base_theta = g.theta
        self._last_spot = spot
        self._estimated_price = g.price
        self._estimated_delta = g.delta

    def on_tick(self, spot: float) -> None:
        """The hot path: delta-gamma update from the anchor. No transcendental
        math — two multiplies and three adds."""
        d_s = spot - self._base_spot
        self._estimated_price = (self._base_price + self._base_delta * d_s
                                 + 0.5 * self._base_gamma * d_s * d_s)
        self._estimated_delta = self._base_delta + self._base_gamma * d_s
        self._last_spot = spot

    def needs_reprice(self, max_spot_drift: float) -> bool:
        """Whether spot has drifted beyond ``max_spot_drift`` from the anchor —
        the signal for the slow path to ``reprice``. The tick thread only
        reads a flag-style comparison; it never re-anchors itself."""
        return abs(self._last_spot - self._base_spot) > max_spot_drift

    def estimated_price(self) -> float:
        """Tick-fresh price estimate (per unit; scale by position externally)."""
        return self._estimated_price

    def estimated_delta(self) -> float:
        """Tick-fresh delta estimate."""
        return self._estimated_delta

    def gamma(self) -> float:
        """Anchor gamma (constant between reprices — second order is the anchor's)."""
        return self._base_gamma

    def vega(self) -> float:
        """Anchor vega — vol risk only changes on reprice, not per tick."""
        return self._base_vega

    def theta(self) -> float:
        """Anchor theta."""
        return self._base_theta

    def anchor_spot(self) -> float:
        """The spot the expansion is anchored at (NaN before the first reprice)."""
        return self._base_spot
