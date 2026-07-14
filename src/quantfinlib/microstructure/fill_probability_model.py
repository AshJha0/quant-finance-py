"""Passive-fill probability for a limit order (port of Java
``microstructure.FillProbabilityModel``) resting AWAY from the touch
-- the placement question :class:`~quantfinlib.microstructure.queue_model.QueueModel`
alone can't answer. A resting order fills only if two things happen:

1. **The price reaches the level** -- under a driftless diffusion with
   volatility sigma, the probability the price travels a distance
   ``d`` within horizon ``T`` is the reflection-principle
   barrier-touch probability ``2*Phi(-d/(sigma*sqrt(T)))`` (1 when
   you're already at/through the level);
2. **The queue at the level clears to you** --
   :class:`~quantfinlib.microstructure.queue_model.QueueModel`'s
   territory: ``P = exp(-(qty_ahead + qty)/expected_traded)``.

:meth:`FillProbabilityModel.passive_fill_probability` composes the two
under an independence approximation -- documented honestly: touch and
queue-clearing are positively correlated (the flow that moves price
also eats queues), so the composition is a mild UNDERestimate; treat
it as a conservative placement score, not a calibrated probability.

Volatility enters as return-per-sqrt(second), converted to price units
against the current price. Static, cross-asset.
"""

from __future__ import annotations

import math

from quantfinlib.microstructure.queue_model import QueueModel
from quantfinlib.util import math_utils


class FillProbabilityModel:
    """Static entry points; see the module docstring."""

    @staticmethod
    def touch_probability(distance: float, vol_per_sqrt_second: float,
                          horizon_seconds: float, price: float) -> float:
        """Probability the price touches a level ``distance`` away (in
        price units, >= 0) within ``horizon_seconds``, given
        volatility ``vol_per_sqrt_second`` (return per sqrt(second))
        at ``price``. 1 at/through the level; 0 for degenerate inputs
        (no vol, no time, no price -- a dead market never reaches
        anything)."""
        if distance <= 0:
            return 1.0
        if (not (vol_per_sqrt_second > 0) or not (horizon_seconds > 0)
                or not (price > 0) or vol_per_sqrt_second == math.inf
                or distance == math.inf):
            return 0.0
        sigma_abs = price * vol_per_sqrt_second * math.sqrt(horizon_seconds)
        return math_utils.clamp(2 * math_utils.norm_cdf(-distance / sigma_abs),
                                0, 1)

    @staticmethod
    def passive_fill_probability(distance: float, vol_per_sqrt_second: float,
                                 horizon_seconds: float, price: float,
                                 qty_ahead: float, order_qty: float,
                                 expected_traded_qty: float) -> float:
        """Probability a passive order ``distance`` from the current
        price fills within the horizon: touch x queue-clear
        (independence approximation, mildly conservative -- see the
        module doc).

        Args:
            qty_ahead: shares ahead in the queue at the level (from
                :class:`~quantfinlib.microstructure.queue_position_estimator.QueuePositionEstimator`,
                or displayed size before joining).
            order_qty: our order size.
            expected_traded_qty: volume expected to execute at the
                level over the horizon (e.g. from
                :class:`~quantfinlib.microstructure.volume_curve.VolumeCurve`).
        """
        touch = FillProbabilityModel.touch_probability(
            distance, vol_per_sqrt_second, horizon_seconds, price)
        return touch * QueueModel.fill_probability(qty_ahead, order_qty,
                                                   expected_traded_qty)
