"""Queue positioning and priority analytics (port of Java
``microstructure.QueueModel``) -- how position in the price-time queue
-- and small latency differences in reaching it -- translate into fill
probability.

Model: executed volume at a price level over a horizon is treated as
exponentially distributed with mean ``expected_traded_qty``; an order
fills when cumulative executions reach the quantity ahead of it plus
its own size, giving
``P(fill) = exp(-(qty_ahead + order_qty) / expected_traded_qty)``. A
deliberately simple, closed-form model -- calibrate
``expected_traded_qty`` from observed level turnover.

Not itself named in the port's task list, but required plumbing for
:mod:`quantfinlib.microstructure.queue_position_estimator` and
:mod:`quantfinlib.microstructure.fill_probability_model`, both of
which call it in the Java source.
"""

from __future__ import annotations

import math


class QueueModel:
    """Static queue-fill-probability entry points; see the module
    docstring."""

    @staticmethod
    def fill_probability(qty_ahead: float, order_qty: float,
                         expected_traded_qty: float) -> float:
        """Probability the order fully fills within the horizon.

        Args:
            qty_ahead: resting quantity ahead in the queue.
            order_qty: our order size.
            expected_traded_qty: expected volume to execute at this
                level over the horizon.
        """
        if expected_traded_qty <= 0:
            return 0.0
        return math.exp(-(qty_ahead + order_qty) / expected_traded_qty)

    @staticmethod
    def queue_growth(join_rate_qty_per_sec: float,
                     latency_nanos: int) -> float:
        """Extra quantity that joins the queue ahead of an order
        arriving ``latency_nanos`` later, given the rate at which
        others join."""
        return max(0.0, join_rate_qty_per_sec) * latency_nanos / 1e9

    @staticmethod
    def latency_fill_advantage(qty_ahead: float, order_qty: float,
                               expected_traded_qty: float,
                               join_rate_qty_per_sec: float,
                               latency_advantage_nanos: int) -> float:
        """Fill-probability edge from being
        ``latency_advantage_nanos`` faster to the queue:
        ``P(fill | fast arrival) - P(fill | slow arrival)``."""
        fast = QueueModel.fill_probability(qty_ahead, order_qty,
                                           expected_traded_qty)
        # Java Math.round: half-up (floor(x + 0.5)), not Python's
        # banker's-rounding round() -- queue_growth can be negative
        # (a latency DISADVANTAGE), and the two only agree at half
        # integers by coincidence, e.g. round(-1.5) is -2 in Python
        # (round-half-to-even) but -1 in Java.
        extra = math.floor(QueueModel.queue_growth(
            join_rate_qty_per_sec, latency_advantage_nanos) + 0.5)
        slow = QueueModel.fill_probability(qty_ahead + extra, order_qty,
                                           expected_traded_qty)
        return fast - slow
