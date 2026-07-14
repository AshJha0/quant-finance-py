"""Execution models for the execution-aware backtester (ports of Java
``backtest.ExecutionModel``, ``InstantExecution``, ``IcebergExecution``,
``LastLookExecution`` and ``execution.IcebergOrder``).

How parent orders turn into fills: the engine calls
:meth:`ExecutionModel.execute` once per bar while a parent order is
working; anything not filled carries over to the next bar. Returned
fill prices are **all-in** (fees and spread folded in), so the engine's
cash accounting is simply price x quantity.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional

from quantfinlib.data.bar_series import BarSeries
from quantfinlib.microstructure.execution import Execution, Side


class ExecutionModel(ABC):
    """How parent orders turn into fills; see the module docstring."""

    def on_parent_order(self, side: Side, total_quantity: int,
                        signal_index: int) -> None:
        """Notification that a new parent order has been created (entry
        or exit). Stateful models (e.g. :class:`IcebergExecution`) reset
        per-parent state here."""

    @abstractmethod
    def execute(self, side: Side, requested_qty: int, series: BarSeries,
                index: int) -> List[Execution]:
        """Executes up to ``requested_qty`` on this bar. Must never fill
        more than requested; may fill less (or nothing) — the remainder
        is retried on subsequent bars."""

    def reference_price(self, series: BarSeries, index: int) -> float:
        """The price this model's fills are anchored to on the given bar
        — the engine budgets entry requests as
        ``cash / (reference_price * (1 + worst_case_cost_fraction()))``.
        Default: the bar close. A model that fills off a different price
        point (e.g. :class:`LastLookExecution` fills at the OPEN) must
        override this, or a gap between close and its actual anchor lets
        a fully-filled request overdraw cash."""
        return series.close(index)

    def worst_case_cost_fraction(self) -> float:
        """Upper bound on this model's all-in cost as a fraction of
        :meth:`reference_price` (spread + fees + slippage). The engine
        uses it to size entries so that a fully-filled parent can never
        overdraw cash — a model whose fills can cost more than
        ``reference_price * (1 + worst_case_cost_fraction())`` MUST
        override one or both methods, or the backtest silently trades on
        margin it doesn't have. Wrappers must DELEGATE both to the model
        that actually prices the fills. Default 1%."""
        return 0.01


class InstantExecution(ExecutionModel):
    """Baseline execution model: the full quantity fills at the bar
    close with commission and slippage folded into the all-in price —
    equivalent to the classic :class:`~quantfinlib.backtest.backtester.
    Backtester` fill assumption."""

    __slots__ = ("_cost_rate",)

    def __init__(self, commission_rate: float, slippage_rate: float) -> None:
        self._cost_rate = commission_rate + slippage_rate

    @staticmethod
    def from_config(config) -> "InstantExecution":
        return InstantExecution(config.commission_rate, config.slippage_rate)

    def worst_case_cost_fraction(self) -> float:
        # The all-in fill is exactly close * (1 + cost_rate); declare it
        # so entry sizing can spend cash to the last share without
        # overdrawing.
        return self._cost_rate

    def execute(self, side: Side, requested_qty: int, series: BarSeries,
                index: int) -> List[Execution]:
        if requested_qty <= 0:
            return []
        all_in = series.close(index) * (1 + side.sign() * self._cost_rate)
        return [Execution(series.symbol(), side, all_in, requested_qty,
                          series.timestamp(index), "PRIMARY")]


class IcebergOrder:
    """Iceberg order state machine (port of Java
    ``execution.IcebergOrder``): shows only a small display tranche of
    the full quantity and reloads automatically when the visible portion
    fills. Display sizes can be randomized to make the iceberg harder to
    detect."""

    __slots__ = ("_display_qty", "_randomize_pct", "_rnd", "_remaining",
                 "_visible")

    def __init__(self, total_qty: int, display_qty: int,
                 randomize_pct: float = 0.0, seed: int = 0) -> None:
        """``randomize_pct``: display-size jitter, e.g. 0.2 = +/-20%
        (0 = fixed)."""
        if total_qty <= 0 or display_qty <= 0:
            raise ValueError("quantities must be positive")
        self._display_qty = display_qty
        self._randomize_pct = randomize_pct
        self._rnd = random.Random(seed)
        self._remaining = total_qty   # total unexecuted (visible + hidden)
        self._visible = self._next_tranche()

    def _next_tranche(self) -> int:
        base = self._display_qty
        if self._randomize_pct > 0:
            # Java Math.round: half-up (floor(x + 0.5)), not Python's
            # banker's-rounding round().
            base = max(1, math.floor(
                self._display_qty
                * (1 + self._randomize_pct * (2 * self._rnd.random() - 1))
                + 0.5))
        return min(base, self._remaining)

    def on_fill(self, qty: int) -> bool:
        """Records a fill against the visible tranche. Returns True when
        the tranche was exhausted and a fresh one was loaded (i.e. the
        working order should be re-submitted at the back of the queue).
        """
        if qty <= 0 or qty > self._visible:
            raise ValueError(f"fill {qty} exceeds visible {self._visible}")
        self._visible -= qty
        self._remaining -= qty
        if self._visible == 0 and self._remaining > 0:
            self._visible = self._next_tranche()
            return True
        return False

    def visible_qty(self) -> int:
        return self._visible

    def hidden_qty(self) -> int:
        return self._remaining - self._visible

    def remaining_qty(self) -> int:
        return self._remaining

    def is_complete(self) -> bool:
        return self._remaining == 0


class IcebergExecution(ExecutionModel):
    """Iceberg execution: wraps another :class:`ExecutionModel` and caps
    each bar's execution at the :class:`IcebergOrder` state machine's
    visible tranche (optionally randomized), plus an optional
    participation cap versus the bar's volume. A fresh iceberg is loaded
    for every parent order, so entries and exits are both worked
    patiently across bars."""

    __slots__ = ("_inner", "_display_qty", "_randomize_pct",
                 "_max_participation", "_seed", "_iceberg", "_parent_seq")

    def __init__(self, inner: ExecutionModel, display_qty: int,
                 randomize_pct: float = 0.0, max_participation: float = 0.0,
                 seed: int = 0) -> None:
        """``randomize_pct``: display-size jitter, e.g. 0.2 = +/-20%;
        ``max_participation``: cap per bar as a fraction of bar volume
        (0 = off)."""
        if display_qty <= 0:
            raise ValueError("display_qty must be positive")
        self._inner = inner
        self._display_qty = display_qty
        self._randomize_pct = randomize_pct
        self._max_participation = max_participation
        self._seed = seed
        self._iceberg: Optional[IcebergOrder] = None
        self._parent_seq = 0

    def worst_case_cost_fraction(self) -> float:
        # The INNER model prices the fills; a wrapper that reports the
        # 1% default while wrapping a costlier model re-opens the cash
        # overdraw the engine's sizing exists to prevent.
        return self._inner.worst_case_cost_fraction()

    def reference_price(self, series: BarSeries, index: int) -> float:
        return self._inner.reference_price(series, index)

    def on_parent_order(self, side: Side, total_quantity: int,
                        signal_index: int) -> None:
        self._iceberg = IcebergOrder(
            total_quantity, self._display_qty, self._randomize_pct,
            self._seed + self._parent_seq * 1_000_003)
        self._parent_seq += 1
        self._inner.on_parent_order(side, total_quantity, signal_index)

    def execute(self, side: Side, requested_qty: int, series: BarSeries,
                index: int) -> List[Execution]:
        iceberg = self._iceberg
        if iceberg is None or iceberg.is_complete() or requested_qty <= 0:
            return []
        cap = min(requested_qty, iceberg.visible_qty())
        if self._max_participation > 0:
            cap = min(cap, int(series.volume(index)
                               * self._max_participation))
        if cap <= 0:
            return []
        fills = self._inner.execute(side, cap, series, index)
        filled = sum(f.quantity for f in fills)
        if filled > 0:
            iceberg.on_fill(filled)
        return fills


class LastLookExecution(ExecutionModel):
    """Last-look execution model — the missing realism for FX backtests.

    On ECN and single-dealer FX liquidity, the provider holds your order
    briefly and may *reject* it if the price moves against them during
    the hold. A backtest that fills every FX order unconditionally is
    fiction; rejects cluster exactly on the flow that was about to be
    profitable.

    Bar-level model of the hold window: the order arrives at the bar
    open and the LP watches the intra-bar move.

    **Signal-bar handling**: when worked through the execution-aware
    backtester, the parent is created at the signal bar's CLOSE — so on
    that first bar there is no hold window left to observe, and filling
    at that bar's *open* would credit a price from before the signal
    existed (intrabar time travel). The model therefore HOLDS on the
    parent's signal bar (no fill, no reject counted — it is pure
    latency, the LP has seen nothing yet); the first real attempt is the
    next bar, whose open is the price standing when the order actually
    arrived. Direct calls without :meth:`on_parent_order` keep the plain
    arrives-at-the-open semantics.

    * Move in the taker's favor beyond ``reject_threshold_bps`` (price
      rising on a buy — adverse to the LP who would sell) -> **reject**;
      the parent quantity carries to the next bar, exactly like real
      requote-and-chase.
    * Otherwise -> full fill at the open plus the taker pays
      ``spread_bps`` half-spread (all-in price).

    The asymmetry is deliberate *as a taker's worst-case model*: it
    simulates the adverse LP behavior the FX Global Code prohibits but a
    taker must still budget for. Rejection statistics are exposed for
    TCA — a live desk watches its reject rate per LP for exactly this
    pattern.
    """

    __slots__ = ("_spread_bps", "_reject_threshold_bps", "_fills",
                 "_rejects", "_parent_signal_bar")

    def __init__(self, spread_bps: float,
                 reject_threshold_bps: float) -> None:
        """``spread_bps``: half-spread paid on accepted fills;
        ``reject_threshold_bps``: intra-bar move (bps, in the taker's
        favor) beyond which the LP rejects."""
        if spread_bps < 0 or reject_threshold_bps <= 0:
            raise ValueError(
                "spread_bps must be >= 0 and reject_threshold_bps > 0")
        self._spread_bps = spread_bps
        self._reject_threshold_bps = reject_threshold_bps
        self._fills = 0
        self._rejects = 0
        self._parent_signal_bar = -1

    def on_parent_order(self, side: Side, total_quantity: int,
                        signal_index: int) -> None:
        self._parent_signal_bar = signal_index

    def reference_price(self, series: BarSeries, index: int) -> float:
        # Fills anchor to the bar OPEN, not the close: sizing against
        # the close would overdraw cash on any accepted bar that gaps
        # open-high (the engine budgets
        # request * reference_price * (1 + spread)).
        return series.open(index)

    def worst_case_cost_fraction(self) -> float:
        # Exact: the all-in fill is open * (1 + spread), and
        # reference_price hands the engine that same open.
        return self._spread_bps / 1e4

    def execute(self, side: Side, requested_qty: int, series: BarSeries,
                index: int) -> List[Execution]:
        if requested_qty <= 0:
            return []
        if index == self._parent_signal_bar:
            # Order decided at this bar's close: no hold window has
            # elapsed and this bar's open predates the signal. Hold —
            # not a reject.
            return []
        open_ = series.open(index)
        close = series.close(index)
        # Signed move in the TAKER's favor: up for a buy, down for a sell.
        favorable_move_bps = side.sign() * (close - open_) / open_ * 1e4
        if favorable_move_bps > self._reject_threshold_bps:
            # The LP saw the market run away and pulled the quote. The
            # engine retries the remainder on later bars — chasing the
            # move.
            self._rejects += 1
            return []
        self._fills += 1
        all_in = open_ * (1 + side.sign() * self._spread_bps / 1e4)
        return [Execution(series.symbol(), side, all_in, requested_qty,
                          series.timestamp(index), "LASTLOOK")]

    def fill_count(self) -> int:
        """Accepted fills (parent-bar attempts, not shares)."""
        return self._fills

    def reject_count(self) -> int:
        """Last-look rejections — the number a real desk tracks per LP."""
        return self._rejects

    def reject_rate(self) -> float:
        """Reject rate across all attempts, 0 when nothing was attempted."""
        attempts = self._fills + self._rejects
        return 0.0 if attempts == 0 else self._rejects / attempts
