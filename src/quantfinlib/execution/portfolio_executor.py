"""True multi-symbol portfolio-level scheduling (port of Java
``execution.PortfolioExecutor``).

A basket (rebalance, transition, program trade) executed as one
coordinated schedule rather than N independent parents. Each symbol
keeps its own
:class:`~quantfinlib.execution.benchmark_executor.BenchmarkExecutor`
child -- its benchmark, curve and per-symbol shaping stay intact -- and
the portfolio layer applies the two overlays that only exist at basket
level:

1. **Leg balance** -- the defining constraint of a two-sided
   transition: the buy leg and the sell leg must stay in step, or the
   basket carries unintended net market exposure mid-flight. When the
   projected net filled notional (buys - sells, plus this interval's
   dues) would breach ``max_net_notional``, the interval throttles the
   leg that is ahead. It never accelerates the lagging leg -- pushing a
   child past its own schedule would break the benchmark it is
   measured against.
2. **Capacity allocation** -- ``max_interval_notional`` caps the
   basket's total demand per interval (participation budget, cash
   constraint). When it binds, capacity goes to the symbols carrying
   the most residual risk. By default that is the diagonal
   approximation of multi-asset Almgren-Chriss -- weight ~ (1 +
   volatility regime) * due notional. Plug in a streaming
   :class:`~quantfinlib.microstructure.ewma_covariance.EwmaCovariance`
   via :meth:`PortfolioExecutor.use_risk_model` and it becomes the real
   thing: weight ~ (1 + marginal contribution to BASKET variance) * due
   notional, so two correlated legs are recognized as one concentrated
   risk and a natural hedge earns no urgency.

Both overlays only ever REDUCE a child's own due quantity, so
per-symbol benchmark integrity holds by construction, and anything
deferred reappears through each child's behind-schedule catch-up next
interval. A binding cap can therefore leave a residual at the horizon
-- that is the constraint's honest meaning, not a bug.

Usage: :meth:`PortfolioExecutor.add` each child once (buys and sells
mixed freely), then each interval call :meth:`PortfolioExecutor.decide`
with per-symbol
:class:`~quantfinlib.execution.benchmark_executor.MarketState`
snapshots and route the returned dues; report fills via
:meth:`PortfolioExecutor.on_fill` (which also maintains the net
ledger). Notional arithmetic needs a price: the layer remembers the
last finite mid per symbol (and fill prices); a symbol that has never
shown a price passes through unscaled -- the caps cannot see what they
cannot price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from quantfinlib.execution.benchmark_executor import BenchmarkExecutor, MarketState
from quantfinlib.microstructure.execution import Side
from quantfinlib.microstructure.ewma_covariance import EwmaCovariance
from quantfinlib.util import math_utils


@dataclass(frozen=True, slots=True)
class Config:
    """
    Attributes:
        max_net_notional: leg-balance band: |filled buys - filled
            sells| (projected through this interval) stays within
            this; +inf disables.
        max_interval_notional: total basket demand per interval; +inf
            disables.
    """

    max_net_notional: float
    max_interval_notional: float

    def __post_init__(self) -> None:
        if not (self.max_net_notional > 0) or not (self.max_interval_notional > 0):
            raise ValueError(
                "need maxNetNotional > 0 and maxIntervalNotional > 0 "
                "(+Inf disables)")

    @staticmethod
    def unconstrained() -> "Config":
        """No portfolio constraints: children pass through untouched."""
        return Config(math.inf, math.inf)


class PortfolioExecutor:
    """Multi-symbol basket scheduler; see the module docstring."""

    def __init__(self, max_symbols: int, config: Config) -> None:
        if max_symbols < 1:
            raise ValueError("need maxSymbols >= 1")
        self._config = config
        self._children: List[Optional[BenchmarkExecutor]] = [None] * max_symbols
        self._last_mid = np.zeros(max_symbols)
        self._due_notional = np.zeros(max_symbols)
        self._signed_remaining = np.zeros(max_symbols)
        self._risk_factor = np.zeros(max_symbols)
        self._count = 0
        self._risk_model: Optional[EwmaCovariance] = None
        self._buy_filled_notional = 0.0
        self._sell_filled_notional = 0.0

    def use_risk_model(self, model: EwmaCovariance) -> None:
        """Upgrades the capacity allocation from the diagonal
        approximation to true basket risk: with a covariance model, a
        binding ``max_interval_notional`` flows to the symbols whose
        REMAINING position contributes most to portfolio variance --
        two correlated buys carry more joint timing risk than their
        individual vols admit, and a natural hedge carries less. Handle
        ``i`` maps to covariance symbol ``i``; feed the model one
        return vector per interval on your own clock. Without a model
        (or before it has learned), the weight falls back to the
        per-symbol volatility regime."""
        if model.symbols() != len(self._children):
            raise ValueError(
                f"risk model covers {model.symbols()} symbols; this "
                f"portfolio was sized for {len(self._children)}")
        self._risk_model = model

    def add(self, child: BenchmarkExecutor) -> int:
        """Registers a child parent order; returns its handle for
        decide/on_fill."""
        if self._count == len(self._children):
            raise RuntimeError(f"portfolio is full ({len(self._children)} symbols)")
        self._children[self._count] = child
        handle = self._count
        self._count += 1
        return handle

    # ------------------------------------------------------------------
    # The interval decision
    # ------------------------------------------------------------------

    def decide(self, schedule_fraction: float, states: Sequence[MarketState],
              due_out) -> None:
        """One portfolio interval: asks every child for its own due
        quantity, then applies the leg-balance band and the capacity
        allocation. ``due_out[handle]`` receives the shares to send per
        symbol.

        Args:
            schedule_fraction: elapsed fraction of the execution horizon.
            states: per-handle market snapshots (index = handle).
            due_out: per-handle output, length >= :meth:`size`.
        """
        if len(states) < self._count or len(due_out) < self._count:
            raise ValueError(
                f"states/dueOut must cover all {self._count} symbols")
        # 1. Each child's own decision, and its notional at the
        #    best-known price.
        for i in range(self._count):
            due_out[i] = self._children[i].due_quantity(schedule_fraction, states[i])
            mid = states[i].mid
            if 0 < mid < math.inf:
                self._last_mid[i] = mid            # NaN fails the > 0 test

        # 2. Leg balance: throttle the leg that would push |net| past the band.
        self._apply_leg_band(due_out)

        # 3. Capacity: when total demand exceeds the interval budget,
        #    allocate it risk-weighted and cut each symbol to its share.
        if self._config.max_interval_notional < math.inf:
            self._fill_risk_factors(states)
            total = 0.0
            sum_weight = 0.0
            for i in range(self._count):
                notional = due_out[i] * self._last_mid[i]
                self._due_notional[i] = notional
                total += notional
                sum_weight += self._risk_factor[i] * notional
            if total > self._config.max_interval_notional and sum_weight > 0:
                for i in range(self._count):
                    if self._due_notional[i] <= 0:
                        continue
                    weight = self._risk_factor[i] * self._due_notional[i]
                    allocation = self._config.max_interval_notional * weight / sum_weight
                    if self._due_notional[i] > allocation:
                        due_out[i] = math.floor(
                            due_out[i] * allocation / self._due_notional[i])
                # 4. The risk weights cut the two legs asymmetrically, so
                #    the capacity pass can push |net| back over the band
                #    the first pass enforced. Re-apply it -- the band
                #    only reduces dues, so it can never re-violate the
                #    budget, and the sequence terminates here by
                #    construction.
                self._apply_leg_band(due_out)

    def _fill_risk_factors(self, states: Sequence[MarketState]) -> None:
        """The capacity weight multiplier per symbol. With a
        covariance model that has a live risk picture: ``1 +
        clamp(MRC, 0, 1)`` where MRC is the remaining position's
        marginal contribution to basket variance -- a natural hedge
        (negative MRC) earns no extra capacity, because executing it
        INCREASES the risk left behind. Otherwise the diagonal fallback
        ``1 + volatility_regime``. Same bounded [1, 2] shape either
        way, so the two modes are interchangeable mid-flight."""
        modeled = False
        if self._risk_model is not None:
            for i in range(self._count):
                sign = 1 if self._children[i].side() == Side.BUY else -1
                self._signed_remaining[i] = (
                    sign * self._children[i].remaining() * self._last_mid[i])
            if self._risk_model.marginal_contribution(
                    self._signed_remaining, self._risk_factor) > 0:
                for i in range(self._count):
                    self._risk_factor[i] = 1 + math_utils.clamp(self._risk_factor[i], 0, 1)
                modeled = True
        if not modeled:
            for i in range(self._count):
                vol = states[i].volatility
                self._risk_factor[i] = 1 + (vol if vol > 0 else 0)   # NaN -> 0

    def _apply_leg_band(self, due_out) -> None:
        """The leg-balance band: if this interval's dues would carry
        the net filled notional past ``max_net_notional``, scale down
        the leg that is pushing it over. Reads the CURRENT dues, so it
        is safe to apply again after another overlay has changed
        them."""
        if self._config.max_net_notional == math.inf:
            return
        buy_due = 0.0
        sell_due = 0.0
        for i in range(self._count):
            notional = due_out[i] * self._last_mid[i]
            if self._children[i].side() == Side.BUY:
                buy_due += notional
            else:
                sell_due += notional
        net = self._buy_filled_notional - self._sell_filled_notional
        projected = net + buy_due - sell_due
        if projected > self._config.max_net_notional and buy_due > 0:
            allowed = max(0.0, self._config.max_net_notional - net + sell_due)
            self._scale_side(Side.BUY, allowed / buy_due, due_out)
        elif projected < -self._config.max_net_notional and sell_due > 0:
            allowed = max(0.0, self._config.max_net_notional + net + buy_due)
            self._scale_side(Side.SELL, allowed / sell_due, due_out)

    def _scale_side(self, side: Side, scale: float, due_out) -> None:
        """Scales one side's dues by ``scale`` in [0,1] (floor
        rounding). Symbols without a known price are skipped -- they
        contributed nothing to the notional being reduced, so cutting
        them would shrink flow without moving the ledger."""
        s = math_utils.clamp(scale, 0, 1)
        for i in range(self._count):
            if (self._children[i].side() == side and due_out[i] > 0
                    and self._last_mid[i] > 0):
                due_out[i] = math.floor(due_out[i] * s)

    # ------------------------------------------------------------------
    # Fills and progress
    # ------------------------------------------------------------------

    def on_fill(self, handle: int, qty: int, price: float) -> None:
        """A fill for one child: forwards to its executor and
        maintains the net ledger. A non-positive or non-finite price
        still advances the child's schedule but cannot enter the
        notional ledger.

        Report every fill through THIS method, never through
        ``child(h).on_fill(...)`` -- the child call advances that
        symbol's schedule but silently bypasses the buy/sell notional
        ledger the leg-balance band reads, leaving the basket's net
        exposure uncontrolled while every per-child number looks
        healthy.
        """
        self._children[handle].on_fill(qty)
        if qty > 0 and 0 < price < math.inf:
            self._last_mid[handle] = price
            notional = qty * price
            if self._children[handle].side() == Side.BUY:
                self._buy_filled_notional += notional
            else:
                self._sell_filled_notional += notional

    def net_notional(self) -> float:
        """Signed net filled notional: buys - sells. The leg-balance
        ledger."""
        return self._buy_filled_notional - self._sell_filled_notional

    def done(self) -> bool:
        return all(self._children[i].done() for i in range(self._count))

    def size(self) -> int:
        return self._count

    def child(self, handle: int) -> BenchmarkExecutor:
        """The child executor behind a handle -- for progress/drift
        reads and for feeding ``on_market_volume`` to VWAP/POV
        children. Do NOT report fills via ``child(h).on_fill(...)``:
        fills must go through :meth:`on_fill` so the leg-balance ledger
        sees them."""
        return self._children[handle]
