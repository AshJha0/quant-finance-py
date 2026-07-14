"""Quote-driven paper trading venue (port of Java
``trading.PaperTradingGateway``).

Closes the research-to-production loop by running real strategy +
risk-gate code against simulated fills.

* Feed top-of-book quotes with :meth:`PaperTradingGateway.on_quote`;
  market orders fill at the touch, resting limit orders fill when the
  market crosses them.
* Every order passes an optional pre-trade limit checker first --
  rejected orders never reach the market, exactly like production.
* Full account tracking: signed positions with average cost, cash,
  realized/unrealized P&L, commission, and mark-to-market equity.

Port note (concurrency): the Java class is ``synchronized`` on every
method so a dashboard thread can safely read :meth:`snapshot` while a
trading thread fills orders. This port is a **single mutex-free Python
port** -- CPython's GIL already serializes attribute mutation within
one interpreter, and this library targets single-process
research/backtest use, not concurrent live trading across OS threads.
If you do drive this gateway from multiple Python threads, add your
own lock around the call sites; nothing here does it for you.

Port note (limit checker): the Java class gates orders through
``risk.PreTradeLimitChecker``, which has not been ported to this
Python library (out of scope for this port -- see the execution/
trading port notes). ``limit_checker`` therefore accepts any object
duck-typing that interface: a ``check(order_request, reference_mid,
current_position_qty, counterparty_exposure)`` method returning an
object with ``.approved`` (bool) and ``.violations`` (sequence of str).
:class:`OrderRequest` below is the minimal structural stand-in for the
request Java builds internally.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Protocol

from quantfinlib.microstructure.execution import Side

#: A soak with a misconfigured limit checker rejects every order, and
#: an unbounded log would grow the heap for the whole run. The first
#: entries carry the diagnosis; the count keeps the total.
REJECTION_LOG_CAP = 1_000


class OrderStatus(Enum):
    """Lifecycle state of a gateway order."""

    NEW = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELED = auto()
    REJECTED = auto()


#: All-primitive fill callback: ``(order_id, symbol, side, price,
#: quantity, timestamp_nanos) -> None``.
ExecutionListener = Callable[[int, str, Side, float, int, int], None]


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """Structural stand-in for ``risk.PreTradeLimitChecker.OrderRequest``
    (see the module docstring) -- the request handed to an optional
    ``limit_checker``."""

    symbol: str
    side: Side
    quantity: int
    price: float
    counterparty: str


class _LimitChecker(Protocol):
    def check(self, order: OrderRequest, reference_mid: float,
             current_position_qty: int, counterparty_exposure: float):
        ...


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """One internally consistent view of the whole account."""

    cash: float
    equity: float
    realized_pnl: float
    rejection_count: int
    positions: Dict[str, float]


class _WorkingOrder:
    __slots__ = ("id", "symbol", "side", "limit_price", "quantity", "status")

    def __init__(self, order_id: int, symbol: str, side: Side, quantity: int,
                limit_price: float) -> None:
        self.id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.limit_price = limit_price      # NaN = market
        self.status = OrderStatus.NEW


class _Position:
    __slots__ = ("quantity", "avg_cost")

    def __init__(self) -> None:
        self.quantity = 0.0
        self.avg_cost = 0.0


class _Quote:
    __slots__ = ("bid", "ask")

    def __init__(self, bid: float, ask: float) -> None:
        self.bid = bid
        self.ask = ask

    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class PaperTradingGateway:
    """Quote-driven paper trading venue; see the module docstring."""

    def __init__(self, initial_cash: float, commission_rate: float = 0.0,
                limit_checker: Optional[_LimitChecker] = None) -> None:
        self._limit_checker = limit_checker
        self._commission_rate = commission_rate
        self._quotes: Dict[str, _Quote] = {}
        self._orders: Dict[int, _WorkingOrder] = {}
        self._resting: List[_WorkingOrder] = []
        self._positions: Dict[str, _Position] = {}
        self._listeners: List[ExecutionListener] = []
        self._rejection_log: List[str] = []
        self._rejection_count = 0
        self._cash = initial_cash
        self._realized_pnl = 0.0
        self._next_id = 1

    # ------------------------------------------------------------------
    # Market data in
    # ------------------------------------------------------------------

    def on_quote(self, symbol: str, bid: float, ask: float) -> None:
        """Updates the top of book and fills any resting limit orders
        that now cross."""
        self._quotes[symbol] = _Quote(bid, ask)
        still_resting = []
        for order in self._resting:
            if order.symbol != symbol:
                still_resting.append(order)
                continue
            if order.side == Side.BUY and ask <= order.limit_price:
                self._fill(order, min(order.limit_price, ask))
            elif order.side == Side.SELL and bid >= order.limit_price:
                self._fill(order, max(order.limit_price, bid))
            else:
                still_resting.append(order)
        self._resting = still_resting

    # ------------------------------------------------------------------
    # OrderGateway
    # ------------------------------------------------------------------

    def submit_limit(self, symbol: str, side: Side, quantity: int, price: float) -> int:
        order = _WorkingOrder(self._next_id, symbol, side, quantity, price)
        self._next_id += 1
        self._orders[order.id] = order
        if not self._passes_risk_gate(order, price):
            return order.id
        q = self._quotes.get(symbol)
        if q is not None and (q.ask <= price if side == Side.BUY else q.bid >= price):
            self._fill(order, q.ask if side == Side.BUY else q.bid)   # marketable: fill at touch
        else:
            self._resting.append(order)
        return order.id

    def submit_market(self, symbol: str, side: Side, quantity: int) -> int:
        order = _WorkingOrder(self._next_id, symbol, side, quantity, math.nan)
        self._next_id += 1
        self._orders[order.id] = order
        q = self._quotes.get(symbol)
        if q is None:
            order.status = OrderStatus.REJECTED
            self._log_rejection(f"order {order.id}: NO_QUOTE for {symbol}")
            return order.id
        touch = q.ask if side == Side.BUY else q.bid
        if not self._passes_risk_gate(order, touch):
            return order.id
        self._fill(order, touch)
        return order.id

    def cancel(self, order_id: int) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status != OrderStatus.NEW:
            return False
        order.status = OrderStatus.CANCELED
        if order in self._resting:
            self._resting.remove(order)
        return True

    def status(self, order_id: int) -> OrderStatus:
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"unknown order {order_id}")
        return order.status

    def add_execution_listener(self, listener: ExecutionListener) -> None:
        self._listeners.append(listener)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def position(self, symbol: str) -> float:
        p = self._positions.get(symbol)
        return 0.0 if p is None else p.quantity

    def cash(self) -> float:
        return self._cash

    def realized_pnl(self) -> float:
        return self._realized_pnl

    def equity(self) -> float:
        """Mark-to-market equity at current mids."""
        value = self._cash
        for symbol, p in self._positions.items():
            q = self._quotes.get(symbol)
            if q is not None:
                value += p.quantity * q.mid()
        return value

    def rejection_log(self) -> List[str]:
        """The first :data:`REJECTION_LOG_CAP` rejection messages; the
        snapshot's ``rejection_count`` keeps counting past the cap."""
        return list(self._rejection_log)

    def positions_snapshot(self) -> Dict[str, float]:
        """Snapshot of non-zero positions by symbol (for
        dashboards/monitoring)."""
        return {symbol: p.quantity for symbol, p in self._positions.items()
               if p.quantity != 0}

    def snapshot(self) -> AccountSnapshot:
        """One internally consistent view of the whole account."""
        return AccountSnapshot(self._cash, self.equity(), self._realized_pnl,
                               self._rejection_count, self.positions_snapshot())

    # ------------------------------------------------------------------

    def _passes_risk_gate(self, order: _WorkingOrder, reference_price: float) -> bool:
        if self._limit_checker is None:
            return True
        q = self._quotes.get(order.symbol)
        mid = math.nan if q is None else q.mid()
        request = OrderRequest(order.symbol, order.side, order.quantity,
                               reference_price, "PAPER")
        # Java Math.round: half-up (floor(x + 0.5)), not Python's
        # banker's-rounding round() -- position can be negative (short)
        # and can land exactly on a half share, where the two disagree.
        check = self._limit_checker.check(
            request, mid, math.floor(self.position(order.symbol) + 0.5), 0.0)
        if not check.approved:
            order.status = OrderStatus.REJECTED
            self._log_rejection(f"order {order.id}: {check.violations}")
            return False
        return True

    def _log_rejection(self, entry: str) -> None:
        self._rejection_count += 1
        if len(self._rejection_log) < REJECTION_LOG_CAP:
            self._rejection_log.append(entry)

    def _fill(self, order: _WorkingOrder, price: float) -> None:
        qty = order.quantity
        self._apply_fill(order.symbol, order.side, price, qty)
        order.quantity = 0
        order.status = OrderStatus.FILLED
        ts = time.perf_counter_ns()
        for listener in self._listeners:
            listener(order.id, order.symbol, order.side, price, qty, ts)

    def _apply_fill(self, symbol: str, side: Side, price: float, qty: int) -> None:
        """Signed average-cost accounting with realized P&L on position
        reduction."""
        signed = qty if side == Side.BUY else -qty
        notional = price * qty
        self._cash += -notional if side == Side.BUY else notional
        fee = notional * self._commission_rate
        self._cash -= fee

        p = self._positions.setdefault(symbol, _Position())
        if p.quantity == 0 or math.copysign(1, p.quantity) == math.copysign(1, signed):
            # Opening or adding: blend the average cost.
            new_qty = p.quantity + signed
            p.avg_cost = ((p.avg_cost * abs(p.quantity) + price * qty)
                         / abs(new_qty))
            p.quantity = new_qty
        else:
            closing = min(abs(p.quantity), qty)
            self._realized_pnl += (price - p.avg_cost) * closing * math.copysign(1, p.quantity)
            new_qty = p.quantity + signed
            if math.copysign(1, new_qty) != math.copysign(1, p.quantity) and new_qty != 0:
                p.avg_cost = price      # position flipped: remainder opened at this fill
            p.quantity = new_qty
            if new_qty == 0:
                p.avg_cost = 0.0
