"""Quote-driven paper trading gateway: fills, average costing,
position flips, risk-gate rejection, commission. Ported from Java
PaperTradingTest.

Java gates orders through ``risk.PreTradeLimitChecker``, which has not
been ported to this Python library (see
:mod:`quantfinlib.trading.paper_trading_gateway`'s module docstring).
``_SimpleLimitChecker`` below is a minimal duck-typed stand-in used
only to exercise :class:`PaperTradingGateway`'s risk-gate wiring.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Set

import pytest

from quantfinlib.microstructure.execution import Side
from quantfinlib.trading.paper_trading_gateway import (OrderStatus,
                                                        PaperTradingGateway)


@dataclass
class _CheckResult:
    approved: bool
    violations: List[str] = field(default_factory=list)


class _SimpleLimitChecker:
    """Minimal stand-in for ``risk.PreTradeLimitChecker`` covering just
    the two checks Java's ``PaperTradingTest.riskGateRejectsBeforeTheMarket``
    exercises."""

    def __init__(self, max_order_quantity: Optional[int] = None,
                restricted: Optional[Set[str]] = None) -> None:
        self._max_order_quantity = max_order_quantity
        self._restricted = restricted or set()

    def check(self, order, reference_mid, current_position_qty, counterparty_exposure):
        violations = []
        if order.symbol in self._restricted:
            violations.append(f"RESTRICTED_SYMBOL: {order.symbol}")
        if (self._max_order_quantity is not None
                and order.quantity > self._max_order_quantity):
            violations.append(
                f"MAX_ORDER_QTY: {order.quantity} > {self._max_order_quantity}")
        return _CheckResult(len(violations) == 0, violations)


def test_market_orders_fill_at_the_touch():
    gw = PaperTradingGateway(100_000)
    gw.on_quote("EURUSD", 1.0848, 1.0852)

    fills = []

    def on_fill(order_id, symbol, side, price, qty, ts):
        fills.append((order_id, symbol, side, price, qty))
        assert price == pytest.approx(1.0852, abs=1e-12)   # buy pays the ask

    gw.add_execution_listener(on_fill)
    order_id = gw.submit_market("EURUSD", Side.BUY, 10_000)

    assert gw.status(order_id) == OrderStatus.FILLED
    assert len(fills) == 1
    assert gw.position("EURUSD") == pytest.approx(10_000, abs=1e-9)
    assert gw.cash() == pytest.approx(100_000 - 10_000 * 1.0852, abs=1e-6)
    assert gw.equity() == pytest.approx(gw.cash() + 10_000 * 1.0850, abs=1e-6)


def test_limit_orders_rest_and_fill_when_market_crosses():
    gw = PaperTradingGateway(100_000)
    gw.on_quote("AAPL", 99.98, 100.02)

    order_id = gw.submit_limit("AAPL", Side.BUY, 100, 99.50)   # passive
    assert gw.status(order_id) == OrderStatus.NEW
    assert gw.position("AAPL") == pytest.approx(0, abs=1e-9)

    gw.on_quote("AAPL", 99.30, 99.40)                           # market drops through
    assert gw.status(order_id) == OrderStatus.FILLED
    assert gw.position("AAPL") == pytest.approx(100, abs=1e-9)

    order_id2 = gw.submit_limit("AAPL", Side.BUY, 50, 101)      # marketable
    assert gw.status(order_id2) == OrderStatus.FILLED


def test_cancel_only_works_before_fill():
    gw = PaperTradingGateway(100_000)
    gw.on_quote("X", 99, 101)
    resting = gw.submit_limit("X", Side.BUY, 10, 95)
    assert gw.cancel(resting)
    assert gw.status(resting) == OrderStatus.CANCELED
    assert not gw.cancel(resting)

    filled = gw.submit_market("X", Side.BUY, 10)
    assert not gw.cancel(filled)
    gw.on_quote("X", 90, 91)
    assert gw.position("X") == pytest.approx(10, abs=1e-9)


def test_round_trip_realizes_pnl_with_average_costing():
    gw = PaperTradingGateway(100_000)
    gw.on_quote("X", 99.99, 100.01)
    gw.submit_market("X", Side.BUY, 100)         # buy at 100.01
    gw.on_quote("X", 109.99, 110.01)
    gw.submit_market("X", Side.SELL, 100)        # sell at 109.99

    assert gw.position("X") == pytest.approx(0, abs=1e-9)
    assert gw.realized_pnl() == pytest.approx(100 * (109.99 - 100.01), abs=1e-6)
    assert gw.cash() == pytest.approx(100_000 + gw.realized_pnl(), abs=1e-6)


def test_shorts_and_position_flips_account_correctly():
    gw = PaperTradingGateway(100_000)
    gw.on_quote("X", 100, 100)                    # zero spread for clean numbers
    gw.submit_market("X", Side.SELL, 50)          # open short 50 @ 100
    assert gw.position("X") == pytest.approx(-50, abs=1e-9)

    gw.on_quote("X", 90, 90)
    gw.submit_market("X", Side.BUY, 80)           # close 50 (+10*50 pnl), flip long 30 @ 90
    assert gw.position("X") == pytest.approx(30, abs=1e-9)
    assert gw.realized_pnl() == pytest.approx(500, abs=1e-6)


def test_risk_gate_rejects_before_the_market():
    checker = _SimpleLimitChecker(max_order_quantity=1_000, restricted={"BANNED"})
    gw = PaperTradingGateway(100_000, 0, checker)
    gw.on_quote("OK", 99, 101)
    gw.on_quote("BANNED", 99, 101)

    too_big = gw.submit_market("OK", Side.BUY, 5_000)
    assert gw.status(too_big) == OrderStatus.REJECTED
    banned = gw.submit_market("BANNED", Side.BUY, 10)
    assert gw.status(banned) == OrderStatus.REJECTED
    fine = gw.submit_market("OK", Side.BUY, 500)
    assert gw.status(fine) == OrderStatus.FILLED

    assert len(gw.rejection_log()) == 2
    assert gw.position("BANNED") == pytest.approx(0, abs=1e-9)


class _RecordingLimitChecker:
    """Records the ``current_position_qty`` the gateway passed in, so
    the risk-gate rounding can be inspected directly."""

    def __init__(self) -> None:
        self.seen_position_qty: List[float] = []

    def check(self, order, reference_mid, current_position_qty, counterparty_exposure):
        self.seen_position_qty.append(current_position_qty)
        return _CheckResult(True, [])


def test_risk_gate_rounds_a_half_share_position_up_like_java():
    # Java Math.round is half-up (floor(x + 0.5)); Python's builtin
    # round() is banker's-rounding and would round 4.5 DOWN to 4 here
    # (4 is even). Build a position that lands exactly on a half
    # share, then check the NEXT order sees it rounded to 5, not 4.
    checker = _RecordingLimitChecker()
    gw = PaperTradingGateway(100_000, 0, checker)
    gw.on_quote("X", 99, 101)
    gw.submit_market("X", Side.BUY, 4.5)
    assert gw.position("X") == pytest.approx(4.5, abs=1e-9)
    assert round(4.5) == 4   # documents the Python trap this guards against

    gw.submit_market("X", Side.BUY, 1)
    # The SECOND call's risk check is the one that observed the 4.5
    # position from the first fill.
    assert checker.seen_position_qty[1] == 5


def test_market_order_without_quote_is_rejected():
    gw = PaperTradingGateway(100_000)
    order_id = gw.submit_market("UNKNOWN", Side.BUY, 10)
    assert gw.status(order_id) == OrderStatus.REJECTED


def test_commission_reduces_cash_only():
    gw = PaperTradingGateway(100_000, 0.001, None)
    gw.on_quote("X", 100, 100)
    gw.submit_market("X", Side.BUY, 100)
    assert gw.cash() == pytest.approx(100_000 - 10_000 - 10, abs=1e-6)   # notional + 10bps fee


def test_status_of_unknown_order_raises():
    gw = PaperTradingGateway(100_000)
    with pytest.raises(ValueError):
        gw.status(9999)


def test_rejection_log_is_capped_but_count_keeps_going():
    from quantfinlib.trading.paper_trading_gateway import REJECTION_LOG_CAP

    checker = _SimpleLimitChecker(max_order_quantity=0)
    gw = PaperTradingGateway(100_000, 0, checker)
    gw.on_quote("X", 99, 101)
    n = REJECTION_LOG_CAP + 5
    for _ in range(n):
        gw.submit_market("X", Side.BUY, 10)
    assert len(gw.rejection_log()) == REJECTION_LOG_CAP
    assert gw.snapshot().rejection_count == n
