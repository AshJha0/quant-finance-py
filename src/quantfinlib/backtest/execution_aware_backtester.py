"""Execution-aware backtesting engine (ports of Java
``backtest.ExecutionAwareBacktester`` and ``backtest.ExecutionAwareResult``).

Strategy signals create **parent orders** that are worked through an
:class:`~quantfinlib.backtest.execution_models.ExecutionModel` — sliced
by iceberg, held by last-look, or filled instantly. Fills can span
multiple bars, the position accumulates gradually, and every child fill
is recorded so execution cost (TCA) is measurable per parent order.

Semantics (long-only, single instrument, like the classic Backtester):

* BUY signal while flat -> entry parent sized to available cash; worked
  from the signal bar until filled or superseded.
* SELL signal -> cancels any unfilled entry remainder and works an exit
  parent for the whole position.
* Stop-loss / take-profit are evaluated intrabar against the
  *volume-weighted average entry price*; the triggered exit is worked
  through the execution model (a patient model exits slowly — that
  realism is the point).
* Any position left at the end of data is force-closed at the last
  close less the model's worst-case cost fraction — unconditional (it
  bypasses the model's fill logic) but not free.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from quantfinlib.backtest.backtest_config import BacktestConfig
from quantfinlib.backtest.backtester import BacktestResult
from quantfinlib.backtest.execution_models import ExecutionModel
from quantfinlib.backtest.parent_order import ParentOrder
from quantfinlib.backtest.signal import Signal
from quantfinlib.backtest.trade import Trade
from quantfinlib.backtest.trading_strategy import TradingStrategy
from quantfinlib.data.bar_series import BarSeries
from quantfinlib.microstructure.execution import Execution, Side
from quantfinlib.microstructure.transaction_cost_analyzer import (
    TransactionCostAnalyzer)


class _ParentState:
    """Mutable accumulator for one parent order."""

    __slots__ = ("side", "signal_index", "arrival_price", "reason",
                 "fills", "bars")

    def __init__(self, side: Side, signal_index: int, arrival_price: float,
                 reason: str) -> None:
        self.side = side
        self.signal_index = signal_index
        self.arrival_price = arrival_price
        self.reason = reason
        self.fills: List[Execution] = []
        self.bars: List[int] = []

    def to_record(self) -> ParentOrder:
        return ParentOrder(self.side, self.signal_index, self.arrival_price,
                           self.reason, tuple(self.fills), tuple(self.bars))


class ExecutionAwareResult:
    """Result of an execution-aware backtest: the standard
    :class:`BacktestResult` (equity curve, trades, performance metrics)
    plus the full parent-order / child-fill history, with one-call TCA
    per parent order."""

    __slots__ = ("_backtest", "_parent_orders", "_series")

    def __init__(self, backtest: BacktestResult,
                 parent_orders: List[ParentOrder],
                 series: BarSeries) -> None:
        self._backtest = backtest
        self._parent_orders = tuple(parent_orders)
        self._series = series

    def backtest(self) -> BacktestResult:
        return self._backtest

    def parent_orders(self) -> tuple:
        return self._parent_orders

    def all_fills(self) -> List[Execution]:
        """Every child fill of the run, in execution order."""
        out: List[Execution] = []
        for p in self._parent_orders:
            out.extend(p.fills)
        return out

    def tca(self, parent: ParentOrder) -> TransactionCostAnalyzer.TcaReport:
        """Transaction cost analysis for one parent order: arrival mid =
        close at the signal bar, market VWAP = volume-weighted close
        over the fill interval, per-fill mid = close of the fill bar.

        Raises:
            ValueError: if the parent has no fills.
        """
        n = len(parent.fills)
        if n == 0:
            raise ValueError("parent order has no fills")
        series = self._series
        mids = [series.close(idx) for idx in parent.fill_bar_indices]
        first = parent.fill_bar_indices[0]
        last = parent.fill_bar_indices[-1]
        pv = 0.0
        vol = 0.0
        for i in range(first, last + 1):
            pv += series.close(i) * series.volume(i)
            vol += series.volume(i)
        market_vwap = parent.arrival_price if vol == 0 else pv / vol
        return TransactionCostAnalyzer.analyze(
            parent.fills, parent.arrival_price, market_vwap, mids)


class ExecutionAwareBacktester:
    """Engine; see the module docstring. Use :meth:`run`."""

    def __init__(self, strategy: TradingStrategy, series: BarSeries,
                 config: BacktestConfig, model: ExecutionModel) -> None:
        self._strategy = strategy
        self._series = series
        self._config = config
        self._model = model
        self._stop_loss_pct = (strategy.stop_loss_pct()
                               if strategy.stop_loss_pct() > 0
                               else config.stop_loss_pct)
        self._take_profit_pct = (strategy.take_profit_pct()
                                 if strategy.take_profit_pct() > 0
                                 else config.take_profit_pct)
        self._trades: List[Trade] = []
        self._parents: List[_ParentState] = []
        self._cash = 0.0
        self._position = 0
        self._pending_entry = 0
        self._pending_exit = 0
        # All-in cash spent on the current round trip's entries.
        self._entry_cost = 0.0
        self._exit_proceeds = 0.0
        self._total_entry_qty = 0
        self._first_entry_bar = -1
        self._entry: Optional[_ParentState] = None
        self._exit: Optional[_ParentState] = None

    @staticmethod
    def run(strategy: TradingStrategy, series: BarSeries,
            config: BacktestConfig,
            model: ExecutionModel) -> ExecutionAwareResult:
        return ExecutionAwareBacktester(strategy, series, config,
                                        model)._execute()

    def _execute(self) -> ExecutionAwareResult:
        strategy = self._strategy
        series = self._series
        strategy.init(series)
        n = series.size()
        equity = np.zeros(n)
        self._cash = self._config.initial_capital

        for i in range(n):
            # 1. Keep working whatever parent order is open.
            if self._pending_exit > 0:
                self._work_exit(i)
            elif self._pending_entry > 0:
                self._work_entry(i)
            # 2. Intrabar risk exits on the accumulated position.
            self._check_risk_exits(i)
            # 3. Strategy signal at the close.
            self._on_signal(strategy.on_bar(i), i)

            equity[i] = self._cash + self._position * series.close(i)
        self._force_close(n - 1, equity)

        parent_records = [p.to_record() for p in self._parents]
        result = BacktestResult(strategy.name(), series.symbol(), equity,
                                self._trades, self._config.periods_per_year)
        return ExecutionAwareResult(result, parent_records, series)

    # ------------------------------------------------------------------

    def _work_entry(self, i: int) -> None:
        # Cap the request by what cash can actually pay for, priced at
        # the model's own fill anchor (close for most models, the OPEN
        # for last-look) with its declared worst-case all-in cost on top
        # — a flat close-based 1% buffer overdraws cash the moment a
        # model charges more, or fills off a gapped open.
        model = self._model
        series = self._series
        ref = model.reference_price(series, i)
        affordable = int(self._cash
                         / (ref * (1 + model.worst_case_cost_fraction())))
        request = min(self._pending_entry, affordable)
        if request <= 0:
            return
        for f in model.execute(Side.BUY, request, series, i):
            cost = f.notional()
            self._cash -= cost
            self._entry_cost += cost
            self._position += f.quantity
            self._total_entry_qty += f.quantity
            self._pending_entry -= f.quantity
            self._entry.fills.append(f)
            self._entry.bars.append(i)
            if self._first_entry_bar < 0:
                self._first_entry_bar = i

    def _work_exit(self, i: int) -> None:
        for f in self._model.execute(Side.SELL, self._pending_exit,
                                     self._series, i):
            proceeds = f.notional()
            self._cash += proceeds
            self._exit_proceeds += proceeds
            self._position -= f.quantity
            self._pending_exit -= f.quantity
            self._exit.fills.append(f)
            self._exit.bars.append(i)
        if self._position == 0 and self._pending_exit == 0:
            self._close_round_trip(i)

    def _check_risk_exits(self, i: int) -> None:
        if (self._position <= 0 or self._pending_exit > 0
                or self._first_entry_bar < 0 or i <= self._first_entry_bar):
            return
        series = self._series
        avg_entry = self._entry_cost / self._total_entry_qty
        reason = None
        if (self._stop_loss_pct > 0
                and series.low(i) <= avg_entry * (1 - self._stop_loss_pct)):
            reason = Trade.REASON_STOP_LOSS
        elif (self._take_profit_pct > 0
              and series.high(i) >= avg_entry * (1 + self._take_profit_pct)):
            reason = Trade.REASON_TAKE_PROFIT
        if reason is not None:
            self._start_exit(i, reason)
            self._work_exit(i)

    def _on_signal(self, signal: Signal, i: int) -> None:
        if (signal is Signal.BUY and self._position == 0
                and self._pending_entry == 0 and self._pending_exit == 0):
            close = self._series.close(i)
            target = int(self._cash
                         / (close
                            * (1 + self._model.worst_case_cost_fraction())))
            if target <= 0:
                return
            self._entry = _ParentState(Side.BUY, i, close,
                                       ParentOrder.REASON_ENTRY)
            self._parents.append(self._entry)
            self._entry_cost = 0.0
            self._exit_proceeds = 0.0
            self._total_entry_qty = 0
            self._first_entry_bar = -1
            self._pending_entry = target
            self._model.on_parent_order(Side.BUY, target, i)
            self._work_entry(i)
        elif (signal is Signal.SELL and self._pending_exit == 0
              and (self._position > 0 or self._pending_entry > 0)):
            self._pending_entry = 0   # cancel unfilled entry remainder
            if self._position > 0:
                self._start_exit(i, Trade.REASON_SIGNAL)
                self._work_exit(i)
            else:
                self._entry = None    # nothing filled; abandon the parent

    def _start_exit(self, i: int, reason: str) -> None:
        self._pending_entry = 0
        self._exit = _ParentState(Side.SELL, i, self._series.close(i),
                                  reason)
        self._parents.append(self._exit)
        self._pending_exit = self._position
        self._model.on_parent_order(Side.SELL, self._position, i)

    def _close_round_trip(self, i: int) -> None:
        series = self._series
        avg_entry = self._entry_cost / self._total_entry_qty
        avg_exit = self._exit_proceeds / self._total_entry_qty
        pnl = self._exit_proceeds - self._entry_cost
        self._trades.append(Trade(
            series.symbol(), self._first_entry_bar, i,
            series.timestamp(self._first_entry_bar), series.timestamp(i),
            avg_entry, avg_exit, self._total_entry_qty, pnl,
            0.0 if self._entry_cost == 0 else pnl / self._entry_cost,
            self._exit.reason))
        self._entry = None
        self._exit = None
        self._first_entry_bar = -1
        self._entry_cost = 0.0
        self._exit_proceeds = 0.0
        self._total_entry_qty = 0

    def _force_close(self, last: int, equity: np.ndarray) -> None:
        self._pending_entry = 0
        if self._position <= 0:
            return
        series = self._series
        if self._exit is None:
            self._exit = _ParentState(Side.SELL, last, series.close(last),
                                      Trade.REASON_END_OF_DATA)
            self._parents.append(self._exit)
        # Even a forced close pays to trade: charge the model's
        # worst-case cost rather than exiting for free — a run that ends
        # holding must not get its last round trip's exit cost waived.
        px = (series.close(last)
              * (1 - self._model.worst_case_cost_fraction()))
        fill = Execution(series.symbol(), Side.SELL, px, self._position,
                         series.timestamp(last), "FORCED_CLOSE")
        self._exit.fills.append(fill)
        self._exit.bars.append(last)
        self._cash += px * self._position
        self._exit_proceeds += px * self._position
        self._position = 0
        self._pending_exit = 0
        self._close_round_trip(last)
        equity[last] = self._cash
