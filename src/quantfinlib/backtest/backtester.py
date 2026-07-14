"""Event-driven backtesting engine (port of Java ``backtest.Backtester``
and ``backtest.BacktestResult``).

Execution model: signals fill at the bar close (adjusted for slippage);
stop-loss / take-profit levels are evaluated intrabar against the bar's
low/high on bars after entry, with gap-aware fills (a gap through the
level fills at the open). Commission is charged on both entry and exit
notional.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from quantfinlib.backtest.backtest_config import BacktestConfig
from quantfinlib.backtest.performance_analytics import (PerformanceAnalytics,
                                                        PerformanceMetrics)
from quantfinlib.backtest.signal import Signal
from quantfinlib.backtest.trade import Trade
from quantfinlib.backtest.trading_strategy import TradingStrategy
from quantfinlib.data.bar_series import BarSeries


class BacktestResult:
    """Result of a backtest run: full equity curve (one point per bar),
    completed trade history, and derived performance metrics."""

    __slots__ = ("_strategy_name", "_symbol", "_equity_curve", "_trades",
                 "_metrics")

    def __init__(self, strategy_name: str, symbol: str, equity_curve,
                 trades: Sequence[Trade], periods_per_year: int) -> None:
        self._strategy_name = strategy_name
        self._symbol = symbol
        self._equity_curve = np.asarray(equity_curve, dtype=float)
        self._equity_curve.setflags(write=False)
        self._trades = tuple(trades)
        self._metrics = PerformanceAnalytics.compute(
            self._equity_curve, self._trades, periods_per_year)

    def strategy_name(self) -> str:
        return self._strategy_name

    def symbol(self) -> str:
        return self._symbol

    def equity_curve(self) -> np.ndarray:
        return self._equity_curve

    def trades(self) -> tuple:
        return self._trades

    def metrics(self) -> PerformanceMetrics:
        return self._metrics

    def __str__(self) -> str:
        m = self._metrics
        return (f"{self._strategy_name} on {self._symbol}: "
                f"totalReturn={m.total_return * 100:.2f}%, "
                f"CAGR={m.cagr * 100:.2f}%, sharpe={m.sharpe_ratio:.2f}, "
                f"sortino={m.sortino_ratio:.2f}, calmar={m.calmar_ratio:.2f}, "
                f"maxDD={m.max_drawdown * 100:.2f}%, "
                f"profitFactor={m.profit_factor:.2f}, "
                f"winRate={m.win_rate * 100:.1f}%, trades={m.trade_count}")


class Backtester:
    """Event-driven, single-instrument, long-only backtesting engine."""

    @staticmethod
    def run(strategy: TradingStrategy, series: BarSeries,
            config: BacktestConfig, trade_from: int = 0) -> BacktestResult:
        """Runs the backtest, optionally with a WARM-UP prefix.

        Indicators are initialized over the whole series, but no signal
        is acted on (and no equity recorded) before ``trade_from``. This
        is how walk-forward analysis avoids the cold-start bias —
        evaluating a fold on a bare test slice re-computes every
        indicator from scratch, silently forcing HOLD through the first
        ``lookback`` bars of *every* fold; feeding the preceding bars as
        warm-up (they are the past — no look-ahead) lets the strategy
        enter the test window with warm indicators, the way it would
        trade live. The returned equity curve covers ``[trade_from, n)``
        only.

        Scope of the warm-up: it warms whatever
        :meth:`TradingStrategy.init` precomputes over the series (all
        shipped strategies). ``on_bar`` is NOT called for warm-up bars,
        so a strategy that accumulates state inside ``on_bar`` still
        starts cold at ``trade_from``.
        """
        n = series.size()
        if trade_from < 0 or trade_from >= n:
            raise ValueError(
                f"trade_from must be in [0, {n}), got {trade_from}")
        strategy.init(series)
        equity = np.zeros(n - trade_from)
        trades: List[Trade] = []

        stop_loss = (strategy.stop_loss_pct() if strategy.stop_loss_pct() > 0
                     else config.stop_loss_pct)
        take_profit = (strategy.take_profit_pct()
                       if strategy.take_profit_pct() > 0
                       else config.take_profit_pct)

        cash = config.initial_capital
        qty = 0.0
        entry_price = 0.0
        entry_cost = 0.0
        entry_index = -1

        for i in range(trade_from, n):
            # 1. Intrabar risk exits (only on bars after the entry bar).
            if qty > 0 and i > entry_index:
                if stop_loss > 0:
                    stop_price = entry_price * (1 - stop_loss)
                    if series.low(i) <= stop_price:
                        fill = min(stop_price, series.open(i))
                        cash = Backtester._close_position(
                            trades, series, config, qty, entry_price,
                            entry_cost, entry_index, i, fill,
                            Trade.REASON_STOP_LOSS, cash)
                        qty = 0.0
                if qty > 0 and take_profit > 0:
                    target_price = entry_price * (1 + take_profit)
                    if series.high(i) >= target_price:
                        fill = max(target_price, series.open(i))
                        cash = Backtester._close_position(
                            trades, series, config, qty, entry_price,
                            entry_cost, entry_index, i, fill,
                            Trade.REASON_TAKE_PROFIT, cash)
                        qty = 0.0

            # 2. Strategy signal at the close.
            sig = strategy.on_bar(i)
            close = series.close(i)
            if sig is Signal.BUY and qty == 0 and cash > 0:
                fill = close * (1 + config.slippage_rate)
                fee = cash * config.commission_rate
                qty = (cash - fee) / fill
                entry_price = fill
                entry_cost = cash
                entry_index = i
                cash = 0.0
            elif sig is Signal.SELL and qty > 0:
                fill = close * (1 - config.slippage_rate)
                cash = Backtester._close_position(
                    trades, series, config, qty, entry_price, entry_cost,
                    entry_index, i, fill, Trade.REASON_SIGNAL, cash)
                qty = 0.0

            equity[i - trade_from] = cash + qty * close

        # 3. Force-close any open position at the final bar.
        if qty > 0:
            fill = series.close(n - 1) * (1 - config.slippage_rate)
            cash = Backtester._close_position(
                trades, series, config, qty, entry_price, entry_cost,
                entry_index, n - 1, fill, Trade.REASON_END_OF_DATA, cash)
            equity[-1] = cash

        return BacktestResult(strategy.name(), series.symbol(), equity,
                              trades, config.periods_per_year)

    @staticmethod
    def _close_position(trades: List[Trade], series: BarSeries,
                        config: BacktestConfig, qty: float,
                        entry_price: float, entry_cost: float,
                        entry_index: int, exit_index: int, fill_price: float,
                        reason: str, cash: float) -> float:
        proceeds = qty * fill_price
        fee = proceeds * config.commission_rate
        net = proceeds - fee
        pnl = net - entry_cost
        trades.append(Trade(series.symbol(), entry_index, exit_index,
                            series.timestamp(entry_index),
                            series.timestamp(exit_index),
                            entry_price, fill_price, qty, pnl,
                            0.0 if entry_cost == 0 else pnl / entry_cost,
                            reason))
        return cash + net
