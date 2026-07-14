"""Pins for the classic backtest engine.

Java sources: Backtester.java, BacktestConfig.java, TradeCostModel.java
and SmaCrossStrategy.java. The SMA-cross series below is hand-designed
so every cross (and therefore every trade) is known in closed form; the
equity pins are exact arithmetic, not tolerances of a recorded run.
"""

import math

import numpy as np
import pytest

from quantfinlib.backtest import (Backtester, BacktestConfig, Signal, Trade,
                                  TradeCostModel, TradingStrategy)
from quantfinlib.backtest.strategies import (BollingerBandsStrategy,
                                             EmaCrossStrategy, MacdStrategy,
                                             RsiStrategy, SmaCrossStrategy)
from quantfinlib.data import BarSeries


class ScriptedStrategy(TradingStrategy):
    """Emits a fixed per-bar signal list — isolates engine mechanics."""

    def __init__(self, signals):
        self._signals = list(signals)

    def name(self):
        return "SCRIPTED"

    def init(self, series):
        pass

    def on_bar(self, i):
        return self._signals[i]


# Hand-designed closes for SMA(1, 2): fast = close, slow = 2-bar mean.
# Cross up at bar 2 (BUY @110), cross down at bar 4 (SELL @115),
# cross up again at bar 6 (BUY @110 -> force-closed END_OF_DATA @110).
SMA_CLOSES = [100.0, 100.0, 110.0, 120.0, 115.0, 100.0, 110.0]


def test_sma_cross_exact_trades_and_equity_no_costs():
    series = BarSeries.of("SMA", SMA_CLOSES)
    config = BacktestConfig(100_000, 0.0, 0.0, 0.0, 0.0, 252)
    result = Backtester.run(SmaCrossStrategy(1, 2), series, config)

    trades = result.trades()
    assert len(trades) == 2
    t1, t2 = trades
    assert (t1.entry_index, t1.exit_index) == (2, 4)
    assert t1.entry_price == 110.0 and t1.exit_price == 115.0
    assert t1.exit_reason == Trade.REASON_SIGNAL
    assert (t2.entry_index, t2.exit_index) == (6, 6)
    assert t2.exit_reason == Trade.REASON_END_OF_DATA

    # Exact equity path: flat 100k, in at 110, marked at 120, out at 115.
    q = 100_000 / 110.0
    expected = [100_000, 100_000, 100_000, q * 120, q * 115,
                q * 115, q * 115]
    assert np.allclose(result.equity_curve(), expected, rtol=0, atol=1e-9)
    assert result.metrics().total_return == pytest.approx(115 / 110 - 1)


def test_sma_cross_commission_and_slippage_pin():
    series = BarSeries.of("SMA", SMA_CLOSES)
    c, s = 0.001, 0.0005
    config = BacktestConfig(100_000, c, s, 0.0, 0.0, 252)
    result = Backtester.run(SmaCrossStrategy(1, 2), series, config)

    # Entry at bar 2: fill = 110*(1+s); fee charged on committed cash.
    fill_in = 110 * (1 + s)
    qty = (100_000 - 100_000 * c) / fill_in
    # Exit at bar 4: fill = 115*(1-s), commission on proceeds.
    net_out = qty * 115 * (1 - s) * (1 - c)
    t1 = result.trades()[0]
    assert t1.quantity == pytest.approx(qty)
    assert t1.pnl == pytest.approx(net_out - 100_000)
    assert t1.return_pct == pytest.approx((net_out - 100_000) / 100_000)
    # Costs always hurt versus the frictionless run.
    assert net_out < 100_000 * 115 / 110


def test_scripted_engine_stop_loss_gap_fill_and_take_profit():
    # Bar 2 gaps down through the stop: fill at the OPEN, not the level.
    b = BarSeries.builder("GAP")
    b.add(0, 100, 100, 100, 100, 1e6)
    b.add(1, 100, 100, 100, 100, 1e6)   # BUY here at close 100
    b.add(2, 80, 82, 78, 80, 1e6)       # gap open 80 < stop 95
    b.add(3, 80, 80, 80, 80, 1e6)
    series = b.build()
    signals = [Signal.HOLD, Signal.BUY, Signal.HOLD, Signal.HOLD]
    config = BacktestConfig(10_000, 0.0, 0.0, 0.05, 0.0, 252)
    result = Backtester.run(ScriptedStrategy(signals), series, config)
    t = result.trades()[0]
    assert t.exit_reason == Trade.REASON_STOP_LOSS
    assert t.exit_price == 80.0          # min(stop 95, open 80)

    # Take profit fills at the level when the bar trades through it.
    b = BarSeries.builder("TP")
    b.add(0, 100, 100, 100, 100, 1e6)
    b.add(1, 100, 100, 100, 100, 1e6)   # BUY at 100
    b.add(2, 101, 112, 100, 105, 1e6)   # high 112 >= target 110
    b.add(3, 105, 105, 105, 105, 1e6)
    series = b.build()
    config = BacktestConfig(10_000, 0.0, 0.0, 0.0, 0.10, 252)
    result = Backtester.run(ScriptedStrategy(signals), series, config)
    t = result.trades()[0]
    assert t.exit_reason == Trade.REASON_TAKE_PROFIT
    # Target level is entry * (1 + tp): 100 * 1.10 is 110.00000000000001
    # in binary -- pin the computed level, not the decimal literal.
    assert t.exit_price == pytest.approx(110.0, abs=1e-9)  # max(target, open 101)


def test_warm_start_trades_only_from_boundary():
    series = BarSeries.of("SMA", SMA_CLOSES)
    config = BacktestConfig(100_000, 0.0, 0.0, 0.0, 0.0, 252)
    # trade_from=5 skips both the bar-2 entry and the bar-4 exit; only
    # the bar-6 cross remains, and the equity curve covers [5, 7).
    result = Backtester.run(SmaCrossStrategy(1, 2), series, config,
                            trade_from=5)
    assert result.equity_curve().shape[0] == 2
    assert len(result.trades()) == 1
    assert result.trades()[0].entry_index == 6
    with pytest.raises(ValueError):
        Backtester.run(SmaCrossStrategy(1, 2), series, config,
                       trade_from=len(SMA_CLOSES))
    with pytest.raises(ValueError):
        Backtester.run(SmaCrossStrategy(1, 2), series, config,
                       trade_from=-1)


def test_all_shipped_strategies_run_and_hold_through_warmup():
    rng = np.random.default_rng(7)
    closes = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(400))
    series = BarSeries.of("MIX", closes)
    config = BacktestConfig.defaults()
    for strategy in (SmaCrossStrategy(10, 30), EmaCrossStrategy(10, 30),
                     MacdStrategy(), RsiStrategy(14, 30, 70),
                     BollingerBandsStrategy()):
        result = Backtester.run(strategy, series, config)
        assert np.all(np.isfinite(result.equity_curve()))
        # NaN indicator gates: no strategy may signal on bar 0.
        strategy.init(series)
        assert strategy.on_bar(0) is Signal.HOLD
    with pytest.raises(ValueError):
        SmaCrossStrategy(30, 10)
    with pytest.raises(ValueError):
        EmaCrossStrategy(20, 20)


def test_trade_cost_model_flat_and_institutional():
    rng = np.random.default_rng(3)
    closes = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(60))
    series = BarSeries.of("COST", closes)

    flat = TradeCostModel.flat(25.0)
    assert flat.cost_bps(series, 10, 1e6) == 25.0
    with pytest.raises(ValueError):
        TradeCostModel.flat(-1.0)

    inst = TradeCostModel.institutional(1.0, 2.0, 3.0, 20)
    # Before the impact window only the flat components are charged.
    assert inst.cost_bps(series, 10, 1e6) == pytest.approx(6.0)
    # Past the window the square-root impact term is added, and it
    # GROWS with size — capacity as a number.
    small = inst.cost_bps(series, 40, 1e5)
    large = inst.cost_bps(series, 40, 1e7)
    assert small > 6.0
    assert large > small
    with pytest.raises(ValueError):
        TradeCostModel.institutional(1.0, 2.0, 3.0, 1)

    # A series with no volume degrades to the flat components.
    b = BarSeries.builder("NOVOL")
    for i, px in enumerate(closes):
        b.add(i, px, px, px, px, 0.0)
    assert inst.cost_bps(b.build(), 40, 1e6) == pytest.approx(6.0)


def test_backtest_config_with_methods():
    cfg = BacktestConfig.defaults()
    assert cfg.initial_capital == 100_000
    assert cfg.with_initial_capital(5e5).initial_capital == 5e5
    assert cfg.with_commission(0.002).commission_rate == 0.002
    assert cfg.with_stop_loss(0.05).stop_loss_pct == 0.05
    assert cfg.with_take_profit(0.1).take_profit_pct == 0.1
    # The originals are untouched (frozen value semantics).
    assert cfg.commission_rate == 0.001 and cfg.stop_loss_pct == 0.0
