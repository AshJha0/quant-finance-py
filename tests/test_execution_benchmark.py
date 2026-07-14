"""BenchmarkExecutor and LiquiditySeekingAlgo. Ported from Java
BenchmarkExecutorTest.
"""

import math

import pytest

from quantfinlib.execution.benchmark_executor import (Benchmark,
                                                       BenchmarkExecutor,
                                                       MarketState)
from quantfinlib.execution.liquidity_seeking_algo import (
    Config as SeekConfig, LiquiditySeekingAlgo)
from quantfinlib.microstructure.execution import Side


def _neutral(frac):
    return MarketState.neutral(100.0, frac)


def _run_neutral(benchmark, parent, steps):
    e = BenchmarkExecutor(Side.BUY, parent, benchmark, 0.1, 0, 1.0)
    for i in range(1, steps + 1):
        due = e.due_quantity(i / steps, _neutral(i / steps))
        e.on_fill(due)
    return e.executed()


# ------------------------------------------------------------------
# Each benchmark's completion curve
# ------------------------------------------------------------------


def test_twap_is_linear_in_time():
    e = BenchmarkExecutor(Side.BUY, 10_000, Benchmark.TWAP, 0.1, 0, 1.0)
    assert e.due_quantity(0.5, _neutral(0.5)) == 5_000
    e.on_fill(5_000)
    assert e.due_quantity(0.5, _neutral(0.5)) == 0, "on schedule: nothing due"
    assert e.due_quantity(1.0, _neutral(1.0)) == 5_000


def test_all_time_benchmarks_finish_the_parent():
    for b in (Benchmark.TWAP, Benchmark.ARRIVAL_PRICE,
             Benchmark.IMPLEMENTATION_SHORTFALL, Benchmark.CLOSING_PRICE,
             Benchmark.OPENING_PRICE):
        assert _run_neutral(b, 100_000, 50) == 100_000, f"{b} must complete the parent"


def test_front_loaded_benchmarks_lead_back_loaded_ones_early():
    def front_progress(b):
        e = BenchmarkExecutor(Side.BUY, 100_000, b, 0.1, 0, 1.0)
        return e.due_quantity(0.25, _neutral(0.25))

    open_ = front_progress(Benchmark.OPENING_PRICE)
    is_ = front_progress(Benchmark.IMPLEMENTATION_SHORTFALL)
    twap = front_progress(Benchmark.TWAP)
    close = front_progress(Benchmark.CLOSING_PRICE)
    assert open_ > is_, "open leads IS early"
    assert is_ > twap, "IS front-loads vs TWAP"
    assert twap > close, "TWAP leads the back-loaded close"


def test_closing_price_keeps_weight_near_the_close():
    e = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.CLOSING_PRICE, 0.1, 0, 1.0)
    assert e.due_quantity(0.5, _neutral(0.5)) == 25_000
    e.on_fill(25_000)
    assert e.due_quantity(0.9, _neutral(0.9)) > 25_000


def test_vwap_tracks_the_volume_curve_not_wall_clock():
    e = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.VWAP, 0.1, 0, 1.0)
    busy_open = MarketState(100, 0, 0, math.inf, 0.5, 0, 0)
    assert e.due_quantity(0.2, busy_open) == 50_000


def test_participation_chases_realized_volume():
    e = BenchmarkExecutor.pov(Side.BUY, 1_000_000, 0.10)
    assert e.due_quantity(0.0, _neutral(0.0)) == 0
    e.on_market_volume(500_000)
    assert e.due_quantity(0.0, _neutral(0.0)) == 50_000
    e.on_fill(50_000)
    e.on_market_volume(300_000)
    due = e.due_quantity(0.0, _neutral(0.0))
    assert due == 30_000
    e.on_fill(due)
    assert e.realized_participation() == pytest.approx(0.10, abs=1e-9)


# ------------------------------------------------------------------
# Dynamic layer
# ------------------------------------------------------------------


def test_adverse_alpha_speeds_up_and_favorable_alpha_slows_down():
    def due_with_alpha(alpha):
        e = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 10, 1.0)
        return e.due_quantity(0.5, MarketState(100, 0, 0, math.inf, 0.5, alpha, 0))

    base = due_with_alpha(0.0)
    adverse = due_with_alpha(0.01)
    favorable = due_with_alpha(-0.01)
    assert adverse > base
    assert favorable < base


def test_sell_side_flips_the_alpha_sign():
    buy = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 10, 1.0)
    sell = BenchmarkExecutor(Side.SELL, 100_000, Benchmark.TWAP, 0.1, 10, 1.0)
    price_falling = MarketState(100, 0, 0, math.inf, 0.5, -0.01, 0)
    assert sell.due_quantity(0.5, price_falling) > buy.due_quantity(0.5, price_falling)


def test_liquidity_cap_limits_the_child():
    e = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 0, 0.25)
    thin = MarketState(100, 0, 0, 40_000, 0.5, 0, 0)
    assert e.due_quantity(0.5, thin) == 10_000


def test_wide_spread_damps_aggression():
    tight = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 0, 1.0)
    wide = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 0, 1.0)
    tight_due = tight.due_quantity(0.5, MarketState(100, 0.001, 0, math.inf, 0.5, 0, 0))
    wide_due = wide.due_quantity(0.5, MarketState(100, 1.0, 0, math.inf, 0.5, 0, 0))
    assert wide_due < tight_due


def test_estimated_impact_damps_the_pace_like_a_spread():
    free = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 0, 1.0)
    costly = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 0, 1.0)
    free_due = free.due_quantity(0.5, MarketState(100, 0, 0, math.inf, 0.5, 0, 0))
    costly_due = costly.due_quantity(0.5, MarketState(100, 0, 0, math.inf, 0.5, 0, 100))
    assert costly_due == free_due // 2


def test_volatility_raises_urgency_for_shortfall_but_lowers_it_for_twap():
    volatile = MarketState(100, 0, 0.5, math.inf, 0.5, 0, 0)
    is_due = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.IMPLEMENTATION_SHORTFALL,
                               0.1, 0, 1.0).due_quantity(0.5, volatile)
    is_base = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.IMPLEMENTATION_SHORTFALL,
                                0.1, 0, 1.0).due_quantity(0.5, _neutral(0.5))
    twap_due = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP,
                                 0.1, 0, 1.0).due_quantity(0.5, volatile)
    twap_base = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP,
                                  0.1, 0, 1.0).due_quantity(0.5, _neutral(0.5))
    assert is_due > is_base
    assert twap_due < twap_base


def test_schedule_drift_reports_ahead_behind():
    e = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 0, 1.0)
    e.on_fill(60_000)
    assert e.schedule_drift(0.5, _neutral(0.5)) > 0
    assert e.schedule_drift(0.8, _neutral(0.8)) < 0


def test_catches_up_when_behind_schedule():
    e = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 0, 1.0)
    due = e.due_quantity(0.8, _neutral(0.8))
    assert due == 80_000


def test_works_on_fx_sized_notionals_and_rates():
    e = BenchmarkExecutor(Side.SELL, 50_000_000, Benchmark.VWAP, 0.1, 5, 0.25)
    fx = MarketState(1.08501, 0.00002, 0.0001, 20_000_000, 0.4, 0, 0)
    due = e.due_quantity(0.3, fx)
    assert 0 < due <= 5_000_000


def test_nan_market_inputs_are_neutral_not_a_silent_stall():
    e = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.TWAP, 0.1, 10, 1.0)
    poisoned = MarketState(100, math.nan, math.nan, math.inf, 0.5, math.nan, 0)
    assert e.due_quantity(0.5, poisoned) == 50_000

    v = BenchmarkExecutor(Side.BUY, 100_000, Benchmark.VWAP, 0.1, 0, 1.0)
    nan_vol = MarketState(100, 0, 0, math.inf, math.nan, 0, 0)
    assert v.due_quantity(0.5, nan_vol) == 50_000


def test_of_rejects_participation_which_needs_an_explicit_rate():
    with pytest.raises(ValueError):
        BenchmarkExecutor.of(Side.BUY, 1000, Benchmark.PARTICIPATION)


def test_benchmark_executor_validates_inputs():
    with pytest.raises(ValueError):
        BenchmarkExecutor(Side.BUY, 0, Benchmark.TWAP, 0.1, 10, 0.25)
    with pytest.raises(ValueError):
        BenchmarkExecutor.pov(Side.BUY, 1000, 1.5)
    with pytest.raises(ValueError):
        BenchmarkExecutor(Side.BUY, 1000, Benchmark.TWAP, 0.1, -1, 0.25)
    with pytest.raises(ValueError):
        BenchmarkExecutor(Side.BUY, 1000, Benchmark.TWAP, 0.1, 10, 0)


# ------------------------------------------------------------------
# LiquiditySeekingAlgo
# ------------------------------------------------------------------


def test_liquidity_seeking_bursts_when_cheap_and_floors_completion():
    algo = LiquiditySeekingAlgo(10_000, SeekConfig.defaults())
    cheap_state = MarketState(100, 0.01, 0.1, 5_000, 0.0, 0, 2.0)
    due = algo.due_quantity(0.1, cheap_state, forecast_spread=0.02)
    assert due == pytest.approx(1_250)   # 25% of 5,000 displayed depth


def test_liquidity_seeking_floor_ramps_in_late_and_guarantees_completion():
    algo = LiquiditySeekingAlgo(1_000)
    # Expensive/volatile: never cheap, so only the completion floor fires.
    hostile = MarketState(100, 10, 1.0, 0, 0.0, 0, 100)
    assert algo.due_quantity(0.5, hostile, forecast_spread=math.nan) == 0
    due_late = algo.due_quantity(0.95, hostile, forecast_spread=math.nan)
    assert due_late > 0
    assert algo.due_quantity(1.0, hostile, forecast_spread=math.nan) == 1_000


def test_liquidity_seeking_validates_config():
    with pytest.raises(ValueError):
        SeekConfig(-1, 0.5, 5, 0.25, 0.7)
    with pytest.raises(ValueError):
        LiquiditySeekingAlgo(0)
