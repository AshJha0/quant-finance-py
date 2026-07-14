"""Dark pool, mid-peg, iceberg, order placement policy, spread
legging, futures roll, anti-gaming jitter. Ported from Java
ExecutionTest / AdvancedExecutionAlgosTest.
"""

import math

import pytest

from quantfinlib.execution.anti_gaming_jitter import AntiGamingJitter
from quantfinlib.execution.dark_pool_simulator import DarkPoolSimulator
from quantfinlib.execution.futures_roll_algo import FuturesRollAlgo
from quantfinlib.execution.iceberg_order import IcebergOrder
from quantfinlib.execution.mid_peg_tracker import MidPegTracker
from quantfinlib.execution.order_placement_policy import decide, post_region
from quantfinlib.execution.spread_execution_algo import SpreadExecutionAlgo
from quantfinlib.microstructure.execution import Side

# ------------------------------------------------------------------
# Dark pool
# ------------------------------------------------------------------


def test_dark_pool_crosses_at_lit_mid():
    pool = DarkPoolSimulator()
    pool.on_quote(99.99, 100.01)
    assert pool.submit(Side.SELL, 500, 0) == []
    assert pool.resting_qty(Side.SELL) == 500

    fills = pool.submit(Side.BUY, 300, 0)
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(100.00)
    assert fills[0].quantity == 300
    assert pool.resting_qty(Side.SELL) == 200


def test_dark_pool_honors_minimum_execution_quantity():
    pool = DarkPoolSimulator()
    pool.on_quote(99.99, 100.01)
    pool.submit(Side.SELL, 200, 500)     # resting min 500 > its qty: never fillable
    fills = pool.submit(Side.BUY, 300, 0)
    assert fills == []
    assert pool.resting_qty(Side.BUY) == 300


def test_dark_pool_aggregate_meq_counts_only_what_would_actually_fill():
    # Incoming BUY 100 with aggregate MEQ 100 against SELL 60 (no MEQ)
    # and SELL 60 (MEQ 50). A static pre-scan against the ORIGINAL
    # remaining counts 60 + 60 = 120 and crosses -- but after the first
    # fill only 40 remain, below the second seller's MEQ 50, so the
    # true executable total is 60 < 100: the cross must not happen.
    pool = DarkPoolSimulator()
    pool.on_quote(99.99, 100.01)
    pool.submit(Side.SELL, 60, 0)
    pool.submit(Side.SELL, 60, 50)
    fills = pool.submit(Side.BUY, 100, 100)
    assert fills == [], "aggregate MEQ 100 cannot be met: only 60 can execute"
    assert pool.resting_qty(Side.SELL) == 120
    assert pool.resting_qty(Side.BUY) == 100

    # Same book, MEQ 60: exactly satisfiable, fills the first seller only.
    ok = DarkPoolSimulator()
    ok.on_quote(99.99, 100.01)
    ok.submit(Side.SELL, 60, 0)
    ok.submit(Side.SELL, 60, 50)
    ok_fills = ok.submit(Side.BUY, 100, 60)
    assert len(ok_fills) == 1
    assert ok_fills[0].quantity == 60
    assert ok.resting_qty(Side.SELL) == 60     # the MEQ-50 seller untouched
    assert ok.resting_qty(Side.BUY) == 40       # remainder rests


def test_dark_pool_pauses_on_locked_or_crossed_market():
    pool = DarkPoolSimulator()
    pool.submit(Side.SELL, 100, 0)
    pool.on_quote(100.02, 100.00)          # crossed: bid > ask
    fills = pool.submit(Side.BUY, 50, 0)
    assert fills == [], "a crossed reference must not execute"
    assert pool.resting_qty(Side.BUY) == 50


def test_dark_pool_validates_inputs():
    pool = DarkPoolSimulator()
    with pytest.raises(ValueError):
        pool.submit(Side.BUY, 0, 0)
    with pytest.raises(ValueError):
        pool.submit(Side.BUY, 10, -1)


# ------------------------------------------------------------------
# Mid peg
# ------------------------------------------------------------------


def test_mid_peg_reprices_only_beyond_threshold():
    peg = MidPegTracker(Side.BUY, -0.01, math.nan, 0.02)
    assert peg.on_quote(99.98, 100.02) == pytest.approx(99.99)      # first quote always prices
    assert math.isnan(peg.on_quote(99.99, 100.02))                  # mid +0.005: below threshold
    assert peg.on_quote(100.03, 100.07) == pytest.approx(100.04)    # big move: reprice
    assert peg.current_price() == pytest.approx(100.04)


def test_mid_peg_respects_limit_cap():
    peg = MidPegTracker(Side.BUY, 0, 100.00, 0.001)
    assert peg.on_quote(100.10, 100.20) == pytest.approx(100.00)    # mid 100.15 capped at limit


def test_mid_peg_validates_inputs():
    with pytest.raises(ValueError):
        MidPegTracker(Side.BUY, math.nan, math.nan, 0.01)
    with pytest.raises(ValueError):
        MidPegTracker(Side.BUY, 0, math.inf, 0.01)
    with pytest.raises(ValueError):
        MidPegTracker(Side.BUY, 0, math.nan, -0.01)


# ------------------------------------------------------------------
# Iceberg
# ------------------------------------------------------------------


def test_iceberg_reloads_tranches():
    ice = IcebergOrder(1_000, 100)
    assert ice.visible_qty() == 100
    assert ice.hidden_qty() == 900

    assert ice.on_fill(100) is True             # tranche exhausted -> reload
    assert ice.visible_qty() == 100
    assert ice.remaining_qty() == 900

    assert ice.on_fill(40) is False              # partial: no reload
    assert ice.visible_qty() == 60

    safety = 0
    while not ice.is_complete() and safety < 100:
        ice.on_fill(ice.visible_qty())
        safety += 1
    assert ice.is_complete()
    assert ice.remaining_qty() == 0


def test_randomized_iceberg_stays_within_bounds_and_total():
    ice = IcebergOrder(1_000, 100, 0.2, 7)
    filled = 0
    while not ice.is_complete():
        v = ice.visible_qty()
        assert 1 <= v <= min(120, 1_000 - filled) + 1
        ice.on_fill(v)
        filled += v
    assert filled == 1_000


def test_iceberg_validates_inputs():
    with pytest.raises(ValueError):
        IcebergOrder(0, 10)
    with pytest.raises(ValueError):
        IcebergOrder(10, 0)
    ice = IcebergOrder(100, 10)
    with pytest.raises(ValueError):
        ice.on_fill(0)
    with pytest.raises(ValueError):
        ice.on_fill(11)


# ------------------------------------------------------------------
# Order placement policy
# ------------------------------------------------------------------


def test_post_or_cross_follows_expected_cost_exactly():
    friendly = decide(0.05, 0.6, 0.02, 0.01, 0.002)
    assert friendly.expected_post_cost == pytest.approx(0.0048, abs=1e-12)
    assert friendly.cross_cost == pytest.approx(0.05, abs=1e-12)
    assert friendly.post, "posting is 10x cheaper here"

    toxic = decide(0.05, 0.6, 0.15, 0.01, 0.002)
    assert toxic.expected_post_cost == pytest.approx(0.0828, abs=1e-12)
    assert not toxic.post, "cross and be done"

    normal = post_region(0.05, 0.02, 0.01, 0.002)
    assert normal.from_ == pytest.approx(0.01 / 0.092, abs=1e-12)
    assert normal.to == pytest.approx(1, abs=1e-12)
    assert decide(0.05, normal.from_ + 0.01, 0.02, 0.01, 0.002).post
    assert not decide(0.05, normal.from_ - 0.01, 0.02, 0.01, 0.002).post

    assert post_region(0.05, 0.15, 0.01, 0.002).is_empty(), \
        "adverse selection swamps everything: never post"

    flipped = post_region(0.05, 0.3, -0.05, 0)
    assert flipped.from_ == pytest.approx(0, abs=1e-12)
    assert flipped.to == pytest.approx(0.2, abs=1e-12)
    assert decide(0.05, 0.1, 0.3, -0.05, 0).post, "below the flipped threshold: post"
    assert not decide(0.05, 0.3, 0.3, -0.05, 0).post, "above it: cross"

    assert not post_region(1, 1, -1, 0).is_empty()
    assert post_region(1, 2.1, 0.1, 0).is_empty()

    assert not decide(0.05, 0, 0.02, 0, 0).post, "p=0, d=0: a tie, resolved to CROSS"
    assert decide(0.05, 1, 0.02, 0.01, 0.002).expected_post_cost == \
        pytest.approx(0.02 - 0.05 - 0.002, abs=1e-12)

    assert decide(0.05, 0.1, 0.02, -0.03, 0).post


def test_order_placement_policy_validates_inputs():
    with pytest.raises(ValueError):
        decide(0, 0.5, 0.02, 0.01, 0)
    with pytest.raises(ValueError):
        decide(0.05, 1.5, 0.02, 0.01, 0)
    with pytest.raises(ValueError):
        decide(0.05, 0.5, 0.02, math.nan, 0)


# ------------------------------------------------------------------
# Spread execution
# ------------------------------------------------------------------


def test_hedge_leg_chases_and_the_legging_cap_is_hard():
    spread = SpreadExecutionAlgo(10_000, 2.0, 3_000, 1_000)

    c = spread.decide()
    assert c.lead_qty == 1_000, "fresh: work the lead patiently"
    assert c.hedge_qty == 0
    assert not c.at_risk_cap

    spread.on_lead_fill(1_000)
    c = spread.decide()
    assert c.hedge_qty == 2_000
    assert c.lead_qty == 500

    spread.on_lead_fill(500)
    c = spread.decide()
    assert c.at_risk_cap
    assert c.lead_qty == 0
    assert c.hedge_qty == 3_000

    spread.on_hedge_fill(3_000)
    assert spread.imbalance_hedge_units() == 0
    c = spread.decide()
    assert c.lead_qty == 1_000
    assert not spread.done()

    spread.on_lead_fill(8_500)
    assert not spread.done()
    spread.on_hedge_fill(spread.imbalance_hedge_units())
    assert spread.done()
    assert spread.hedge_executed() == 20_000

    with pytest.raises(ValueError):
        spread.on_lead_fill(1)
    with pytest.raises(ValueError):
        SpreadExecutionAlgo(0, 2, 3_000, 100)
    with pytest.raises(ValueError):
        SpreadExecutionAlgo(100, math.nan, 3_000, 100)


def test_fractional_ratios_overfills_and_impossible_caps_are_handled():
    frac = SpreadExecutionAlgo(3, 1.5, 4, 3)
    while not frac.done():
        c = frac.decide()
        frac.on_lead_fill(c.lead_qty)
        frac.on_hedge_fill(c.hedge_qty)
    assert frac.lead_executed() == 3
    assert frac.hedge_executed() == 5, "round(3 x 1.5) = 5, half-up"

    with pytest.raises(ValueError):
        SpreadExecutionAlgo(100, 2.0, 1, 10)

    over = SpreadExecutionAlgo(10, 1.0, 5, 10)
    over.on_lead_fill(10)
    with pytest.raises(ValueError):
        over.on_hedge_fill(11)
    over.on_hedge_fill(10)
    assert over.done()


# ------------------------------------------------------------------
# Anti-gaming jitter
# ------------------------------------------------------------------


def test_jitter_kills_the_pattern_but_never_the_total():
    clockwork = [1_000] * 12
    jitter = AntiGamingJitter(42, 0.3, 0.4)
    jittered = jitter.jitter_sizes(clockwork)

    assert sum(int(x) for x in jittered) == 12_000
    assert all(x >= 0 for x in jittered)
    assert any(x != 1_000 for x in jittered), "the metronome is gone"

    assert list(AntiGamingJitter(42, 0.3, 0.4).jitter_sizes(clockwork)) == list(jittered), \
        "same seed, same schedule"
    assert list(AntiGamingJitter(43, 0.3, 0.4).jitter_sizes(clockwork)) != list(jittered), \
        "different seed, different pattern"

    times = [(i + 1) * 60_000_000_000 for i in range(10)]
    jt = AntiGamingJitter(7, 0, 0.4).jitter_times(times, 0)
    for i in range(1, len(jt)):
        assert jt[i] > jt[i - 1], "children never reorder"
    assert jt[0] >= 0
    assert jt[9] <= times[9], "the schedule never runs long"
    assert list(jt) != times, "the clock is no longer a metronome"

    with pytest.raises(ValueError):
        AntiGamingJitter(1, 0.6, 0.1)
    with pytest.raises(ValueError):
        jitter.jitter_times([5, 5], 0)

    off = AntiGamingJitter(99, 0, 0)
    assert list(off.jitter_sizes(clockwork)) == clockwork
    assert list(off.jitter_times(times, 0)) == times

    assert list(jitter.jitter_sizes([5_000])) == [5_000]


# ------------------------------------------------------------------
# Futures roll
# ------------------------------------------------------------------


def test_roll_follows_the_migration_curve_and_always_completes():
    curve = FuturesRollAlgo.default_migration(5)
    assert curve[4] == 1.0, "the roll ENDS complete, exactly"
    for d in range(1, 5):
        assert curve[d] > curve[d - 1], "migration only goes forward"
    assert curve[2] - curve[1] > curve[0], "concentrated middle"

    roll = FuturesRollAlgo(10_000, curve)
    day0 = roll.due_on_day(0)
    assert day0 == math.floor(10_000 * curve[0] + 0.5)
    roll.on_rolled(day0)

    assert roll.due_on_day(2) == math.floor(10_000 * curve[2] + 0.5) - day0, \
        "falling behind just makes later days bigger"
    roll.on_rolled(roll.due_on_day(2))
    roll.on_rolled(roll.due_on_day(3))
    roll.on_rolled(roll.due_on_day(4))
    assert roll.done()
    assert roll.rolled() == 10_000

    calendar = SpreadExecutionAlgo(day0, 1.0, 200, 300)
    while not calendar.done():
        c = calendar.decide()
        calendar.on_lead_fill(c.lead_qty)
        calendar.on_hedge_fill(c.hedge_qty)
    assert calendar.lead_executed() == calendar.hedge_executed()

    with pytest.raises(ValueError):
        FuturesRollAlgo(1_000, [0.5, 0.9])
    with pytest.raises(ValueError):
        FuturesRollAlgo(1_000, [0.5, 0.4, 1.0])
    with pytest.raises(ValueError):
        roll.on_rolled(1)

    one_day = FuturesRollAlgo(500, FuturesRollAlgo.default_migration(1))
    assert one_day.due_on_day(0) == 500
    one_day.on_rolled(500)
    assert one_day.done()

    one_lot = FuturesRollAlgo(1, FuturesRollAlgo.default_migration(5))
    total = 0
    for d in range(5):
        due = one_lot.due_on_day(d)
        one_lot.on_rolled(due)
        total += due
    assert total == 1
    assert one_lot.done()

    late_start = FuturesRollAlgo(1_000, [0, 0, 1])
    assert late_start.due_on_day(0) == 0
    assert late_start.due_on_day(2) == 1_000
