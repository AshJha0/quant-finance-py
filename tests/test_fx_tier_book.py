"""Tiered multi-LP FX book, ported from Java FxTierBookTest.

(The Java allocation-benchmark test is JVM-specific and not ported.)
"""

import math

import pytest

from quantfinlib.fx import FxTierBook


def _book():
    """Two LPs with two-tier ladders around 1.0850, LP2 silent."""
    b = FxTierBook(3, 4)
    # LP0: tight small tier, wider big tier.
    b.tier(0, False, 0, 1.08502, 1_000_000)
    b.tier(0, False, 1, 1.08504, 5_000_000)
    b.tier_count(0, False, 2)
    b.tier(0, True, 0, 1.08500, 1_000_000)
    b.tier(0, True, 1, 1.08498, 5_000_000)
    b.tier_count(0, True, 2)
    # LP1: slightly worse top, deeper second tier.
    b.tier(1, False, 0, 1.08503, 2_000_000)
    b.tier(1, False, 1, 1.08505, 10_000_000)
    b.tier_count(1, False, 2)
    b.tier(1, True, 0, 1.08499, 2_000_000)
    b.tier(1, True, 1, 1.08497, 10_000_000)
    b.tier_count(1, True, 2)
    return b


def test_composite_top_of_book_scans_tier_zero():
    b = _book()
    assert b.best_bid() == pytest.approx(1.08500, abs=1e-12)
    assert b.best_ask() == pytest.approx(1.08502, abs=1e-12)


def test_sweep_takes_cheapest_tiers_across_lps():
    b = _book()
    # Buy 4M: 1M @ .08502 (LP0 t0), 2M @ .08503 (LP1 t0), 1M @ .08504 (LP0 t1).
    cost = b.sweep_buy_cost(4_000_000)
    expected = 1_000_000 * 1.08502 + 2_000_000 * 1.08503 + 1_000_000 * 1.08504
    assert cost == pytest.approx(expected, abs=1e-4)

    # Sell 4M: 1M @ .08500 (LP0), 2M @ .08499 (LP1), 1M @ .08498 (LP0 t1).
    proceeds = b.sweep_sell_proceeds(4_000_000)
    exp_sell = 1_000_000 * 1.08500 + 2_000_000 * 1.08499 + 1_000_000 * 1.08498
    assert proceeds == pytest.approx(exp_sell, abs=1e-4)


def test_sweep_plan_attributes_quantity_per_lp():
    b = _book()
    plan = [0.0, 0.0, 0.0]
    cost = b.sweep_plan(True, 4_000_000, plan)
    assert cost > 0
    assert plan[0] == pytest.approx(2_000_000, abs=1e-9)  # LP0: 1M t0 + 1M t1
    assert plan[1] == pytest.approx(2_000_000, abs=1e-9)
    assert plan[2] == pytest.approx(0, abs=1e-9)


def test_unfillable_size_is_nan_not_a_partial_price():
    b = _book()
    assert math.isnan(b.sweep_buy_cost(50_000_000))
    assert math.isnan(b.sweep_sell_proceeds(50_000_000))
    assert math.isnan(b.sweep_buy_cost(0))


def test_full_amount_finds_the_single_lp_covering_the_clip():
    b = _book()
    # 8M: only LP1's second tiers cover it.
    assert b.best_full_amount_ask(8_000_000) == pytest.approx(1.08505, abs=1e-12)
    assert b.best_full_amount_ask_lp(8_000_000) == 1
    assert b.best_full_amount_bid(8_000_000) == pytest.approx(1.08497, abs=1e-12)
    # 4M: LP0 (.08504) vs LP1 (.08505) -> LP0 wins on the ask.
    assert b.best_full_amount_ask(4_000_000) == pytest.approx(1.08504, abs=1e-12)
    assert b.best_full_amount_ask_lp(4_000_000) == 0
    # 1M: tier-0 prices compete; LP0 tightest ask.
    assert b.best_full_amount_ask(1_000_000) == pytest.approx(1.08502, abs=1e-12)
    # 20M: nobody quotes it full-amount.
    assert math.isnan(b.best_full_amount_ask(20_000_000))
    assert b.best_full_amount_ask_lp(20_000_000) == -1


def test_clearing_an_lp_removes_it_everywhere():
    b = _book()
    b.clear(0)
    assert b.best_bid() == pytest.approx(1.08499, abs=1e-12)
    assert b.best_ask() == pytest.approx(1.08503, abs=1e-12)
    cost = b.sweep_buy_cost(4_000_000)
    expected = 2_000_000 * 1.08503 + 2_000_000 * 1.08505
    assert cost == pytest.approx(expected, abs=1e-4)
    b.clear(1)
    assert math.isnan(b.best_bid())
    assert math.isnan(b.sweep_buy_cost(1))


def test_full_amount_respects_tier_order_within_an_lp():
    b = FxTierBook(1, 3)
    b.tier(0, False, 0, 1.0001, 1_000_000)
    b.tier(0, False, 1, 1.0003, 5_000_000)
    b.tier(0, False, 2, 1.0006, 20_000_000)
    b.tier_count(0, False, 3)
    assert b.best_full_amount_ask(500_000) == pytest.approx(1.0001, abs=1e-12)
    assert b.best_full_amount_ask(2_000_000) == pytest.approx(1.0003, abs=1e-12)
    assert b.best_full_amount_ask(19_000_000) == pytest.approx(1.0006, abs=1e-12)


def test_out_of_range_tier_or_lp_fails_loudly_instead_of_corrupting_a_neighbor():
    # The flat layout means an unchecked tier overflow would land in the
    # NEXT LP's ladder -- the one failure a price store must never have.
    b = FxTierBook(2, 2)
    with pytest.raises(ValueError):
        b.tier(0, True, 2, 1.0, 1_000_000)
    with pytest.raises(ValueError):
        b.tier(0, True, -1, 1.0, 1_000_000)
    with pytest.raises(ValueError):
        b.tier(2, True, 0, 1.0, 1_000_000)
    # Neighbor untouched.
    assert math.isnan(b.best_bid())


def test_nan_and_zero_tiers_never_win_a_sweep():
    # AggregatedBook convention: "NaN never wins". A NaN-priced or
    # zero/NaN-size tier is skipped, not swept.
    b = FxTierBook(2, 2)
    b.tier(0, False, 0, math.nan, 5_000_000)   # pulled via NaN price
    b.tier_count(0, False, 1)
    b.tier(1, False, 0, 1.08505, 5_000_000)
    b.tier_count(1, False, 1)
    assert b.sweep_buy_cost(5_000_000) == pytest.approx(5_000_000 * 1.08505, abs=1e-4)
    plan = [0.0, 0.0]
    b.sweep_plan(True, 1_000_000, plan)
    assert plan[0] == pytest.approx(0, abs=1e-12), "the withdrawn LP must get no quantity"
    assert plan[1] == pytest.approx(1_000_000, abs=1e-12)
    # NaN size behaves as absent too.
    b.tier(0, False, 0, 1.08000, math.nan)
    b.tier_count(0, False, 1)
    assert b.sweep_buy_cost(5_000_000) == pytest.approx(5_000_000 * 1.08505, abs=1e-4)
    # Zero SIZE is absent...
    b.tier(0, False, 0, 1.08000, 0)
    b.tier_count(0, False, 1)
    assert b.sweep_buy_cost(5_000_000) == pytest.approx(5_000_000 * 1.08505, abs=1e-4)
    # ...and zero PRICE must never win a buy sweep as an infinitely cheap ask.
    b.tier(0, False, 0, 0.0, 5_000_000)
    b.tier_count(0, False, 1)
    assert b.sweep_buy_cost(5_000_000) == pytest.approx(5_000_000 * 1.08505, abs=1e-4)


def test_top_of_book_uses_the_frontier_not_blind_tier_zero():
    # A malformed tier 0 (NaN/zero) must not mask a live deeper quote.
    b = FxTierBook(1, 2)
    b.tier(0, True, 0, math.nan, 1_000_000)
    b.tier(0, True, 1, 1.08498, 5_000_000)
    b.tier_count(0, True, 2)
    b.tier(0, False, 0, 0.0, 1_000_000)
    b.tier(0, False, 1, 1.08504, 5_000_000)
    b.tier_count(0, False, 2)
    assert b.best_bid() == pytest.approx(1.08498, abs=1e-12)
    assert b.best_ask() == pytest.approx(1.08504, abs=1e-12)


def test_zero_clip_is_not_a_quote_request():
    b = _book()
    assert math.isnan(b.full_amount_price(0, True, 0))
    assert math.isnan(b.full_amount_price(0, True, -5))
    assert math.isnan(b.best_full_amount_ask(0))


def test_per_lp_full_amount_price_is_the_public_routing_primitive():
    b = _book()
    assert b.full_amount_price(0, True, 1_000_000) == pytest.approx(1.08502, abs=1e-12)
    assert b.full_amount_price(0, True, 4_000_000) == pytest.approx(1.08504, abs=1e-12)
    assert math.isnan(b.full_amount_price(2, True, 1_000_000))
    assert math.isnan(b.full_amount_price(0, True, 50_000_000))
