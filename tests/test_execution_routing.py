"""Hot-lane and adaptive smart order routing, UCB1 bandit selection.
Ported from Java ExecutionTest / AdvancedExecutionAlgosTest (routing
sections) plus new coverage for the streaming venue scorecard.
"""

import math

import numpy as np
import pytest

from quantfinlib.execution.adaptive_sor import AdaptiveSor, Config
from quantfinlib.execution.hft_sor import HftSor
from quantfinlib.execution.ucb1_selector import Ucb1Selector
from quantfinlib.execution.venue_quote import VenueQuote
from quantfinlib.execution.venue_scorecard import VenueScorecard
from quantfinlib.microstructure.execution import Side

# ------------------------------------------------------------------
# HftSor (hot-lane greedy routing)
# ------------------------------------------------------------------


def test_hft_sor_routes_by_best_all_in_price():
    sor = HftSor(2)
    sor.fee(0, 0.0)
    sor.fee(1, 0.0)
    sor.venue_quote(0, 100, 500, 101, 500)
    sor.venue_quote(1, 99, 300, 102, 300)
    out = np.zeros(2, dtype=np.int64)
    routed = sor.route(Side.BUY, 700, 2 ** 31 - 1, out)
    assert routed == 700
    assert list(out) == [500, 200]
    assert sor.route_count() == 1


def test_hft_sor_fees_shift_the_ranking():
    sor = HftSor(2)
    sor.venue_quote(0, 100, 500, 100, 500)   # ask 100, no fee
    sor.venue_quote(1, 100, 500, 100, 500)   # ask 100, but rebate (-1 tick)
    sor.fee(0, 0.0)
    sor.fee(1, -1.0)
    out = np.zeros(2, dtype=np.int64)
    routed = sor.route(Side.BUY, 100, 2 ** 31 - 1, out)
    assert routed == 100
    assert out[1] == 100, "the rebate venue is cheaper all-in and goes first"
    assert out[0] == 0


def test_hft_sor_respects_limit_and_reports_shortfall():
    sor = HftSor(1)
    sor.venue_quote(0, 100, 200, 105, 200)
    out = np.zeros(1, dtype=np.int64)
    routed = sor.route(Side.BUY, 500, 104, out)  # limit below the ask: no fill
    assert routed == 0
    assert out[0] == 0


def test_hft_sor_venue_down_removes_liquidity():
    sor = HftSor(2)
    sor.venue_quote(0, 100, 500, 101, 500)
    sor.venue_quote(1, 99, 500, 100, 500)
    sor.venue_down(1)
    out = np.zeros(2, dtype=np.int64)
    routed = sor.route(Side.BUY, 500, 2 ** 31 - 1, out)
    assert routed == 500
    assert out[1] == 0
    assert out[0] == 500


def test_hft_sor_validates_venue_count():
    with pytest.raises(ValueError):
        HftSor(0)


# ------------------------------------------------------------------
# VenueScorecard
# ------------------------------------------------------------------


def test_venue_scorecard_fill_rate_seeds_from_prior():
    sc = VenueScorecard(1, alpha=0.1, fill_rate_prior=0.8)
    assert sc.fill_rate(0) == pytest.approx(0.8)
    sc.on_fill(0, 1_000)
    # First event seeds at the prior then blends toward 1.
    assert sc.fill_rate(0) == pytest.approx(0.8 + 0.1 * (1 - 0.8))
    assert sc.sent(0) == 1
    assert sc.filled(0) == 1


def test_venue_scorecard_miss_decays_fill_rate():
    sc = VenueScorecard(1, alpha=0.5, fill_rate_prior=0.9)
    sc.on_miss(0, 500)
    assert sc.fill_rate(0) == pytest.approx(0.45)  # 0.9 + 0.5*(0-0.9)


def test_venue_scorecard_markout_matures_after_horizon():
    sc = VenueScorecard(1, alpha=0.5, fill_rate_prior=0.9,
                        markout_horizon_nanos=100)
    sc.on_fill(0, 1_000, True, 100.0, 0)
    assert sc.matured_fill_markouts() == 0
    sc.on_mid(100.0, 50)          # too early
    assert sc.matured_fill_markouts() == 0
    sc.on_mid(100.5, 150)         # horizon elapsed, mid moved +0.5 with the buy
    assert sc.matured_fill_markouts() == 1
    assert sc.post_fill_markout(0) == pytest.approx(0.5)


def test_venue_scorecard_dark_probe_seeds_from_first_observation():
    sc = VenueScorecard(1)
    sc.on_dark_probe(0, 200)
    assert sc.expected_hidden_shares(0) == pytest.approx(200)
    sc.on_dark_probe(0, 0)
    assert 0 < sc.expected_hidden_shares(0) < 200


def test_venue_scorecard_persistence_round_trips():
    from quantfinlib.persist import BinReader, BinWriter

    sc = VenueScorecard(2, alpha=0.1, fill_rate_prior=0.9)
    sc.on_fill(0, 1_000)
    sc.on_miss(1, 2_000)
    sc.on_dark_probe(0, 300)

    w = BinWriter()
    sc.write_state(w)
    r = BinReader(w.to_bytes())

    sc2 = VenueScorecard(2, alpha=0.1, fill_rate_prior=0.9)
    sc2.read_state(r)
    assert sc2.fill_rate(0) == pytest.approx(sc.fill_rate(0))
    assert sc2.fill_rate(1) == pytest.approx(sc.fill_rate(1))
    assert sc2.expected_hidden_shares(0) == pytest.approx(300)


# ------------------------------------------------------------------
# AdaptiveSor
# ------------------------------------------------------------------


def test_adaptive_sor_vetoes_low_reliability_venues():
    sc = VenueScorecard(2, alpha=0.5, fill_rate_prior=0.95)
    sor = AdaptiveSor(sc, Config(2.0, 1.0, 0.9, 5_000, 0.5))
    sor.register("A", 0)
    sor.register("B", 1)
    # Hammer B with misses until its fill rate drops below the 0.9 floor.
    for _ in range(10):
        sc.on_miss(1, 100)
    quotes = [
        VenueQuote("A", 99.99, 100, 100.01, 500, 0.0, 100, False),
        VenueQuote("B", 99.99, 100, 100.00, 500, 0.0, 100, False),
    ]
    decision = sor.route(Side.BUY, 500, quotes)
    venues_used = {leg.venue for leg in decision.lit}
    assert "B" not in venues_used, "low fill rate must veto the venue"
    assert decision.routed_qty == 500


def test_adaptive_sor_probes_dark_venues_additively():
    sc = VenueScorecard(1)
    sor = AdaptiveSor(sc)
    sor.register("DARK", 0)
    sc.on_dark_probe(0, 1_000)
    quotes = [VenueQuote("DARK", 99.99, 500, 100.01, 500, 0.0, 100, True)]
    decision = sor.route(Side.BUY, 2_000, quotes)
    assert len(decision.probes) == 1
    assert decision.probes[0].quantity == 1_000
    assert decision.lit == []
    assert decision.routed_qty == 0
    assert decision.unrouted == 2_000


def test_adaptive_sor_config_validates():
    with pytest.raises(ValueError):
        Config(-1, 1, 0.5, 100, 0.5)
    with pytest.raises(ValueError):
        Config(1, 1, 1.5, 100, 0.5)


def test_adaptive_sor_register_validates_scorecard_id():
    sc = VenueScorecard(1)
    sor = AdaptiveSor(sc)
    with pytest.raises(ValueError):
        sor.register("X", 5)


def test_adaptive_sor_passive_fill_probability_delegates_to_queue_model():
    p = AdaptiveSor.passive_fill_probability(100, 50, 200)
    assert 0 < p < 1
    assert AdaptiveSor.passive_fill_probability(0, 0, 0) == 0.0


# ------------------------------------------------------------------
# Ucb1Selector
# ------------------------------------------------------------------


def test_ucb1_tries_every_arm_before_exploiting():
    u = Ucb1Selector(3)
    seen = {u.select() for _ in range(1)}
    u.record(0, 0.1)
    seen.add(u.select())
    u.record(1, 0.1)
    seen.add(u.select())
    u.record(2, 0.9)
    assert seen == {0, 1, 2}, "every arm earns one look before UCB kicks in"


def test_ucb1_prefers_higher_reward_after_warmup():
    u = Ucb1Selector(2)
    u.select()
    u.record(0, 0.1)
    u.select()
    u.record(1, 0.9)
    assert u.select() == 1


def test_ucb1_validates_inputs():
    with pytest.raises(ValueError):
        Ucb1Selector(1)
    u = Ucb1Selector(2)
    with pytest.raises(ValueError):
        u.record(5, 0.5)
    with pytest.raises(ValueError):
        u.record(0, 1.5)


def test_ucb1_mean_reward_is_nan_before_first_pull():
    u = Ucb1Selector(2)
    assert math.isnan(u.mean_reward(0))
    u.record(0, 0.5)
    assert u.mean_reward(0) == pytest.approx(0.5)
