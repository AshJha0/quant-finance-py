"""LP scorecard and last-look-aware router, ported from Java
LpScorecardAndRouterTest.

(The Java allocation-benchmark test is JVM-specific and not ported.)
"""

import math

import pytest

from quantfinlib.fx import FxTierBook, LpRouter, LpScorecard

MS = 1_000_000  # nanos per millisecond


# ------------------------------------------------------------------
# Scorecard
# ------------------------------------------------------------------

def test_reject_rate_tracks_recent_behavior():
    c = LpScorecard(2, 0.5, 100 * MS)
    assert c.reject_rate(0) == pytest.approx(0, abs=1e-12)
    c.on_fill(0, True, 1.08502, 1.08501, MS)
    assert c.reject_rate(0) == pytest.approx(0, abs=1e-12)
    c.on_reject(0, True, 1.08501, 0, 2 * MS)
    assert c.reject_rate(0) == pytest.approx(0.5, abs=1e-12)
    c.on_reject(0, True, 1.08501, 0, 2 * MS)
    assert c.reject_rate(0) == pytest.approx(0.75, abs=1e-12)
    c.on_fill(0, True, 1.08502, 1.08501, MS)
    assert c.reject_rate(0) == pytest.approx(0.375, abs=1e-12)
    assert c.attempts(0) == 4
    assert c.fills(0) == 2
    assert c.rejects(0) == 2
    # LP1 untouched.
    assert c.attempts(1) == 0


def test_post_reject_markout_matures_at_the_horizon():
    c = LpScorecard(1, 1.0, 100 * MS)   # alpha 1: exact values
    # Buy rejected at mid 1.08500; 100ms later mid is 1.08510: +10 pips
    # of markout -- the reject cost us the move we were chasing.
    c.on_reject(0, True, 1.08500, 0, MS)
    c.on_mid(1.08505, 50 * MS)          # too early: pending
    assert c.post_reject_markout(0) == pytest.approx(0, abs=1e-12)
    c.on_mid(1.08510, 100 * MS)
    assert c.post_reject_markout(0) == pytest.approx(0.00010, abs=1e-9)
    # A sell reject with the market falling is equally adverse (+).
    c.on_reject(0, False, 1.08510, 200 * MS, MS)
    c.on_mid(1.08495, 300 * MS)
    assert c.post_reject_markout(0) == pytest.approx(0.00015, abs=1e-9)


def test_reject_bursts_are_sampled_not_overwritten():
    # Bursts happen when the market runs -- exactly when markouts are
    # largest -- so the pending ring must mature EVERY burst reject.
    c = LpScorecard(1, 1.0, 100 * MS)
    c.on_reject(0, True, 1.08500, 0, MS)
    c.on_reject(0, True, 1.08508, 60 * MS, MS)   # within the horizon
    c.on_mid(1.08512, 160 * MS)                  # matures BOTH
    assert c.matured_markouts() == 2
    # EWMA at alpha=1 ends on the last matured (ring order): +0.00004.
    assert c.post_reject_markout(0) == pytest.approx(0.00004, abs=1e-9)


def test_nan_reference_mids_never_start_a_markout():
    # A reject whose mid_at_request is NaN must count against the rate
    # but never create a pending markout.
    c = LpScorecard(1, 1.0, 100 * MS)
    c.on_reject(0, True, math.nan, 0, MS)
    assert c.rejects(0) == 1, "the reject itself still counts"
    c.on_mid(1.08510, 200 * MS)
    assert c.matured_markouts() == 0, "no pending markout was created"
    assert c.post_reject_markout(0) == pytest.approx(0.0, abs=1e-12)
    # And the LP remains routable on real stats afterwards.
    c.on_reject(0, True, 1.08500, 300 * MS, MS)
    c.on_mid(1.08505, 400 * MS)
    assert c.post_reject_markout(0) == pytest.approx(0.00005, abs=1e-9)


def test_nan_mids_never_poison_the_markout():
    c = LpScorecard(1, 1.0, 100 * MS)
    c.on_reject(0, True, 1.08500, 0, MS)
    c.on_mid(math.nan, 200 * MS)   # ignored
    assert c.matured_markouts() == 0
    c.on_mid(1.08510, 300 * MS)    # real mid matures it
    assert c.post_reject_markout(0) == pytest.approx(0.00010, abs=1e-9)
    assert c.matured_markouts() == 1


def _two_lp_book():
    b = FxTierBook(2, 2)
    b.tier(0, False, 0, 1.08502, 5_000_000)   # LP0: tighter ask
    b.tier_count(0, False, 1)
    b.tier(0, True, 0, 1.08499, 5_000_000)
    b.tier_count(0, True, 1)
    b.tier(1, False, 0, 1.08504, 5_000_000)   # LP1: 2 pips wider
    b.tier_count(1, False, 1)
    b.tier(1, True, 0, 1.08497, 5_000_000)
    b.tier_count(1, True, 1)
    return b


def test_poisoned_or_unquoted_expected_prices_never_capture_the_router():
    # Regression: a NaN expected price for the first candidate must not
    # win the empty-best branch and freeze routing onto that LP.
    b = _two_lp_book()
    c = LpScorecard(2, 1.0, 100 * MS)
    r = LpRouter(b, c, 1.0)
    # LP0's ladder is pulled: its full-amount price is NaN.
    b.tier_count(0, False, 0)
    assert r.route(True, 1_000_000) == 1, "NaN candidate must lose"
    assert r.last_quoted_price() == pytest.approx(1.08504, abs=1e-12)


def test_effective_spread_and_hold_are_ewmas():
    c = LpScorecard(1, 1.0, 100 * MS)
    c.on_fill(0, True, 1.08503, 1.08501, 5 * MS)   # paid 2 pips over mid
    assert c.effective_spread(0) == pytest.approx(0.00002, abs=1e-9)
    assert c.avg_hold_nanos(0) == pytest.approx(5 * MS, abs=1e-6)
    c.on_fill(0, False, 1.08499, 1.08501, 3 * MS)  # sell 2 pips under mid
    assert c.effective_spread(0) == pytest.approx(0.00002, abs=1e-9)
    assert c.avg_hold_nanos(0) == pytest.approx(3 * MS, abs=1e-6)


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------

def test_clean_books_route_to_the_tightest_quote():
    b = _two_lp_book()
    c = LpScorecard(2)
    r = LpRouter(b, c, 0.25)
    assert r.route(True, 1_000_000) == 0
    assert r.last_quoted_price() == pytest.approx(1.08502, abs=1e-12)
    assert r.last_expected_price() == pytest.approx(1.08502, abs=1e-12)
    assert r.route(False, 1_000_000) == 0, "LP0 has the better bid too"


def test_rejecty_lp_loses_despite_the_tighter_quote():
    b = _two_lp_book()
    c = LpScorecard(2, 1.0, 100 * MS)
    # LP0 always rejects (rate 1.0 at alpha 1) with 30 pips of adverse
    # markout: expected LP0 ask = 1.08502 + 1.0 x 0.0030 = 1.08532,
    # worse than LP1's firm 1.08504 despite the tighter display.
    c.on_reject(0, True, 1.08500, 0, MS)
    c.on_mid(1.08530, 100 * MS)
    r = LpRouter(b, c, 1.0)   # no veto: pure pricing
    assert r.route(True, 1_000_000) == 1
    assert r.last_quoted_price() == pytest.approx(1.08504, abs=1e-12)


def test_reject_rate_cap_vetoes_outright():
    b = _two_lp_book()
    c = LpScorecard(2, 1.0, 100 * MS)
    c.on_reject(0, True, 1.08500, 0, MS)   # rate -> 1.0
    r = LpRouter(b, c, 0.25)
    assert r.route(True, 1_000_000) == 1
    assert r.veto_count() > 0
    # Both vetoed/unquoting -> -1 and NaN prices.
    c.on_reject(1, True, 1.08500, 0, MS)
    assert r.route(True, 1_000_000) == -1
    assert math.isnan(r.last_quoted_price())


def test_hold_time_is_priced_like_latency_when_urgency_is_set():
    # Two LPs, identical quotes and zero rejects; LP0 holds requests
    # 50ms, LP1 decides in 1ms. With hold urgency, the slow holder
    # loses the tie -- FX's latency dimension, priced.
    b = FxTierBook(2, 1)
    for lp in range(2):
        b.tier(lp, False, 0, 1.08502, 5_000_000)
        b.tier_count(lp, False, 1)
    c = LpScorecard(2, 1.0, 100 * MS)
    c.on_fill(0, True, 1.08502, 1.08501, 50 * MS)
    c.on_fill(1, True, 1.08502, 1.08501, MS)
    # Without urgency: a pure price tie (first candidate wins).
    assert LpRouter(b, c, 1.0).route(True, 1_000_000) == 0
    # With urgency: the fast decider wins.
    urgent = LpRouter(b, c, 1.0, 1.0)   # 1 bp per ms held
    assert urgent.route(True, 1_000_000) == 1
    assert urgent.last_expected_price() > urgent.last_quoted_price(), \
        "the hold penalty must be visible in the expected price"


def test_first_markout_seeds_at_full_strength_not_five_percent():
    # Regression: ramping the markout EWMA from 0 under-penalized a
    # toxic LP for its first ~1/alpha rejects. The first matured
    # markout now seeds.
    c = LpScorecard(2, 0.05, 100 * MS)
    c.on_reject(0, True, 1.08500, 0, MS)
    c.on_mid(1.08530, 200 * MS)   # +30 pips against us
    assert c.post_reject_markout(0) == pytest.approx(
        0.00030, abs=1e-12), "full strength on observation one, not 5% of it"
    # And the second one blends at alpha as before.
    c.on_reject(0, True, 1.08530, 300 * MS, MS)
    c.on_mid(1.08540, 500 * MS)   # +10 pips
    assert c.post_reject_markout(0) == pytest.approx(
        0.00030 + 0.05 * (0.00010 - 0.00030), abs=1e-12)


def test_router_respects_clip_size_against_tiers():
    b = _two_lp_book()
    c = LpScorecard(2)
    r = LpRouter(b, c, 0.5)
    assert r.route(True, 20_000_000) == -1, "nobody quotes 20M full-amount"
    assert r.route(True, 5_000_000) == 0
