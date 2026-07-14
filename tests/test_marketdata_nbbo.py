"""Nbbo tests, pinning Java ``marketdata.Nbbo``: differential updates
verified against a brute-force recompute, change-only listener firing,
venue bitmasks, and locked/crossed detection.
"""

import math

import pytest

from quantfinlib.marketdata.nbbo import NO_ASK, NO_BID, Nbbo


def brute_force_nbbo(bid_ticks, bid_sizes, ask_ticks, ask_sizes):
    """Independent reimplementation used to cross-check Nbbo's
    incremental recompute."""
    bb, bb_sz = NO_BID, 0
    for t, s in zip(bid_ticks, bid_sizes):
        if t == NO_BID or s <= 0:
            continue
        if t > bb:
            bb, bb_sz = t, s
        elif t == bb:
            bb_sz += s
    bo, bo_sz = NO_ASK, 0
    for t, s in zip(ask_ticks, ask_sizes):
        if t == NO_ASK or s <= 0:
            continue
        if t < bo:
            bo, bo_sz = t, s
        elif t == bo:
            bo_sz += s
    return bb, bb_sz, bo, bo_sz


def test_constructor_validates_venue_count():
    with pytest.raises(ValueError):
        Nbbo(0)
    with pytest.raises(ValueError):
        Nbbo(65)
    Nbbo(1)
    Nbbo(64)


def test_single_venue_quote_becomes_the_nbbo():
    n = Nbbo(3)
    n.on_venue_quote(0, 100, 10, 105, 20, 1)
    assert n.bid_tick() == 100
    assert n.bid_size() == 10
    assert n.ask_tick() == 105
    assert n.ask_size() == 20
    assert n.bid_venues() == 0b001
    assert n.ask_venues() == 0b001


def test_differential_update_matches_brute_force_across_random_venue_updates():
    import random
    rng = random.Random(42)
    venues = 5
    n = Nbbo(venues)
    bid_ticks = [NO_BID] * venues
    bid_sizes = [0] * venues
    ask_ticks = [NO_ASK] * venues
    ask_sizes = [0] * venues

    for _ in range(500):
        v = rng.randrange(venues)
        has_bid = rng.random() > 0.1
        has_ask = rng.random() > 0.1
        bid = rng.randint(95, 105) if has_bid else NO_BID
        bid_sz = rng.randint(1, 100) if has_bid else 0
        ask = rng.randint(95, 115) if has_ask else NO_ASK
        ask_sz = rng.randint(1, 100) if has_ask else 0

        n.on_venue_quote(v, bid, bid_sz, ask, ask_sz, _)
        bid_ticks[v], bid_sizes[v] = bid, bid_sz
        ask_ticks[v], ask_sizes[v] = ask, ask_sz

        exp_bb, exp_bb_sz, exp_bo, exp_bo_sz = brute_force_nbbo(
            bid_ticks, bid_sizes, ask_ticks, ask_sizes)
        assert n.bid_tick() == exp_bb
        assert n.bid_size() == exp_bb_sz
        assert n.ask_tick() == exp_bo
        assert n.ask_size() == exp_bo_sz


def test_listener_fires_only_on_actual_change():
    n = Nbbo(2)
    events = []
    n.listener(lambda bt, bs, at, asz, ts: events.append((bt, bs, at, asz, ts)))
    n.on_venue_quote(0, 100, 10, 105, 10, 1)
    assert len(events) == 1
    # Venue 1 quotes strictly off the inside: fast path should skip
    # recompute entirely (no NBBO change, no listener firing).
    n.on_venue_quote(1, 90, 5, 110, 5, 2)
    assert len(events) == 1
    assert n.change_count() == 1
    assert n.update_count() == 2


def test_on_venue_down_clears_that_venue_and_can_change_nbbo():
    n = Nbbo(2)
    n.on_venue_quote(0, 100, 10, 105, 10, 1)
    n.on_venue_quote(1, 102, 5, 104, 5, 2)
    assert n.bid_tick() == 102
    assert n.ask_tick() == 104
    changed = n.on_venue_down(1, 3)
    assert changed is True
    assert n.bid_tick() == 100
    assert n.ask_tick() == 105


def test_locked_and_crossed_detection():
    n = Nbbo(2)
    n.on_venue_quote(0, 100, 10, 100, 10, 1)
    assert n.locked() is True
    assert n.crossed() is False

    n2 = Nbbo(2)
    n2.on_venue_quote(0, 101, 10, 100, 10, 1)
    assert n2.crossed() is True
    assert n2.locked() is False


def test_mid_tick_is_nan_when_one_side_absent():
    n = Nbbo(1)
    assert math.isnan(n.mid_tick())
    n.on_venue_quote(0, 100, 10, 102, 10, 1)
    assert n.mid_tick() == pytest.approx(101.0)


def test_bid_and_ask_venue_bitmasks_aggregate_ties():
    n = Nbbo(3)
    n.on_venue_quote(0, 100, 10, 105, 5, 1)
    n.on_venue_quote(1, 100, 20, 105, 5, 2)   # ties both sides
    n.on_venue_quote(2, 95, 1, 110, 1, 3)     # off the inside both sides
    assert n.bid_tick() == 100
    assert n.bid_size() == 30
    assert n.bid_venues() == 0b011
    assert n.ask_tick() == 105
    assert n.ask_size() == 10
    assert n.ask_venues() == 0b011
