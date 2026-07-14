"""Multi-venue aggregation, ported from Java AggregatedBookTest.

Composite BBO with venue attribution, venue pulls, crossed-composite
reporting, and sweepable-size sums. (The JVM allocation-benchmark test
is not ported.)
"""

import math

import pytest

from quantfinlib.fx import AggregatedBook


def test_composite_tracks_best_across_venues():
    book = AggregatedBook(3)
    book.on_quote(0, 1.08500, 1_000_000, 1.08520, 1_000_000)
    book.on_quote(1, 1.08505, 2_000_000, 1.08515, 500_000)   # best both sides
    book.on_quote(2, 1.08490, 5_000_000, 1.08530, 3_000_000)

    assert book.best_bid() == 1.08505
    assert book.best_ask() == 1.08515
    assert book.best_bid_venue() == 1
    assert book.best_ask_venue() == 1
    assert book.best_bid_size() == 2_000_000
    assert book.best_ask_size() == 500_000
    assert book.mid() == pytest.approx((1.08505 + 1.08515) / 2, abs=1e-12)
    assert book.spread() == pytest.approx(1.08515 - 1.08505, abs=1e-12)
    assert not book.is_crossed()
    assert book.update_count() == 3
    assert book.venue_count() == 3


def test_pulling_the_best_venue_promotes_the_next():
    book = AggregatedBook(2)
    book.on_quote(0, 1.0850, 1_000_000, 1.0852, 1_000_000)
    book.on_quote(1, 1.0851, 2_000_000, 1.0853, 2_000_000)
    assert book.best_bid_venue() == 1
    # Venue 1 disconnects: venue 0 must own the composite again.
    book.clear(1)
    assert book.best_bid_venue() == 0
    assert book.best_bid() == 1.0850
    assert book.best_ask() == 1.0852
    # Empty book: NaN composite, no venue, zero sizes.
    book.clear(0)
    assert math.isnan(book.best_bid())
    assert book.best_bid_venue() == -1
    assert book.best_bid_size() == 0
    assert book.total_bid_size_at_best(0) == 0


def test_crossed_composites_are_reported_not_hidden():
    book = AggregatedBook(2)
    # Venue 1's stale bid crosses venue 0's fresh ask -- real e-FX life.
    book.on_quote(0, 1.0850, 1_000_000, 1.0852, 1_000_000)
    book.on_quote(1, 1.0853, 1_000_000, 1.0855, 1_000_000)
    assert book.is_crossed()
    assert book.best_bid() == 1.0853
    assert book.best_ask() == 1.0852
    assert book.spread() < 0


def test_sweepable_size_sums_venues_within_tolerance():
    book = AggregatedBook(3)
    book.on_quote(0, 1.08500, 1_000_000, 1.08520, 1_000_000)
    book.on_quote(1, 1.08500, 2_000_000, 1.08521, 500_000)
    book.on_quote(2, 1.08490, 4_000_000, 1.08540, 100_000)
    # Exactly at best: venues 0 and 1 on the bid.
    assert book.total_bid_size_at_best(0) == pytest.approx(3_000_000, abs=1e-9)
    # One pip of tolerance pulls venue 2 in too.
    assert book.total_bid_size_at_best(0.0001) == pytest.approx(7_000_000, abs=1e-9)
    # Ask side: half-pip tolerance covers 1.08520 and 1.08521.
    assert book.total_ask_size_at_best(0.00005) == pytest.approx(1_500_000, abs=1e-9)


def test_one_sided_quotes_participate_on_their_side_only():
    book = AggregatedBook(2)
    book.on_quote(0, 1.0850, 1_000_000, math.nan, 0)   # bid-only venue
    book.on_quote(1, math.nan, 0, 1.0853, 2_000_000)   # ask-only venue
    assert book.best_bid() == 1.0850
    assert book.best_ask() == 1.0853
    assert book.best_bid_venue() == 0
    assert book.best_ask_venue() == 1
    assert not book.is_crossed()


def test_validation():
    with pytest.raises(ValueError):
        AggregatedBook(0)
