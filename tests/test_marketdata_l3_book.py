"""L3BookBuilder tests, pinning Java ``marketdata.L3BookBuilder``:
exact FIFO queue-position tracking (init walk + O(1) maintenance),
best bid/ask/depth queries, and the unknown/out-of-band/duplicate-ref
counters.
"""

import pytest

from quantfinlib.marketdata import itch_codec
from quantfinlib.marketdata.l3_book_builder import L3BookBuilder
from quantfinlib.microstructure.execution import Side


def make_book(max_orders=32):
    return L3BookBuilder(stock_locate=1, min_price_tick=900_000,
                         max_price_tick=1_100_000, max_orders=max_orders)


def test_constructor_validates_tick_range_and_capacity():
    with pytest.raises(ValueError):
        L3BookBuilder(1, 1_000_000, 900_000, 10)
    with pytest.raises(ValueError):
        L3BookBuilder(1, 900_000, 1_000_000, 0)


def test_best_bid_ask_track_the_inside_after_adds():
    book = make_book()
    assert book.on_add(1, Side.BUY, 100, 1_000_000)
    assert book.on_add(2, Side.BUY, 200, 999_000)
    assert book.on_add(3, Side.SELL, 150, 1_001_000)
    assert book.on_add(4, Side.SELL, 50, 1_002_000)

    assert book.best_bid_tick() == 1_000_000
    assert book.best_bid_size() == 100
    assert book.best_ask_tick() == 1_001_000
    assert book.best_ask_size() == 150
    assert book.add_count() == 4


def test_no_bid_or_ask_sentinels_when_book_empty():
    book = make_book()
    assert book.best_bid_tick() == -(1 << 31)
    assert book.best_ask_tick() == (1 << 31) - 1
    assert book.best_bid_size() == 0
    assert book.best_ask_size() == 0


def test_queue_position_initial_walk_and_o1_maintenance():
    book = make_book()
    book.on_add(1, Side.BUY, 500, 1_000_000)
    book.on_add(2, Side.BUY, 300, 1_000_000)   # arrives after ref 1
    book.on_add(3, Side.BUY, 100, 1_000_000)   # arrives after ref 2

    assert book.track(2) is True
    assert book.shares_ahead(2) == 500          # only ref 1 is ahead

    assert book.track(3) is True
    assert book.shares_ahead(3) == 800           # refs 1 and 2 are ahead

    # An execution against the head (ref 1) is ahead of both trackers.
    book.on_execute(1, 200)
    assert book.shares_ahead(2) == 300
    assert book.shares_ahead(3) == 600

    # A cancel of ref 1's remainder finishes clearing it out.
    book.on_cancel(1, 300)
    assert book.shares_ahead(2) == 0
    assert book.shares_ahead(3) == 300


def test_cancel_only_credits_trackers_it_actually_preceded():
    book = make_book()
    book.on_add(1, Side.BUY, 100, 1_000_000)
    book.on_add(2, Side.BUY, 100, 1_000_000)   # tracked after this point
    assert book.track(2) is True
    assert book.shares_ahead(2) == 100
    book.on_add(3, Side.BUY, 50, 1_000_000)    # arrives AFTER ref 2 in queue
    # Cancelling ref 3 (which came after ref 2) must not reduce ref 2's
    # shares-ahead -- it never queued ahead of it.
    book.on_cancel(3, 50)
    assert book.shares_ahead(2) == 100


def test_tracking_ends_on_fill_delete_and_replace():
    book = make_book()
    book.on_add(1, Side.BUY, 100, 1_000_000)
    book.track(1)
    assert book.shares_ahead(1) == 0
    book.on_execute(1, 100)   # fully filled -> tracking auto-ends
    assert book.shares_ahead(1) == -1

    book.on_add(2, Side.BUY, 100, 1_000_000)
    book.track(2)
    book.on_delete(2)
    assert book.shares_ahead(2) == -1

    book.on_add(3, Side.BUY, 100, 1_000_000)
    book.track(3)
    book.on_replace(3, 4, 100, 1_000_500)
    assert book.shares_ahead(3) == -1
    # The replacement gets a fresh (back-of-queue) position, not tracked.
    assert book.shares_ahead(4) == -1


def test_track_is_idempotent():
    book = make_book()
    book.on_add(1, Side.BUY, 100, 1_000_000)
    assert book.track(1) is True
    assert book.track(1) is True   # retried ack: no duplicate row
    book.on_execute(1, 50)
    assert book.open_quantity(1) == 50


def test_track_returns_false_for_unknown_ref():
    book = make_book()
    assert book.track(999) is False
    assert book.shares_ahead(999) == -1


def test_out_of_band_price_is_rejected():
    book = make_book()
    assert book.on_add(1, Side.BUY, 100, 500_000) is False   # below min tick
    assert book.out_of_band_count() == 1
    assert book.resting_orders() == 0


def test_duplicate_ref_is_rejected():
    book = make_book()
    assert book.on_add(1, Side.BUY, 100, 1_000_000) is True
    assert book.on_add(1, Side.SELL, 50, 1_001_000) is False
    assert book.duplicate_ref_count() == 1
    # The original order is untouched.
    assert book.open_quantity(1) == 100


def test_unknown_ref_events_are_counted_not_raised():
    book = make_book()
    book.on_execute(999, 10)
    book.on_cancel(999, 10)
    book.on_delete(999)
    book.on_replace(999, 1000, 10, 1_000_000)
    assert book.unknown_ref_count() == 4


def test_pool_exhaustion_counts_as_out_of_band():
    book = make_book(max_orders=2)
    assert book.on_add(1, Side.BUY, 10, 1_000_000) is True
    assert book.on_add(2, Side.BUY, 10, 1_000_000) is True
    assert book.on_add(3, Side.BUY, 10, 1_000_000) is False
    assert book.out_of_band_count() == 1


def test_snapshot_returns_best_first_depth():
    book = make_book()
    book.on_add(1, Side.BUY, 100, 1_000_000)
    book.on_add(2, Side.BUY, 50, 999_000)
    book.on_add(3, Side.BUY, 25, 998_000)
    levels = book.snapshot(Side.BUY, 10)
    assert levels == [(1_000_000, 100), (999_000, 50), (998_000, 25)]
    assert book.snapshot(Side.BUY, 2) == [(1_000_000, 100), (999_000, 50)]


def test_on_trade_records_last_trade_tick_without_touching_book():
    book = make_book()
    book.on_add(1, Side.BUY, 100, 1_000_000)
    book.on_trade(1_050_000)
    assert book.trade_count() == 1
    assert book.last_trade_tick() == 1_050_000
    assert book.best_bid_tick() == 1_000_000   # book unaffected


def test_on_message_dispatches_and_ignores_other_stock_locate():
    book = make_book()
    buf = bytearray(64)
    n = itch_codec.encode_add(buf, 0, stock_locate=1, timestamp_nanos=1,
                              order_ref=1, side=itch_codec.BUY, shares=100,
                              packed_stock=itch_codec.pack_stock("X"),
                              price_tick=1_000_000)
    assert book.on_message(bytes(buf), 0) == n
    assert book.best_bid_tick() == 1_000_000

    other_buf = bytearray(64)
    itch_codec.encode_add(other_buf, 0, stock_locate=2, timestamp_nanos=1,
                          order_ref=2, side=itch_codec.BUY, shares=100,
                          packed_stock=itch_codec.pack_stock("Y"),
                          price_tick=1_000_000)
    assert book.on_message(bytes(other_buf), 0) == 0
    assert book.add_count() == 1


def test_replace_moves_order_to_new_level_and_loses_priority():
    book = make_book()
    book.on_add(1, Side.BUY, 100, 1_000_000)
    book.on_add(2, Side.BUY, 100, 1_000_000)
    book.on_replace(1, 3, 150, 999_500)
    assert book.qty_at_tick(Side.BUY, 1_000_000) == 100   # ref 2 remains
    assert book.qty_at_tick(Side.BUY, 999_500) == 150
    assert book.replace_count() == 1
