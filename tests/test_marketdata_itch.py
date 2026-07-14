"""ITCH codec tests, pinning Java ``marketdata.ItchCodec``: message
round trips (encode -> decode via the flyweight view), the signed price
ceiling, and the 8-char symbol pack/unpack truncation.
"""

import pytest

from quantfinlib.marketdata import itch_codec


def test_pack_stock_pads_and_truncates_to_eight_chars():
    packed = itch_codec.pack_stock("AAPL")
    assert itch_codec.unpack_stock(packed) == "AAPL"
    # A 12-char symbol is truncated to its first 8 chars on pack.
    long_packed = itch_codec.pack_stock("ABCDEFGHIJKL")
    assert itch_codec.unpack_stock(long_packed) == "ABCDEFGH"


def test_pack_stock_round_trips_exactly_eight_chars():
    packed = itch_codec.pack_stock("ABCDEFGH")
    assert itch_codec.unpack_stock(packed) == "ABCDEFGH"


def test_length_of_each_message_type():
    assert itch_codec.length(itch_codec.ADD) == 36
    assert itch_codec.length(itch_codec.ADD_MPID) == 40
    assert itch_codec.length(itch_codec.EXECUTED) == 31
    assert itch_codec.length(itch_codec.CANCEL) == 23
    assert itch_codec.length(itch_codec.DELETE) == 19
    assert itch_codec.length(itch_codec.REPLACE) == 35
    assert itch_codec.length(itch_codec.TRADE) == 44
    assert itch_codec.length(ord("Z")) == -1


def test_add_message_round_trip():
    buf = bytearray(64)
    n = itch_codec.encode_add(buf, 0, stock_locate=7, timestamp_nanos=123456789,
                              order_ref=555, side=itch_codec.BUY, shares=1000,
                              packed_stock=itch_codec.pack_stock("AAPL"),
                              price_tick=1_500_000)
    assert n == 36
    v = itch_codec.ItchView().wrap(bytes(buf), 0)
    assert v.type() == itch_codec.ADD
    assert v.stock_locate() == 7
    assert v.timestamp_nanos() == 123456789
    assert v.order_ref() == 555
    assert v.side() == itch_codec.BUY
    assert v.shares() == 1000
    assert itch_codec.unpack_stock(v.stock()) == "AAPL"
    assert v.price_tick() == 1_500_000


def test_executed_message_round_trip():
    buf = bytearray(64)
    n = itch_codec.encode_executed(buf, 0, stock_locate=1, timestamp_nanos=99,
                                   order_ref=42, executed_shares=100,
                                   match_number=777)
    assert n == 31
    v = itch_codec.ItchView().wrap(bytes(buf), 0)
    assert v.type() == itch_codec.EXECUTED
    assert v.order_ref() == 42
    assert v.delta_shares() == 100
    assert v.match_number() == 777


def test_cancel_message_round_trip():
    buf = bytearray(64)
    itch_codec.encode_cancel(buf, 0, 1, 1, 42, 50)
    v = itch_codec.ItchView().wrap(bytes(buf), 0)
    assert v.type() == itch_codec.CANCEL
    assert v.order_ref() == 42
    assert v.delta_shares() == 50


def test_delete_message_round_trip():
    buf = bytearray(64)
    itch_codec.encode_delete(buf, 0, 1, 1, 42)
    v = itch_codec.ItchView().wrap(bytes(buf), 0)
    assert v.type() == itch_codec.DELETE
    assert v.order_ref() == 42


def test_replace_message_round_trip():
    buf = bytearray(64)
    itch_codec.encode_replace(buf, 0, 1, 1, orig_ref=42, new_ref=43,
                              shares=200, price_tick=1_600_000)
    v = itch_codec.ItchView().wrap(bytes(buf), 0)
    assert v.type() == itch_codec.REPLACE
    assert v.orig_ref() == 42
    assert v.new_ref() == 43
    assert v.shares() == 200
    assert v.price_tick() == 1_600_000


def test_trade_message_round_trip():
    buf = bytearray(64)
    itch_codec.encode_trade(buf, 0, 1, 1, order_ref=42, side=itch_codec.SELL,
                            shares=300, packed_stock=itch_codec.pack_stock("MSFT"),
                            price_tick=2_000_000, match_number=9)
    v = itch_codec.ItchView().wrap(bytes(buf), 0)
    assert v.type() == itch_codec.TRADE
    assert v.side() == itch_codec.SELL
    assert v.shares() == 300
    assert itch_codec.unpack_stock(v.stock()) == "MSFT"
    assert v.price_tick() == 2_000_000
    assert v.match_number() == 9


def test_price_tick_ceiling_decodes_negative_above_signed_32bit_max():
    # Domain limit documented on ItchCodec: wire prices above 2^31-1
    # ticks decode negative under this port's signed-int semantics.
    buf = bytearray(64)
    huge_price = (1 << 31) + 100     # just past INT32_MAX
    itch_codec.encode_add(buf, 0, 1, 1, 1, itch_codec.BUY, 1,
                          itch_codec.pack_stock("X"), huge_price)
    v = itch_codec.ItchView().wrap(bytes(buf), 0)
    assert v.price_tick() < 0


def test_price_tick_at_signed_32bit_max_is_still_positive():
    buf = bytearray(64)
    max_price = (1 << 31) - 1
    itch_codec.encode_add(buf, 0, 1, 1, 1, itch_codec.BUY, 1,
                          itch_codec.pack_stock("X"), max_price)
    v = itch_codec.ItchView().wrap(bytes(buf), 0)
    assert v.price_tick() == max_price


def test_view_wrap_returns_self_for_chaining():
    buf = bytearray(64)
    itch_codec.encode_delete(buf, 0, 1, 1, 42)
    view = itch_codec.ItchView()
    assert view.wrap(bytes(buf), 0) is view
