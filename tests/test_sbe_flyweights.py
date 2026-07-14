"""SBE-style flyweight tests, pinning Java ``sbe`` package byte
offsets: exact little-endian field layout for each message, block
lengths, and the type-discriminator dispatch.
"""

import math
import struct

import pytest

from quantfinlib.microstructure.execution import Side
from quantfinlib.sbe import OrderFlyweight, QuoteFlyweight, TradeFlyweight


def test_trade_flyweight_block_length_and_type():
    assert TradeFlyweight.BLOCK_LENGTH == 32
    assert TradeFlyweight.MESSAGE_TYPE == 1


def test_trade_flyweight_offsets_match_java_layout():
    buf = bytearray(TradeFlyweight.BLOCK_LENGTH)
    TradeFlyweight().wrap(buf, 0).encode(symbol_id=7, price=1.2345,
                                         size=100.0, timestamp_nanos=123456789)
    assert struct.unpack_from("<i", buf, 0)[0] == 1          # messageType
    assert struct.unpack_from("<i", buf, 4)[0] == 7           # symbolId
    assert struct.unpack_from("<d", buf, 8)[0] == pytest.approx(1.2345)
    assert struct.unpack_from("<d", buf, 16)[0] == pytest.approx(100.0)
    assert struct.unpack_from("<q", buf, 24)[0] == 123456789


def test_trade_flyweight_round_trip_via_getters():
    buf = bytearray(TradeFlyweight.BLOCK_LENGTH)
    tf = TradeFlyweight().wrap(buf, 0).encode(3, 99.5, 10.0, 42)
    assert tf.symbol_id() == 3
    assert tf.price() == pytest.approx(99.5)
    assert tf.size() == pytest.approx(10.0)
    assert tf.timestamp_nanos() == 42
    assert TradeFlyweight.type_at(buf, 0) == TradeFlyweight.MESSAGE_TYPE


def test_order_flyweight_block_length_and_type():
    assert OrderFlyweight.BLOCK_LENGTH == 44
    assert OrderFlyweight.MESSAGE_TYPE == 2


def test_order_flyweight_offsets_match_java_layout():
    buf = bytearray(OrderFlyweight.BLOCK_LENGTH)
    OrderFlyweight().wrap(buf, 0).encode(order_id=999, symbol_id=3,
                                         side=Side.SELL, quantity=500,
                                         price=math.nan, timestamp_nanos=42)
    assert struct.unpack_from("<i", buf, 0)[0] == 2           # messageType
    assert struct.unpack_from("<q", buf, 4)[0] == 999          # orderId
    assert struct.unpack_from("<i", buf, 12)[0] == 3           # symbolId
    assert buf[16] == 1                                        # side: SELL
    assert struct.unpack_from("<q", buf, 20)[0] == 500          # quantity
    assert math.isnan(struct.unpack_from("<d", buf, 28)[0])     # price
    assert struct.unpack_from("<q", buf, 36)[0] == 42           # timestampNanos


def test_order_flyweight_side_round_trips_buy_and_sell():
    buf = bytearray(OrderFlyweight.BLOCK_LENGTH)
    of = OrderFlyweight().wrap(buf, 0)
    of.encode(1, 0, Side.BUY, 1, 1.0, 0)
    assert of.side() == Side.BUY
    assert buf[16] == 0
    of.encode(1, 0, Side.SELL, 1, 1.0, 0)
    assert of.side() == Side.SELL
    assert buf[16] == 1


def test_order_flyweight_market_order_price_is_nan():
    buf = bytearray(OrderFlyweight.BLOCK_LENGTH)
    of = OrderFlyweight().wrap(buf, 0).encode(1, 0, Side.BUY, 100, math.nan, 0)
    assert math.isnan(of.price())


def test_quote_flyweight_block_length_and_type():
    assert QuoteFlyweight.BLOCK_LENGTH == 48
    assert QuoteFlyweight.MESSAGE_TYPE == 3


def test_quote_flyweight_offsets_match_java_layout():
    buf = bytearray(QuoteFlyweight.BLOCK_LENGTH)
    QuoteFlyweight().wrap(buf, 0).encode(symbol_id=5, bid_price=1.10,
                                         bid_size=10.0, ask_price=1.12,
                                         ask_size=20.0, timestamp_nanos=99)
    assert struct.unpack_from("<i", buf, 0)[0] == 3            # messageType
    assert struct.unpack_from("<i", buf, 4)[0] == 5             # symbolId
    assert struct.unpack_from("<d", buf, 8)[0] == pytest.approx(1.10)
    assert struct.unpack_from("<d", buf, 16)[0] == pytest.approx(10.0)
    assert struct.unpack_from("<d", buf, 24)[0] == pytest.approx(1.12)
    assert struct.unpack_from("<d", buf, 32)[0] == pytest.approx(20.0)
    assert struct.unpack_from("<q", buf, 40)[0] == 99


def test_quote_flyweight_one_sided_quote_carries_nan():
    buf = bytearray(QuoteFlyweight.BLOCK_LENGTH)
    qf = QuoteFlyweight().wrap(buf, 0).encode(1, math.nan, math.nan, 1.5, 10.0, 0)
    assert math.isnan(qf.bid_price())
    assert math.isnan(qf.bid_size())
    assert qf.ask_price() == pytest.approx(1.5)


def test_flyweights_can_be_wrapped_at_nonzero_offset_in_a_shared_buffer():
    buf = bytearray(200)
    TradeFlyweight().wrap(buf, 0).encode(1, 1.0, 1.0, 1)
    QuoteFlyweight().wrap(buf, TradeFlyweight.BLOCK_LENGTH).encode(
        2, 1.1, 5.0, 1.2, 5.0, 2)
    tf = TradeFlyweight().wrap(buf, 0)
    qf = QuoteFlyweight().wrap(buf, TradeFlyweight.BLOCK_LENGTH)
    assert tf.symbol_id() == 1
    assert qf.symbol_id() == 2
    assert TradeFlyweight.type_at(buf, 0) == TradeFlyweight.MESSAGE_TYPE
    assert QuoteFlyweight.type_at(buf, TradeFlyweight.BLOCK_LENGTH) == QuoteFlyweight.MESSAGE_TYPE
