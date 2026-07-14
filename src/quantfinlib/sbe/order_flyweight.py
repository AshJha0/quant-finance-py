"""SBE-style flyweight codec for an order-entry message (port of Java
``sbe.OrderFlyweight``) -- the binary counterpart of a FIX
NewOrderSingle, at fixed offsets (see :mod:`~quantfinlib.sbe.trade_flyweight`
for the pattern).

Wire layout (little-endian, 44 bytes; 3 bytes padding keep the 64-bit
fields naturally aligned)::

    offset  0  int32   messageType    = 2
    offset  4  int64   orderId
    offset 12  int32   symbolId
    offset 16  int8    side           (0 = BUY, 1 = SELL)
    offset 17  int8[3] padding
    offset 20  int64   quantity
    offset 28  double  price          (NaN = market order)
    offset 36  int64   timestampNanos
"""

from __future__ import annotations

import struct

from quantfinlib.microstructure.execution import Side

MESSAGE_TYPE = 2
BLOCK_LENGTH = 44

_TYPE_OFFSET = 0
_ORDER_ID_OFFSET = 4
_SYMBOL_OFFSET = 12
_SIDE_OFFSET = 16
_QUANTITY_OFFSET = 20
_PRICE_OFFSET = 28
_TIMESTAMP_OFFSET = 36


class OrderFlyweight:
    """Flyweight over one order message in a caller-owned buffer."""

    __slots__ = ("_buffer", "_offset")

    MESSAGE_TYPE = MESSAGE_TYPE
    BLOCK_LENGTH = BLOCK_LENGTH

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()
        self._offset = 0

    def wrap(self, buffer: bytearray, offset: int) -> "OrderFlyweight":
        self._buffer = buffer
        self._offset = offset
        return self

    def encode(self, order_id: int, symbol_id: int, side: Side, quantity: int,
              price: float, timestamp_nanos: int) -> "OrderFlyweight":
        """Encodes a full order message at the wrap position (writes
        the type header)."""
        o = self._offset
        struct.pack_into("<i", self._buffer, o + _TYPE_OFFSET, MESSAGE_TYPE)
        struct.pack_into("<q", self._buffer, o + _ORDER_ID_OFFSET, order_id)
        struct.pack_into("<i", self._buffer, o + _SYMBOL_OFFSET, symbol_id)
        self._buffer[o + _SIDE_OFFSET] = 0 if side == Side.BUY else 1
        struct.pack_into("<q", self._buffer, o + _QUANTITY_OFFSET, quantity)
        struct.pack_into("<d", self._buffer, o + _PRICE_OFFSET, price)
        struct.pack_into("<q", self._buffer, o + _TIMESTAMP_OFFSET, timestamp_nanos)
        return self

    def order_id(self) -> int:
        return struct.unpack_from("<q", self._buffer, self._offset + _ORDER_ID_OFFSET)[0]

    def symbol_id(self) -> int:
        return struct.unpack_from("<i", self._buffer, self._offset + _SYMBOL_OFFSET)[0]

    def side(self) -> Side:
        return Side.BUY if self._buffer[self._offset + _SIDE_OFFSET] == 0 else Side.SELL

    def quantity(self) -> int:
        return struct.unpack_from("<q", self._buffer, self._offset + _QUANTITY_OFFSET)[0]

    def price(self) -> float:
        return struct.unpack_from("<d", self._buffer, self._offset + _PRICE_OFFSET)[0]

    def timestamp_nanos(self) -> int:
        return struct.unpack_from("<q", self._buffer, self._offset + _TIMESTAMP_OFFSET)[0]
