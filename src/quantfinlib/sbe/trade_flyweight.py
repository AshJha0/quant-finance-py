"""SBE-style flyweight codec for a market-data trade message (port of
Java ``sbe.TradeFlyweight``): fixed field offsets over a caller-owned
``bytearray``, so encode/decode is a handful of absolute primitive
reads/writes -- no parsing, no copying. This is the wire format real
HFT feeds use (ITCH/SBE family), as opposed to the text protocols
(JSON/FIX tag-value) of the retail edges.

Wire layout (little-endian, 32 bytes)::

    offset  0  int32   messageType   = 1
    offset  4  int32   symbolId          (dense id shared by both ends)
    offset  8  double  price
    offset 16  double  size
    offset 24  int64   timestampNanos    (exchange event time)

Usage: ``wrap(buffer, offset)`` then read/write fields. The flyweight
holds no state besides the wrap position -- reuse one instance for
millions of messages.
"""

from __future__ import annotations

import struct

MESSAGE_TYPE = 1
BLOCK_LENGTH = 32

_TYPE_OFFSET = 0
_SYMBOL_OFFSET = 4
_PRICE_OFFSET = 8
_SIZE_OFFSET = 16
_TIMESTAMP_OFFSET = 24


class TradeFlyweight:
    """Flyweight over one trade message in a caller-owned buffer."""

    __slots__ = ("_buffer", "_offset")

    MESSAGE_TYPE = MESSAGE_TYPE
    BLOCK_LENGTH = BLOCK_LENGTH

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()
        self._offset = 0

    def wrap(self, buffer: bytearray, offset: int) -> "TradeFlyweight":
        """Positions this flyweight over ``buffer`` at ``offset``."""
        self._buffer = buffer
        self._offset = offset
        return self

    def encode(self, symbol_id: int, price: float, size: float,
              timestamp_nanos: int) -> "TradeFlyweight":
        """Encodes a full trade message at the wrap position (writes
        the type header)."""
        o = self._offset
        struct.pack_into("<i", self._buffer, o + _TYPE_OFFSET, MESSAGE_TYPE)
        struct.pack_into("<i", self._buffer, o + _SYMBOL_OFFSET, symbol_id)
        struct.pack_into("<d", self._buffer, o + _PRICE_OFFSET, price)
        struct.pack_into("<d", self._buffer, o + _SIZE_OFFSET, size)
        struct.pack_into("<q", self._buffer, o + _TIMESTAMP_OFFSET, timestamp_nanos)
        return self

    def symbol_id(self) -> int:
        return struct.unpack_from("<i", self._buffer, self._offset + _SYMBOL_OFFSET)[0]

    def price(self) -> float:
        return struct.unpack_from("<d", self._buffer, self._offset + _PRICE_OFFSET)[0]

    def size(self) -> float:
        return struct.unpack_from("<d", self._buffer, self._offset + _SIZE_OFFSET)[0]

    def timestamp_nanos(self) -> int:
        return struct.unpack_from("<q", self._buffer, self._offset + _TIMESTAMP_OFFSET)[0]

    @staticmethod
    def type_at(buffer, offset: int) -> int:
        """Reads the message-type discriminator at ``offset`` without
        wrapping a flyweight."""
        return struct.unpack_from("<i", buffer, offset)[0]
