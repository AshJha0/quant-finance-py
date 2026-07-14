"""SBE-style flyweight codec for a two-sided quote message (port of
Java ``sbe.QuoteFlyweight``) -- the outbound format of a market maker
(and the inbound format of venue top-of-book feeds), completing the
binary codec family: trade in, order out, quote (two-sided) out.

Wire layout (little-endian, 48 bytes)::

    offset  0  int32   messageType   = 3
    offset  4  int32   symbolId          (dense id shared by both ends)
    offset  8  double  bidPrice
    offset 16  double  bidSize
    offset 24  double  askPrice
    offset 32  double  askSize
    offset 40  int64   timestampNanos    (quote creation time)

One-sided quotes carry NaN on the pulled side -- the same convention
``fx.AggregatedBook`` consumes.
"""

from __future__ import annotations

import struct

MESSAGE_TYPE = 3
BLOCK_LENGTH = 48

_TYPE_OFFSET = 0
_SYMBOL_OFFSET = 4
_BID_OFFSET = 8
_BID_SIZE_OFFSET = 16
_ASK_OFFSET = 24
_ASK_SIZE_OFFSET = 32
_TIMESTAMP_OFFSET = 40


class QuoteFlyweight:
    """Flyweight over one quote message in a caller-owned buffer."""

    __slots__ = ("_buffer", "_offset")

    MESSAGE_TYPE = MESSAGE_TYPE
    BLOCK_LENGTH = BLOCK_LENGTH

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()
        self._offset = 0

    def wrap(self, buffer: bytearray, offset: int) -> "QuoteFlyweight":
        self._buffer = buffer
        self._offset = offset
        return self

    def encode(self, symbol_id: int, bid_price: float, bid_size: float,
              ask_price: float, ask_size: float,
              timestamp_nanos: int) -> "QuoteFlyweight":
        """Encodes a full quote message at the wrap position (writes
        the type header)."""
        o = self._offset
        struct.pack_into("<i", self._buffer, o + _TYPE_OFFSET, MESSAGE_TYPE)
        struct.pack_into("<i", self._buffer, o + _SYMBOL_OFFSET, symbol_id)
        struct.pack_into("<d", self._buffer, o + _BID_OFFSET, bid_price)
        struct.pack_into("<d", self._buffer, o + _BID_SIZE_OFFSET, bid_size)
        struct.pack_into("<d", self._buffer, o + _ASK_OFFSET, ask_price)
        struct.pack_into("<d", self._buffer, o + _ASK_SIZE_OFFSET, ask_size)
        struct.pack_into("<q", self._buffer, o + _TIMESTAMP_OFFSET, timestamp_nanos)
        return self

    def symbol_id(self) -> int:
        return struct.unpack_from("<i", self._buffer, self._offset + _SYMBOL_OFFSET)[0]

    def bid_price(self) -> float:
        return struct.unpack_from("<d", self._buffer, self._offset + _BID_OFFSET)[0]

    def bid_size(self) -> float:
        return struct.unpack_from("<d", self._buffer, self._offset + _BID_SIZE_OFFSET)[0]

    def ask_price(self) -> float:
        return struct.unpack_from("<d", self._buffer, self._offset + _ASK_OFFSET)[0]

    def ask_size(self) -> float:
        return struct.unpack_from("<d", self._buffer, self._offset + _ASK_SIZE_OFFSET)[0]

    def timestamp_nanos(self) -> int:
        return struct.unpack_from("<q", self._buffer, self._offset + _TIMESTAMP_OFFSET)[0]

    @staticmethod
    def type_at(buffer, offset: int) -> int:
        """Reads the type discriminator without wrapping a flyweight."""
        return struct.unpack_from("<i", buffer, offset)[0]
