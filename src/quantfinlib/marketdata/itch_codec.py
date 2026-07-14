"""ITCH 5.0-style binary market-data codec (port of Java
``marketdata.ItchCodec``): the message subset that drives a full-depth
(L3) book -- add, add-with-attribution, execute, cancel, delete,
replace, and off-book trade -- with the exact field layout and
big-endian encoding of the Nasdaq TotalView-ITCH 5.0 specification.
Styled after the spec for realism; not a certified implementation.

Decoding is a flyweight: :meth:`ItchView.wrap` points at a message
inside a caller-owned buffer and every getter reads the bytes directly
-- no parsing step. Symbols travel as an int of 8 ASCII bytes
(:func:`pack_stock`) so the hot path never touches a Python string.

Prices are unsigned 32-bit integers with four implied decimals -- i.e.
the raw value *is* the price in 0.0001 ticks, which plugs straight into
tick-indexed books like :class:`~quantfinlib.marketdata.l3_book_builder.L3BookBuilder`.
**Domain limit**: this port's tick-indexed pipeline uses signed 32-bit
semantics throughout (matching the Java ``int`` original), so prices
above 2^31-1 ticks ($214,748.36) decode negative and are dropped by
band checks downstream.

Encoders exist for simulators, replay tooling and tests; a production
participant only decodes.
"""

from __future__ import annotations

import struct

ADD = ord("A")
ADD_MPID = ord("F")
EXECUTED = ord("E")
CANCEL = ord("X")
DELETE = ord("D")
REPLACE = ord("U")
TRADE = ord("P")

BUY = ord("B")
SELL = ord("S")

_HAS_ORDER_REF = {ADD, ADD_MPID, EXECUTED, CANCEL, DELETE, REPLACE, TRADE}
_HAS_SIDE = {ADD, ADD_MPID, TRADE}
_HAS_SHARES = {ADD, ADD_MPID, TRADE, REPLACE}
_HAS_STOCK = {ADD, ADD_MPID, TRADE}
_HAS_PRICE = {ADD, ADD_MPID, TRADE, REPLACE}


def length(msg_type: int) -> int:
    """Wire length of a message type; -1 for types outside the subset."""
    return {
        ADD: 36,
        ADD_MPID: 40,
        EXECUTED: 31,
        CANCEL: 23,
        DELETE: 19,
        REPLACE: 35,
        TRADE: 44,
    }.get(msg_type, -1)


def pack_stock(symbol: str) -> int:
    """Packs up to 8 ASCII chars into a big-endian int, space-padded
    (ITCH alpha style)."""
    v = 0
    for i in range(8):
        c = ord(symbol[i]) if i < len(symbol) else ord(" ")
        v = (v << 8) | (c & 0xFF)
    return v


def unpack_stock(packed: int) -> str:
    """Inverse of :func:`pack_stock`: trailing spaces stripped.
    Test/logging use only."""
    chars = [chr((packed >> (i * 8)) & 0xFF) for i in range(7, -1, -1)]
    return "".join(chars).rstrip(" ")


class ItchView:
    """Mutable flyweight over one message in a caller-owned buffer.
    Reuse a single instance per decoding "thread"; every getter is a
    direct big-endian read. Getters are only meaningful for the
    message types that carry the field per the ITCH layout -- reading
    a field the type does not carry raises ``AssertionError`` (the
    Python stand-in for the Java ``assert``, always active here since
    these are cheap bookkeeping checks, not a hot per-tick cost)."""

    __slots__ = ("_buf", "_off")

    def __init__(self) -> None:
        self._buf: bytes = b""
        self._off = 0

    def wrap(self, buffer: bytes, offset: int) -> "ItchView":
        """Points this view at a message; returns self for chaining."""
        self._buf = buffer
        self._off = offset
        return self

    def type(self) -> int:
        return self._buf[self._off]

    def stock_locate(self) -> int:
        """Per-symbol locate code -- the feed's symbol id for the day."""
        return self._u16(self._off + 1)

    def tracking_number(self) -> int:
        return self._u16(self._off + 3)

    def timestamp_nanos(self) -> int:
        """Nanoseconds since midnight (48-bit wire field)."""
        return self._u48(self._off + 5)

    def order_ref(self) -> int:
        """Order reference (A/F/E/X/D/U/P, and the original ref of U
        via :meth:`orig_ref`)."""
        t = self._buf[self._off]
        assert t in _HAS_ORDER_REF, f"no orderRef on type {chr(t)}"
        return self._u64(self._off + 11)

    def orig_ref(self) -> int:
        """U only: the replaced (original) order reference."""
        t = self._buf[self._off]
        assert t == REPLACE, f"origRef only on U, not {chr(t)}"
        return self._u64(self._off + 11)

    def new_ref(self) -> int:
        """U only: the new order reference."""
        t = self._buf[self._off]
        assert t == REPLACE, f"newRef only on U, not {chr(t)}"
        return self._u64(self._off + 19)

    def side(self) -> int:
        """A/F/P: BUY or SELL."""
        t = self._buf[self._off]
        assert t in _HAS_SIDE, f"no side on type {chr(t)}"
        return self._buf[self._off + 19]

    def shares(self) -> int:
        """A/F/P: displayed shares. U: the new total shares."""
        t = self._buf[self._off]
        assert t in _HAS_SHARES, f"no shares on type {chr(t)}"
        return self._u32(self._off + 27) if t == REPLACE else self._u32(self._off + 20)

    def delta_shares(self) -> int:
        """E: executed shares. X: cancelled shares."""
        t = self._buf[self._off]
        assert t in (EXECUTED, CANCEL), f"no deltaShares on type {chr(t)}"
        return self._u32(self._off + 19)

    def stock(self) -> int:
        """A/F/P: symbol as 8 packed ASCII bytes (compare against
        :func:`pack_stock`)."""
        t = self._buf[self._off]
        assert t in _HAS_STOCK, f"no stock on type {chr(t)}"
        return self._u64(self._off + 24)

    def price_tick(self) -> int:
        """Price in 0.0001 ticks (A/F/P; U: the new price). Signed
        32-bit domain: wire prices above 2^31-1 ticks decode negative
        (see the module doc's domain-limit note)."""
        t = self._buf[self._off]
        assert t in _HAS_PRICE, f"no price on type {chr(t)}"
        raw = self._u32(self._off + 31) if t == REPLACE else self._u32(self._off + 32)
        return raw - (1 << 32) if raw >= (1 << 31) else raw

    def match_number(self) -> int:
        """E/P: the venue's match (execution) number."""
        t = self._buf[self._off]
        assert t in (EXECUTED, TRADE), f"no matchNumber on type {chr(t)}"
        return self._u64(self._off + 36) if t == TRADE else self._u64(self._off + 23)

    def _u16(self, i: int) -> int:
        return struct.unpack_from(">H", self._buf, i)[0]

    def _u32(self, i: int) -> int:
        return struct.unpack_from(">I", self._buf, i)[0]

    def _u48(self, i: int) -> int:
        return (self._u16(i) << 32) | self._u32(i + 2)

    def _u64(self, i: int) -> int:
        return struct.unpack_from(">Q", self._buf, i)[0]


# ----------------------------------------------------------------------
# Encoders (simulator / replay / test side)
# ----------------------------------------------------------------------

def _header(buf: bytearray, off: int, msg_type: int, locate: int, ts_nanos: int) -> None:
    buf[off] = msg_type
    struct.pack_into(">H", buf, off + 1, locate & 0xFFFF)
    buf[off + 3] = 0
    buf[off + 4] = 0
    buf[off + 5] = (ts_nanos >> 40) & 0xFF
    buf[off + 6] = (ts_nanos >> 32) & 0xFF
    struct.pack_into(">I", buf, off + 7, ts_nanos & 0xFFFFFFFF)


def encode_add(buf: bytearray, off: int, stock_locate: int, timestamp_nanos: int,
              order_ref: int, side: int, shares: int, packed_stock: int,
              price_tick: int) -> int:
    """Encodes an Add Order (A); returns bytes written."""
    _header(buf, off, ADD, stock_locate, timestamp_nanos)
    struct.pack_into(">Q", buf, off + 11, order_ref)
    buf[off + 19] = side
    struct.pack_into(">I", buf, off + 20, shares)
    struct.pack_into(">Q", buf, off + 24, packed_stock)
    struct.pack_into(">I", buf, off + 32, price_tick & 0xFFFFFFFF)
    return 36


def encode_executed(buf: bytearray, off: int, stock_locate: int, timestamp_nanos: int,
                    order_ref: int, executed_shares: int, match_number: int) -> int:
    """Encodes an Order Executed (E); returns bytes written."""
    _header(buf, off, EXECUTED, stock_locate, timestamp_nanos)
    struct.pack_into(">Q", buf, off + 11, order_ref)
    struct.pack_into(">I", buf, off + 19, executed_shares)
    struct.pack_into(">Q", buf, off + 23, match_number)
    return 31


def encode_cancel(buf: bytearray, off: int, stock_locate: int, timestamp_nanos: int,
                  order_ref: int, cancelled_shares: int) -> int:
    """Encodes an Order Cancel (X, partial cancel); returns bytes written."""
    _header(buf, off, CANCEL, stock_locate, timestamp_nanos)
    struct.pack_into(">Q", buf, off + 11, order_ref)
    struct.pack_into(">I", buf, off + 19, cancelled_shares)
    return 23


def encode_delete(buf: bytearray, off: int, stock_locate: int, timestamp_nanos: int,
                  order_ref: int) -> int:
    """Encodes an Order Delete (D); returns bytes written."""
    _header(buf, off, DELETE, stock_locate, timestamp_nanos)
    struct.pack_into(">Q", buf, off + 11, order_ref)
    return 19


def encode_replace(buf: bytearray, off: int, stock_locate: int, timestamp_nanos: int,
                   orig_ref: int, new_ref: int, shares: int, price_tick: int) -> int:
    """Encodes an Order Replace (U); returns bytes written."""
    _header(buf, off, REPLACE, stock_locate, timestamp_nanos)
    struct.pack_into(">Q", buf, off + 11, orig_ref)
    struct.pack_into(">Q", buf, off + 19, new_ref)
    struct.pack_into(">I", buf, off + 27, shares)
    struct.pack_into(">I", buf, off + 31, price_tick & 0xFFFFFFFF)
    return 35


def encode_trade(buf: bytearray, off: int, stock_locate: int, timestamp_nanos: int,
                 order_ref: int, side: int, shares: int, packed_stock: int,
                 price_tick: int, match_number: int) -> int:
    """Encodes a non-cross Trade (P); returns bytes written."""
    _header(buf, off, TRADE, stock_locate, timestamp_nanos)
    struct.pack_into(">Q", buf, off + 11, order_ref)
    buf[off + 19] = side
    struct.pack_into(">I", buf, off + 20, shares)
    struct.pack_into(">Q", buf, off + 24, packed_stock)
    struct.pack_into(">I", buf, off + 32, price_tick & 0xFFFFFFFF)
    struct.pack_into(">Q", buf, off + 36, match_number)
    return 44
