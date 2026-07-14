"""FIX 4.4 ExecutionReport reader (port of Java
``fix.FixExecReportView``) -- the inbound half of the FIX hot path,
completing the round trip :class:`~quantfinlib.fix.fix_order_encoder.FixOrderEncoder`
started: order out, fill in.

Where :class:`~quantfinlib.fix.fix_message.FixMessage` materializes a
``dict`` of tag -> string (right for session management and research),
this is a *flyweight view*: :meth:`wrap` performs one pass over the
framed bytes recording primitive values, and every getter returns a
primitive. The fields extracted are exactly what a fill handler needs:
ClOrdID (numeric, as issued by ``FixOrderEncoder``), ExecType,
OrdStatus, Side, LastQty, LastPx, CumQty, LeavesQty. Prices come back as
scaled ints (``last_px_mantissa()`` x 10^-``last_px_decimals()``) -- the
same representation the encoder takes in, so the round trip never
touches a float. The symbol is exposed as bytes-in-place for comparison
against a registered table, never as a string.

Assumes an already-framed message (:class:`~quantfinlib.fix.fix_decoder.FixDecoder`
owns framing and checksum).
"""

from __future__ import annotations

from quantfinlib.fix import fix_parse
from quantfinlib.fix.fix_message import CL_ORD_ID, EXEC_TYPE, LAST_PX, LAST_QTY
from quantfinlib.fix.fix_message import (
    CUM_QTY, LEAVES_QTY, MSG_TYPE, ORD_STATUS, SIDE, SYMBOL,
)

_SOH_BYTE = 1
_EQUALS = ord("=")
_ZERO = ord("0")


class FixExecReportView:
    """Flyweight over a framed FIX message buffer; reuse one instance
    per session thread, re-:meth:`wrap` per message."""

    __slots__ = ("_buf", "_cl_ord_id", "_exec_type", "_ord_status", "_side",
                 "_last_qty", "_cum_qty", "_leaves_qty", "_last_px_mantissa",
                 "_last_px_decimals", "_symbol_offset", "_symbol_length")

    def __init__(self) -> None:
        self._buf = b""
        self._cl_ord_id = -1
        self._exec_type = 0
        self._ord_status = 0
        self._side = 0
        self._last_qty = 0
        self._cum_qty = 0
        self._leaves_qty = 0
        self._last_px_mantissa = 0
        self._last_px_decimals = 0
        self._symbol_offset = -1
        self._symbol_length = 0

    def wrap(self, buffer: bytes, offset: int, length: int) -> bool:
        """Parses one framed message in place. Returns True when it is
        an ExecutionReport (35=8); other message types return False and
        leave the getters undefined."""
        self._buf = buffer
        exec_report = False
        self._cl_ord_id = -1
        self._exec_type = 0
        self._ord_status = 0
        self._side = 0
        self._last_qty = 0
        self._cum_qty = 0
        self._leaves_qty = 0
        self._last_px_mantissa = 0
        self._last_px_decimals = 0
        self._symbol_offset = -1
        self._symbol_length = 0

        end = offset + length
        p = offset
        while p < end:
            tag = 0
            while p < end and buffer[p] != _EQUALS:
                tag = tag * 10 + (buffer[p] - _ZERO)
                p += 1
            p += 1  # skip '='
            value_start = p
            while p < end and buffer[p] != _SOH_BYTE:
                p += 1
            value_end = p
            p += 1  # skip SOH

            if tag == MSG_TYPE:
                exec_report = (value_end - value_start == 1
                               and buffer[value_start] == ord("8"))
                if not exec_report:
                    return False   # not ours: stop scanning immediately
            elif tag == CL_ORD_ID:
                # ClOrdID is free-format FIX: numeric when WE issued it
                # (FixOrderEncoder), counterparty-format on unsolicited
                # reports -- those map to the existing -1 "not ours"
                # sentinel instead of raising mid-wrap.
                self._cl_ord_id = fix_parse.parse_long_or_else(
                    self._buf, value_start, value_end, -1)
            elif tag == EXEC_TYPE:
                self._exec_type = buffer[value_start]
            elif tag == ORD_STATUS:
                self._ord_status = buffer[value_start]
            elif tag == SIDE:
                self._side = buffer[value_start]
            elif tag == LAST_QTY:
                self._last_qty = fix_parse.parse_long(self._buf, value_start, value_end)
            elif tag == CUM_QTY:
                self._cum_qty = fix_parse.parse_long(self._buf, value_start, value_end)
            elif tag == LEAVES_QTY:
                self._leaves_qty = fix_parse.parse_long(self._buf, value_start, value_end)
            elif tag == LAST_PX:
                self._last_px_mantissa = fix_parse.price_mantissa(
                    self._buf, value_start, value_end)
                self._last_px_decimals = fix_parse.price_decimals(
                    self._buf, value_start, value_end)
            elif tag == SYMBOL:
                self._symbol_offset = value_start
                self._symbol_length = value_end - value_start
            # Every other tag is skipped without materializing it.
        return exec_report

    # ------------------------------------------------------------------
    # Primitive getters (valid until the next wrap)
    # ------------------------------------------------------------------

    def cl_ord_id(self) -> int:
        """Numeric ClOrdID as issued by FixOrderEncoder; -1 when absent."""
        return self._cl_ord_id

    def exec_type(self) -> int:
        """Tag 150 as its ASCII byte value ('0' new, 'F' trade, ...)."""
        return self._exec_type

    def ord_status(self) -> int:
        """Tag 39 as its ASCII byte value."""
        return self._ord_status

    def side(self) -> int:
        """Tag 54: ord('1') buy, ord('2') sell."""
        return self._side

    def last_qty(self) -> int:
        return self._last_qty

    def cum_qty(self) -> int:
        return self._cum_qty

    def leaves_qty(self) -> int:
        return self._leaves_qty

    def last_px_mantissa(self) -> int:
        """LastPx as a scaled int: mantissa x 10^-decimals."""
        return self._last_px_mantissa

    def last_px_decimals(self) -> int:
        return self._last_px_decimals

    def symbol_equals(self, ascii_symbol: bytes) -> bool:
        """Compares the in-place symbol bytes against a registered
        ASCII symbol (e.g. the same table ``FixOrderEncoder`` holds) --
        the getter that replaces a string: resolve the dense id by
        probing your table."""
        if self._symbol_offset < 0 or len(ascii_symbol) != self._symbol_length:
            return False
        start = self._symbol_offset
        return self._buf[start:start + self._symbol_length] == ascii_symbol
