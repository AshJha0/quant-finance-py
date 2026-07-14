"""Incremental stream framer for FIX messages (port of Java
``fix.FixDecoder``): feed raw socket bytes in any fragmentation, poll
complete validated :class:`~quantfinlib.fix.fix_message.FixMessage`\\ s
out. Framing uses BodyLength(9), so message boundaries are exact
regardless of TCP segmentation.
"""

from __future__ import annotations

from typing import Optional

from quantfinlib.fix.fix_message import SOH_BYTE, FixMessage

_ZERO = ord("0")
_NINE = ord("9")
_BODY_LENGTH_CAP = 1 << 20


class FixDecoder:
    """Stateful incremental framer: :meth:`feed` appends bytes,
    :meth:`poll` extracts complete messages one at a time."""

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes, offset: int = 0, count: Optional[int] = None) -> None:
        if count is None:
            count = len(data) - offset
        self._buffer += data[offset:offset + count]

    def poll(self) -> Optional[FixMessage]:
        """Next complete message, or None if more bytes are needed."""
        buf = self._buffer
        length = len(buf)
        if length < 20:
            return None
        first_soh = buf.find(SOH_BYTE, 0)
        if first_soh < 0:
            return None
        if buf[first_soh + 1] != ord("9") or buf[first_soh + 2] != ord("="):
            raise RuntimeError(
                "stream corrupt: BodyLength not after BeginString")
        len_end = buf.find(SOH_BYTE, first_soh + 3)
        if len_end < 0:
            return None
        if len_end == first_soh + 3:
            raise RuntimeError("stream corrupt: empty BodyLength")
        body_len = 0
        for i in range(first_soh + 3, len_end):
            digit = buf[i] - _ZERO
            # A single corrupted byte here would silently inflate the
            # expected frame length and the framer would wait FOREVER
            # for bytes that never come -- a zombie session swallowing
            # every later valid message. Corruption must fail loudly so
            # the session layer disconnects.
            if digit < 0 or digit > 9:
                raise RuntimeError(
                    f"stream corrupt: BodyLength contains 0x{buf[i]:02x}")
            body_len = body_len * 10 + digit
            # Checked INSIDE the loop: ten corrupt-but-numeric digits
            # would wrap a fixed-width int and could sneak under a
            # post-loop cap (Python ints don't wrap, but the cap must
            # still trip as early as the Java original does).
            if body_len > _BODY_LENGTH_CAP:
                raise RuntimeError(
                    f"stream corrupt: BodyLength {body_len} is not a "
                    f"FIX message")
        total = len_end + 1 + body_len + 7   # "10=xxx" + SOH trailer
        if length < total:
            return None
        message = bytes(buf[:total])
        del buf[:total]
        return FixMessage.parse(message)
