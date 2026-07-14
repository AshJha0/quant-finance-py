"""FIX 4.4 NewOrderSingle encoder (port of Java ``fix.FixOrderEncoder``)
-- the hot-lane counterpart of the string-based
:class:`~quantfinlib.fix.fix_message.FixMessage` codec, for venues that
only speak FIX.

The Java original is a zero-allocation, backwards-written byte-buffer
encoder (a garbage-free-JVM concern); this port keeps the exact field
semantics and wire layout -- one reusable buffer per encoder instance,
prices as scaled longs (mantissa x 10^-decimals, never a double),
symbols pre-registered as ASCII bytes, a per-UTC-day cached timestamp
prefix -- but builds the message with ordinary byte-string
concatenation, the Python idiom, rather than chasing the JVM's
allocation-free tricks (backwards prefix writing, digit-by-digit
integer rendering).

Correctness is pinned by round-trip tests: every encoded message is
parsed back by the validated
:meth:`~quantfinlib.fix.fix_message.FixMessage.parse` (which checks
BodyLength and CheckSum) and field-compared.
"""

from __future__ import annotations

import datetime
from typing import List, Optional

from quantfinlib.fix.fix_message import SOH
from quantfinlib.microstructure.execution import Side

_SOH_BYTES = SOH.encode("ascii")
_EPOCH = datetime.date(1970, 1, 1)


class FixOrderEncoder:
    """Encodes limit/market NewOrderSingle messages for one FIX
    session against a dense, pre-registered symbol table."""

    __slots__ = ("_sender_comp_id", "_target_comp_id", "_symbols",
                 "_header34", "_date_prefix", "_cached_epoch_day",
                 "_last")

    def __init__(self, sender_comp_id: str, target_comp_id: str,
                 max_symbols: int, buffer_size: int = 512) -> None:
        """
        Args:
            sender_comp_id: session sender (tag 49).
            target_comp_id: session target (tag 56).
            max_symbols: dense symbol-id capacity.
            buffer_size: kept for API parity with the Java constructor
                (512 is ample for a NewOrderSingle); this port does not
                pre-size a buffer.
        """
        if (not sender_comp_id or not target_comp_id or max_symbols <= 0
                or buffer_size < 256):
            raise ValueError(
                "need comp ids, maxSymbols > 0, bufferSize >= 256")
        self._sender_comp_id = sender_comp_id
        self._target_comp_id = target_comp_id
        self._symbols: List[Optional[bytes]] = [None] * max_symbols
        self._header34 = (
            f"35=D{SOH}49={sender_comp_id}{SOH}56={target_comp_id}{SOH}34="
        ).encode("ascii")
        self._date_prefix = ""
        self._cached_epoch_day = None
        self._last = b""

    def register_symbol(self, symbol_id: int, symbol: str) -> "FixOrderEncoder":
        """Registers a tradeable symbol (cold path, before trading)."""
        self._symbols[symbol_id] = symbol.encode("ascii")
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_limit(self, msg_seq_num: int, cl_ord_id: int, symbol_id: int,
                     side: Side, quantity: int, price_mantissa: int,
                     price_decimals: int, epoch_millis: int) -> int:
        """Encodes a limit NewOrderSingle. Returns the message length
        in bytes; the encoded bytes are available from :meth:`buffer`.

        Args:
            price_mantissa: price x 10^price_decimals as an int.
            price_decimals: decimal places (5 for EURUSD, 3 for
                USDJPY).
            epoch_millis: UTC time for tags 52/60.
        """
        return self._encode(msg_seq_num, cl_ord_id, symbol_id, side,
                            quantity, price_mantissa, price_decimals,
                            False, epoch_millis)

    def encode_market(self, msg_seq_num: int, cl_ord_id: int, symbol_id: int,
                      side: Side, quantity: int, epoch_millis: int) -> int:
        """Market NewOrderSingle (40=1, no price tag)."""
        return self._encode(msg_seq_num, cl_ord_id, symbol_id, side,
                            quantity, 0, 0, True, epoch_millis)

    def _encode(self, msg_seq_num: int, cl_ord_id: int, symbol_id: int,
               side: Side, quantity: int, price_mantissa: int,
               price_decimals: int, market: bool, epoch_millis: int) -> int:
        symbol = self._symbols[symbol_id]
        if symbol is None:
            raise RuntimeError(f"symbol id {symbol_id} not registered")
        self._refresh_date(epoch_millis)

        parts = [
            self._header34, str(msg_seq_num).encode("ascii"), _SOH_BYTES,
            b"52=", self._timestamp(epoch_millis), _SOH_BYTES,
            b"11=", str(cl_ord_id).encode("ascii"), _SOH_BYTES,
            b"55=", symbol, _SOH_BYTES,
            b"54=", b"1" if side == Side.BUY else b"2", _SOH_BYTES,
            b"38=", str(quantity).encode("ascii"), _SOH_BYTES,
            b"40=", b"1" if market else b"2", _SOH_BYTES,
        ]
        if not market:
            parts += [b"44=", self._price(price_mantissa, price_decimals), _SOH_BYTES]
        parts += [b"60=", self._timestamp(epoch_millis), _SOH_BYTES]
        body = b"".join(parts)

        head = b"8=FIX.4.4" + _SOH_BYTES + b"9=" + str(len(body)).encode("ascii") + _SOH_BYTES
        payload = head + body
        checksum = sum(payload) % 256
        full = payload + b"10=" + f"{checksum:03d}".encode("ascii") + _SOH_BYTES
        self._last = full
        return len(full)

    # ------------------------------------------------------------------
    # Buffer access (valid until the next encode)
    # ------------------------------------------------------------------

    def buffer(self) -> bytes:
        """The buffer holding the last encoded message."""
        return self._last

    def offset(self) -> int:
        """Start offset of the last message within :meth:`buffer`."""
        return 0

    def length(self) -> int:
        """Length of the last message."""
        return len(self._last)

    # ------------------------------------------------------------------
    # Field rendering
    # ------------------------------------------------------------------

    def _price(self, mantissa: int, decimals: int) -> bytes:
        """Scaled decimal: mantissa 108505, decimals 5 -> "1.08505"."""
        if mantissa < 0 or decimals < 0:
            raise ValueError("mantissa and decimals must be >= 0")
        if decimals == 0:
            return str(mantissa).encode("ascii")
        scale = 10 ** decimals
        integer_part = mantissa // scale
        frac = mantissa % scale
        return f"{integer_part}.{frac:0{decimals}d}".encode("ascii")

    def _refresh_date(self, epoch_millis: int) -> None:
        """Refreshes the cached "yyyyMMdd-" prefix when the UTC day
        rolls."""
        epoch_day = epoch_millis // 86_400_000
        if epoch_day != self._cached_epoch_day:
            date = _EPOCH + datetime.timedelta(days=epoch_day)
            self._date_prefix = date.strftime("%Y%m%d") + "-"
            self._cached_epoch_day = epoch_day

    def _timestamp(self, epoch_millis: int) -> bytes:
        """"yyyyMMdd-HH:mm:ss.SSS" -- date part cached per UTC day."""
        millis_of_day = epoch_millis - self._cached_epoch_day * 86_400_000
        hh = millis_of_day // 3_600_000
        mm = (millis_of_day // 60_000) % 60
        ss = (millis_of_day // 1_000) % 60
        ms = millis_of_day % 1_000
        return f"{self._date_prefix}{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}".encode("ascii")
