"""FIX 4.4 wire-format message codec (port of Java ``fix.FixMessage``):
tag=value fields delimited by SOH, framed by BeginString(8) /
BodyLength(9) / CheckSum(10). Parsing validates the checksum and body
length; building computes them. No repeating-group support -- sufficient
for the session and single-order message types this port speaks.
"""

from __future__ import annotations

import decimal
import math
from typing import Dict, Optional

#: The FIX field delimiter -- always built from chr(1), never a literal
#: control character in source.
SOH = chr(1)
SOH_BYTE = 1
BEGIN_STRING = "FIX.4.4"

# Common tags.
AVG_PX = 6
BEGIN_SEQ_NO = 7
BEGIN_STRING_TAG = 8
BODY_LENGTH = 9
CHECK_SUM = 10
CL_ORD_ID = 11
CUM_QTY = 14
END_SEQ_NO = 16
EXEC_ID = 17
LAST_PX = 31
LAST_QTY = 32
MSG_SEQ_NUM = 34
MSG_TYPE = 35
NEW_SEQ_NO = 36
ORDER_ID = 37
ORDER_QTY = 38
ORD_STATUS = 39
ORD_TYPE = 40
ORIG_CL_ORD_ID = 41
POSS_DUP_FLAG = 43
PRICE = 44
REF_SEQ_NUM = 45
SENDER_COMP_ID = 49
SENDING_TIME = 52
SIDE = 54
SYMBOL = 55
TARGET_COMP_ID = 56
TEXT = 58
TIME_IN_FORCE = 59
TRANSACT_TIME = 60
ENCRYPT_METHOD = 98
HEART_BT_INT = 108
TEST_REQ_ID = 112
ORIG_SENDING_TIME = 122
GAP_FILL_FLAG = 123
RESET_SEQ_NUM_FLAG = 141
EXEC_TYPE = 150
LEAVES_QTY = 151
SESSION_REJECT_REASON = 373
USERNAME = 553
PASSWORD = 554

# Message types.
HEARTBEAT = "0"
TEST_REQUEST = "1"
RESEND_REQUEST = "2"
REJECT = "3"
SEQUENCE_RESET = "4"
LOGON = "A"
LOGOUT = "5"
NEW_ORDER_SINGLE = "D"
EXECUTION_REPORT = "8"
ORDER_CANCEL_REQUEST = "F"
ORDER_CANCEL_REPLACE_REQUEST = "G"

_NO_DEFAULT = object()


def _printable(raw: str) -> str:
    return raw.replace(SOH, "|")


def _format_double(value: float) -> str:
    """Plain-decimal rendering (FIX forbids scientific notation),
    mirroring Java's ``BigDecimal.valueOf(value).stripTrailingZeros()
    .toPlainString()``."""
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"cannot encode non-finite FIX value: {value}")
    d = decimal.Decimal(repr(float(value)))
    text = format(d, "f")
    if d != 0 and "." in text:
        text = text.rstrip("0").rstrip(".")
        if text in ("", "-"):
            text += "0"
    return text


class FixMessage:
    """An immutable, parsed FIX message: tag -> string value, in wire
    order. Construct via :meth:`parse` (inbound) or :meth:`builder`
    (outbound)."""

    __slots__ = ("_fields",)

    def __init__(self, fields: Dict[int, str]) -> None:
        self._fields = fields

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse(data: bytes) -> "FixMessage":
        """Parses and validates one complete framed message."""
        raw = data.decode("latin-1")
        if not raw.startswith("8=" + BEGIN_STRING + SOH):
            raise ValueError(f"missing BeginString: {_printable(raw)}")
        checksum_field = raw.rfind(SOH + "10=")
        if checksum_field < 0 or not raw.endswith(SOH):
            raise ValueError(f"missing CheckSum: {_printable(raw)}")
        total = 0
        for i in range(checksum_field + 1):
            total += data[i]
        declared = raw[checksum_field + 4:len(raw) - 1]
        if f"{total % 256:03d}" != declared:
            raise ValueError(
                f"checksum mismatch: declared {declared} computed "
                f"{total % 256}")

        fields: Dict[int, str] = {}
        for pair in raw[:len(raw) - 1].split(SOH):
            eq = pair.find("=")
            if eq <= 0:
                raise ValueError(f"malformed field: {_printable(pair)}")
            fields[int(pair[:eq])] = pair[eq + 1:]

        body_start = raw.index(SOH, raw.index("9=")) + 1
        declared_len = int(fields[BODY_LENGTH])
        actual_len = checksum_field + 1 - body_start
        if declared_len != actual_len:
            raise ValueError(
                f"body length mismatch: declared {declared_len} actual "
                f"{actual_len}")
        return FixMessage(fields)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def msg_type(self) -> Optional[str]:
        return self._fields.get(MSG_TYPE)

    def has(self, tag: int) -> bool:
        return tag in self._fields

    def get_string(self, tag: int, default=_NO_DEFAULT) -> str:
        v = self._fields.get(tag)
        if v is None:
            if default is _NO_DEFAULT:
                raise ValueError(f"missing tag {tag} in {self}")
            return default
        return v

    def get_long(self, tag: int) -> int:
        return int(self.get_string(tag))

    def get_double(self, tag: int, default=_NO_DEFAULT) -> float:
        v = self._fields.get(tag)
        if v is None:
            if default is _NO_DEFAULT:
                raise ValueError(f"missing tag {tag} in {self}")
            return default
        return float(v)

    def get_char(self, tag: int) -> str:
        return self.get_string(tag)[0]

    def __str__(self) -> str:
        return "".join(f"{k}={v}|" for k, v in self._fields.items())

    __repr__ = __str__

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @staticmethod
    def builder(msg_type: str) -> "FixMessageBuilder":
        return FixMessageBuilder(msg_type)


# Re-exposed as class attributes for parity with the Java
# ``public static final`` tag/message-type constants on
# ``fix.FixMessage`` (e.g. ``FixMessage.EXECUTION_REPORT``); the
# module-level names above remain the canonical definitions used
# internally by the rest of :mod:`quantfinlib.fix`.
for _name in (
    "SOH", "BEGIN_STRING", "AVG_PX", "BEGIN_SEQ_NO", "BEGIN_STRING_TAG",
    "BODY_LENGTH", "CHECK_SUM", "CL_ORD_ID", "CUM_QTY", "END_SEQ_NO",
    "EXEC_ID", "LAST_PX", "LAST_QTY", "MSG_SEQ_NUM", "MSG_TYPE",
    "NEW_SEQ_NO", "ORDER_ID", "ORDER_QTY", "ORD_STATUS", "ORD_TYPE",
    "ORIG_CL_ORD_ID", "POSS_DUP_FLAG", "PRICE", "REF_SEQ_NUM",
    "SENDER_COMP_ID", "SENDING_TIME", "SIDE", "SYMBOL", "TARGET_COMP_ID",
    "TEXT", "TIME_IN_FORCE", "TRANSACT_TIME", "ENCRYPT_METHOD",
    "HEART_BT_INT", "TEST_REQ_ID", "ORIG_SENDING_TIME", "GAP_FILL_FLAG",
    "RESET_SEQ_NUM_FLAG", "EXEC_TYPE", "LEAVES_QTY",
    "SESSION_REJECT_REASON", "USERNAME", "PASSWORD", "HEARTBEAT",
    "TEST_REQUEST", "RESEND_REQUEST", "REJECT", "SEQUENCE_RESET", "LOGON",
    "LOGOUT", "NEW_ORDER_SINGLE", "EXECUTION_REPORT",
    "ORDER_CANCEL_REQUEST", "ORDER_CANCEL_REPLACE_REQUEST",
):
    setattr(FixMessage, _name, globals()[_name])
del _name


class FixMessageBuilder:
    """Body-field builder; the session supplies header fields at
    encode time. Corresponds to Java's ``FixMessage.Builder``."""

    __slots__ = ("_msg_type", "_body")

    def __init__(self, msg_type: str) -> None:
        self._msg_type = msg_type
        self._body: list[str] = []

    def field(self, tag: int, value) -> "FixMessageBuilder":
        if isinstance(value, float):
            rendered = _format_double(value)
        else:
            rendered = str(value)
        self._body.append(f"{tag}={rendered}{SOH}")
        return self

    def msg_type(self) -> str:
        return self._msg_type

    def body(self) -> str:
        """Raw body fields (for a session's outbound message store)."""
        return "".join(self._body)

    def encode(self, sender_comp_id: str, target_comp_id: str, seq_num: int,
               sending_time_utc: str, poss_dup: bool = False,
               orig_sending_time: Optional[str] = None) -> bytes:
        """Frames the message with header, body length and checksum.
        ``poss_dup``/``orig_sending_time`` set PossDupFlag(43)=Y and
        OrigSendingTime(122) -- required when replaying stored
        messages."""
        after_length = (
            f"35={self._msg_type}{SOH}"
            f"49={sender_comp_id}{SOH}"
            f"56={target_comp_id}{SOH}"
            f"34={seq_num}{SOH}"
            + (f"43=Y{SOH}" if poss_dup else "")
            + f"52={sending_time_utc}{SOH}"
            + (f"122={orig_sending_time}{SOH}" if orig_sending_time is not None else "")
            + self.body()
        )
        head = f"8={BEGIN_STRING}{SOH}9={len(after_length)}{SOH}"
        payload = (head + after_length).encode("latin-1")
        total = sum(payload)
        full = payload + f"10={total % 256:03d}{SOH}".encode("latin-1")
        return full
