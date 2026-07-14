"""FIX 4.4 codec tests, pinning the Java ``fix`` package's behavior:
message round trips, checksum/body-length validation, the incremental
decoder's partial-feed and corrupt-BodyLength-resync semantics, and the
garbage-free order-encoder / exec-report-view pair.
"""

import math

import pytest

from quantfinlib.fix import (
    FixDecoder,
    FixExecReportView,
    FixMessage,
    FixOrderEncoder,
)
from quantfinlib.microstructure.execution import Side

SOH = chr(1)


# ------------------------------------------------------------------
# FixMessage: building + parsing round trip
# ------------------------------------------------------------------

def _build_new_order_single() -> bytes:
    b = FixMessage.builder(FixMessage.NEW_ORDER_SINGLE)
    b.field(11, 12345).field(55, "EUR/USD").field(54, "1") \
        .field(38, 100).field(44, 1.08505)
    return b.encode("SENDER", "TARGET", 1, "20260101-00:00:00.000")


def test_round_trip_new_order_single():
    msg_bytes = _build_new_order_single()
    m = FixMessage.parse(msg_bytes)
    assert m.msg_type() == "D"
    assert m.get_string(11) == "12345"
    assert m.get_long(11) == 12345
    assert m.get_string(55) == "EUR/USD"
    assert m.get_double(44) == pytest.approx(1.08505)
    assert m.get_char(54) == "1"
    assert m.has(9999) is False
    assert m.get_string(9999, "fallback") == "fallback"


def test_checksum_and_body_length_are_correct():
    msg_bytes = _build_new_order_single()
    raw = msg_bytes.decode("latin-1")
    checksum_field = raw.rfind(SOH + "10=")
    total = sum(msg_bytes[: checksum_field + 1])
    declared = raw[checksum_field + 4: len(raw) - 1]
    assert declared == f"{total % 256:03d}"

    body_start = raw.index(SOH, raw.index("9=")) + 1
    declared_len = int(FixMessage.parse(msg_bytes).get_string(9))
    assert declared_len == checksum_field + 1 - body_start


def test_parse_rejects_bad_checksum():
    msg_bytes = bytearray(_build_new_order_single())
    # Flip the checksum's last digit.
    idx = msg_bytes.rfind(b"10=")
    msg_bytes[idx + 3] = ord("9") if msg_bytes[idx + 3] != ord("9") else ord("8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        FixMessage.parse(bytes(msg_bytes))


def test_parse_rejects_bad_body_length():
    msg_bytes = _build_new_order_single().decode("latin-1")
    # Corrupt the declared BodyLength value itself.
    bad = msg_bytes.replace(SOH + "9=", SOH + "9=999999", 1)
    with pytest.raises(ValueError):
        FixMessage.parse(bad.encode("latin-1"))


def test_parse_rejects_missing_begin_string():
    with pytest.raises(ValueError, match="missing BeginString"):
        FixMessage.parse(b"not a fix message")


def test_double_field_renders_plain_no_scientific_and_strips_zeros():
    b = FixMessage.builder("D").field(44, 100.0).field(6, 1.50000)
    encoded = b.encode("A", "B", 1, "20260101-00:00:00.000")
    m = FixMessage.parse(encoded)
    assert m.get_string(44) == "100"
    assert m.get_string(6) == "1.5"


# ------------------------------------------------------------------
# FixDecoder: incremental framing
# ------------------------------------------------------------------

def test_decoder_polls_none_until_full_message_fed():
    msg_bytes = _build_new_order_single()
    dec = FixDecoder()
    dec.feed(msg_bytes[:15])
    assert dec.poll() is None
    dec.feed(msg_bytes[15:])
    m = dec.poll()
    assert m is not None
    assert m.get_long(11) == 12345
    assert dec.poll() is None


def test_decoder_frames_two_back_to_back_messages_from_one_feed():
    msg1 = _build_new_order_single()
    msg2 = FixMessage.builder("D").field(11, 99).field(55, "GBP/USD") \
        .field(54, "2").field(38, 50).field(44, 1.2500) \
        .encode("SENDER", "TARGET", 2, "20260101-00:00:01.000")
    dec = FixDecoder()
    dec.feed(msg1 + msg2)
    m1 = dec.poll()
    m2 = dec.poll()
    assert m1.get_long(11) == 12345
    assert m2.get_long(11) == 99
    assert dec.poll() is None


def test_decoder_byte_at_a_time_feed_matches_bulk_feed():
    msg_bytes = _build_new_order_single()
    dec = FixDecoder()
    result = None
    for i in range(len(msg_bytes)):
        dec.feed(msg_bytes[i:i + 1])
        result = dec.poll() or result
    assert result is not None
    assert result.get_long(11) == 12345


def test_decoder_raises_on_corrupt_body_length_digit():
    # This is the zombie-session regression: a non-digit byte inside
    # the BodyLength field must fail loudly, not wait forever.
    msg_bytes = bytearray(_build_new_order_single())
    nine_eq = msg_bytes.find(b"9=")
    msg_bytes[nine_eq + 2] = ord("X")   # corrupt the first BodyLength digit
    dec = FixDecoder()
    dec.feed(bytes(msg_bytes))
    with pytest.raises(RuntimeError, match="stream corrupt"):
        dec.poll()


def test_decoder_raises_when_body_length_exceeds_cap():
    # Ten corrupt-but-numeric digits must trip the 1<<20 cap rather
    # than wrapping into something that looks like a small, valid
    # length.
    header = f"8={FixMessage.BEGIN_STRING}{SOH}9=9999999999{SOH}".encode("ascii")
    dec = FixDecoder()
    dec.feed(header)
    with pytest.raises(RuntimeError, match="not a FIX message"):
        dec.poll()


def test_decoder_raises_on_empty_body_length():
    # Padded past the decoder's 20-byte minimum-buffer guard without
    # adding another SOH, so the empty-BodyLength check is actually
    # reached rather than short-circuited by "need more bytes".
    header = (f"8={FixMessage.BEGIN_STRING}{SOH}9={SOH}".encode("ascii")
              + b"PADDING-PADDING")
    dec = FixDecoder()
    dec.feed(header)
    with pytest.raises(RuntimeError, match="empty BodyLength"):
        dec.poll()


def test_decoder_raises_when_beginstring_not_followed_by_bodylength():
    bad = (f"8={FixMessage.BEGIN_STRING}{SOH}99=1{SOH}".encode("ascii")
           + b"PADDING-PADDING")
    dec = FixDecoder()
    dec.feed(bad)
    with pytest.raises(RuntimeError, match="BodyLength not after BeginString"):
        dec.poll()


# ------------------------------------------------------------------
# FixOrderEncoder / FixExecReportView: the garbage-free hot path
# ------------------------------------------------------------------

def test_order_encoder_round_trips_through_fix_message_parse():
    enc = FixOrderEncoder("SENDER", "TARGET", 4, 512)
    enc.register_symbol(0, "EURUSD")
    n = enc.encode_limit(1, 42, 0, Side.BUY, 1_000_000, 108505, 5,
                        1_700_000_000_000)
    encoded = enc.buffer()[enc.offset():enc.offset() + n]
    m = FixMessage.parse(encoded)
    assert m.msg_type() == "D"
    assert m.get_long(11) == 42
    assert m.get_string(55) == "EURUSD"
    assert m.get_char(54) == "1"
    assert m.get_long(38) == 1_000_000
    assert m.get_string(44) == "1.08505"


def test_order_encoder_market_order_omits_price_tag():
    enc = FixOrderEncoder("SENDER", "TARGET", 4, 512)
    enc.register_symbol(0, "EURUSD")
    n = enc.encode_market(1, 7, 0, Side.SELL, 500_000, 1_700_000_000_000)
    m = FixMessage.parse(enc.buffer()[:n])
    assert m.get_char(40) == "1"
    assert m.has(44) is False
    assert m.get_char(54) == "2"


def test_order_encoder_unregistered_symbol_raises():
    enc = FixOrderEncoder("SENDER", "TARGET", 2, 512)
    with pytest.raises(RuntimeError):
        enc.encode_limit(1, 1, 0, Side.BUY, 100, 10000, 4, 0)


def test_order_encoder_timestamp_rolls_over_utc_day():
    enc = FixOrderEncoder("SENDER", "TARGET", 1, 512)
    enc.register_symbol(0, "EURUSD")
    day_ms = 86_400_000
    n1 = enc.encode_limit(1, 1, 0, Side.BUY, 1, 10000, 4, day_ms - 1000)
    m1 = FixMessage.parse(enc.buffer()[:n1])
    n2 = enc.encode_limit(2, 2, 0, Side.BUY, 1, 10000, 4, day_ms + 1000)
    m2 = FixMessage.parse(enc.buffer()[:n2])
    assert m1.get_string(52).startswith("19700101-")
    assert m2.get_string(52).startswith("19700102-")


def test_exec_report_view_extracts_fill_fields():
    er = FixMessage.builder(FixMessage.EXECUTION_REPORT)
    er.field(11, 42).field(150, "F").field(39, "2").field(54, "1") \
        .field(32, 500_000).field(31, 1.08510).field(14, 500_000) \
        .field(151, 500_000).field(55, "EURUSD")
    er_bytes = er.encode("TARGET", "SENDER", 2, "20260101-00:00:00.100")

    view = FixExecReportView()
    assert view.wrap(er_bytes, 0, len(er_bytes)) is True
    assert view.cl_ord_id() == 42
    assert view.exec_type() == ord("F")
    assert view.ord_status() == ord("2")
    assert view.side() == ord("1")
    assert view.last_qty() == 500_000
    assert view.cum_qty() == 500_000
    assert view.leaves_qty() == 500_000
    assert view.symbol_equals(b"EURUSD") is True
    assert view.symbol_equals(b"GBPUSD") is False


def test_exec_report_view_rejects_non_exec_report_message_types():
    m = FixMessage.builder(FixMessage.HEARTBEAT).encode(
        "A", "B", 1, "20260101-00:00:00.000")
    view = FixExecReportView()
    assert view.wrap(m, 0, len(m)) is False


def test_exec_report_view_clordid_sentinel_for_counterparty_format_id():
    # Unsolicited reports may carry a non-numeric ClOrdID (venue's own
    # format); that must fall back to the -1 "not ours" sentinel
    # instead of raising mid-wrap.
    er = FixMessage.builder(FixMessage.EXECUTION_REPORT)
    er.field(11, "ORD-2024-17").field(150, "0").field(39, "0").field(54, "1")
    er_bytes = er.encode("TARGET", "SENDER", 1, "20260101-00:00:00.000")
    view = FixExecReportView()
    assert view.wrap(er_bytes, 0, len(er_bytes)) is True
    assert view.cl_ord_id() == -1


def test_exec_report_view_price_mantissa_and_decimals():
    er = FixMessage.builder(FixMessage.EXECUTION_REPORT)
    er.field(11, 1).field(150, "F").field(39, "2").field(54, "2").field(31, -0.5)
    er_bytes = er.encode("TARGET", "SENDER", 1, "20260101-00:00:00.000")
    view = FixExecReportView()
    assert view.wrap(er_bytes, 0, len(er_bytes)) is True
    assert view.last_px_mantissa() == -5
    assert view.last_px_decimals() == 1


# ------------------------------------------------------------------
# fix_parse internals (used by FixExecReportView)
# ------------------------------------------------------------------

def test_fix_parse_rejects_negative_quantity_field():
    from quantfinlib.fix import fix_parse
    buf = b"-5"
    with pytest.raises(ValueError, match="negative quantity"):
        fix_parse.parse_long(buf, 0, len(buf))


def test_fix_parse_tolerates_zero_fraction_but_rejects_real_fraction():
    from quantfinlib.fix import fix_parse
    assert fix_parse.parse_long(b"100.0", 0, 5) == 100
    with pytest.raises(ValueError, match="fractional quantity"):
        fix_parse.parse_long(b"100.5", 0, 5)


def test_fix_parse_long_or_else_falls_back_on_non_numeric():
    from quantfinlib.fix import fix_parse
    buf = b"ORD-17"
    assert fix_parse.parse_long_or_else(buf, 0, len(buf), -1) == -1
    buf2 = b"12345"
    assert fix_parse.parse_long_or_else(buf2, 0, len(buf2), -1) == 12345
