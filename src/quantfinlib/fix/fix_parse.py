"""Shared primitive parsers for the garbage-free FIX flyweights (port of
Java ``fix.FixParse``) -- the fraction-rejection and scaled-long price
rules are load-bearing for the "feed-to-order never touches a double"
invariant, so they exist exactly once. Internal to :mod:`quantfinlib.fix`.
"""

from __future__ import annotations

_MINUS = ord("-")
_DOT = ord(".")
_ZERO = ord("0")
_NINE = ord("9")


def parse_long(buf: bytes, frm: int, to: int) -> int:
    """Non-negative integer field (quantities, counts). Tolerates a
    zero fraction ("100.0"); a real fraction or a '-' sign fails loudly
    -- negative quantities are protocol violations that must never
    parse into silent garbage."""
    v = 0
    i = frm
    while i < to:
        b = buf[i]
        if b == _MINUS:
            raise ValueError("negative quantity field")
        if b == _DOT:
            for j in range(i + 1, to):
                if buf[j] != _ZERO:
                    raise ValueError(
                        "fractional quantity not representable as long")
            return v
        v = v * 10 + (b - _ZERO)
        i += 1
    return v


def parse_long_or_else(buf: bytes, frm: int, to: int, fallback: int) -> int:
    """Numeric field that may legitimately be non-numeric on the wire
    -- e.g. ClOrdID (tag 11), which is only numeric by OUR encoder's
    convention; unsolicited venue messages carry counterparty formats
    ("ORD-2024-17", UUIDs). Returns ``fallback`` on any non-digit
    instead of raising: a foreign id must never kill the message
    pump."""
    v = 0
    for i in range(frm, to):
        b = buf[i]
        if b < _ZERO or b > _NINE:
            return fallback
        v = v * 10 + (b - _ZERO)
    return fallback if frm == to else v


def price_mantissa(buf: bytes, frm: int, to: int) -> int:
    """Price mantissa as a signed scaled long: "1.08505" -> 108505,
    "-0.5" -> -5 (with :func:`price_decimals` = 1). Negative prices are
    real in FX (forward points, negative rates) and must round-trip."""
    negative = frm < to and buf[frm] == _MINUS
    mantissa = 0
    start = frm + 1 if negative else frm
    for i in range(start, to):
        b = buf[i]
        if b != _DOT:
            mantissa = mantissa * 10 + (b - _ZERO)
    return -mantissa if negative else mantissa


def price_decimals(buf: bytes, frm: int, to: int) -> int:
    """Digits after the decimal point: "1.08505" -> 5; "99" -> 0."""
    for i in range(frm, to):
        if buf[i] == _DOT:
            return to - i - 1
    return 0
