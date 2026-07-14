"""FIX 4.4 codec layer (port of Java ``com.quantfinlib.fix``).

:class:`FixMessage` is the string-based tag/value codec (parse +
checksum/body-length validation, and a builder that frames outbound
messages). :class:`FixDecoder` is the incremental stream framer that
turns arbitrarily-fragmented socket bytes into complete validated
messages. :class:`FixOrderEncoder`/:class:`FixExecReportView` are the
garbage-free-styled hot-path pair: NewOrderSingle out, ExecutionReport
in, both keyed on scaled-int prices and a dense pre-registered symbol
table.

``FixSession`` (the threaded TCP session state machine -- logon/
heartbeat/resend/sequence-number bookkeeping) is **not ported**: it is
a stateful network protocol on top of these primitives, out of scope
for a library port. ``FixParse`` is ported as the internal
:mod:`quantfinlib.fix.fix_parse` module (not re-exported: it backs
:class:`FixExecReportView` only, as in the Java package-private
original).
"""

from quantfinlib.fix.fix_decoder import FixDecoder
from quantfinlib.fix.fix_exec_report_view import FixExecReportView
from quantfinlib.fix.fix_message import FixMessage, FixMessageBuilder
from quantfinlib.fix.fix_order_encoder import FixOrderEncoder

__all__ = [
    "FixDecoder",
    "FixExecReportView",
    "FixMessage",
    "FixMessageBuilder",
    "FixOrderEncoder",
]
