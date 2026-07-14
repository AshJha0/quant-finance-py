"""Writer for the QFLT binary tick format (port of Java
``data.TickFileWriter``): compact capture of live tick streams for
deterministic replay (28 bytes per tick).

Format: magic ``"QFLT"`` + version byte, then framed records: type 1 =
symbol definition (int32 id, uint16-length-prefixed UTF-8 name), type 0
= tick (int32 symbolId, double price, double size, int64 timestampNanos).
Symbol definitions may appear anywhere before the first tick that
references them, so symbols can be added mid-capture. All multi-byte
fields are big-endian, matching Java's ``DataOutputStream``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Set

MAGIC = 0x51464C54   # "QFLT"
VERSION = 1
TYPE_TICK = 0
TYPE_SYMBOL = 1


class TickFileWriter:
    """Sequential writer for one QFLT file. Use as a context manager
    or call :meth:`close` explicitly."""

    def __init__(self, path) -> None:
        path = Path(path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self._out: BinaryIO = open(path, "wb")
        self._defined: Set[int] = set()
        self._tick_count = 0
        self._out.write(struct.pack(">i", MAGIC))
        self._out.write(struct.pack(">b", VERSION))

    def define_symbol(self, symbol_id: int, symbol: str) -> None:
        """Registers a symbol id (idempotent; must precede its first tick)."""
        if symbol_id in self._defined:
            return
        name = symbol.encode("utf-8")
        self._out.write(struct.pack(">b", TYPE_SYMBOL))
        self._out.write(struct.pack(">i", symbol_id))
        self._out.write(struct.pack(">H", len(name)))
        self._out.write(name)
        self._defined.add(symbol_id)

    def write(self, symbol_id: int, price: float, size: float,
             timestamp_nanos: int) -> None:
        if symbol_id not in self._defined:
            raise RuntimeError(
                f"symbol id {symbol_id} not defined before first tick")
        self._out.write(struct.pack(">b", TYPE_TICK))
        self._out.write(struct.pack(">i", symbol_id))
        self._out.write(struct.pack(">d", price))
        self._out.write(struct.pack(">d", size))
        self._out.write(struct.pack(">q", timestamp_nanos))
        self._tick_count += 1

    def tick_count(self) -> int:
        return self._tick_count

    def close(self) -> None:
        self._out.close()

    def __enter__(self) -> "TickFileWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False
