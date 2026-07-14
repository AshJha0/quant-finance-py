"""Reader/replayer for QFLT tick files (port of Java
``data.TickFileReader``; see :mod:`~quantfinlib.data.tick_file_writer`
for the format). Replay is deterministic -- record a live session once,
then run strategy, latency, and microstructure experiments against
identical real tick sequences.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Callable, Optional

from quantfinlib.data.tick_file_writer import MAGIC, TYPE_SYMBOL, TYPE_TICK, VERSION

#: on_tick(symbol_id, price, size, timestamp_nanos)
OnTick = Callable[[int, float, float, int], None]
#: on_symbol(symbol_id, symbol)
OnSymbol = Callable[[int, str], None]


def _default_on_symbol(symbol_id: int, symbol: str) -> None:
    pass


def _read_exact(f, n: int, path) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError(f"truncated QFLT file: {path}")
    return data


def replay(path, on_tick: OnTick, on_symbol: OnSymbol = _default_on_symbol) -> int:
    """Replays the file as fast as possible. Returns the number of
    ticks delivered."""
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4 or struct.unpack(">i", header)[0] != MAGIC:
            raise ValueError(f"not a QFLT tick file: {path}")
        version = _read_exact(f, 1, path)[0]
        if version != VERSION:
            raise ValueError(f"unsupported QFLT version {version}")
        ticks = 0
        while True:
            type_byte = f.read(1)
            if not type_byte:
                return ticks   # clean EOF
            record_type = type_byte[0]
            if record_type == TYPE_SYMBOL:
                symbol_id = struct.unpack(">i", _read_exact(f, 4, path))[0]
                name_len = struct.unpack(">H", _read_exact(f, 2, path))[0]
                name = _read_exact(f, name_len, path).decode("utf-8")
                on_symbol(symbol_id, name)
            elif record_type == TYPE_TICK:
                symbol_id = struct.unpack(">i", _read_exact(f, 4, path))[0]
                price = struct.unpack(">d", _read_exact(f, 8, path))[0]
                size = struct.unpack(">d", _read_exact(f, 8, path))[0]
                timestamp_nanos = struct.unpack(">q", _read_exact(f, 8, path))[0]
                on_tick(symbol_id, price, size, timestamp_nanos)
                ticks += 1
            else:
                raise ValueError(
                    f"corrupt record type {record_type} after {ticks} ticks")


def replay_paced(path, on_tick: OnTick, speed_multiplier: float,
                 on_symbol: OnSymbol = _default_on_symbol) -> int:
    """Replays reproducing the recorded inter-tick gaps scaled by
    ``speed_multiplier`` (2.0 = twice real time; individual gaps are
    capped at 10s). For live-like feeds into a market-data bus."""
    if speed_multiplier <= 0:
        raise ValueError("speedMultiplier must be positive")
    prev_ts: Optional[int] = None

    def paced_on_tick(symbol_id: int, price: float, size: float,
                      timestamp_nanos: int) -> None:
        nonlocal prev_ts
        if prev_ts is not None:
            gap_nanos = (timestamp_nanos - prev_ts) / speed_multiplier
            if gap_nanos > 0:
                time.sleep(min(gap_nanos, 10_000_000_000.0) / 1e9)
        prev_ts = timestamp_nanos
        on_tick(symbol_id, price, size, timestamp_nanos)

    return replay(path, paced_on_tick, on_symbol)
