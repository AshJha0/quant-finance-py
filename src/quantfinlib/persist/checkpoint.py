"""Multi-day persistence of learned state (port of Java
``persist.Checkpoint``).

Everything a model learns across sessions -- volume/vol/spread
baselines, alpha weights, venue and LP scorecards -- is exactly what a
desk does NOT want to relearn from zero every morning. A checkpoint is
one binary file of named sections, written at end of day and restored
at the next session start::

    with Checkpoint.writer(path) as w:               # end of day
        w.section("volume.AAPL", volume_curve.write_state)
        w.section("alpha.AAPL", learner.write_state)
        w.section("venues", scorecard.write_state)
    # commits atomically on __exit__

    r = Checkpoint.reader(path)                       # next morning
    r.section("volume.AAPL", volume_curve.read_state)  # False if absent

**Contract.** Each model persists its *learned* (cross-day) state only;
intraday state resets on read -- restore at session start, not
mid-stream. The reading instance must be constructed with the same
configuration (bucket count, venue count, ...): a mismatch raises
``ValueError`` rather than silently misaligning arrays. Each section
payload carries its own version byte so models can evolve their format
independently of the file format.

**Durability.** The writer buffers sections in memory and commits in
:meth:`CheckpointWriter.close`: temp file beside the target, then
``os.replace`` over the old checkpoint -- a crash mid-save leaves
yesterday's file intact, never a torn one (``os.replace`` is an atomic
rename on both POSIX and Windows). If any section writer raised,
nothing is committed. The reader loads the whole file up front (these
files are kilobytes), and rejects a section the model did not fully
consume -- the loudest possible signal of a writer/reader format
drift.

**Wire format** mirrors the Java ``DataOutputStream``/``DataInputStream``
byte layout so files are cross-compatible in header/section framing
(everything is big-endian, matching Java's ``DataOutput``):

    header:  int32 MAGIC ("QFLC") + int32 FORMAT_VERSION
    section: uint16 name-length + UTF-8 name bytes
             + int32 payload-length + payload bytes

One deviation: section names are encoded as plain UTF-8, not Java's
"modified UTF-8" (``DataOutputStream.writeUTF``) -- the two differ only
for embedded NUL bytes and characters outside the Basic Multilingual
Plane, neither of which occurs in the ``model.symbol`` section names
this library uses.

Everything here is cold-path (end of day / session start); the hot
lanes never see it. Naming convention: ``model.symbol``
("volume.EURUSD", "venues").
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Callable, Dict, List


class BinWriter:
    """Growable big-endian byte sink -- the Python counterpart of
    Java's ``DataOutput``, as passed to a model's ``write_state``."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def write_byte(self, value: int) -> None:
        self._buf += struct.pack(">b", value)

    def write_int(self, value: int) -> None:
        self._buf += struct.pack(">i", value)

    def write_long(self, value: int) -> None:
        self._buf += struct.pack(">q", value)

    def write_double(self, value: float) -> None:
        self._buf += struct.pack(">d", value)

    def to_bytes(self) -> bytes:
        return bytes(self._buf)


class BinReader:
    """Fixed byte source with a cursor -- the Python counterpart of
    Java's ``DataInput``, as passed to a model's ``read_state``."""

    __slots__ = ("_buf", "_pos")

    def __init__(self, data: bytes) -> None:
        self._buf = data
        self._pos = 0

    def read_byte(self) -> int:
        v = struct.unpack_from(">b", self._buf, self._pos)[0]
        self._pos += 1
        return v

    def read_int(self) -> int:
        v = struct.unpack_from(">i", self._buf, self._pos)[0]
        self._pos += 4
        return v

    def read_long(self) -> int:
        v = struct.unpack_from(">q", self._buf, self._pos)[0]
        self._pos += 8
        return v

    def read_double(self) -> float:
        v = struct.unpack_from(">d", self._buf, self._pos)[0]
        self._pos += 8
        return v

    def remaining(self) -> int:
        return len(self._buf) - self._pos


StateWriter = Callable[[BinWriter], None]
StateReader = Callable[[BinReader], None]


class CheckpointWriter:
    """Collects named sections and commits them atomically on
    :meth:`close` (or on ``with`` exit). See the module docstring."""

    def __init__(self, target) -> None:
        self._target = Path(target)
        self._sections: Dict[str, bytes] = {}
        self._failed = False

    def section(self, name: str, body: StateWriter) -> "CheckpointWriter":
        """Serializes one model's state under ``name``. A duplicate
        name or a raising ``body`` marks the writer failed: ``close``
        then commits nothing, so a half-written model can never
        replace a good file."""
        if name in self._sections:
            self._failed = True
            raise ValueError(f"duplicate checkpoint section: {name}")
        w = BinWriter()
        try:
            body(w)
        except Exception:
            self._failed = True
            raise
        self._sections[name] = w.to_bytes()
        return self

    def close(self) -> None:
        """Commits: temp file beside the target, then an atomic
        replace over it. No-op if a section body raised."""
        if self._failed:
            return
        temp = self._target.with_name(self._target.name + ".tmp")
        if temp.parent and not temp.parent.exists():
            temp.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(temp, "wb") as f:
                f.write(struct.pack(">i", Checkpoint.MAGIC))
                f.write(struct.pack(">i", Checkpoint.FORMAT_VERSION))
                for name, payload in self._sections.items():
                    name_bytes = name.encode("utf-8")
                    f.write(struct.pack(">H", len(name_bytes)))
                    f.write(name_bytes)
                    f.write(struct.pack(">i", len(payload)))
                    f.write(payload)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        os.replace(temp, self._target)

    def __enter__(self) -> "CheckpointWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


class CheckpointReader:
    """Random access to a loaded checkpoint's sections by name."""

    def __init__(self, path) -> None:
        source = Path(path)
        data = source.read_bytes()
        if len(data) < 8:
            raise ValueError(f"{source} is not a quantfinlib checkpoint")
        magic, version = struct.unpack_from(">ii", data, 0)
        if magic != Checkpoint.MAGIC:
            raise ValueError(f"{source} is not a quantfinlib checkpoint")
        if version > Checkpoint.FORMAT_VERSION:
            raise ValueError(
                f"{source} uses checkpoint format {version}; this build "
                f"reads up to {Checkpoint.FORMAT_VERSION}")
        self._sections: Dict[str, bytes] = {}
        pos = 8
        n = len(data)
        while pos < n:
            name_len = struct.unpack_from(">H", data, pos)[0]
            pos += 2
            name = data[pos:pos + name_len].decode("utf-8")
            pos += name_len
            length = struct.unpack_from(">i", data, pos)[0]
            pos += 4
            if length < 0 or pos + length > n:
                raise ValueError(
                    f"{source}: section '{name}' declares {length} bytes, "
                    f"{n - pos} remain")
            self._sections[name] = bytes(data[pos:pos + length])
            pos += length

    def section(self, name: str, body: StateReader) -> bool:
        """Restores one model from the named section. Returns False
        when the section is absent (the caller decides whether that is
        a cold start or an error). Raises when the model does not
        consume the payload exactly."""
        payload = self._sections.get(name)
        if payload is None:
            return False
        r = BinReader(payload)
        body(r)
        if r.remaining() > 0:
            raise ValueError(
                f"section '{name}': {r.remaining()} bytes left unread "
                f"-- writer/reader format mismatch")
        return True

    def names(self) -> List[str]:
        """The section names present, in file order."""
        return list(self._sections.keys())


class Checkpoint:
    """Namespace matching Java ``persist.Checkpoint`` -- not
    instantiated; use :meth:`writer`/:meth:`reader` plus the array
    helpers shared by models' state formats."""

    MAGIC = 0x51464C43       # "QFLC"
    FORMAT_VERSION = 1

    def __new__(cls):
        raise TypeError("Checkpoint is a namespace; use its static methods")

    @staticmethod
    def writer(path) -> CheckpointWriter:
        """Opens a writer; nothing touches ``path`` until ``close()``."""
        return CheckpointWriter(path)

    @staticmethod
    def reader(path) -> CheckpointReader:
        """Loads a checkpoint fully into memory and validates the header."""
        return CheckpointReader(path)

    @staticmethod
    def require_version(inp: BinReader, expected: int, model: str) -> None:
        """Reads and checks a model's leading state-version byte -- the
        shared first line of every ``read_state``."""
        v = inp.read_byte()
        if v != expected:
            raise ValueError(
                f"{model} state version {v} not supported (this build "
                f"reads version {expected})")

    @staticmethod
    def write_doubles(out: BinWriter, arr) -> None:
        """Length-prefixed double array."""
        seq = list(arr)
        out.write_int(len(seq))
        for v in seq:
            out.write_double(float(v))

    @staticmethod
    def read_doubles_into(inp: BinReader, arr) -> None:
        """Reads a length-prefixed double array INTO ``arr`` -- a
        length mismatch means the checkpoint was written by a
        differently-configured instance and raises before touching
        it."""
        n = inp.read_int()
        if n != len(arr):
            raise ValueError(
                f"checkpoint array has {n} entries, this instance has "
                f"{len(arr)} -- incompatible configuration")
        for i in range(n):
            arr[i] = inp.read_double()

    @staticmethod
    def write_longs(out: BinWriter, arr) -> None:
        """Length-prefixed long array."""
        seq = list(arr)
        out.write_int(len(seq))
        for v in seq:
            out.write_long(int(v))

    @staticmethod
    def read_longs_into(inp: BinReader, arr) -> None:
        """Long-array counterpart of :meth:`read_doubles_into`."""
        n = inp.read_int()
        if n != len(arr):
            raise ValueError(
                f"checkpoint array has {n} entries, this instance has "
                f"{len(arr)} -- incompatible configuration")
        for i in range(n):
            arr[i] = inp.read_long()
