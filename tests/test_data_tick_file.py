"""QFLT tick-file tests, pinning Java ``data.TickFileWriter``/
``data.TickFileReader``: round trip through the binary format, the
magic/version reject path, and truncated-file detection.
"""

import struct

import pytest

from quantfinlib.data import TickFileWriter, replay, replay_paced
from quantfinlib.data.tick_file_writer import MAGIC, VERSION


def test_round_trip_ticks_and_symbols(tmp_path):
    path = tmp_path / "session.qflt"
    with TickFileWriter(path) as w:
        w.define_symbol(0, "AAPL")
        w.define_symbol(1, "MSFT")
        w.write(0, 100.5, 10.0, 1_000)
        w.write(1, 200.25, 5.0, 2_000)
        w.write(0, 100.6, 20.0, 3_000)
        assert w.tick_count() == 3

    ticks = []
    symbols = []
    n = replay(path, lambda sid, p, s, t: ticks.append((sid, p, s, t)),
              lambda sid, name: symbols.append((sid, name)))
    assert n == 3
    assert symbols == [(0, "AAPL"), (1, "MSFT")]
    assert ticks == [
        (0, 100.5, 10.0, 1_000),
        (1, 200.25, 5.0, 2_000),
        (0, 100.6, 20.0, 3_000),
    ]


def test_define_symbol_is_idempotent(tmp_path):
    path = tmp_path / "session.qflt"
    with TickFileWriter(path) as w:
        w.define_symbol(0, "AAPL")
        w.define_symbol(0, "AAPL")   # repeated: no second record written
        w.write(0, 1.0, 1.0, 0)

    symbols = []
    replay(path, lambda *a: None, lambda sid, name: symbols.append((sid, name)))
    assert symbols == [(0, "AAPL")]


def test_write_before_define_raises(tmp_path):
    path = tmp_path / "session.qflt"
    with TickFileWriter(path) as w:
        with pytest.raises(RuntimeError, match="not defined"):
            w.write(5, 1.0, 1.0, 0)


def test_replay_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.qflt"
    path.write_bytes(b"XXXX" + bytes([VERSION]))
    with pytest.raises(ValueError, match="not a QFLT tick file"):
        replay(path, lambda *a: None)


def test_replay_rejects_unsupported_version(tmp_path):
    path = tmp_path / "bad_version.qflt"
    path.write_bytes(struct.pack(">i", MAGIC) + bytes([99]))
    with pytest.raises(ValueError, match="unsupported QFLT version"):
        replay(path, lambda *a: None)


def test_replay_rejects_truncated_file(tmp_path):
    path = tmp_path / "truncated.qflt"
    with TickFileWriter(path) as w:
        w.define_symbol(0, "AAPL")
        w.write(0, 1.0, 1.0, 0)
    # Chop off the last few bytes mid-record.
    data = path.read_bytes()
    path.write_bytes(data[:-3])
    with pytest.raises(ValueError, match="truncated QFLT file"):
        replay(path, lambda *a: None)


def test_replay_rejects_corrupt_record_type(tmp_path):
    path = tmp_path / "corrupt.qflt"
    with TickFileWriter(path) as w:
        w.define_symbol(0, "AAPL")
        w.write(0, 1.0, 1.0, 0)
    data = bytearray(path.read_bytes())
    # The first record byte right after the header is the TYPE_SYMBOL
    # tag (1); stomp it with an unrecognized type.
    data[5] = 9
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="corrupt record type"):
        replay(path, lambda *a: None)


def test_replay_paced_reproduces_gaps_scaled_by_speed(tmp_path, monkeypatch):
    path = tmp_path / "paced.qflt"
    with TickFileWriter(path) as w:
        w.define_symbol(0, "AAPL")
        w.write(0, 1.0, 1.0, 0)
        w.write(0, 1.1, 1.0, 100_000_000)   # 100ms gap

    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    ticks = []
    n = replay_paced(path, lambda sid, p, s, t: ticks.append(t), speed_multiplier=2.0)
    assert n == 2
    assert len(slept) == 1
    assert slept[0] == pytest.approx(0.05, abs=1e-9)   # 100ms / 2x speed


def test_replay_paced_rejects_nonpositive_speed(tmp_path):
    path = tmp_path / "x.qflt"
    with TickFileWriter(path) as w:
        w.define_symbol(0, "AAPL")
    with pytest.raises(ValueError):
        replay_paced(path, lambda *a: None, speed_multiplier=0)
