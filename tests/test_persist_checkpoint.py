"""Checkpoint tests, pinning Java ``persist.Checkpoint``: named-section
round trip, duplicate-section and version-byte rejection, corrupt-
header detection, and the three wired models' write/mutate/restore
round trips (LpScorecard, EwmaCovariance, VolumeCurve).
"""

import struct

import pytest

from quantfinlib.fx import LpScorecard
from quantfinlib.microstructure import EwmaCovariance, VolumeCurve
from quantfinlib.persist import BinReader, BinWriter, Checkpoint


# ------------------------------------------------------------------
# Low-level section framing
# ------------------------------------------------------------------

def test_round_trip_multiple_sections(tmp_path):
    path = tmp_path / "ckpt.bin"

    def write_a(out: BinWriter) -> None:
        out.write_int(42)
        Checkpoint.write_doubles(out, [1.0, 2.0, 3.0])

    def write_b(out: BinWriter) -> None:
        out.write_long(-7)

    with Checkpoint.writer(path) as w:
        w.section("a", write_a)
        w.section("b", write_b)

    r = Checkpoint.reader(path)
    assert r.names() == ["a", "b"]

    read_back = {}

    def read_a(inp: BinReader) -> None:
        read_back["a_int"] = inp.read_int()
        arr = [0.0, 0.0, 0.0]
        Checkpoint.read_doubles_into(inp, arr)
        read_back["a_doubles"] = arr

    def read_b(inp: BinReader) -> None:
        read_back["b_long"] = inp.read_long()

    assert r.section("a", read_a) is True
    assert r.section("b", read_b) is True
    assert read_back == {"a_int": 42, "a_doubles": [1.0, 2.0, 3.0], "b_long": -7}


def test_section_absent_returns_false(tmp_path):
    path = tmp_path / "ckpt.bin"
    with Checkpoint.writer(path) as w:
        w.section("present", lambda out: out.write_byte(1))
    r = Checkpoint.reader(path)
    assert r.section("absent", lambda inp: None) is False


def test_duplicate_section_name_raises_and_commits_nothing(tmp_path):
    path = tmp_path / "ckpt.bin"
    with pytest.raises(ValueError, match="duplicate checkpoint section"):
        with Checkpoint.writer(path) as w:
            w.section("x", lambda out: out.write_byte(1))
            w.section("x", lambda out: out.write_byte(2))
    assert not path.exists()


def test_failing_section_body_commits_nothing(tmp_path):
    path = tmp_path / "ckpt.bin"

    def bad_writer(out: BinWriter) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        with Checkpoint.writer(path) as w:
            w.section("good", lambda out: out.write_byte(1))
            w.section("bad", bad_writer)
    assert not path.exists()


def test_reader_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"XXXXYYYY")
    with pytest.raises(ValueError, match="not a quantfinlib checkpoint"):
        Checkpoint.reader(path)


def test_reader_rejects_newer_format_version(tmp_path):
    path = tmp_path / "future.bin"
    path.write_bytes(struct.pack(">ii", Checkpoint.MAGIC, 99))
    with pytest.raises(ValueError, match="checkpoint format 99"):
        Checkpoint.reader(path)


def test_section_leftover_bytes_raises_format_mismatch(tmp_path):
    path = tmp_path / "ckpt.bin"
    with Checkpoint.writer(path) as w:
        w.section("a", lambda out: (out.write_int(1), out.write_int(2)))
    r = Checkpoint.reader(path)
    with pytest.raises(ValueError, match="bytes left unread"):
        r.section("a", lambda inp: inp.read_int())   # only consumes half


def test_writer_commits_atomically_leaving_no_tmp_file(tmp_path):
    path = tmp_path / "ckpt.bin"
    with Checkpoint.writer(path) as w:
        w.section("a", lambda out: out.write_byte(1))
    assert path.exists()
    assert not (tmp_path / "ckpt.bin.tmp").exists()


# ------------------------------------------------------------------
# Model round trips: write, mutate, restore, assert equality of behavior
# ------------------------------------------------------------------

def test_lp_scorecard_write_mutate_restore_round_trip(tmp_path):
    path = tmp_path / "venues.bin"
    MS = 1_000_000
    c = LpScorecard(2, 0.5, 100 * MS)
    c.on_fill(0, True, 1.08502, 1.08501, MS)
    c.on_reject(0, True, 1.08501, 0, 2 * MS)
    c.on_reject(1, False, 1.0900, 10 * MS, MS)
    c.on_mid(1.0910, 200 * MS)

    with Checkpoint.writer(path) as w:
        w.section("venues", c.write_state)

    # Mutate the original after checkpointing to prove restore gives an
    # independent, correct snapshot rather than aliasing state.
    c.on_fill(0, True, 1.0, 1.0, 0)

    restored = LpScorecard(2, 0.5, 100 * MS)
    r = Checkpoint.reader(path)
    assert r.section("venues", restored.read_state) is True

    assert restored.attempts(0) == 2
    assert restored.fills(0) == 1
    assert restored.rejects(0) == 1
    assert restored.attempts(1) == 1
    # Cross-check against a freshly-built card fed the identical
    # pre-mutation event sequence.
    expected = LpScorecard(2, 0.5, 100 * MS)
    expected.on_fill(0, True, 1.08502, 1.08501, MS)
    expected.on_reject(0, True, 1.08501, 0, 2 * MS)
    expected.on_reject(1, False, 1.0900, 10 * MS, MS)
    expected.on_mid(1.0910, 200 * MS)
    for lp in (0, 1):
        assert restored.attempts(lp) == expected.attempts(lp)
        assert restored.fills(lp) == expected.fills(lp)
        assert restored.rejects(lp) == expected.rejects(lp)
        assert restored.reject_rate(lp) == pytest.approx(expected.reject_rate(lp))
        assert restored.avg_hold_nanos(lp) == pytest.approx(expected.avg_hold_nanos(lp))
        assert restored.effective_spread(lp) == pytest.approx(expected.effective_spread(lp))
        assert restored.post_reject_markout(lp) == pytest.approx(expected.post_reject_markout(lp))
    assert restored.matured_markouts() == expected.matured_markouts()


def test_lp_scorecard_rejects_unknown_version(tmp_path):
    path = tmp_path / "venues.bin"

    def bad_write(out) -> None:
        out.write_byte(3)   # unsupported version

    with Checkpoint.writer(path) as w:
        w.section("venues", bad_write)
    r = Checkpoint.reader(path)
    c = LpScorecard(1, 0.1, 1)
    with pytest.raises(ValueError, match="version 3 not supported"):
        r.section("venues", c.read_state)


def test_ewma_covariance_write_mutate_restore_round_trip(tmp_path):
    path = tmp_path / "cov.bin"
    ec = EwmaCovariance(3, 0.9)
    ec.on_returns([0.01, -0.02, 0.005])
    ec.on_returns([0.02, 0.01, -0.01])

    with Checkpoint.writer(path) as w:
        w.section("cov.basket", ec.write_state)

    ec.on_returns([1.0, 1.0, 1.0])   # mutate after checkpointing

    restored = EwmaCovariance(3, 0.9)
    r = Checkpoint.reader(path)
    assert r.section("cov.basket", restored.read_state) is True

    expected = EwmaCovariance(3, 0.9)
    expected.on_returns([0.01, -0.02, 0.005])
    expected.on_returns([0.02, 0.01, -0.01])

    assert restored.samples() == expected.samples() == 2
    for i in range(3):
        for j in range(3):
            assert restored.covariance(i, j) == pytest.approx(expected.covariance(i, j))
    assert restored.variance(0) == pytest.approx(expected.variance(0))
    assert restored.correlation(0, 1) == pytest.approx(expected.correlation(0, 1))


def test_ewma_covariance_rejects_basket_size_mismatch(tmp_path):
    path = tmp_path / "cov.bin"
    ec = EwmaCovariance(3, 0.9)
    ec.on_returns([0.01, 0.02, 0.03])
    with Checkpoint.writer(path) as w:
        w.section("cov", ec.write_state)

    wrong_size = EwmaCovariance(4, 0.9)
    r = Checkpoint.reader(path)
    with pytest.raises(ValueError, match="incompatible configuration"):
        r.section("cov", wrong_size.read_state)


def test_volume_curve_write_mutate_restore_round_trip(tmp_path):
    path = tmp_path / "volume.bin"
    vc = VolumeCurve(4, 0.2)
    vc.on_volume(0, 100)
    vc.on_volume(1, 50)
    vc.on_volume(2, 200)
    vc.on_volume(3, 25)
    vc.roll_day()
    vc.on_volume(0, 10)   # intraday state after roll_day: must NOT persist

    with Checkpoint.writer(path) as w:
        w.section("volume.AAPL", vc.write_state)

    vc.roll_day()   # mutate after checkpointing

    restored = VolumeCurve(4, 0.2)
    r = Checkpoint.reader(path)
    assert r.section("volume.AAPL", restored.read_state) is True

    assert restored.days_learned() == 1
    for b, expected in enumerate([100.0, 50.0, 200.0, 25.0]):
        assert restored.profile_volume(b) == pytest.approx(expected)
    # Intraday state resets on restore, regardless of what was live
    # before the checkpoint was taken.
    assert restored.realized_today() == 0.0


def test_volume_curve_rejects_bucket_count_mismatch(tmp_path):
    path = tmp_path / "volume.bin"
    vc = VolumeCurve(4, 0.2)
    vc.on_volume(0, 10)
    vc.roll_day()
    with Checkpoint.writer(path) as w:
        w.section("volume", vc.write_state)

    wrong = VolumeCurve(5, 0.2)
    r = Checkpoint.reader(path)
    with pytest.raises(ValueError, match="incompatible configuration"):
        r.section("volume", wrong.read_state)


def test_all_three_models_round_trip_in_one_checkpoint_file(tmp_path):
    path = tmp_path / "eod.bin"
    c = LpScorecard(1, 0.5, 1_000_000)
    c.on_fill(0, True, 1.0, 1.0, 0)
    ec = EwmaCovariance(2, 0.9)
    ec.on_returns([0.01, 0.02])
    vc = VolumeCurve(2, 0.1)
    vc.on_volume(0, 5)
    vc.roll_day()

    with Checkpoint.writer(path) as w:
        w.section("venues", c.write_state)
        w.section("cov.basket", ec.write_state)
        w.section("volume.AAPL", vc.write_state)

    r = Checkpoint.reader(path)
    assert set(r.names()) == {"venues", "cov.basket", "volume.AAPL"}

    c2 = LpScorecard(1, 0.5, 1_000_000)
    ec2 = EwmaCovariance(2, 0.9)
    vc2 = VolumeCurve(2, 0.1)
    assert r.section("venues", c2.read_state)
    assert r.section("cov.basket", ec2.read_state)
    assert r.section("volume.AAPL", vc2.read_state)

    assert c2.fills(0) == 1
    assert ec2.samples() == 1
    assert vc2.days_learned() == 1
