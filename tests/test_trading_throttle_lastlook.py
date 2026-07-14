"""OrderThrottle token-bucket arithmetic and LastLookGate symmetric
accept/reject. Ported from Java OrderThrottleTest / LastLookGateTest.
"""

import math

import numpy as np
import pytest

from quantfinlib.trading.last_look_gate import LastLookGate
from quantfinlib.trading.order_throttle import OrderThrottle

MS = 1_000_000
PIP = 0.0001


# ------------------------------------------------------------------
# OrderThrottle
# ------------------------------------------------------------------


def test_burst_then_sustained_rate():
    t = OrderThrottle(1_000, 5)      # 1k/s, burst 5
    now = 0
    for i in range(5):
        assert t.try_acquire(now), f"burst permit {i}"
    assert not t.try_acquire(now), "bucket exhausted"
    # 1 ms refills exactly one token at 1k/s.
    assert t.try_acquire(now + MS)
    assert not t.try_acquire(now + MS)
    assert t.acquired_count() == 6
    assert t.throttled_count() == 2


def test_bucket_never_banks_more_than_the_burst():
    t = OrderThrottle(1_000, 3)
    idle = 10_000 * MS                # 10 s idle
    granted = 0
    while t.try_acquire(idle):
        granted += 1
    assert granted == 3, "idle time must not bank beyond the burst"


def test_nanos_until_available_is_an_exact_wait_hint():
    t = OrderThrottle(100, 1)          # 10 ms per token
    assert t.try_acquire(0)
    wait = t.nanos_until_available(0)
    assert wait > 0
    assert not t.try_acquire(wait - 1_000)
    assert t.try_acquire(wait)


def test_sustained_throughput_matches_the_configured_rate():
    t = OrderThrottle(50_000, 10)
    granted = 0
    now = 0
    while now < 1_000_000_000:
        if t.try_acquire(now):
            granted += 1
        now += 10_000
    assert abs(granted - 50_000) <= 11, f"granted {granted} permits at a 50k/s limit"


def test_order_throttle_validates_inputs():
    with pytest.raises(ValueError):
        OrderThrottle(0, 5)
    with pytest.raises(ValueError):
        OrderThrottle(100, 0)


# ------------------------------------------------------------------
# LastLookGate
# ------------------------------------------------------------------


def test_within_tolerance_always_fills():
    g = LastLookGate(PIP)
    assert g.accept(True, 1.08500, 1.08505)     # half a pip
    assert g.accept(True, 1.08500, 1.08495)
    assert g.accept(False, 1.08500, 1.08510)    # exactly at tolerance
    assert g.accepts() == 3
    assert g.rejects() == 0


def test_symmetric_rejection_in_both_directions():
    g = LastLookGate(PIP)
    # Maker sells; fair jumps 3 pips against the maker: classic pick-off.
    assert not g.accept(True, 1.08500, 1.08530)
    assert g.maker_protective_rejects() == 1
    # Maker sells; fair DROPS 3 pips (maker would love this fill): a
    # symmetric gate rejects anyway -- that's the Code's requirement.
    assert not g.accept(True, 1.08500, 1.08470)
    assert g.taker_protective_rejects() == 1
    # Maker buys mirror.
    assert not g.accept(False, 1.08500, 1.08470)    # fair fell: hurts buyer-maker
    assert g.maker_protective_rejects() == 2
    assert not g.accept(False, 1.08500, 1.08530)
    assert g.taker_protective_rejects() == 2
    assert g.reject_rate() == pytest.approx(1.0, abs=1e-12)


def test_statistics_split_evenly_under_symmetric_noise():
    g = LastLookGate(PIP)
    rng = np.random.default_rng(3)
    for _ in range(100_000):
        move = (rng.random() - 0.5) * 6 * PIP
        g.accept(bool(rng.integers(0, 2)), 1.08500, 1.08500 + move)
    assert g.rejects() > 0
    asymmetry = g.maker_protective_rejects() / g.rejects()
    assert abs(asymmetry - 0.5) < 0.02, \
        f"symmetric last look must reject both directions equally: {asymmetry}"


def test_last_look_gate_validates_tolerance():
    with pytest.raises(ValueError):
        LastLookGate(0)
    with pytest.raises(ValueError):
        LastLookGate(-0.0001)
    with pytest.raises(ValueError):
        LastLookGate(math.nan)


def test_last_look_gate_reject_rate_is_nan_before_any_decision():
    g = LastLookGate(PIP)
    assert math.isnan(g.reject_rate())
