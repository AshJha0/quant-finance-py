"""Pins for quantfinlib.risk.concentration_risk (hand-derived, house style)."""

import pytest

from quantfinlib.risk import concentration_risk as cr


def test_herfindahl_pins():
    # Two equal names: shares 0.5 each -> HHI = 0.25 + 0.25 = 0.5.
    assert cr.herfindahl_index([50.0, 50.0]) == pytest.approx(0.5, abs=1e-15)
    # Single name: HHI = 1 (maximum concentration).
    assert cr.herfindahl_index([100.0]) == pytest.approx(1.0, abs=1e-15)
    # Signs are irrelevant: |exposure| shares (a short is still exposure).
    assert cr.herfindahl_index([50.0, -50.0]) == pytest.approx(0.5, abs=1e-15)
    # Empty book: 0 by convention, not NaN.
    assert cr.herfindahl_index([0.0, 0.0]) == 0.0


def test_effective_positions():
    # 1/HHI: two equal names -> 2 effective positions.
    assert cr.effective_positions([50.0, 50.0]) == pytest.approx(2.0, abs=1e-12)
    # [75, 25]: HHI = 0.5625 + 0.0625 = 0.625 -> 1.6 effective.
    assert cr.effective_positions([75.0, 25.0]) == pytest.approx(1.6, abs=1e-12)
    assert cr.effective_positions([0.0]) == 0.0


def test_top_n_share():
    # [40, 30, 20, 10]: top 2 = 70/100.
    e = [10.0, 40.0, 20.0, 30.0]
    assert cr.top_n_share(e, 2) == pytest.approx(0.70, abs=1e-15)
    assert cr.top_n_share(e, 4) == pytest.approx(1.0, abs=1e-15)
    assert cr.top_n_share(e, 10) == pytest.approx(1.0, abs=1e-15)  # n > len
    assert cr.top_n_share([0.0], 1) == 0.0


def test_shares_by_group():
    shares = cr.shares({"tech": 60.0, "energy": -30.0, "fin": 10.0})
    assert shares["tech"] == pytest.approx(0.6, abs=1e-15)
    assert shares["energy"] == pytest.approx(0.3, abs=1e-15), "abs of the short"
    assert shares["fin"] == pytest.approx(0.1, abs=1e-15)
    assert list(shares) == ["tech", "energy", "fin"], "insertion order preserved"
    # Zero total: all shares 0.
    assert cr.shares({"a": 0.0})["a"] == 0.0


def test_limit_breaches():
    groups = {"tech": 60.0, "energy": 30.0, "fin": 10.0}
    assert cr.limit_breaches(groups, 0.25) == ["tech", "energy"]
    assert cr.limit_breaches(groups, 0.6) == []      # share must EXCEED the limit
    assert cr.limit_breaches(groups, 0.59) == ["tech"]
