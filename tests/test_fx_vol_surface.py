"""Delta-quoted FX smile, ported from Java FxVolSurfaceTest.

Strike solving round-trips the target delta, the surface reproduces
the broker quote identities (RR/BF), the delta-neutral straddle really
is delta-neutral, and premium-adjusted solving differs from unadjusted
in the documented direction.
"""

import math

import pytest

from quantfinlib.fx import FxVolSurface
from quantfinlib.util import math_utils as mu

F = 1.0850    # EURUSD 3M forward
T = 0.25
ATM = 0.078   # 7.8 vols
RR25 = -0.010  # EURUSD-style: puts over calls
BF25 = 0.0022


def _surface():
    return (FxVolSurface.builder()
            .add(T, F, ATM, RR25, BF25)
            .add(1.0, 1.0920, 0.083, -0.012, 0.0030, -0.020, 0.0095)
            .build())


def _forward_delta(f, k, vol, t, call):
    """Forward delta of a call/put at a strike under Black (forward measure)."""
    sv = vol * math.sqrt(t)
    d1 = math.log(f / k) / sv + 0.5 * sv
    return mu.norm_cdf(d1) if call else -mu.norm_cdf(-d1)


def test_strike_for_delta_round_trips_the_target_delta():
    k25c = FxVolSurface.strike_for_delta(F, ATM, T, 0.25, True, False)
    k25p = FxVolSurface.strike_for_delta(F, ATM, T, -0.25, False, False)
    assert _forward_delta(F, k25c, ATM, T, True) == pytest.approx(0.25, abs=1e-6)
    assert _forward_delta(F, k25p, ATM, T, False) == pytest.approx(-0.25, abs=1e-6)
    assert k25c > F
    assert k25p < F


def test_dns_strike_is_delta_neutral():
    k = FxVolSurface.dns_strike(F, ATM, T, False)
    assert k == pytest.approx(F * math.exp(0.5 * ATM * ATM * T), abs=1e-12)
    dc = _forward_delta(F, k, ATM, T, True)
    dp = _forward_delta(F, k, ATM, T, False)
    assert dc + dp == pytest.approx(0, abs=1e-7)   # straddle delta cancels


def test_surface_reproduces_broker_quote_identities():
    s = _surface()
    p = s.pillar(0)
    # Three pillars at 3M (no 10d): 25P, ATM, 25C, strikes ascending.
    assert len(p.strikes) == 3
    v25p, v_atm, v25c = p.vols
    assert v25c - v25p == pytest.approx(RR25, abs=1e-12)               # risk reversal
    assert (v25c + v25p) / 2 - v_atm == pytest.approx(BF25, abs=1e-12)  # butterfly
    assert v_atm == pytest.approx(ATM, abs=1e-12)
    # Surface lookup at a pillar strike returns the pillar vol.
    assert s.vol(T, p.strikes[2]) == pytest.approx(v25c, abs=1e-10)
    # Negative RR: put wing above call wing (EURUSD skew).
    assert v25p > v25c


def test_ten_delta_wings_extend_the_smile():
    s = _surface()
    p = s.pillar(1)
    assert len(p.strikes) == 5
    # Wing vols beyond the wings are flat (no explosive extrapolation).
    assert s.vol(1.0, p.strikes[0] * 0.90) == pytest.approx(p.vols[0], abs=1e-10)
    assert s.vol(1.0, p.strikes[4] * 1.10) == pytest.approx(p.vols[4], abs=1e-10)


def test_time_interpolation_is_linear_in_total_variance():
    s = _surface()
    t_mid = 0.5
    f = s.forward_at(t_mid)
    k = f  # zero log-moneyness at t_mid
    v_lo = s.vol(T, s.forward_at(T))
    v_hi = s.vol(1.0, s.forward_at(1.0))
    w_expected = v_lo * v_lo * T + (v_hi * v_hi * 1.0 - v_lo * v_lo * T) * (t_mid - T) / (1.0 - T)
    assert s.vol(t_mid, k) == pytest.approx(math.sqrt(w_expected / t_mid), abs=1e-12)
    # Outside the quoted range: flat in the boundary smile.
    assert s.vol(0.05, k) == pytest.approx(s.vol(T, k), abs=1e-12)


def test_premium_adjusted_solving_hits_the_pa_delta():
    sv = ATM * math.sqrt(T)
    k = FxVolSurface.strike_for_delta(F, ATM, T, 0.25, True, True)
    d2 = math.log(F / k) / sv - 0.5 * sv
    assert (k / F) * mu.norm_cdf(d2) == pytest.approx(0.25, abs=1e-9)
    # Premium adjustment lowers the call strike for the same delta.
    k_unadj = FxVolSurface.strike_for_delta(F, ATM, T, 0.25, True, False)
    assert k < k_unadj
    # PA DNS strike is below forward.
    assert FxVolSurface.dns_strike(F, ATM, T, True) < F
    # Put side round-trips too.
    kp = FxVolSurface.strike_for_delta(F, ATM, T, -0.25, False, True)
    d2p = math.log(F / kp) / sv - 0.5 * sv
    assert -(kp / F) * mu.norm_cdf(-d2p) == pytest.approx(-0.25, abs=1e-9)


def test_validation_rejects_bad_input():
    with pytest.raises(RuntimeError):
        FxVolSurface.builder().build()
    with pytest.raises(ValueError):
        FxVolSurface.builder().add(0, F, ATM, 0, 0)
    with pytest.raises(ValueError):
        FxVolSurface.strike_for_delta(F, ATM, T, 1.5, True, False)
    with pytest.raises(ValueError):
        _surface().vol(T, -1)
    # An unattainable premium-adjusted delta is reported, not silently clamped.
    with pytest.raises(ValueError):
        FxVolSurface.strike_for_delta(F, ATM, T, 0.999, True, True)
