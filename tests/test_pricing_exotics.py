"""Pins for digital, touch and barrier pricing, ported from
ExoticOptionsTest.java: exact parities against the vanilla pricer,
limiting behavior, and Monte Carlo cross-checks of the reflection-
principle formulas (with the discrete-monitoring bias of the simulation
accounted for in the assertion direction).

Port note on the MC cross-checks: the Java tests seed java.util.Random;
here the same GBM paths are drawn (vectorized) from
numpy.random.default_rng with fixed seeds. The assertions are
statistical (tolerance 0.03 / 0.002 as in Java), not exact pins, so the
RNG swap is immaterial — the closed forms under test are identical.
"""

import math

import numpy as np
import pytest

from quantfinlib.pricing import (BarrierOption, BlackScholes, DigitalOption,
                                 OptionType, TouchOption)

S = 1.0850   # FX-style levels (EURUSD)
R = 0.045    # domestic (USD) rate
Q = 0.030    # foreign (EUR) rate = carry
VOL = 0.09
T = 0.5

# ------------------------------------------------------------------
# Digitals
# ------------------------------------------------------------------


def test_digital_parities_hold_exactly():
    k = 1.10
    cash_call = DigitalOption.cash_or_nothing(OptionType.CALL, S, k, R, Q, VOL, T, 1)
    cash_put = DigitalOption.cash_or_nothing(OptionType.PUT, S, k, R, Q, VOL, T, 1)
    # Call + put digitals pay 1 in every state: worth the discount factor.
    assert cash_call + cash_put == pytest.approx(math.exp(-R * T), abs=1e-12)

    asset_call = DigitalOption.asset_or_nothing(OptionType.CALL, S, k, R, Q, VOL, T)
    asset_put = DigitalOption.asset_or_nothing(OptionType.PUT, S, k, R, Q, VOL, T)
    assert asset_call + asset_put == pytest.approx(S * math.exp(-Q * T), abs=1e-12)

    # A vanilla is asset-or-nothing minus K cash-or-nothings.
    vanilla = BlackScholes.price(OptionType.CALL, S, k, R, Q, VOL, T)
    assert asset_call - k * cash_call == pytest.approx(vanilla, abs=1e-12)


def test_digital_expiry_and_validation():
    assert DigitalOption.cash_or_nothing(OptionType.CALL, 1.10, 1.05, R, Q, VOL, 0, 5) == 5
    assert DigitalOption.cash_or_nothing(OptionType.CALL, 1.00, 1.05, R, Q, VOL, 0, 5) == 0
    assert DigitalOption.asset_or_nothing(OptionType.CALL, 1.10, 1.05, R, Q, VOL, 0) == 1.10
    with pytest.raises(ValueError):
        DigitalOption.cash_or_nothing(OptionType.CALL, -1, 1, R, Q, VOL, T, 1)


# ------------------------------------------------------------------
# Touches
# ------------------------------------------------------------------


def test_touch_properties_and_complement():
    upper = 1.12
    p = TouchOption.hit_probability(S, upper, R, Q, VOL, T)
    assert 0 < p < 1
    # Barrier at spot: certain touch. Barrier far away: nearly none.
    assert TouchOption.hit_probability(S, S, R, Q, VOL, T) == 1
    assert TouchOption.hit_probability(S, 2.5, R, Q, VOL, T) < 1e-9
    # One-touch + no-touch = discounted payout (complement identity).
    ot = TouchOption.one_touch(S, upper, R, Q, VOL, T, 100)
    nt = TouchOption.no_touch(S, upper, R, Q, VOL, T, 100)
    assert ot + nt == pytest.approx(100 * math.exp(-R * T), abs=1e-9)
    # Longer expiry can only raise the hit probability.
    assert TouchOption.hit_probability(S, upper, R, Q, VOL, 1.0) > p
    # Lower barrier branch.
    p_low = TouchOption.hit_probability(S, 1.05, R, Q, VOL, T)
    assert 0 < p_low < 1


PATHS = 60_000
STEPS = 400


def _mc_paths(seed):
    """Deterministic GBM path matrix (log-Euler is exact for GBM per step)."""
    rng = np.random.default_rng(seed)
    dt = T / STEPS
    drift = (R - Q - 0.5 * VOL * VOL) * dt
    diff = VOL * math.sqrt(dt)
    z = rng.standard_normal((PATHS, STEPS))
    return S * np.exp(np.cumsum(drift + diff * z, axis=1))


def test_touch_probability_matches_monte_carlo():
    paths = _mc_paths(42)
    upper = 1.11
    closed = TouchOption.hit_probability(S, upper, R, Q, VOL, T)
    mc = float(np.mean(np.any(paths >= upper, axis=1)))
    # Discrete monitoring misses intra-step touches: MC is biased LOW.
    assert mc <= closed + 0.005
    assert mc == pytest.approx(closed, abs=0.03)

    lower = 1.06
    closed_low = TouchOption.hit_probability(S, lower, R, Q, VOL, T)
    mc_low = float(np.mean(np.any(paths <= lower, axis=1)))
    assert mc_low == pytest.approx(closed_low, abs=0.03)


# ------------------------------------------------------------------
# Barriers
# ------------------------------------------------------------------


def test_in_out_parity_reconstructs_the_vanilla():
    k = 1.10
    h = 1.05
    vanilla = BlackScholes.price(OptionType.CALL, S, k, R, Q, VOL, T)
    ki = BarrierOption.down_and_in_call(S, k, h, R, Q, VOL, T)
    ko = BarrierOption.down_and_out_call(S, k, h, R, Q, VOL, T)
    assert ki + ko == pytest.approx(vanilla, abs=1e-12)
    assert ki > 0 and ko > 0

    hp = 1.13
    kp = 1.08
    vanilla_put = BlackScholes.price(OptionType.PUT, S, kp, R, Q, VOL, T)
    assert (BarrierOption.up_and_in_put(S, kp, hp, R, Q, VOL, T)
            + BarrierOption.up_and_out_put(S, kp, hp, R, Q, VOL, T)) == pytest.approx(
                vanilla_put, abs=1e-12)


def test_limiting_barriers_recover_vanilla_and_zero():
    k = 1.10
    vanilla = BlackScholes.price(OptionType.CALL, S, k, R, Q, VOL, T)
    # A barrier miles below spot never knocks: KO ~ vanilla, KI ~ 0.
    assert BarrierOption.down_and_out_call(S, k, 0.40, R, Q, VOL, T) == pytest.approx(
        vanilla, abs=1e-9)
    assert BarrierOption.down_and_in_call(S, k, 0.40, R, Q, VOL, T) < 1e-9
    # A barrier just below spot almost surely knocks: KO ~ 0.
    assert BarrierOption.down_and_out_call(S, k, S - 1e-4, R, Q, VOL, T) < 0.002


def test_barrier_price_matches_monte_carlo():
    paths = _mc_paths(4242)
    k = 1.10
    h = 1.05
    closed = BarrierOption.down_and_out_call(S, k, h, R, Q, VOL, T)
    alive = ~np.any(paths <= h, axis=1)
    payoff = np.where(alive, np.maximum(0.0, paths[:, -1] - k), 0.0)
    mc = math.exp(-R * T) * float(np.mean(payoff))
    # Discrete monitoring misses knocks: MC knock-OUT is biased HIGH.
    assert mc >= closed - 0.0005
    assert mc == pytest.approx(closed, abs=0.002)


def test_reverse_and_breached_barriers_are_rejected():
    with pytest.raises(ValueError):  # breached down barrier
        BarrierOption.down_and_out_call(S, 1.10, 1.09, R, Q, VOL, T)
    with pytest.raises(ValueError):  # reverse: H > K on a call
        BarrierOption.down_and_out_call(1.20, 1.02, 1.05, R, Q, VOL, T)
    with pytest.raises(ValueError):  # breached up barrier
        BarrierOption.up_and_out_put(S, 1.08, 1.05, R, Q, VOL, T)
    with pytest.raises(ValueError):  # reverse: H < K on a put
        BarrierOption.up_and_out_put(1.00, 1.15, 1.10, R, Q, VOL, T)
