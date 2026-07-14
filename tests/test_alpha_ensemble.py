"""Pins for the IC-weighted alpha ensemble.

Java source: QuantModels6Test.java (AlphaEnsemble section).
"""

import math

import numpy as np
import pytest

from quantfinlib.alpha.alpha_ensemble import AlphaEnsemble


def test_weight_concentrates_on_the_component_that_predicts():
    ens = AlphaEnsemble(2, 0.01)
    rng = np.random.default_rng(42)
    values = np.zeros(2)
    planted = 0.0
    for _ in range(3_000):
        # The return realized NOW was leaned by component 0's PREVIOUS
        # value.
        ret = 1e-4 * (0.6 * planted + rng.standard_normal())
        planted = rng.random() * 2 - 1
        values[0] = planted                # predictive
        values[1] = rng.random() * 2 - 1   # pure noise
        ens.on_observation(values, ret)
    # Theoretical IC of the plant is close to 0.33; the decayed estimate
    # sits near it.
    assert ens.component_ic(0) > 0.2
    assert abs(ens.component_ic(1)) < 0.15
    # Component 1 votes hard the other way -- and is ignored.
    values[0] = 1
    values[1] = -1
    combined = ens.combined(values)
    assert combined > 0.15
    assert combined <= 1


def test_a_barely_trusted_blend_is_a_barely_sized_signal():
    # The weights are NOT renormalized: with only weak trust, the output
    # must be small -- never a lone IC-0.02 component at full strength.
    ens = AlphaEnsemble(1, 0.01)
    rng = np.random.default_rng(9)
    v = np.zeros(1)
    planted = 0.0
    for _ in range(3_000):
        ret = 1e-4 * (0.1 * planted + rng.standard_normal())
        planted = rng.random() * 2 - 1
        v[0] = planted
        ens.on_observation(v, ret)
    ic = ens.component_ic(0)
    assert 0 < ic < 0.2
    v[0] = 1
    assert ens.combined(v) == pytest.approx(ic, abs=1e-9)


def test_silent_before_a_track_record_and_on_pure_noise():
    young = AlphaEnsemble(2, 0.01)
    v = np.array([1.0, 1.0])
    for _ in range(50):
        young.on_observation(v, 1e-4)
    assert young.combined(v) == 0.0

    noise = AlphaEnsemble(2, 0.01)
    rng = np.random.default_rng(7)
    nv = np.zeros(2)
    for _ in range(3_000):
        nv[0] = rng.random() * 2 - 1
        nv[1] = rng.random() * 2 - 1
        noise.on_observation(nv, 1e-4 * rng.standard_normal())
    nv[0] = 1
    nv[1] = 1
    assert abs(noise.combined(nv)) < 0.15


def test_non_finite_inputs_skip_scoring_but_never_poison_or_inflate():
    ens = AlphaEnsemble(2, 0.5)
    # Establish real IC history on component 0.
    ens.on_observation(np.array([0.8, -0.1]), 0.0)   # first call: snapshot only
    ens.on_observation(np.array([-0.6, 0.2]), 1e-4)
    ens.on_observation(np.array([0.5, 0.5]), -1e-4)
    ic0 = ens.component_ic(0)
    assert ic0 != 0 and math.isfinite(ic0)
    before = ens.samples()

    ens.on_observation(np.array([0.7, 0.7]), math.nan)
    assert ens.samples() == before

    # A NaN COMPONENT in the snapshot: its scoring is skipped (history
    # preserved, not NaN-poisoned to a permanent zero), the finite
    # sibling still scores, and the observation still counts.
    ens.on_observation(np.array([math.nan, 0.3]), 1e-4)   # arms NaN snapshot
    ens.on_observation(np.array([0.1, 0.1]), 1e-4)        # scores comp1 only
    assert ens.samples() == before + 2
    assert ens.component_ic(0) != 0 and math.isfinite(ens.component_ic(0))

    # An observation where NOTHING scores must not inflate the record.
    ens.on_observation(np.array([math.nan, math.nan]), 1e-4)
    all_samples = ens.samples()
    ens.on_observation(np.array([0.2, 0.2]), 1e-4)   # all-NaN snapshot: no scoring
    assert ens.samples() == all_samples

    with pytest.raises(ValueError):
        AlphaEnsemble(0)
    with pytest.raises(ValueError):
        ens.on_observation(np.zeros(1), 0.0)
