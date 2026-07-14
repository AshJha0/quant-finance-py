"""Pins for the online ridge-SGD alpha-weight learner.

Java source: QuantModels3Test.java (OnlineAlphaLearner section). The
Java ``trainFrom``/``predictFrom`` overloads pull features from a live
``SignalEngine`` this port does not carry (see
:mod:`quantfinlib.alpha.online_alpha_learner`); the alignment
regression below uses :meth:`OnlineAlphaLearner.train_snapshot`
instead, which preserves the same "fit the snapshot, not the
contemporaneous features" contract generically.
"""

import numpy as np
import pytest

from quantfinlib.alpha.online_alpha_learner import OnlineAlphaLearner


def test_learner_recovers_planted_weights_and_reports_positive_ic():
    learner = OnlineAlphaLearner()
    rng = np.random.default_rng(42)
    for _ in range(30_000):
        qi = rng.random() * 2 - 1
        ti = rng.random() * 2 - 1
        ofi = rng.random() * 2 - 1
        mz = rng.random() * 2 - 1
        y = 0.6 * qi - 0.4 * mz + 0.1 * rng.standard_normal()
        learner.train(qi, ti, ofi, mz, y)
    assert learner.weight(0) == pytest.approx(0.6, abs=0.05)
    assert learner.weight(3) == pytest.approx(-0.4, abs=0.05)
    assert abs(learner.weight(1)) < 0.05
    assert abs(learner.weight(2)) < 0.05
    assert learner.out_of_sample_ic() > 0.85
    assert learner.samples() == 30_000


def test_pure_noise_target_cannot_masquerade_as_validated():
    learner = OnlineAlphaLearner()
    rng = np.random.default_rng(7)
    for _ in range(30_000):
        learner.train(rng.random() * 2 - 1, rng.random() * 2 - 1,
                      rng.random() * 2 - 1, rng.random() * 2 - 1,
                      0.02 * rng.standard_normal())
    assert abs(learner.out_of_sample_ic()) < 0.2
    for f in range(4):
        assert abs(learner.weight(f)) < 0.05


def test_fx_scale_returns_learn_just_as_well():
    learner = OnlineAlphaLearner()
    rng = np.random.default_rng(11)
    for _ in range(30_000):
        qi = rng.random() * 2 - 1
        mz = rng.random() * 2 - 1
        y = 1e-5 * (0.6 * qi - 0.4 * mz) + 1e-6 * rng.standard_normal()
        learner.train(qi, 0, 0, mz, y)
    assert learner.out_of_sample_ic() > 0.85
    assert learner.weight(0) == pytest.approx(6e-6, abs=1e-6)


def test_nan_inputs_never_poison_weights_or_ic():
    learner = OnlineAlphaLearner()
    learner.train(float("nan"), 0, 0, 0, 0.01)
    learner.train(0.5, float("inf"), 0, 0, 0.01)
    learner.train(0.5, 0, 0, 0, float("nan"))
    assert learner.samples() == 0
    assert learner.weight(0) == 0.0
    assert learner.out_of_sample_ic() == 0.0


def test_a_lucky_first_hour_is_not_a_track_record():
    # Even a perfect early IC emits no signal until the learner has one
    # full IC memory (~1/icAlpha samples) of evidence.
    learner = OnlineAlphaLearner()
    rng = np.random.default_rng(17)
    for _ in range(50):
        qi = rng.random() * 2 - 1
        learner.train(qi, 0, 0, 0, 0.5 * qi)   # noiseless: IC -> 1 fast
    assert learner.out_of_sample_ic() > 0
    assert learner.normalized_prediction(1, 0, 0, 0) == 0.0
    for _ in range(150):
        qi = rng.random() * 2 - 1
        learner.train(qi, 0, 0, 0, 0.5 * qi)
    assert learner.normalized_prediction(1, 0, 0, 0) > 0


def test_normalized_prediction_is_gated_on_demonstrated_ic():
    # Fresh learner: no demonstrated predictive power -> no signal.
    assert OnlineAlphaLearner().normalized_prediction(1, 1, 1, 1) == 0.0

    learner = OnlineAlphaLearner()
    rng = np.random.default_rng(3)
    for _ in range(30_000):
        qi = rng.random() * 2 - 1
        learner.train(qi, 0, 0, 0, 0.5 * qi + 0.05 * rng.standard_normal())
    strong = learner.normalized_prediction(1, 0, 0, 0)
    opposite = learner.normalized_prediction(-1, 0, 0, 0)
    assert strong > 0.5
    assert strong <= 1.0
    assert opposite < -0.5


def test_train_snapshot_fits_the_snapshot_not_contemporaneous_features():
    # The lookahead trap: at t+1 the contemporaneous features already
    # contain the (t, t+1] move. train_snapshot must fit the return
    # against the features snapshotted at t, not the ones visible now.
    learner = OnlineAlphaLearner(0.1, 0, 0.01)

    # State at t: heavy bid queue -> queue imbalance strongly POSITIVE.
    learner.train_snapshot(0.8, 0, 0, 0, 0.0)    # first call: snapshot only
    assert learner.samples() == 0

    # State at t+1: the book flipped -> queue imbalance strongly
    # NEGATIVE, and the interval's return was positive.
    learner.train_snapshot(-0.8, 0, 0, 0, 1.0)
    assert learner.samples() == 1
    assert learner.weight(0) > 0
