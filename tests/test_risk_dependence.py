"""Pins for quantfinlib.risk.dependence, ported from Java
MarketRiskTest.rankCorrelationsSurviveWhatWrecksPearson."""

import math

import numpy as np
import pytest

from quantfinlib.risk import dependence as dep
from quantfinlib.risk import risk_metrics as rm


def test_rank_correlations_survive_what_wrecks_pearson():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500)
    y = np.exp(3 * x)                     # monotone but violently nonlinear
    assert dep.spearman(x, y) == pytest.approx(1.0, abs=1e-9), \
        "a monotone transform is invisible to ranks"
    assert dep.kendall_tau(x, y) == pytest.approx(1.0, abs=1e-9)
    assert rm.correlation(x, y) < 0.75, \
        "while Pearson is dragged around by the convexity"


def test_kendall_hand_pin():
    # One discordant pair among three: (2 concordant - 1 discordant) / 3.
    tau = dep.kendall_tau([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])
    assert tau == pytest.approx(1.0 / 3, abs=1e-12)
    # Ties count as neither (tau-a): pairs (1,2),(1,3) concordant; (2,3)
    # tied in b -> (2 - 0)/3.
    assert dep.kendall_tau([1.0, 2.0, 3.0], [1.0, 2.0, 2.0]) == pytest.approx(
        2.0 / 3, abs=1e-12)


def test_elliptical_bridge_pins():
    assert dep.pearson_from_kendall(0) == pytest.approx(0, abs=1e-12)
    assert dep.pearson_from_kendall(1) == pytest.approx(1, abs=1e-12)
    assert dep.pearson_from_kendall(0.5) == pytest.approx(
        math.sin(math.pi * 0.25), abs=1e-12)
    with pytest.raises(ValueError):
        dep.pearson_from_kendall(1.5)
    with pytest.raises(ValueError):
        dep.pearson_from_kendall(math.nan)   # NaN-rejecting gate


def test_midranks():
    # [1, 2, 2, 3]: the tied 2s share (2+3)/2 = 2.5.
    assert dep.ranks([1.0, 2.0, 2.0, 3.0]).tolist() == [1.0, 2.5, 2.5, 4.0]
    # Order-independence: ranks follow values, not positions.
    assert dep.ranks([3.0, 1.0, 2.0]).tolist() == [3.0, 1.0, 2.0]
    # All tied: everyone gets the middle rank.
    assert dep.ranks([7.0, 7.0, 7.0]).tolist() == [2.0, 2.0, 2.0]


def test_spearman_with_ties_uses_midranks():
    # a = [1,2,2,3] -> ranks [1,2.5,2.5,4]; b = [1,2,3,4] -> [1,2,3,4].
    # Pearson of those ranks: hand-check monotone-ish but not 1.
    r = dep.spearman([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    # cov([1,2.5,2.5,4],[1,2,3,4]) / (sd*sd): means 2.5/2.5; devs
    # [-1.5,0,0,1.5] and [-1.5,-0.5,0.5,1.5]: cov = (2.25+0+0+2.25)/3 = 1.5;
    # sd_a = sqrt(4.5/3), sd_b = sqrt(5/3) -> r = 1.5/sqrt(2.5) = 0.9486833.
    assert r == pytest.approx(1.5 / math.sqrt(4.5 / 3 * 5 / 3), abs=1e-12)


def test_gates():
    with pytest.raises(ValueError):
        dep.spearman([1.0], [1.0])          # need >= 2
    with pytest.raises(ValueError):
        dep.kendall_tau([1.0, 2.0], [1.0])  # misaligned
