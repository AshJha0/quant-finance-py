"""Index construction pins, ported from Java AssetClassRoundTest and
AssetClassEdgeTest (index sections).

Weight schemes match hand arithmetic, the divisor adjustment keeps the
level continuous through a member swap, and turnover is half the
absolute weight shift.
"""

import numpy as np
import pytest

from quantfinlib.markets import IndexConstruction


def test_weight_schemes_match_hand_arithmetic():
    np.testing.assert_allclose(
        IndexConstruction.cap_weights([10, 20], [100, 50], [1, 1]),
        [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(
        IndexConstruction.cap_weights([10, 20], [100, 50], [1, 0.5]),
        [2.0 / 3, 1.0 / 3], atol=1e-12)
    # The Dow accident: the expensive stock owns the index.
    np.testing.assert_allclose(
        IndexConstruction.price_weights([50, 400]), [1.0 / 9, 8.0 / 9], atol=1e-12)
    np.testing.assert_array_equal(
        IndexConstruction.equal_weights(4), [0.25, 0.25, 0.25, 0.25])


def test_divisor_adjustment_keeps_the_level_continuous_through_a_member_swap():
    p = [10, 20]
    s = [100, 50]
    f = [1, 1]
    divisor = 20
    level = IndexConstruction.level(p, s, f, divisor)   # 2000/20 = 100
    assert level == pytest.approx(100, abs=1e-12)

    # Swap member 2 (cap 1000) for a new stock with cap 1200: the
    # divisor rescales 20 * 2200/2000 = 22 and the level holds at 100.
    new_divisor = IndexConstruction.adjust_divisor(divisor, 2000, 2200)
    assert new_divisor == pytest.approx(22, abs=1e-12)
    assert IndexConstruction.level([10, 30], [100, 40], f,
                                   new_divisor) == pytest.approx(level, abs=1e-12), \
        "membership changes must not move the index"


def test_turnover_is_half_the_absolute_weight_shift():
    assert IndexConstruction.turnover([0.5, 0.5], [0.6, 0.4]) == pytest.approx(
        0.1, abs=1e-12)
    assert IndexConstruction.turnover([0.3, 0.7], [0.3, 0.7]) == 0.0


def test_weights_normalize_to_one_and_turnover_matches_a_second_hand_pin():
    # priceWeights {20,30,50} -> {0.2, 0.3, 0.5} exactly.
    np.testing.assert_allclose(
        IndexConstruction.price_weights([20, 30, 50]), [0.2, 0.3, 0.5], atol=1e-15)
    cap = IndexConstruction.cap_weights([11, 23, 47], [9, 13, 3], [1, 0.7, 0.4])
    assert float(np.sum(cap)) == pytest.approx(1.0, abs=1e-15), \
        "cap weights are a probability vector"
    # Equal quarter weights to {0.4, 0.3, 0.2, 0.1}:
    # 0.5 * (0.15 + 0.05 + 0.05 + 0.15) = 0.2.
    assert IndexConstruction.turnover(
        IndexConstruction.equal_weights(4), [0.4, 0.3, 0.2, 0.1]) == pytest.approx(
        0.2, abs=1e-12)


def test_index_gates_refuse_nonsense():
    with pytest.raises(ValueError):
        IndexConstruction.turnover([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        IndexConstruction.cap_weights([10], [100], [1.5])       # float > 1
    with pytest.raises(ValueError):
        IndexConstruction.level([10], [100], [1], 0)            # divisor 0
    with pytest.raises(ValueError):
        IndexConstruction.equal_weights(0)
    with pytest.raises(ValueError):
        IndexConstruction.adjust_divisor(-1, 100, 110)
    with pytest.raises(ValueError):
        IndexConstruction.price_weights([10, -5])
