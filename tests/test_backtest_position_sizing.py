"""Pins for quantfinlib.backtest.portfolio.position_sizing.

Java source: PortfolioBacktestTest.java (positionSizingRules) — every
value hand-derivable: Kelly mu/sigma^2, the $2-stop fixed-fractional
share count, the 0.75/0.25 inverse-vol split, and the vol-target
leverage ratio.
"""

import numpy as np
import pytest

from quantfinlib.backtest.portfolio import PositionSizing


def test_position_sizing_rules():
    # Kelly = mu / sigma^2 = 0.02 / 0.01 = 2; half-Kelly halves it.
    assert PositionSizing.kelly_fraction(0.02, 0.01) == pytest.approx(2.0, abs=1e-12)
    assert PositionSizing.half_kelly(0.02, 0.01) == pytest.approx(1.0, abs=1e-12)
    # Risk 1% of 100k with a $2 stop distance: 500 shares.
    assert PositionSizing.fixed_fractional_quantity(
        100_000, 0.01, 50, 48) == pytest.approx(500, abs=1e-9)
    # Inverse vol: vols 0.1 and 0.3 -> weights 0.75 / 0.25.
    w = PositionSizing.inverse_volatility_weights([0.1, 0.3])
    assert w[0] == pytest.approx(0.75, abs=1e-12)
    assert w[1] == pytest.approx(0.25, abs=1e-12)
    # Scale 20% vol down to a 10% target: leverage 0.5.
    assert PositionSizing.volatility_target_leverage(
        0.20, 0.10) == pytest.approx(0.5, abs=1e-12)


def test_position_sizing_degenerate_branches():
    # Zero variance -> zero Kelly, not a division blow-up.
    assert PositionSizing.kelly_fraction(0.02, 0.0) == 0.0
    # Entry equals stop: undefined risk is refused.
    with pytest.raises(ValueError):
        PositionSizing.fixed_fractional_quantity(100_000, 0.01, 50, 50)
    # Any zero vol degrades to equal weight.
    w = PositionSizing.inverse_volatility_weights([0.0, 0.2, 0.3])
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3])
    # Zero current vol -> zero leverage, not infinity.
    assert PositionSizing.volatility_target_leverage(0.0, 0.10) == 0.0
