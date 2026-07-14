"""Pins for quantfinlib.pricing.vol_surface, ported from VolSurfaceTest.java."""

import math

import pytest

from quantfinlib.pricing import BlackScholes, OptionType, VolSurface


@pytest.fixture()
def surface():
    return (VolSurface.builder()
            .add(0.25, 90, 0.25).add(0.25, 100, 0.20).add(0.25, 110, 0.22)
            .add(1.00, 90, 0.28).add(1.00, 100, 0.24).add(1.00, 110, 0.25)
            .build())


def test_recovers_pillar_quotes_exactly(surface):
    assert surface.vol(0.25, 100) == pytest.approx(0.20, abs=1e-12)
    assert surface.vol(1.00, 90) == pytest.approx(0.28, abs=1e-12)
    assert len(surface.expiries()) == 2
    assert len(surface.strikes(0.25)) == 3


def test_interpolates_linearly_across_strikes(surface):
    # Midway between 90 (0.25) and 100 (0.20).
    assert surface.vol(0.25, 95) == pytest.approx(0.225, abs=1e-12)
    # Flat extrapolation beyond the wings.
    assert surface.vol(0.25, 80) == pytest.approx(0.25, abs=1e-12)
    assert surface.vol(0.25, 130) == pytest.approx(0.22, abs=1e-12)


def test_interpolates_in_total_variance_across_expiries(surface):
    # K=100: w1 = 0.04*0.25 = 0.01, w2 = 0.0576*1.0 = 0.0576.
    # t=0.625 (halfway): w = 0.0338 -> vol = sqrt(0.0338/0.625).
    assert surface.vol(0.625, 100) == pytest.approx(math.sqrt(0.0338 / 0.625), abs=1e-9)
    # Flat vol extrapolation outside the pillar range.
    assert surface.vol(0.10, 100) == pytest.approx(0.20, abs=1e-12)
    assert surface.vol(3.00, 100) == pytest.approx(0.24, abs=1e-12)


def test_smile_shape_is_preserved(surface):
    # Put skew: downside strikes carry higher vol at every expiry.
    assert surface.vol(1.0, 90) > surface.vol(1.0, 100)
    assert surface.vol(0.625, 90) > surface.vol(0.625, 100)
    assert surface.skew(1.0, 90, 100) < 0


def test_builds_from_market_prices_via_implied_vol():
    spot, rate, carry = 100.0, 0.03, 0.01
    p1 = BlackScholes.price(OptionType.CALL, spot, 100, rate, carry, 0.30, 0.5)
    p2 = BlackScholes.price(OptionType.PUT, spot, 90, rate, carry, 0.35, 0.5)

    from_prices = (VolSurface.builder()
                   .add_from_price(OptionType.CALL, p1, spot, 100, rate, carry, 0.5)
                   .add_from_price(OptionType.PUT, p2, spot, 90, rate, carry, 0.5)
                   .build())

    assert from_prices.vol(0.5, 100) == pytest.approx(0.30, abs=1e-6)
    assert from_prices.vol(0.5, 90) == pytest.approx(0.35, abs=1e-6)


def test_prices_with_the_surface_vol(surface):
    expected = BlackScholes.price(OptionType.CALL, 100, 95,
                                  0.02, 0, surface.vol(0.5, 95), 0.5)
    assert surface.price(OptionType.CALL, 100, 95, 0.02, 0, 0.5) == pytest.approx(
        expected, abs=1e-12)


def test_validates_inputs(surface):
    with pytest.raises(RuntimeError):
        VolSurface.builder().build()          # IllegalStateException in Java
    with pytest.raises(ValueError):
        VolSurface.builder().add(-1, 100, 0.2)
    with pytest.raises(ValueError):
        VolSurface.builder().add(0.5, 100, math.nan)  # NaN vol from a bad inversion
    with pytest.raises(ValueError):
        surface.strikes(0.33)
