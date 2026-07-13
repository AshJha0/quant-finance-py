"""Credit curve + CDS pins, ported from Java CreditTest and
CreditRoundTripTest.

Pinned by the identities that define them: every input reprices exactly
(bootstrap), the credit triangle spread ~ h*(1-R) holds on a flat
curve, the flat-hazard ROUND TRIP recovers the planted hazard, upfront
is exactly linear (antisymmetric) in the coupon, and hazards
extrapolate flat beyond the last pillar.
"""

import math

import pytest

from quantfinlib.credit import CdsPricer, CreditCurve
from quantfinlib.rates import YieldCurve


def flat3() -> YieldCurve:
    return YieldCurve.of_zero_rates([1, 2, 3, 5, 7, 10], [0.03] * 6)


def test_flat_spreads_satisfy_the_credit_triangle():
    # 100bp flat, R = 40%: h ~ S/(1-R) = 1.667%. The quarterly
    # discretization and accrual-on-default term move it by well under a
    # basis point of hazard.
    curve = CreditCurve.bootstrap([1, 3, 5], [0.01, 0.01, 0.01], 0.40, flat3())
    assert curve.hazard(2.0) == pytest.approx(0.01 / 0.60, abs=3e-4)
    assert curve.hazard(4.0) == pytest.approx(0.01 / 0.60, abs=3e-4)
    # Survival is monotone from 1 and consistent with its complement.
    assert curve.survival_probability(0) == 1.0
    assert curve.survival_probability(1) > curve.survival_probability(5)
    assert curve.default_probability(3) == 1 - curve.survival_probability(3)


def test_bootstrap_reprices_every_pillar_exactly():
    tenors = [1, 3, 5, 7]
    spreads = [0.008, 0.012, 0.015, 0.016]     # upward credit curve
    discount = flat3()
    curve = CreditCurve.bootstrap(tenors, spreads, 0.40, discount)
    for t, s in zip(tenors, spreads):
        assert CdsPricer.par_spread(curve, discount, t) == pytest.approx(s, abs=1e-10), \
            f"{t}y must reprice"
    # Upward spreads need upward hazards.
    assert curve.hazard(4.0) > curve.hazard(0.5)


def test_upfront_is_zero_at_par_and_positive_for_cheap_coupons():
    discount = flat3()
    curve = CreditCurve.bootstrap([5], [0.03], 0.40, discount)
    par = CdsPricer.par_spread(curve, discount, 5)
    assert CdsPricer.upfront(curve, discount, par, 5) == pytest.approx(0.0, abs=1e-12)
    # Standard 100bp contract coupon on a 300bp name: the buyer pays
    # points up front — roughly (300-100)bp times the risky annuity.
    up = CdsPricer.upfront(curve, discount, 0.01, 5)
    assert up > 0
    assert up == pytest.approx(
        (par - 0.01) * CdsPricer.risky_annuity(curve, discount, 5), abs=1e-12)


def test_credit_gates_refuse_nonsense():
    d = flat3()
    with pytest.raises(ValueError):
        CreditCurve.bootstrap([3, 1], [0.01, 0.01], 0.4, d)     # descending
    with pytest.raises(ValueError):
        CreditCurve.bootstrap([1], [-0.01], 0.4, d)             # negative spread
    with pytest.raises(ValueError):
        CreditCurve.bootstrap([1], [0.01], 1.0, d)              # recovery = 1
    with pytest.raises(ValueError):
        CreditCurve.bootstrap([1], [100.0], 0.4, d)             # no hazard fits


# ------------------------------------------------------------ round trip


def test_spreads_from_one_flat_hazard_bootstrap_back_to_that_hazard():
    # Plant a single flat hazard by bootstrapping ONE pillar, read the
    # par spreads that hazard implies at 1/3/5/7/10y, then bootstrap a
    # fresh multi-pillar curve from those spreads: every recovered
    # pillar hazard must be the planted h, because a flat hazard is the
    # unique piecewise-constant solution repricing all of them.
    discount = flat3()
    base = CreditCurve.bootstrap([5], [0.02], 0.40, discount)
    h = base.hazard(1.0)

    tenors = [1, 3, 5, 7, 10]
    spreads = [CdsPricer.par_spread(base, discount, t) for t in tenors]
    rebuilt = CreditCurve.bootstrap(tenors, spreads, 0.40, discount)
    for t in [0.5, 2, 4, 6, 9]:
        assert rebuilt.hazard(t) == pytest.approx(h, abs=1e-9), \
            f"flat-hazard round trip broke at t={t}"
    # And the survival curves coincide, not just the local hazards.
    assert rebuilt.survival_probability(8) == pytest.approx(
        base.survival_probability(8), abs=1e-9)


def test_bootstrap_reprices_every_pillar_on_a_sloped_discount_curve_too():
    # Every other credit pin discounts at a FLAT curve, which a
    # DF-handling bug could hide behind; an upward-sloping curve makes
    # each period's discount factor distinct, so repricing all pillars
    # to 1e-12 exercises the discounting for real.
    sloped = YieldCurve.of_zero_rates(
        [1, 2, 3, 5, 7, 10], [0.020, 0.024, 0.027, 0.030, 0.032, 0.033])
    tenors = [1, 3, 5, 7, 10]
    spreads = [0.008, 0.011, 0.014, 0.015, 0.016]
    curve = CreditCurve.bootstrap(tenors, spreads, 0.40, sloped)
    for t, s in zip(tenors, spreads):
        assert CdsPricer.par_spread(curve, sloped, t) == pytest.approx(s, abs=1e-12), \
            f"{t}y must reprice on the sloped curve"
        assert CdsPricer.upfront(curve, sloped, s, t) == pytest.approx(0.0, abs=1e-14), \
            f"par coupon means zero upfront at {t}y"


def test_credit_triangle_error_shrinks_with_the_spread_level():
    # The triangle S = h*(1-R) is the flat-curve zeroth-order identity;
    # the discretization/accrual correction is O(spread), so the
    # RELATIVE error at 10bp must be far below the error at 500bp.
    discount = flat3()
    tight = CreditCurve.bootstrap([5], [0.001], 0.40, discount)
    wide = CreditCurve.bootstrap([5], [0.05], 0.40, discount)
    err_tight = abs(tight.hazard(1) - 0.001 / 0.60) / (0.001 / 0.60)
    err_wide = abs(wide.hazard(1) - 0.05 / 0.60) / (0.05 / 0.60)
    assert err_tight < 1e-3, f"10bp triangle relative error {err_tight}"
    assert err_wide > err_tight, \
        f"500bp error {err_wide} must exceed 10bp error {err_tight}"


def test_upfront_is_exactly_antisymmetric_around_the_par():
    # upfront(S_c) = protection - S_c * annuity is LINEAR in the coupon,
    # so equal coupon distances above and below par produce upfronts of
    # equal size and opposite sign: a rich coupon means the protection
    # BUYER receives points (negative upfront).
    discount = flat3()
    curve = CreditCurve.bootstrap([5], [0.03], 0.40, discount)
    par = CdsPricer.par_spread(curve, discount, 5)
    annuity = CdsPricer.risky_annuity(curve, discount, 5)
    d = 0.01
    up_cheap = CdsPricer.upfront(curve, discount, par - d, 5)
    up_rich = CdsPricer.upfront(curve, discount, par + d, 5)
    assert up_cheap == pytest.approx(d * annuity, abs=1e-12)
    assert up_rich == pytest.approx(-d * annuity, abs=1e-12)
    assert up_rich == pytest.approx(-up_cheap, abs=1e-12), \
        "linear in coupon: exact antisymmetry"
    assert up_rich < 0, "coupon above par: seller pays the buyer points"
    # Premium leg is spread times the annuity by definition.
    assert CdsPricer.premium_leg_pv(curve, discount, 0.02, 5) == pytest.approx(
        0.02 * annuity, abs=1e-15)


def test_survival_is_strictly_monotone_and_extrapolates_the_last_hazard_flat():
    discount = flat3()
    curve = CreditCurve.bootstrap([1, 3, 5], [0.01, 0.015, 0.02], 0.40, discount)
    # Strictly decreasing survival on a fine grid.
    prev = 1.0
    t = 0.5
    while t <= 12:
        q = curve.survival_probability(t)
        assert q < prev, f"survival must fall at t={t}"
        assert q > 0
        prev = q
        t += 0.5
    # Beyond the last pillar the hazard is flat, so
    # Q(5+s) = Q(5) * exp(-h_last * s) EXACTLY.
    h_last = curve.hazard(6.0)
    assert curve.hazard(5.0) == h_last, "flat beyond the last pillar"
    assert curve.survival_probability(8) == pytest.approx(
        curve.survival_probability(5) * math.exp(-h_last * 3), abs=1e-15)


def test_gates_the_first_layer_missed():
    d = flat3()
    with pytest.raises(ValueError):
        CreditCurve.bootstrap([1], [0.01], -0.1, d)             # negative recovery
    with pytest.raises(ValueError):
        CreditCurve.bootstrap([0, 1], [0.01, 0.01], 0.4, d)     # zero tenor
    with pytest.raises(ValueError):
        CreditCurve.bootstrap([1], [float("nan")], 0.4, d)      # NaN spread
    curve = CreditCurve.bootstrap([1], [0.01], 0.4, d)
    with pytest.raises(ValueError):
        curve.survival_probability(-1)
    with pytest.raises(ValueError):
        curve.hazard(-0.5)
    with pytest.raises(ValueError):
        CdsPricer.par_spread(curve, d, 0)                       # zero maturity
    with pytest.raises(ValueError):
        CdsPricer.upfront(curve, d, -0.01, 5)                   # negative coupon
    # Below one quarterly grid step the leg sums are EMPTY: par spread
    # would be 0/0 = NaN. The gate raises instead of leaking it.
    with pytest.raises(ValueError):
        CdsPricer.par_spread(curve, d, 0.1)                     # sub-grid maturity
    with pytest.raises(ValueError):
        CdsPricer.risky_annuity(curve, d, 0.2)                  # sub-grid maturity
