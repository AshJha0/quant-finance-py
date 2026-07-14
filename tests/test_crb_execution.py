"""CRB flow economics and execution: skewed quotes, internalization,
hedging, routing. Ported from Java CrbExecutionTest.
"""

import math

import pytest

from quantfinlib.crb import (CentralRiskBook, CrbAutoHedger, CrbRouter, DarkVenue,
                             HedgeOptimizer, InternalizationEngine, SkewedQuoter)
from quantfinlib.pricing.black_scholes import OptionType

# ------------------------------------------------------------------
# Skewed quoting
# ------------------------------------------------------------------


def test_skew_shades_both_quotes_toward_reducing_inventory():
    # Half-limit long, half the spread available as skew: -2.5 bps.
    q = SkewedQuoter.quote(100, 10, 50, 100, 0.5)
    assert q.skew_bps == pytest.approx(-2.5, abs=1e-12), "long book shades DOWN"
    assert q.bid == pytest.approx(100 * (1 - 12.5e-4), abs=1e-12)
    assert q.ask == pytest.approx(100 * (1 + 7.5e-4), abs=1e-12), \
        "the ask is the attractive side: sell what we hold"

    flat = SkewedQuoter.quote(100, 10, 0, 100, 0.5)
    assert flat.skew_bps == pytest.approx(0, abs=1e-12)
    assert flat.ask - 100 == pytest.approx(100 - flat.bid, abs=1e-12), \
        "flat book, symmetric"

    # Beyond-limit inventory clamps: the skew never exceeds
    # skew_fraction of the half spread, and the quote NEVER crosses.
    capped = SkewedQuoter.quote(100, 10, 500, 100, 0.9)
    assert capped.skew_bps == pytest.approx(-9, abs=1e-12)
    assert capped.bid < capped.ask, "never self-crossing"

    short_book = SkewedQuoter.quote(100, 10, -50, 100, 0.5)
    assert short_book.skew_bps == pytest.approx(2.5, abs=1e-12), \
        "short book shades UP to buy"

    with pytest.raises(ValueError):
        SkewedQuoter.quote(100, 10, 0, 100, 1.0)   # fraction 1 could cross
    with pytest.raises(ValueError):
        SkewedQuoter.quote(100, 10, math.nan, 100, 0.5)
    # 100%+ half spreads would quote a zero or NEGATIVE bid.
    with pytest.raises(ValueError):
        SkewedQuoter.quote(100, 12_000, 0, 100, 0)


# ------------------------------------------------------------------
# Internalization
# ------------------------------------------------------------------


def test_risk_reducing_flow_is_internalized_with_improvement():
    engine = InternalizationEngine(10_000_000, 0.5)
    # Book long 5M; client flow shorts the book 3M: pure risk reduction.
    d = engine.decide(5_000_000, -3_000_000, 10)
    assert d.internalized == pytest.approx(-3_000_000, abs=1e-9), "fully crossed"
    assert d.routed == pytest.approx(0, abs=1e-9)
    assert d.improvement_bps == pytest.approx(5, abs=1e-12), \
        "half of the saved 10bps half-spread"
    assert engine.internalization_rate() == pytest.approx(1.0, abs=1e-12)


def test_flow_through_zero_blends_improvement_and_warehouses_the_flip():
    engine = InternalizationEngine(10_000_000, 0.5)
    # Book long 5M; flow -8M: 5M reduces, 3M flips the book short --
    # that excess is warehoused (inside the 10M limit), no improvement
    # earned on it.
    d = engine.decide(5_000_000, -8_000_000, 10)
    assert d.internalized == pytest.approx(-8_000_000, abs=1e-9)
    assert d.routed == pytest.approx(0, abs=1e-9)
    assert d.improvement_bps == pytest.approx(5.0 * 5 / 8, abs=1e-12), \
        "only the reducing 5M of 8M earned improvement"


def test_risk_adding_flow_is_warehoused_only_inside_the_limit():
    engine = InternalizationEngine(10_000_000, 0.5)
    # Book already long 9M; another 5M same-way: 1M of headroom.
    d = engine.decide(9_000_000, 5_000_000, 10)
    assert d.internalized == pytest.approx(1_000_000, abs=1e-9), "warehouse to the limit"
    assert d.routed == pytest.approx(4_000_000, abs=1e-9), "the rest goes to the street"
    assert d.improvement_bps == pytest.approx(0, abs=1e-12), \
        "risk-adding flow earns nothing"
    assert engine.internalization_rate() == pytest.approx(0.2, abs=1e-12), "1M of 5M kept"

    # At (or beyond) the limit: everything routes.
    full = engine.decide(10_000_000, 2_000_000, 10)
    assert full.internalized == pytest.approx(0, abs=1e-9)
    assert full.routed == pytest.approx(2_000_000, abs=1e-9)

    with pytest.raises(ValueError):
        engine.decide(0, 0, 10)
    with pytest.raises(ValueError):
        InternalizationEngine(1e6, 1.5)


# ------------------------------------------------------------------
# Hedge optimizer
# ------------------------------------------------------------------


def test_zero_cost_recovers_the_closed_form_minimum_variance_hedge():
    # Factor A exposure hedged with an instrument that loads factor B:
    # the classic regression hedge h = -E*cov(A,B)/var(B).
    sa2 = 4e-4
    sb2 = 2.25e-4
    cab = 0.6 * math.sqrt(sa2 * sb2)
    cov = [[sa2, cab], [cab, sb2]]
    e = [10_000_000, 0]
    loadings = [[0], [1]]                    # 1 unit -> 1 unit of B
    h = HedgeOptimizer.hedge(e, cov, loadings, [1], 0)
    assert h[0] == pytest.approx(-10_000_000 * cab / sb2, abs=1e-3), \
        "lambda = 0 IS the closed-form minimum-variance hedge"

    before = HedgeOptimizer.risk(e, cov)
    after = HedgeOptimizer.risk(HedgeOptimizer.residual(e, loadings, h), cov)
    assert after < before, f"the hedge cut risk: {after} < {before}"
    # Residual variance = var_A(1 - rho^2) under the optimal proxy hedge.
    assert after == pytest.approx(before * math.sqrt(1 - 0.36), abs=before * 1e-6), \
        "exactly the correlation-limited floor"


def test_cost_term_zeroes_hedges_not_worth_their_price():
    cov = [[4e-4]]
    e = [1_000_000]
    loadings = [[1]]
    # The marginal risk saving at h=0 is |2*L'Se| = 2*4e-4*1e6 = 800;
    # price the instrument above that and the optimizer holds ZERO.
    none = HedgeOptimizer.hedge(e, cov, loadings, [1], 1_000)
    assert none[0] == 0, "an uneconomic hedge is exactly zero, not dust"
    # Moderate cost: hedge less than fully.
    partial = HedgeOptimizer.hedge(e, cov, loadings, [1], 400)
    assert -1_000_000 < partial[0] < 0, f"cost-aware = partially hedged: {partial[0]}"
    # Free: fully flat.
    full = HedgeOptimizer.hedge(e, cov, loadings, [1], 0)
    assert full[0] == pytest.approx(-1_000_000, abs=1e-3)


def test_optimizer_picks_the_cheaper_of_two_identical_instruments():
    cov = [[4e-4]]
    e = [5_000_000]
    loadings = [[1, 1]]                     # twins on the same factor
    h = HedgeOptimizer.hedge(e, cov, loadings, [1, 5], 100)
    assert h[0] < 0, f"the cheap twin does the hedging: {h[0]}"
    assert h[1] == 0, "the expensive twin is exactly zero"
    with pytest.raises(ValueError):
        HedgeOptimizer.hedge(e, cov, loadings, [1, 5], -1)
    with pytest.raises(ValueError):
        HedgeOptimizer.hedge([math.nan], cov, [[1]], [1], 0)
    # A NaN covariance cell must throw, never return a silent all-zero
    # "hedge" for a live breach.
    with pytest.raises(ValueError):
        HedgeOptimizer.hedge([1e7], [[math.nan]], [[1]], [1], 0)
    # Non-PSD covariance is a data error, not a skippable instrument.
    with pytest.raises(ValueError):
        HedgeOptimizer.hedge([1e7], [[-1e-4]], [[1]], [1], 0)


# ------------------------------------------------------------------
# Auto-hedger
# ------------------------------------------------------------------


def test_auto_hedger_hedges_the_excess_back_to_the_band_and_cools_down():
    hedger = CrbAutoHedger([10_000_000], 0.5, 2)
    cov = [[4e-4]]
    loadings = [[1]]
    costs = [1]

    # Inside the band: warehouse, do nothing.
    assert hedger.check([8_000_000], cov, loadings, costs, 0, 0) == [], \
        "inside the band the CRB warehouses"

    # Breach at 12M: hedge the EXCESS beyond 0.5*limit = 5M, i.e. -7M.
    orders = hedger.check([12_000_000], cov, loadings, costs, 0, 10)
    assert len(orders) == 1
    assert orders[0].notional == pytest.approx(-7_000_000, abs=1e-3), \
        "hedge to the reset band, not to flat -- inventory is the edge"
    assert orders[0].instrument == 0
    assert hedger.hedges_emitted() == 1

    # Cooldown: a breach one interval later is suppressed...
    assert hedger.check([12_000_000], cov, loadings, costs, 0, 11) == [], "cooling down"
    # ...and fires again once the cooldown elapses.
    assert len(hedger.check([12_000_000], cov, loadings, costs, 0, 12)) == 1

    with pytest.raises(ValueError):
        CrbAutoHedger([0], 0.5, 1)
    with pytest.raises(ValueError):
        hedger.check([1, 2], cov, loadings, costs, 0, 0)
    # abs(NaN) > limit is false -- an unguarded NaN exposure would
    # silently disable the auto-hedger forever.
    with pytest.raises(ValueError):
        hedger.check([math.nan], cov, loadings, costs, 0, 20)


def test_hard_limit_outranks_cost_thrift():
    hedger = CrbAutoHedger([10_000_000], 0.5, 0)
    cov = [[4e-4]]
    loadings = [[1]]
    # Cost so punitive the cost-aware solve would hold zero -- the
    # hedger must escalate to cost-blind rather than stay breached.
    orders = hedger.check([12_000_000], cov, loadings, [1], 1e9, 0)
    assert len(orders) == 1, "the limit is hard; the cost preference is not"
    assert orders[0].notional == pytest.approx(-7_000_000, abs=1e-3)


# ------------------------------------------------------------------
# Router -- internal cross, then dark by adverse selection, then lit
# ------------------------------------------------------------------


def test_router_crosses_internally_first_then_prices_dark_against_lit():
    # Deliberately pass the venues in the WRONG order to prove the
    # router ranks by adverse selection, not list position.
    toxic = DarkVenue("TOXIC", 10_000_000, 1.0, 20)
    clean = DarkVenue("CLEAN", 4_000_000, 0.5, 2)
    a = CrbRouter.route(10_000_000, 3_000_000, [toxic, clean], 5, 3)

    assert a.internal == pytest.approx(3_000_000, abs=1e-9), \
        "the book itself fills first, free"
    assert a.dark[0] == pytest.approx(0, abs=1e-9), \
        "20bps adverse selection >= 8bps lit cost: the toxic pool gets NOTHING"
    assert a.dark[1] == pytest.approx(2_000_000, abs=1e-9), \
        "clean pool: 4M liquidity x 0.5 fill probability"
    assert a.lit == pytest.approx(5_000_000, abs=1e-9), \
        "the remainder pays the spread but fills"
    # Blended: (3M*0 + 2M*2 + 5M*8) / 10M = 4.4 bps.
    assert a.expected_cost_bps == pytest.approx(4.4, abs=1e-12)


def test_fully_internalized_order_costs_nothing():
    a = CrbRouter.route(2_000_000, 5_000_000, [], 5, 3)
    assert a.internal == pytest.approx(2_000_000, abs=1e-9)
    assert a.lit == pytest.approx(0, abs=1e-9)
    assert a.expected_cost_bps == pytest.approx(0, abs=1e-12), \
        "crossing inventory is free"
    with pytest.raises(ValueError):
        CrbRouter.route(0, 0, [], 5, 3)
    with pytest.raises(ValueError):
        DarkVenue("X", 1e6, 1.2, 5)


# ------------------------------------------------------------------
# The full loop, all six instrument types
# ------------------------------------------------------------------


def test_full_loop_books_nets_quotes_internalizes_hedges_and_routes():
    book = CentralRiskBook()
    # Three desks, six products, one netted book.
    book.book_cash_equity("cash-desk", "SPY", 20_000, 500)
    book.book_equity_option("vol-desk", "SPY", OptionType.PUT,
                            -100, 100, 500, 480, 0.03, 0.015, 0.2, 0.25)  # short puts: long delta
    book.book_fx_spot("fx-desk", "EURUSD", 10_000_000, 1.10)
    book.book_fx_swap("fx-desk", "EURUSD", 20_000_000, 1.1000, 1.1040)
    book.book_ndf("em-desk", "USDINR", 5_000_000, 84.0)
    book.book_fx_option("fx-desk", "EURUSD", OptionType.CALL,
                        -8_000_000, 1.10, 1.12, 0.05, 0.03, 0.10, 0.25)  # short calls: short delta

    assert book.flows_booked() == 6
    assert book.exposure("EQ:SPY") > 10_000_000, "cash + short-put delta stack"
    assert book.exposure("CCY:EUR") < 10_000_000, \
        "the short option delta netted part of the spot leg"
    assert book.netting_efficiency() > 0, "something netted"

    # A client sell of EUR risk-reduces the long CCY:EUR book.
    engine = InternalizationEngine(20_000_000, 0.5)
    eur_net = book.exposure("CCY:EUR")
    d = engine.decide(eur_net, -2_000_000, 4)
    assert d.internalized == pytest.approx(-2_000_000, abs=1e-6), "risk-reducing: kept"
    assert d.improvement_bps > 0, "and the client shares the saving"

    # Auto-hedge the equity factor: limit 8M, book is ~10M+ long.
    n = book.factors().size()
    exposures = book.net_exposures()
    eq_id = book.factors().id_if_present("EQ:SPY")
    limits = [max(1, abs(exposures[f]) * 2) for f in range(n)]   # roomy everywhere...
    limits[eq_id] = 8_000_000                                     # ...except equities
    cov = [[0.0] * n for _ in range(n)]
    for f in range(n):
        cov[f][f] = 1e-4
    loadings = [[0.0] for _ in range(n)]
    loadings[eq_id][0] = 1                                        # an index-futures proxy
    hedger = CrbAutoHedger(limits, 0.75, 0)
    orders = hedger.check(exposures, cov, loadings, [0.5], 1, 0)
    assert len(orders) == 1, "the breach got a hedge"
    residual = HedgeOptimizer.residual(exposures, loadings, [orders[0].notional])
    assert abs(residual[eq_id]) <= 8_000_000 + 1e-3, \
        f"post-hedge equity inside the hard limit: {residual[eq_id]}"

    # The hedge itself routes: internal first, clean dark, lit rest.
    alloc = CrbRouter.route(abs(orders[0].notional), 1_000_000,
                            [DarkVenue("MID", 2_000_000, 0.8, 1.5)], 4, 2)
    assert alloc.internal == pytest.approx(1_000_000, abs=1e-9)
    assert alloc.dark[0] > 0, "cheap dark used before lit"
    assert alloc.lit == pytest.approx(
        abs(orders[0].notional) - 1_000_000 - alloc.dark[0], abs=1e-6), \
        "conservation: every unit lands somewhere"

    # And the report prices the whole arrangement coherently.
    report = book.report(cov, 0.99)
    assert report.var > 0 and report.es > report.var
    assert report.diversification_benefit >= -1e-9, \
        f"netting can only help: {report.diversification_benefit}"
    assert 0 < report.netting_efficiency < 1
