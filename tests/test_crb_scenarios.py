"""The CRB run the way a desk actually runs it, ported from Java
CrbRealWorldScenarioTest: a quiet two-way day (internalization IS the
P&L), a one-way institutional day (the hedge-escalation path earns its
keep), a vol-spike stress day, and an NDF fixing day.

Deviations from Java: the checkpoint round trips are not ported (no
persist lane); the quiet-day random flow uses a fixed deterministic
ticket list instead of java.util.Random(7) -- the assertions are the
same qualitative invariants (warehouse limit held, internalization
rate > 0.5, positive economics). Every pinned number on days 2-4 is
identical to the Java test.
"""

import math

import pytest

from quantfinlib.crb import (CentralRiskBook, CrbAutoHedger, CrbHedgeUniverse,
                             CrbPnlLedger, CrbRouter, DarkVenue,
                             InternalizationEngine, SkewedQuoter)
from quantfinlib.risk import stress_tester, var_engine

# ------------------------------------------------------------------
# Day 1 -- quiet two-way flow: the netting engine pays for itself
# ------------------------------------------------------------------


def test_quiet_two_way_day_internalization_is_the_pnl():
    book = CentralRiskBook()
    # Realistic knobs: 2 bps street half spread, 40% of the saved
    # spread returned to clients, a 5M warehouse.
    half_spread_bps = 2.0
    engine = InternalizationEngine(5_000_000, 0.4)
    ledger = CrbPnlLedger()

    # Client tickets 0.5-3M, both directions all day (deterministic
    # stand-in for the Java Random(7) stream).
    tickets = [1.2, -0.8, 2.5, -1.9, 0.6, -2.2, 3.0, -1.1, 1.7, -2.8,
               0.9, -0.5, 2.1, -1.6, 1.4, -2.4, 0.7, -1.3, 2.9, -0.6,
               1.8, -2.0, 1.0, -1.5]
    price = 500
    for t in tickets:
        notional = t * 1_000_000
        d = engine.decide(book.exposure("EQ:SPY"), notional, half_spread_bps)
        if d.internalized != 0:
            book.book_cash_equity("client-flow", "SPY", d.internalized / price, price)
        ledger.on_decision(d, half_spread_bps)

    # The warehouse limit is a real invariant, not advice.
    assert abs(book.exposure("EQ:SPY")) <= 5_000_000 + 1e-6, \
        f"inventory never exceeds the warehouse: {book.exposure('EQ:SPY')}"
    # Two-way flow mostly crosses itself -- that is the CRB thesis.
    assert engine.internalization_rate() > 0.5, \
        f"most flow crossed internally: {engine.internalization_rate()}"
    assert ledger.spread_captured() > 0, "the day made money"
    assert ledger.improvement_paid() > 0, \
        "and clients were paid to bring the offsetting flow"
    assert ledger.net_economics() == pytest.approx(ledger.spread_captured(), abs=1e-9), \
        "no hedges on a quiet day: economics = captured spread"


# ------------------------------------------------------------------
# Day 2 -- one-way institutional flow: hedge, escalate, still profit
# ------------------------------------------------------------------


def test_one_way_institutional_day_hedges_and_stays_profitable():
    book = CentralRiskBook()
    half_spread_bps = 2.5
    engine = InternalizationEngine(20_000_000, 0.3)
    ledger = CrbPnlLedger()

    # A pension unwinds 18M of SPY through the desk in six tickets --
    # all one way; the warehouse absorbs it (that is the service).
    price = 500
    for _ in range(6):
        d = engine.decide(book.exposure("EQ:SPY"), 3_000_000, half_spread_bps)
        if d.internalized != 0:
            book.book_cash_equity("pension-flow", "SPY", d.internalized / price, price)
        ledger.on_decision(d, half_spread_bps)
    assert book.exposure("EQ:SPY") == pytest.approx(18_000_000, abs=1e-6)

    # Hedge instruments with real cost structure: the ES-proxy is cheap
    # (0.4 bps all-in) but lives on ANOTHER factor; the direct SPY
    # program trade is pricier (2 bps) but hits the band.
    universe = (CrbHedgeUniverse(book.factors())
                .add_single_factor("ES-PROXY", "EQ:SPX", 0.4)
                .add_single_factor("SPY-PROGRAM", "EQ:SPY", 2.0))
    spy_var = 1.44e-4              # 1.2% daily vol
    spx_var = 1.21e-4              # 1.1% daily vol
    cross = 0.97 * math.sqrt(spy_var * spx_var)
    spy = book.factors().id_if_present("EQ:SPY")
    spx = book.factors().id_if_present("EQ:SPX")
    cov = [[0.0, 0.0], [0.0, 0.0]]
    cov[spy][spy] = spy_var
    cov[spx][spx] = spx_var
    cov[spy][spx] = cross
    cov[spx][spy] = cross

    # Band: 10M hard limit on SPY, hedge back to 60%. A punitive cost
    # weight makes the cost-aware pass hold ZERO -- the hedger must
    # escalate to cost-blind, and cost-blind must pick the DIRECT
    # instrument (the proxy cannot satisfy a per-factor band).
    hedger = CrbAutoHedger([10_000_000, 25_000_000], 0.6, 1)
    orders = hedger.check(book.net_exposures(), cov, universe.loadings(),
                          universe.costs(), 6_000, 0)
    assert len(orders) == 1, "one decisive hedge"
    assert universe.name(orders[0].instrument) == "SPY-PROGRAM", \
        "the hard limit demanded the direct hedge, price notwithstanding"
    # Excess beyond 0.6 x 10M: 18M - 6M (to the dollar -- the
    # correlated solve converges to relative tolerance).
    assert orders[0].notional == pytest.approx(-12_000_000, abs=1.0)
    ledger.on_hedge(orders[0].notional, 2.0)

    # The hedge routes like any order: dark midpoint at 1.2 bps
    # undercuts lit (0.5 spread + 0.8 impact); the 15 bps printing
    # pool gets nothing.
    alloc = CrbRouter.route(abs(orders[0].notional), 0,
                            [DarkVenue("PRINT-POOL", 10_000_000, 1.0, 15),
                             DarkVenue("MIDPOINT", 6_000_000, 0.7, 1.2)],
                            0.5, 0.8)
    assert alloc.dark[1] == pytest.approx(4_200_000, abs=1e-6), "clean dark first"
    assert alloc.dark[0] == pytest.approx(0, abs=1e-9), "toxic pool priced out"
    assert alloc.lit == pytest.approx(7_800_000, abs=1.0)
    ledger.on_route(abs(orders[0].notional), alloc)

    # The commercial argument, in one assertion: captured spread paid
    # for the hedge AND its execution, with margin.
    # 18M x (2.5 - 0.75) bps = 3,150 captured; 12M x 2 bps = 2,400
    # hedge; ~1.24 bps blended on 12M routed.
    assert ledger.net_economics() > 0, \
        f"the netting engine paid for its own risk management: {ledger.net_economics()}"
    assert ledger.net_economics() == pytest.approx(
        ledger.spread_captured() - ledger.hedge_cost() - ledger.router_cost(), abs=1e-9)


# ------------------------------------------------------------------
# Day 3 -- the vol spike: what the risk committee asks for
# ------------------------------------------------------------------


def test_vol_spike_day_stresses_the_residual_book():
    # The residual book into the spike: long 8M equities, short 5M USD,
    # short vega (sold calls into the calm) -- mapped onto the stress
    # template's [equity, rates, FX-USD, commodity, vol] factor order.
    vega_per_point = -40_000       # $ per vol point, short
    exposures = [8_000_000, 0, -5_000_000, 0, vega_per_point * 100]
    pnl = stress_tester.scenario_pnl(exposures, stress_tester.covid_march_2020())
    # Hand arithmetic: 8M x -12% - 5M x +5% - 40k x 25 points
    #                = -960k - 250k - 1,000k = -2.21M.
    assert pnl == pytest.approx(-2_210_000, abs=1e-6), \
        "the March-2020 replay on this book, to the dollar"
    assert pnl < 0, "long equity, short USD, short vega -- the spike hurts"

    # The regulator's inverted question on the liquid slice: what move
    # loses 2M, and how implausible is it?
    liquid = [8_000_000, -5_000_000]
    cov = [[1.44e-4, 0.3 * math.sqrt(1.44e-4 * 3.6e-5)],
           [0.3 * math.sqrt(1.44e-4 * 3.6e-5), 3.6e-5]]
    reverse = stress_tester.reverse_stress(liquid, cov, 2_000_000)
    assert stress_tester.scenario_pnl(liquid, reverse.shocks) == pytest.approx(
        -2_000_000, abs=1e-6)
    assert reverse.mahalanobis_sigmas == pytest.approx(
        2_000_000 / var_engine.portfolio_stdev(liquid, cov), abs=1e-9)
    assert reverse.mahalanobis_sigmas > 10, \
        "a 2M loss on this small netted book is a many-sigma event"

    # Quoting through the spike: spread 4x wider, skew working harder,
    # and the quote still never crosses or goes negative.
    crisis = SkewedQuoter.quote(500, 8, 4_000_000, 5_000_000, 0.6)
    assert 0 < crisis.bid < crisis.ask
    assert crisis.skew_bps < -3.5, \
        f"a nearly-full warehouse shades hard: {crisis.skew_bps}"


# ------------------------------------------------------------------
# Day 4 -- NDF fixing day (the Java overnight checkpoint is not
# ported; the fixing lifecycle itself is identical)
# ------------------------------------------------------------------


def test_ndf_fixing_day_releases_and_rebooks():
    book = CentralRiskBook()
    # An EM desk's NDF book: USDINR fixes tomorrow, USDBRL next week.
    book.book_ndf("em-desk", "USDINR", 8_000_000, 84.0)
    book.book_ndf("em-desk", "USDBRL", 4_000_000, 5.60)
    ledger = CrbPnlLedger()
    ledger.on_internalized(8_000_000, 3.0, 1.0)   # the INR flow was internalized
    assert ledger.net_economics() == pytest.approx(
        8_000_000 * (3.0 - 1.0) / 1e4, abs=1e-9)

    # The fixing: release the pending notional, re-book the fixed NDF's
    # delta as the offsetting flow (it cash-settles in USD).
    assert book.pending_fixing("USDINR") == pytest.approx(8_000_000, abs=1e-9)
    book.settle_fixing("USDINR", 8_000_000)
    book.book_ndf("em-desk", "USDINR", -8_000_000, 84.0)
    book.settle_fixing("USDINR", 8_000_000)   # the offset itself fixes too
    assert book.pending_fixing("USDINR") == pytest.approx(0, abs=1e-9), "INR is done"
    assert book.exposure("CCY:INR") == pytest.approx(0, abs=1e-6), "and its delta is flat"
    assert book.pending_fixing("USDBRL") == pytest.approx(4_000_000, abs=1e-9), \
        "BRL still awaits next week's fixing"
    assert book.exposure("CCY:USD") == pytest.approx(4_000_000, abs=1e-6), \
        "only the live BRL forward's USD leg remains"
