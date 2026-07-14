"""CRB hedge-universe builder and economics ledger, ported from the
non-checkpoint halves of Java CrbPersistenceAndUniverseTest (the
persist.Checkpoint round trips are not ported -- no persist lane in
the Python port; the economics/matrix pins are identical).
"""

import pytest

from quantfinlib.crb import (Allocation, CentralRiskBook, CrbHedgeUniverse,
                             CrbPnlLedger, CrbRouter, DarkVenue, Decision,
                             HedgeOptimizer)


def test_ledger_books_every_field_and_ignores_fully_routed_flow():
    ledger = CrbPnlLedger()
    ledger.on_internalized(1_000_000, 4, 1)      # captured 300, paid 100
    ledger.on_hedge(-500_000, 2)                 # cost 100 (abs notional)
    ledger.on_route(200_000, Allocation(0, [], 200_000, 3))
    assert ledger.spread_captured() == pytest.approx(300, abs=1e-9)
    assert ledger.improvement_paid() == pytest.approx(100, abs=1e-9)
    assert ledger.hedge_cost() == pytest.approx(100, abs=1e-9)
    assert ledger.router_cost() == pytest.approx(60, abs=1e-9)
    assert ledger.internalizations() == 1
    assert ledger.hedges() == 1
    assert ledger.net_economics() == pytest.approx(300 - 100 - 60, abs=1e-9)

    # A fully-routed decision earns NOTHING and counts as nothing --
    # the internalization stat must not inflate on flow we passed on.
    ledger.on_decision(Decision(0, 5_000_000, 0), 4)
    assert ledger.internalizations() == 1, "routed flow is not internalization"
    assert ledger.spread_captured() == pytest.approx(300, abs=1e-9)

    # The gate: improvement beyond the half spread means the desk
    # would be PAYING clients to trade.
    with pytest.raises(ValueError):
        ledger.on_internalized(1_000_000, 4, 5)


def test_universe_builds_the_matrix_and_the_optimizer_flattens_what_it_spans():
    book = CentralRiskBook()
    book.book_fx_spot("fx-desk", "EURUSD", 10_000_000, 1.10)
    book.book_ndf("em-desk", "USDINR", 5_000_000, 84.0)
    # CCY:EUR +10M, CCY:USD -6M (spot -11M + NDF +5M), CCY:INR -420M.

    universe = (CrbHedgeUniverse(book.factors())
                .add_fx_forward("EURUSD-1W", "EURUSD", 1.10, 1.0)
                .add_single_factor("INR-OUTRIGHT", "CCY:INR", 2.0))
    assert universe.size() == 2
    assert universe.name(0) == "EURUSD-1W"

    n = book.factors().size()
    cov = [[0.0] * n for _ in range(n)]
    for f in range(n):
        cov[f][f] = 1e-4
    e = book.net_exposures()
    loadings = universe.loadings()
    h = HedgeOptimizer.hedge(e, cov, loadings, universe.costs(), 0)

    residual = HedgeOptimizer.residual(e, loadings, h)
    inr = book.factors().id_if_present("CCY:INR")
    assert h[1] == pytest.approx(420_000_000, abs=1), "the outright flattens INR exactly"
    assert residual[inr] == pytest.approx(0, abs=1e-3)
    before = HedgeOptimizer.risk(e, cov)
    after = HedgeOptimizer.risk(residual, cov)
    assert after < 0.05 * before, \
        f"the universe-built hedge removed the book's risk: {after} vs {before}"


def test_index_future_hedges_single_names_through_the_covariance_not_a_beta_table():
    import math

    book = CentralRiskBook()
    book.book_cash_equity("cash-desk", "AAPL", 50_000, 200)      # +10M
    universe = (CrbHedgeUniverse(book.factors())
                .add_single_factor("ES-FUTURE", "EQ:SPX", 0.5))  # hedge-only factor

    # The hedge-only factor registered cleanly: zero book exposure,
    # arrays stay coherent at the grown registry size.
    assert book.exposure("EQ:SPX") == pytest.approx(0, abs=1e-12)
    n = book.factors().size()
    assert n == 2
    e = book.net_exposures()
    assert len(e) == n

    sa2 = 4e-4                     # AAPL variance
    sb2 = 2.25e-4                  # index variance
    rho = 0.8
    cab = rho * math.sqrt(sa2 * sb2)
    cov = [[sa2, cab], [cab, sb2]]
    h = HedgeOptimizer.hedge(e, cov, universe.loadings(), universe.costs(), 0)
    assert h[0] == pytest.approx(-10_000_000 * cab / sb2, abs=1e-3), \
        "the regression hedge falls out of the covariance"

    # The router then works the hedge: the CRB's own inventory first.
    alloc = CrbRouter.route(abs(h[0]), 2_000_000, [], 4, 1)
    assert alloc.internal == pytest.approx(2_000_000, abs=1e-9)
    assert alloc.lit == pytest.approx(abs(h[0]) - 2_000_000, abs=1e-6)

    with pytest.raises(ValueError):
        universe.add("BAD", -1, ["EQ:SPX"], [1])
    with pytest.raises(ValueError):
        universe.add("BAD", 1, ["EQ:SPX"], [1, 2])
    with pytest.raises(ValueError):
        universe.add_fx_forward("BAD", "EUR", 1.1, 1)
