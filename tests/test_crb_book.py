"""Central risk book: booking, cross-product netting, greeks rollup,
risk report. Ported from Java CrbBookTest.
"""

import math

import pytest

from quantfinlib.crb import CentralRiskBook, CrbHedgeUniverse
from quantfinlib.pricing.black_scholes import BlackScholes, OptionType
from quantfinlib.util import math_utils as mu

# ------------------------------------------------------------------
# Cash equities
# ------------------------------------------------------------------


def test_cash_equity_nets_across_desks():
    book = CentralRiskBook()
    book.book_cash_equity("cash-desk", "AAPL", 10_000, 150)     # +1.5M
    book.book_cash_equity("etf-desk", "AAPL", -8_000, 150)      # -1.2M
    assert book.exposure("EQ:AAPL") == pytest.approx(300_000, abs=1e-9), \
        "two desks' opposite flows net before anyone pays the street"
    assert book.gross_exposure("EQ:AAPL") == pytest.approx(2_700_000, abs=1e-9)
    assert book.desk_exposure("cash-desk", "EQ:AAPL") == pytest.approx(1_500_000, abs=1e-9)
    assert book.desk_exposure("etf-desk", "EQ:AAPL") == pytest.approx(-1_200_000, abs=1e-9)
    assert book.netting_efficiency() == pytest.approx(1 - 300_000.0 / 2_700_000, abs=1e-12)
    assert book.flows_booked() == 2
    assert book.exposure("EQ:MSFT") == 0, "never-booked factor is flat"


# ------------------------------------------------------------------
# Equity options -- greeks decompose onto the SAME factors as cash
# ------------------------------------------------------------------


def test_equity_option_delta_nets_against_cash_shares():
    spot, strike, r, q, vol, t = 100, 105, 0.03, 0.01, 0.25, 0.5
    book = CentralRiskBook()
    book.book_cash_equity("cash-desk", "XYZ", -5_000, spot)             # -500k
    book.book_equity_option("vol-desk", "XYZ", OptionType.CALL,
                            100, 100, spot, strike, r, q, vol, t)         # 10k units

    delta = BlackScholes.delta(OptionType.CALL, spot, strike, r, q, vol, t)
    gamma = BlackScholes.gamma(spot, strike, r, q, vol, t)
    vega = BlackScholes.vega(spot, strike, r, q, vol, t)
    assert book.exposure("EQ:XYZ") == pytest.approx(-500_000 + 10_000 * delta * spot, abs=1e-6), \
        "option delta and cash shares share ONE factor"
    assert book.exposure("EQGAMMA:XYZ") == pytest.approx(
        10_000 * gamma * spot * spot / 100, abs=1e-9)
    assert book.exposure("EQVEGA:XYZ") == pytest.approx(10_000 * vega / 100, abs=1e-9)

    # A put's delta is negative: booking long puts REDUCES a long book.
    put_book = CentralRiskBook()
    put_book.book_equity_option("vol-desk", "XYZ", OptionType.PUT,
                                50, 100, spot, strike, r, q, vol, t)
    assert put_book.exposure("EQ:XYZ") < 0, "long puts are short delta"
    assert put_book.exposure("EQGAMMA:XYZ") > 0, "long options are long gamma"


# ------------------------------------------------------------------
# FX spot -- currency-level decomposition is what nets across pairs
# ------------------------------------------------------------------


def test_fx_spot_decomposes_to_currency_legs_and_nets_across_pairs():
    book = CentralRiskBook()
    book.book_fx_spot("fx-desk", "EURUSD", 10_000_000, 1.10)
    assert book.exposure("CCY:EUR") == pytest.approx(10_000_000, abs=1e-6)
    assert book.exposure("CCY:USD") == pytest.approx(-11_000_000, abs=1e-6)

    # A USDJPY buy from ANOTHER desk nets the USD leg.
    book.book_fx_spot("spot-desk-2", "USDJPY", 5_000_000, 150)
    assert book.exposure("CCY:USD") == pytest.approx(-6_000_000, abs=1e-6), \
        "USD exposure nets across EURUSD and USDJPY"
    assert book.exposure("CCY:JPY") == pytest.approx(-750_000_000, abs=1e-6)


# ------------------------------------------------------------------
# FX swaps -- points risk, not spot risk
# ------------------------------------------------------------------


def test_fx_swap_carries_points_risk_with_zero_base_delta():
    book = CentralRiskBook()
    book.book_fx_swap("fwd-desk", "EURUSD", 20_000_000, 1.1000, 1.1050)
    assert book.exposure("CCY:EUR") == pytest.approx(0, abs=1e-9), \
        "a swap's base legs cancel exactly"
    assert book.exposure("CCY:USD") == pytest.approx(20_000_000 * 0.005, abs=1e-6)
    assert book.exposure("FXPOINTS:EURUSD") == pytest.approx(-20_000_000, abs=1e-9)
    book.book_fx_spot("spot-desk", "EURUSD", 1_000_000, 1.10)
    assert book.exposure("CCY:EUR") == pytest.approx(1_000_000, abs=1e-9)


# ------------------------------------------------------------------
# NDFs -- a forward until the fixing, plus the fixing-notional flag
# ------------------------------------------------------------------


def test_ndf_carries_full_delta_until_fixing_and_tracks_pending_notional():
    book = CentralRiskBook()
    book.book_ndf("em-desk", "USDINR", 5_000_000, 84.0)
    assert book.exposure("CCY:USD") == pytest.approx(5_000_000, abs=1e-6)
    assert book.exposure("CCY:INR") == pytest.approx(-420_000_000, abs=1e-6)
    assert book.pending_fixing("USDINR") == pytest.approx(5_000_000, abs=1e-9)
    book.book_ndf("em-desk", "USDINR", -2_000_000, 84.0)
    assert book.exposure("CCY:USD") == pytest.approx(3_000_000, abs=1e-6), "delta nets"
    assert book.pending_fixing("USDINR") == pytest.approx(7_000_000, abs=1e-9), \
        "fixing exposure is GROSS -- offsetting NDFs still both fix"
    assert book.pending_fixing("USDBRL") == pytest.approx(0, abs=1e-12)

    book.settle_fixing("USDINR", 3_000_000)
    assert book.pending_fixing("USDINR") == pytest.approx(4_000_000, abs=1e-9)
    book.settle_fixing("USDINR", 4_000_000)
    assert book.pending_fixing("USDINR") == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError):
        book.settle_fixing("USDINR", 1)


# ------------------------------------------------------------------
# FX options -- Garman-Kohlhagen delta nets against spot
# ------------------------------------------------------------------


def test_fx_option_delta_nets_against_spot_on_the_currency_legs():
    s, k, rd, rf, vol, t = 1.10, 1.12, 0.05, 0.03, 0.10, 0.25
    book = CentralRiskBook()
    book.book_fx_spot("fx-desk", "EURUSD", -6_000_000, s)
    book.book_fx_option("fx-opt-desk", "EURUSD", OptionType.CALL,
                        10_000_000, s, k, rd, rf, vol, t)

    delta = BlackScholes.delta(OptionType.CALL, s, k, rd, rf, vol, t)
    vega = BlackScholes.vega(s, k, rd, rf, vol, t)
    assert book.exposure("CCY:EUR") == pytest.approx(
        -6_000_000 + 10_000_000 * delta, abs=1e-4)
    assert book.exposure("FXVEGA:EURUSD") == pytest.approx(10_000_000 * vega / 100, abs=1e-9)
    assert book.exposure("FXGAMMA:EURUSD") > 0


# ------------------------------------------------------------------
# The risk report -- pricing the reason the CRB exists
# ------------------------------------------------------------------


def test_report_prices_the_diversification_benefit_of_netting():
    book = CentralRiskBook()
    book.book_cash_equity("desk-a", "SPY", 20_000, 500)    # +10M
    book.book_cash_equity("desk-b", "SPY", -16_000, 500)   # -8M
    daily_var = 4e-4                                       # 2% daily vol
    cov = [[daily_var]]
    report = book.report(cov, 0.99)

    z = mu.norm_inv(0.99)
    sigma = math.sqrt(daily_var)
    assert report.var == pytest.approx(z * 2_000_000 * sigma, abs=1e-6), \
        "the netted book's VaR runs on 2M, not 18M"
    assert report.standalone_desk_var == pytest.approx(z * 18_000_000 * sigma, abs=1e-6)
    assert report.diversification_benefit == pytest.approx(z * 16_000_000 * sigma, abs=1e-6)
    assert report.es > report.var, "ES beyond VaR, always"
    assert report.netting_efficiency == pytest.approx(book.netting_efficiency(), abs=1e-12)
    assert report.gross_exposure == pytest.approx(18_000_000, abs=1e-6)
    assert report.net_exposure == pytest.approx(2_000_000, abs=1e-6)


# ------------------------------------------------------------------
# Gates
# ------------------------------------------------------------------


def test_gates_reject_garbage_loudly():
    book = CentralRiskBook()
    with pytest.raises(ValueError):
        book.book_cash_equity("d", "X", math.nan, 100)
    with pytest.raises(ValueError):
        book.book_cash_equity("d", "X", 100, 0)
    with pytest.raises(ValueError):
        book.book_cash_equity(" ", "X", 100, 100)
    with pytest.raises(ValueError):
        book.book_fx_spot("d", "EUR", 1e6, 1.1)   # pair must be 6 chars
    with pytest.raises(ValueError):
        book.book_fx_swap("d", "EURUSD", 1e6, 1.1, math.inf)
    with pytest.raises(ValueError):
        book.book_equity_option("d", "X", OptionType.CALL,
                                1, 100, 100, 100, 0.02, 0, -0.2, 1)
    book.book_cash_equity("d", "X", 100, 100)
    with pytest.raises(ValueError):
        book.report([[0, 0], [0, 0]], 0.99)   # cov must match factor count

    # Compute-validate-COMMIT: a rejected multi-leg booking must leave
    # the book COMPLETELY untouched, not half-booked.
    flows = book.flows_booked()
    eq = book.exposure("EQ:Y")
    with pytest.raises(ValueError):
        book.book_equity_option("d", "Y", OptionType.CALL,
                                1, 100, 100, 100, 0.03, math.inf, 0.2, 1)
    assert book.flows_booked() == flows, "nothing booked"
    assert book.exposure("EQ:Y") == eq, "no half-booked delta leg"


def test_hedge_only_factors_never_break_the_book_level_views():
    book = CentralRiskBook()
    book.book_fx_spot("fx-desk", "EURUSD", 10_000_000, 1.10)
    efficiency_before = book.netting_efficiency()
    # Register MANY hedge-only factors past the booked arrays' capacity.
    universe = CrbHedgeUniverse(book.factors())
    for i in range(20):
        universe.add_single_factor(f"HEDGE-{i}", f"HEDGEFACTOR:{i}", 1)
    assert book.netting_efficiency() == pytest.approx(efficiency_before, abs=1e-12), \
        "hedge-only factors carry zero exposure, efficiency unchanged"
    n = book.factors().size()
    cov = [[0.0] * n for _ in range(n)]
    for f in range(n):
        cov[f][f] = 1e-4
    report = book.report(cov, 0.99)
    assert report.var > 0, "report survives the grown registry"
    assert len(book.net_exposures()) == n
