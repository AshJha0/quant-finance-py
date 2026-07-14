# Cross-port verification probe: mirrors Probe.java exactly (labels, inputs, order).
import math

from quantfinlib.pricing.black_scholes import BlackScholes, OptionType
from quantfinlib.pricing.black76 import Black76
from quantfinlib.pricing.digital_option import DigitalOption
from quantfinlib.pricing.barrier_option import BarrierOption
from quantfinlib.pricing.touch_option import TouchOption
from quantfinlib.pricing.asian_option import AsianOption
from quantfinlib.pricing.structured_notes import StructuredNotes
from quantfinlib.pricing.exchange_option import ExchangeOption
from quantfinlib.pricing.quanto_option import QuantoOption
from quantfinlib.pricing.variance_swap import VarianceSwap
from quantfinlib.volatility.volatility_index import VolatilityIndex
from quantfinlib.rates.yield_curve import YieldCurve
from quantfinlib.rates.bond_pricer import BondPricer
from quantfinlib.rates.nelson_siegel import NelsonSiegel
from quantfinlib.rates.svensson import Svensson
from quantfinlib.rates.swap_pricer import SwapPricer
from quantfinlib.rates.short_rate_models import ShortRateModels
from quantfinlib.credit.credit_curve import CreditCurve
from quantfinlib.credit.cds_pricer import CdsPricer
from quantfinlib.credit.credit_spreads import CreditSpreads
from quantfinlib.credit.cva_approximator import CvaApproximator
from quantfinlib.commodities.commodity_curve import CommodityCurve
from quantfinlib.markets.index_construction import IndexConstruction
from quantfinlib.markets.private_market_analytics import PrivateMarketAnalytics
from quantfinlib.risk import var_engine, component_var, covariance_shrinkage, \
    extreme_value_theory, frtb_es, var_backtest, risk_metrics
from quantfinlib.risk.pca import Pca
from quantfinlib.volatility.range_volatility import RangeVolatility
from quantfinlib.volatility.garch11 import Garch11
from quantfinlib.volatility.ewma_volatility import EwmaVolatility
from quantfinlib.volatility.information_criteria import InformationCriteria
from quantfinlib.backtest.trade import Trade
from quantfinlib.backtest.trade_analytics import TradeAnalytics
from quantfinlib.backtest.validation.purged_kfold import PurgedKFold
from quantfinlib.backtest.validation.sharpe_validation import SharpeValidation
from quantfinlib.backtest.portfolio.position_sizing import PositionSizing
from quantfinlib.indicators.indicators import Indicators
from quantfinlib.fx.fixing_risk import FixingRisk
from quantfinlib.crb.hedge_optimizer import HedgeOptimizer
from quantfinlib.crb.skewed_quoter import SkewedQuoter
from quantfinlib.alpha.portfolio_construction import PortfolioConstruction
from quantfinlib.microstructure.trade_classifier import TradeClassifier
from quantfinlib.microstructure.vpin import Vpin
from quantfinlib.microstructure.hawkes_intensity import HawkesIntensity
from quantfinlib.microstructure.ewma_covariance import EwmaCovariance
from quantfinlib.microstructure.avellaneda_stoikov import AvellanedaStoikov
from quantfinlib.microstructure.jump_robust_volatility import JumpRobustVolatility
from quantfinlib.microstructure.almgren_chriss import AlmgrenChriss
from quantfinlib.microstructure.execution import Side
from quantfinlib.trading.order_throttle import OrderThrottle
from quantfinlib.trading.last_look_gate import LastLookGate
from quantfinlib.execution import twap_scheduler
from quantfinlib.execution import vwap_scheduler
from quantfinlib.execution.pov_tracker import PovTracker
from quantfinlib.execution import implementation_shortfall_scheduler
from quantfinlib.execution import order_placement_policy
from quantfinlib.execution.dark_pool_simulator import DarkPoolSimulator
from quantfinlib.execution.ucb1_selector import Ucb1Selector
from quantfinlib.execution.benchmark_executor import BenchmarkExecutor, Benchmark, MarketState
from quantfinlib.execution.venue_scorecard import VenueScorecard


def p(label, v):
    print(f"{label}={float(v):.15e}")


def main():
    # ---- Section 1: Black-Scholes ----
    S, K, r, q, sig, T = 100, 105, 0.03, 0.01, 0.25, 0.75
    call = BlackScholes.price(OptionType.CALL, S, K, r, q, sig, T)
    p("bs.call", call)
    p("bs.put", BlackScholes.price(OptionType.PUT, S, K, r, q, sig, T))
    p("bs.delta.call", BlackScholes.delta(OptionType.CALL, S, K, r, q, sig, T))
    p("bs.delta.put", BlackScholes.delta(OptionType.PUT, S, K, r, q, sig, T))
    p("bs.gamma", BlackScholes.gamma(S, K, r, q, sig, T))
    p("bs.vega", BlackScholes.vega(S, K, r, q, sig, T))
    p("bs.theta.call", BlackScholes.theta(OptionType.CALL, S, K, r, q, sig, T))
    p("bs.rho.call", BlackScholes.rho(OptionType.CALL, S, K, r, q, sig, T))
    p("bs.rho.put", BlackScholes.rho(OptionType.PUT, S, K, r, q, sig, T))
    p("bs.iv.roundtrip", BlackScholes.implied_vol(OptionType.CALL, call, S, K, r, q, T))

    # ---- Section 2: Black-76 ----
    p("b76.call", Black76.price(OptionType.CALL, 100, 95, 0.02, 0.3, 0.5))
    p("b76.put", Black76.price(OptionType.PUT, 100, 95, 0.02, 0.3, 0.5))

    # ---- Section 3: exotics & structured ----
    p("dig.cash.call", DigitalOption.cash_or_nothing(OptionType.CALL, S, K, r, q, sig, T, 10))
    p("bar.doc", BarrierOption.down_and_out_call(100, 105, 85, r, q, sig, T))
    p("touch.one", TouchOption.one_touch(100, 90, r, q, sig, T, 10))
    p("asian.geo", AsianOption.geometric_price(OptionType.CALL, 100, 100, r, q, sig, 1.0, 12))
    p("asian.tw", AsianOption.arithmetic_price(OptionType.CALL, 100, 100, r, q, sig, 1.0, 12))
    p("sn.rc", StructuredNotes.reverse_convertible(1000, 0.09, 100, 95, r, q, sig, 1.0))
    p("sn.rc.delta", StructuredNotes.reverse_convertible_delta(1000, 100, 95, r, q, sig, 1.0))
    p("sn.cpn", StructuredNotes.capital_protected_note(1000, 0.9, 0.5, 100, r, q, sig, 1.0))
    p("sn.part", StructuredNotes.participation_for(1000, 0.9, 980, 100, r, q, sig, 1.0))
    p("sn.dc", StructuredNotes.discount_certificate(100, 110, r, q, sig, 1.0))
    p("sn.dc.delta", StructuredNotes.discount_certificate_delta(100, 110, r, q, sig, 1.0))
    p("ex.margrabe", ExchangeOption.margrabe(100, 95, 0.01, 0.02, 0.25, 0.2, 0.5, 1.0))
    p("ex.kirk", ExchangeOption.kirk_spread_call(100, 95, 5, 0.02, 0.25, 0.2, 0.5, 1.0))
    p("q.fwd", QuantoOption.quanto_forward(100, 0.03, 0.01, 0.25, 0.12, -0.3, 0.75))
    p("q.price", QuantoOption.price(OptionType.CALL, 100, 105, 0.03, 0.01, 0.25, 0.12, -0.3, 0.75))
    vs_k = [80, 90, 100, 110, 120]
    vs_p = [0.5, 1.5, 4.5, 10.5, 19.0]
    vs_c = [20.5, 11.5, 4.8, 1.6, 0.4]
    # Python port has no VarianceSwap.fair_variance wrapper; Java defines it as
    # VolatilityIndex.index(...)^2 — computed identically here.
    vix = VolatilityIndex.index(vs_k, vs_p, vs_c, 100, 0.02, 0.5)
    p("vs.fairvar", vix * vix)
    p("vs.volstrike", VarianceSwap.vol_swap_strike(0.05, 0.004))

    # ---- Section 4: rates ----
    yc = YieldCurve.bootstrap_annual_par_swaps(
        [1, 2, 3, 4, 5], [0.020, 0.023, 0.025, 0.027, 0.028])
    p("yc.zero1", yc.zero_rate(1))
    p("yc.zero3", yc.zero_rate(3))
    p("yc.zero5", yc.zero_rate(5))
    p("yc.fwd23", yc.forward_rate(2, 3))
    p("yc.df5", yc.discount_factor(5))
    bp = BondPricer.price_from_yield(100, 0.05, 2, 10, 0.04)
    p("bond.price", bp)
    p("bond.ytm", BondPricer.yield_to_maturity(bp, 100, 0.05, 2, 10))
    fit_t = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
    fit_z = [0.020, 0.021, 0.0225, 0.024, 0.025, 0.0265, 0.027, 0.0275]
    ns = NelsonSiegel.fit(fit_t, fit_z)
    p("ns.b0", ns.b0)
    p("ns.rmse", ns.rmse)
    p("ns.z5", ns.zero_rate(5))
    sv = Svensson.fit(fit_t, fit_z)
    p("sv.b0", sv.b0)
    p("sv.rmse", sv.rmse)
    p("sv.z5", sv.zero_rate(5))
    p("swap.par5", SwapPricer.par_rate(yc, 5))
    p("swap.annuity5", SwapPricer.annuity(yc, 5))
    p("swap.dv01", SwapPricer.dv01(yc, 5, 0.025))
    p("srm.vasicek", ShortRateModels.vasicek_bond(0.02, 0.5, 0.03, 0.01, 5))
    p("srm.cir", ShortRateModels.cir_bond(0.02, 0.5, 0.03, 0.05, 5))
    p("srm.hw", ShortRateModels.hull_white_bond(yc, 1.0, 4.0, 0.025, 0.1, 0.01))
    p("srm.feller", ShortRateModels.cir_feller(0.5, 0.03, 0.05))

    # ---- Section 5: credit ----
    cc = CreditCurve.bootstrap([1, 3, 5], [0.010, 0.012, 0.015], 0.4, yc)
    p("cc.s1", cc.survival_probability(1))
    p("cc.s3", cc.survival_probability(3))
    p("cc.s5", cc.survival_probability(5))
    p("cc.h2", cc.hazard(2))
    p("cds.par5", CdsPricer.par_spread(cc, yc, 5))
    p("cds.upfront", CdsPricer.upfront(cc, yc, 0.01, 5))
    zp = CreditSpreads.price_with_z_spread(100, 0.05, 2, 5, yc, 0.012)
    p("zs.price", zp)
    p("zs.roundtrip", CreditSpreads.z_spread(zp, 100, 0.05, 2, 5, yc))
    p("cva", CvaApproximator.cva([1e6, 9e5, 7e5, 4e5], [1, 2, 3, 4], cc, yc, 0.6))

    # ---- Section 6: commodities / index / private markets ----
    cmd = CommodityCurve.of(50, [0.25, 0.5, 1, 2], [51, 52, 53.5, 55])
    p("cmd.roll", cmd.annualized_roll_yield(0.25, 1))
    p("cmd.carry", cmd.implied_carry(1, 0.02))
    iw = IndexConstruction.cap_weights([10, 20, 30], [1000, 500, 200], [0.9, 0.8, 1.0])
    p("idx.w0", iw[0])
    p("idx.w2", iw[2])
    p("idx.div", IndexConstruction.adjust_divisor(1000, 50000, 52000))
    p("idx.turn", IndexConstruction.turnover([0.5, 0.3, 0.2], [0.4, 0.4, 0.2]))
    p("pm.irr", PrivateMarketAnalytics.irr([-100, -50, 30, 40, 150]))
    p("pm.tvpi", PrivateMarketAnalytics.tvpi(150, 120, 80))
    p("pm.kspme", PrivateMarketAnalytics.ks_pme([100, 50, 0, 0],
            [0, 0, 30, 40], 150, [100, 105, 110, 120]))
    p("pm.geltner2", PrivateMarketAnalytics.geltner_desmooth(
            [0.02, 0.01, -0.005, 0.015], 0.4)[2])

    # ---- Section 7: risk / vol / backtest ----
    sin500 = [math.sin(i + 1.0) * 0.01 for i in range(500)]
    expo = [1e6, 2e6, 1.5e6]
    cov3 = [[0.04, 0.006, 0.012], [0.006, 0.09, 0.009], [0.012, 0.009, 0.0625]]
    gam3 = [[0.01, 0, 0], [0, 0.02, 0], [0, 0, 0.005]]
    p("var.dn", var_engine.delta_normal_var(expo, cov3, 0.99))
    p("var.dnes", var_engine.delta_normal_es(expo, cov3, 0.99))
    p("var.dges", var_engine.delta_gamma_es(expo, gam3, cov3, 0.99))
    fr = [[math.sin(float(i) * j) * 0.01 for j in range(1, 4)] for i in range(1, 501)]
    hv = var_engine.historical_var(expo, fr, 0.99)
    p("var.hist", hv.var)
    p("var.hist.es", hv.expected_shortfall)
    al = component_var.allocate([0.5, 0.3, 0.2], cov3, 0.99)
    p("cvar.pvar", al.portfolio_var)
    p("cvar.c0", al.components[0])
    p("cvar.c1", al.components[1])
    p("cvar.c2", al.components[2])
    lw_r = [[math.sin(float(i) * j) * 0.01 for j in range(1, 4)] for i in range(1, 61)]
    lw = covariance_shrinkage.ledoit_wolf(lw_r)
    p("lw.trace", lw.matrix[0][0] + lw.matrix[1][1] + lw.matrix[2][2])
    p("lw.intensity", lw.intensity)
    losses = [abs(math.sin(i + 1.0)) * 0.01 for i in range(200)]
    evt = extreme_value_theory.fit_pot(losses, 0.9)
    p("evt.shape", evt.shape)
    p("evt.scale", evt.scale)
    p("frtb.lh", frtb_es.liquidity_horizon_es([1e5, 5e4, 3e4, 2e4, 1e4],
            [10, 20, 40, 60, 120]))
    vb = var_backtest.test(sin500, [0.008] * 500, 0.99)
    p("kup.stat", vb.kupiec_statistic)
    p("kup.pval", vb.kupiec_p_value)
    p("pca.e1", Pca(cov3).eigenvalue(0))
    o = [0.0] * 10
    h = [0.0] * 10
    low = [0.0] * 10
    c = [0.0] * 10
    for i in range(1, 11):
        o[i - 1] = 100 + i
        c[i - 1] = o[i - 1] + math.sin(float(i))
        h[i - 1] = max(o[i - 1], c[i - 1]) + 1
        low[i - 1] = min(o[i - 1], c[i - 1]) - 1
    p("rv.park", RangeVolatility.parkinson(h, low, 252))
    p("rv.gk", RangeVolatility.garman_klass(o, h, low, c, 252))
    p("rv.rs", RangeVolatility.rogers_satchell(o, h, low, c, 252))
    p("rv.yz", RangeVolatility.yang_zhang(o, h, low, c, 252))
    gp = Garch11.fit(sin500)
    p("garch.omega", gp.omega)
    p("garch.alpha", gp.alpha)
    p("garch.beta", gp.beta)
    p("garch.fc1", Garch11.forecast_variance(sin500, gp, 1))
    p("ewma.vol", EwmaVolatility.risk_metrics().latest_vol(sin500))
    p("ic.aic", InformationCriteria.aic(-1234.5, 3))
    p("ic.bic", InformationCriteria.bic(-1234.5, 3, 500))
    p("rm.sharpe", risk_metrics.sharpe_ratio(sin500, 0.0, 252))
    p("rm.sortino", risk_metrics.sortino_ratio(sin500, 0.0, 252))
    eq = []
    e = 100.0
    for i in range(100):
        e *= 1 + sin500[i]
        eq.append(e)
    p("rm.maxdd", risk_metrics.max_drawdown(eq))
    trades = [
        Trade("X", 0, 5, 0, 0, 100, 105, 1, 500, 0.05, Trade.REASON_SIGNAL),
        Trade("X", 6, 8, 0, 0, 105, 103, 1, -200, -0.02, Trade.REASON_SIGNAL),
        Trade("X", 9, 15, 0, 0, 103, 106, 1, 300, 0.03, Trade.REASON_SIGNAL),
        Trade("X", 16, 18, 0, 0, 106, 105, 1, -100, -0.01, Trade.REASON_SIGNAL),
        Trade("X", 19, 30, 0, 0, 105, 112, 1, 700, 0.07, Trade.REASON_SIGNAL),
        Trade("X", 31, 33, 0, 0, 112, 111.5, 1, -50, -0.005, Trade.REASON_SIGNAL),
    ]
    ta = TradeAnalytics.analyze(trades)
    p("ta.winrate", ta.win_rate)
    p("ta.expect", ta.expectancy)
    p("ta.payoff", ta.payoff_ratio)
    p("ta.kelly", ta.kelly_fraction)
    sp = PurgedKFold.splits(100, 5, 3, 2)
    s0 = sp[0]
    p("pkf.f0.from", s0.test_from)
    p("pkf.f0.to", s0.test_to)
    p("pkf.f0.trainlen", len(s0.train_indices))
    p("pkf.f0.trainsum", sum(s0.train_indices))
    p("sv.psr", SharpeValidation.probabilistic_sharpe(0.1, 0.0, 252, -0.5, 4.0))
    p("sv.dsr", SharpeValidation.deflated_sharpe(0.1,
            [0.02, 0.05, 0.1, 0.03, -0.01, 0.07], 252, -0.5, 4.0))
    p("kelly", PositionSizing.kelly_fraction(0.08, 0.04))
    v = [100 + 5 * math.sin(float(i)) for i in range(1, 101)]
    p("ind.sma", Indicators.sma(v, 20)[99])
    p("ind.ema", Indicators.ema(v, 20)[99])
    p("ind.rsi", Indicators.rsi(v, 14)[99])
    p("ind.macd", Indicators.macd(v, 12, 26, 9).line[99])
    p("ind.boll.up", Indicators.bollinger(v, 20, 2).upper[99])

    # ---- Section 8: fx / crb / alpha / microstructure (Phase 3) ----
    fix_prices = [100.1, 100.3, 100.2, 100.5]
    fix_sizes = [10.0, 20.0, 15.0, 5.0]
    p("fx.twap", FixingRisk.window_twap(fix_prices))
    p("fx.vwap", FixingRisk.window_vwap(fix_prices, fix_sizes))
    p("fx.trackerr", FixingRisk.tracking_error_std(0.002, 5.0))
    p("fx.partrate", FixingRisk.participation_rate(50000, 400000))

    hedge_exposures = [1000000.0, -500000.0]
    hedge_cov = [[0.04, 0.01], [0.01, 0.09]]
    hedge_loadings = [[1.0, 0.5], [0.3, 1.0]]
    hedge_cost = [200.0, 300.0]
    hedge0 = HedgeOptimizer.hedge(hedge_exposures, hedge_cov, hedge_loadings, hedge_cost, 0.0)
    p("crb.hedge0.h0", hedge0[0])
    p("crb.hedge0.h1", hedge0[1])
    hedge_l1 = HedgeOptimizer.hedge(hedge_exposures, hedge_cov, hedge_loadings, hedge_cost, 400.0)
    p("crb.hedgeL1.h0", hedge_l1[0])
    p("crb.hedgeL1.h1", hedge_l1[1])
    skq = SkewedQuoter.quote(100.0, 5.0, 300000, 1000000, 0.6)
    p("crb.quote.bid", skq.bid)
    p("crb.quote.ask", skq.ask)
    p("crb.quote.skew", skq.skew_bps)

    alpha_scores = [1.2, -0.5, 0.3, 2.1, -1.8, 0.9]
    alpha_w = PortfolioConstruction.z_score_weights(alpha_scores, 1.0, 0.3)
    p("alpha.zw0", alpha_w[0])
    p("alpha.zw3", alpha_w[3])

    tc = TradeClassifier()
    tc.on_quote(99.0, 101.0)
    trade_prices = [101.0, 100.5, 99.0, 100.0, 100.0, 102.0]
    lee_ready_sum = sum(tc.classify(tp) for tp in trade_prices)
    p("micro.leeready.sum", lee_ready_sum)

    vp = Vpin(100, 3)
    vp.on_trade(60, True)
    vp.on_trade(40, False)
    vp.on_trade(30, True)
    vp.on_trade(70, False)
    vp.on_trade(90, True)
    vp.on_trade(10, True)
    p("micro.vpin.value", vp.vpin())

    hi = HawkesIntensity(3.0, 0.2, 1_000_000_000)
    hi.on_event(0)
    hi.on_event(500_000_000)
    hi.on_event(1_200_000_000)
    hi.on_event(1_300_000_000)
    p("micro.hawkes.intensity", hi.intensity(2_000_000_000))
    p("micro.hawkes.burst", hi.burst_score(2_000_000_000))

    ec = EwmaCovariance(2, 0.9)
    ec.on_returns([0.01, 0.02])
    ec.on_returns([-0.015, 0.005])
    ec.on_returns([0.02, -0.01])
    ec.on_returns([0.005, 0.015])
    p("micro.ewmacov.cov01", ec.covariance(0, 1))
    p("micro.ewmacov.corr01", ec.correlation(0, 1))

    avs = AvellanedaStoikov(0.1, 1.5)
    p("micro.as.bid", avs.bid_quote(100.0, 500.0, 0.0004, 300.0))
    p("micro.as.ask", avs.ask_quote(100.0, 500.0, 0.0004, 300.0))

    jrv = JumpRobustVolatility(5_000_000_000)
    jrv.on_return(0.001, 1_000_000_000)
    jrv.on_return(-0.0008, 1_000_000_000)
    jrv.on_return(0.0015, 1_000_000_000)
    jrv.on_return(-0.002, 1_000_000_000)
    p("micro.jrv.vol", jrv.vol_per_sqrt_second())
    p("micro.jrv.jumpfrac", jrv.jump_fraction())

    # ---- Section 9: execution ----
    twap = twap_scheduler.schedule(1003, 10000, 4)
    p("exec.twap.first", twap[0].quantity)
    p("exec.twap.last", twap[3].quantity)

    vwap_profile = [0.2, 0.3, 0.51]
    vwap = vwap_scheduler.schedule(1000, vwap_profile, 9000)
    p("exec.vwap.q0", vwap[0].quantity)
    p("exec.vwap.q2", vwap[2].quantity)

    pov = PovTracker(100000, 0.1, 0, 100000)
    pov.on_market_volume(1000)
    pov_due = pov.due_quantity()
    pov.on_executed(pov_due)
    pov.on_market_volume(1000)
    p("exec.pov.due", pov_due)
    p("exec.pov.realized", pov.realized_participation())

    is_params = AlmgrenChriss.Params(10000, 1.0, 5, 0.3, 0.1, 0.01, 0.5)
    is_slices = implementation_shortfall_scheduler.schedule(is_params, 5000)
    p("exec.is.first", is_slices[0].quantity)
    p("exec.is.last", is_slices[-1].quantity)

    region = order_placement_policy.post_region(0.01, 0.02, -0.005, 0.001)
    p("exec.opp.from", region.from_)
    p("exec.opp.to", region.to)

    dp = DarkPoolSimulator()
    dp.on_quote(99.99, 100.01)
    dp.submit(Side.SELL, 60, 0)
    dp.submit(Side.SELL, 60, 50)
    dark_fills = dp.submit(Side.BUY, 100, 60)
    dark_filled = sum(f.quantity for f in dark_fills)
    p("exec.dark.filled", dark_filled)
    p("exec.dark.restsell", dp.resting_qty(Side.SELL))

    ucb = Ucb1Selector(3)
    ucb.select()
    ucb.record(0, 0.4)
    ucb.select()
    ucb.record(1, 0.9)
    ucb.select()
    ucb.record(2, 0.2)
    ucb_arm = ucb.select()
    p("exec.ucb.arm", ucb_arm)

    be = BenchmarkExecutor.of(Side.BUY, 1000, Benchmark.ARRIVAL_PRICE)
    bench_drift = be.schedule_drift(0.3, MarketState.neutral(100.0, 0.3))
    p("exec.bench.drift", bench_drift)

    throttle = OrderThrottle(10, 5)
    for _ in range(5):
        throttle.try_acquire(0)
    throttle.try_acquire(0)
    p("exec.throttle.nanosuntil", throttle.nanos_until_available(0))

    gate = LastLookGate(0.0001)
    llg_up = gate.accept(True, 1.2000, 1.2002)
    llg_down = gate.accept(True, 1.2000, 1.1998)
    p("exec.llg.up", 1 if llg_up else 0)
    p("exec.llg.down", 1 if llg_down else 0)

    sc = VenueScorecard(2)
    sc.on_fill(0, 1_000_000, True, 100.0, 0)
    sc.on_mid(100.05, 100_000_000)
    sc.on_fill(0, 1_200_000, True, 100.05, 100_000_000)
    sc.on_mid(100.02, 200_000_000)
    p("exec.venue.markout", sc.post_fill_markout(0))


if __name__ == "__main__":
    main()
