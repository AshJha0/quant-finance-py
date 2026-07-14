// Cross-port verification probe: mirrors Probe.java exactly (labels, inputs, order).
#include <cmath>
#include <cstdio>
#include <vector>

#include "quantfinlib/pricing/black_scholes.hpp"
#include "quantfinlib/pricing/black76.hpp"
#include "quantfinlib/pricing/digital_option.hpp"
#include "quantfinlib/pricing/barrier_option.hpp"
#include "quantfinlib/pricing/touch_option.hpp"
#include "quantfinlib/pricing/asian_option.hpp"
#include "quantfinlib/pricing/structured_notes.hpp"
#include "quantfinlib/pricing/exchange_option.hpp"
#include "quantfinlib/pricing/quanto_option.hpp"
#include "quantfinlib/pricing/variance_swap.hpp"
#include "quantfinlib/rates/yield_curve.hpp"
#include "quantfinlib/rates/bond_pricer.hpp"
#include "quantfinlib/rates/nelson_siegel.hpp"
#include "quantfinlib/rates/svensson.hpp"
#include "quantfinlib/rates/swap_pricer.hpp"
#include "quantfinlib/rates/short_rate_models.hpp"
#include "quantfinlib/credit/credit_curve.hpp"
#include "quantfinlib/credit/cds_pricer.hpp"
#include "quantfinlib/credit/credit_spreads.hpp"
#include "quantfinlib/credit/cva_approximator.hpp"
#include "quantfinlib/commodities/commodity_curve.hpp"
#include "quantfinlib/markets/index_construction.hpp"
#include "quantfinlib/markets/private_market_analytics.hpp"
#include "quantfinlib/risk/var_engine.hpp"
#include "quantfinlib/risk/component_var.hpp"
#include "quantfinlib/risk/covariance_shrinkage.hpp"
#include "quantfinlib/risk/extreme_value_theory.hpp"
#include "quantfinlib/risk/frtb_es.hpp"
#include "quantfinlib/risk/var_backtest.hpp"
#include "quantfinlib/risk/pca.hpp"
#include "quantfinlib/risk/risk_metrics.hpp"
#include "quantfinlib/volatility/range_volatility.hpp"
#include "quantfinlib/volatility/garch11.hpp"
#include "quantfinlib/volatility/ewma_volatility.hpp"
#include "quantfinlib/volatility/information_criteria.hpp"
#include "quantfinlib/backtest/trade.hpp"
#include "quantfinlib/backtest/trade_analytics.hpp"
#include "quantfinlib/backtest/validation/purged_k_fold.hpp"
#include "quantfinlib/backtest/validation/sharpe_validation.hpp"
#include "quantfinlib/backtest/portfolio/position_sizing.hpp"
#include "quantfinlib/indicators/indicators.hpp"
#include "quantfinlib/fx/fixing_risk.hpp"
#include "quantfinlib/crb/hedge_optimizer.hpp"
#include "quantfinlib/crb/skewed_quoter.hpp"
#include "quantfinlib/alpha/portfolio_construction.hpp"
#include "quantfinlib/microstructure/trade_classifier.hpp"
#include "quantfinlib/microstructure/vpin.hpp"
#include "quantfinlib/microstructure/hawkes_intensity.hpp"
#include "quantfinlib/microstructure/ewma_covariance.hpp"
#include "quantfinlib/microstructure/avellaneda_stoikov.hpp"
#include "quantfinlib/microstructure/jump_robust_volatility.hpp"
#include "quantfinlib/microstructure/almgren_chriss.hpp"
#include "quantfinlib/trading/order_throttle.hpp"
#include "quantfinlib/trading/last_look_gate.hpp"
#include "quantfinlib/orderbook/side.hpp"
#include "quantfinlib/execution/slice.hpp"
#include "quantfinlib/execution/twap_scheduler.hpp"
#include "quantfinlib/execution/vwap_scheduler.hpp"
#include "quantfinlib/execution/pov_tracker.hpp"
#include "quantfinlib/execution/implementation_shortfall_scheduler.hpp"
#include "quantfinlib/execution/order_placement_policy.hpp"
#include "quantfinlib/execution/dark_pool_simulator.hpp"
#include "quantfinlib/execution/ucb1_selector.hpp"
#include "quantfinlib/execution/benchmark_executor.hpp"
#include "quantfinlib/execution/venue_scorecard.hpp"

using namespace quantfinlib;

static void p(const char* label, double v) {
    std::printf("%s=%.15e\n", label, v);
}

int main() {
    // ---- Section 1: Black-Scholes ----
    double S = 100, K = 105, r = 0.03, q = 0.01, sig = 0.25, T = 0.75;
    double call = black_scholes::price(OptionType::CALL, S, K, r, q, sig, T);
    p("bs.call", call);
    p("bs.put", black_scholes::price(OptionType::PUT, S, K, r, q, sig, T));
    p("bs.delta.call", black_scholes::delta(OptionType::CALL, S, K, r, q, sig, T));
    p("bs.delta.put", black_scholes::delta(OptionType::PUT, S, K, r, q, sig, T));
    p("bs.gamma", black_scholes::gamma(S, K, r, q, sig, T));
    p("bs.vega", black_scholes::vega(S, K, r, q, sig, T));
    p("bs.theta.call", black_scholes::theta(OptionType::CALL, S, K, r, q, sig, T));
    p("bs.rho.call", black_scholes::rho(OptionType::CALL, S, K, r, q, sig, T));
    p("bs.rho.put", black_scholes::rho(OptionType::PUT, S, K, r, q, sig, T));
    p("bs.iv.roundtrip", black_scholes::impliedVol(OptionType::CALL, call, S, K, r, q, T));

    // ---- Section 2: Black-76 ----
    p("b76.call", black76::price(OptionType::CALL, 100, 95, 0.02, 0.3, 0.5));
    p("b76.put", black76::price(OptionType::PUT, 100, 95, 0.02, 0.3, 0.5));

    // ---- Section 3: exotics & structured ----
    p("dig.cash.call", digital_option::cashOrNothing(OptionType::CALL, S, K, r, q, sig, T, 10));
    p("bar.doc", barrier_option::downAndOutCall(100, 105, 85, r, q, sig, T));
    p("touch.one", touch_option::oneTouch(100, 90, r, q, sig, T, 10));
    p("asian.geo", asian_option::geometricPrice(OptionType::CALL, 100, 100, r, q, sig, 1.0, 12));
    p("asian.tw", asian_option::arithmeticPrice(OptionType::CALL, 100, 100, r, q, sig, 1.0, 12));
    p("sn.rc", structured_notes::reverseConvertible(1000, 0.09, 100, 95, r, q, sig, 1.0));
    p("sn.rc.delta", structured_notes::reverseConvertibleDelta(1000, 100, 95, r, q, sig, 1.0));
    p("sn.cpn", structured_notes::capitalProtectedNote(1000, 0.9, 0.5, 100, r, q, sig, 1.0));
    p("sn.part", structured_notes::participationFor(1000, 0.9, 980, 100, r, q, sig, 1.0));
    p("sn.dc", structured_notes::discountCertificate(100, 110, r, q, sig, 1.0));
    p("sn.dc.delta", structured_notes::discountCertificateDelta(100, 110, r, q, sig, 1.0));
    p("ex.margrabe", exchange_option::margrabe(100, 95, 0.01, 0.02, 0.25, 0.2, 0.5, 1.0));
    p("ex.kirk", exchange_option::kirkSpreadCall(100, 95, 5, 0.02, 0.25, 0.2, 0.5, 1.0));
    p("q.fwd", quanto_option::quantoForward(100, 0.03, 0.01, 0.25, 0.12, -0.3, 0.75));
    p("q.price", quanto_option::price(OptionType::CALL, 100, 105, 0.03, 0.01, 0.25, 0.12, -0.3, 0.75));
    std::vector<double> vsK = {80, 90, 100, 110, 120};
    std::vector<double> vsP = {0.5, 1.5, 4.5, 10.5, 19.0};
    std::vector<double> vsC = {20.5, 11.5, 4.8, 1.6, 0.4};
    p("vs.fairvar", variance_swap::fairVariance(vsK, vsP, vsC, 100, 0.02, 0.5));
    p("vs.volstrike", variance_swap::volSwapStrike(0.05, 0.004));

    // ---- Section 4: rates ----
    YieldCurve yc = YieldCurve::bootstrapAnnualParSwaps(
            {1, 2, 3, 4, 5}, {0.020, 0.023, 0.025, 0.027, 0.028});
    p("yc.zero1", yc.zeroRate(1));
    p("yc.zero3", yc.zeroRate(3));
    p("yc.zero5", yc.zeroRate(5));
    p("yc.fwd23", yc.forwardRate(2, 3));
    p("yc.df5", yc.discountFactor(5));
    double bp = bond_pricer::priceFromYield(100, 0.05, 2, 10, 0.04);
    p("bond.price", bp);
    p("bond.ytm", bond_pricer::yieldToMaturity(bp, 100, 0.05, 2, 10));
    std::vector<double> fitT = {0.25, 0.5, 1, 2, 3, 5, 7, 10};
    std::vector<double> fitZ = {0.020, 0.021, 0.0225, 0.024, 0.025, 0.0265, 0.027, 0.0275};
    nelson_siegel::Fit ns = nelson_siegel::fit(fitT, fitZ);
    p("ns.b0", ns.b0);
    p("ns.rmse", ns.rmse);
    p("ns.z5", ns.zeroRate(5));
    svensson::Fit sv = svensson::fit(fitT, fitZ);
    p("sv.b0", sv.b0);
    p("sv.rmse", sv.rmse);
    p("sv.z5", sv.zeroRate(5));
    p("swap.par5", swap_pricer::parRate(yc, 5));
    p("swap.annuity5", swap_pricer::annuity(yc, 5));
    p("swap.dv01", swap_pricer::dv01(yc, 5, 0.025));
    p("srm.vasicek", short_rate_models::vasicekBond(0.02, 0.5, 0.03, 0.01, 5));
    p("srm.cir", short_rate_models::cirBond(0.02, 0.5, 0.03, 0.05, 5));
    p("srm.hw", short_rate_models::hullWhiteBond(yc, 1.0, 4.0, 0.025, 0.1, 0.01));
    p("srm.feller", short_rate_models::cirFeller(0.5, 0.03, 0.05));

    // ---- Section 5: credit ----
    CreditCurve cc = CreditCurve::bootstrap({1, 3, 5}, {0.010, 0.012, 0.015}, 0.4, yc);
    p("cc.s1", cc.survivalProbability(1));
    p("cc.s3", cc.survivalProbability(3));
    p("cc.s5", cc.survivalProbability(5));
    p("cc.h2", cc.hazard(2));
    p("cds.par5", cds_pricer::parSpread(cc, yc, 5));
    p("cds.upfront", cds_pricer::upfront(cc, yc, 0.01, 5));
    double zp = credit_spreads::priceWithZSpread(100, 0.05, 2, 5, yc, 0.012);
    p("zs.price", zp);
    p("zs.roundtrip", credit_spreads::zSpread(zp, 100, 0.05, 2, 5, yc));
    p("cva", cva_approximator::cva({1e6, 9e5, 7e5, 4e5}, {1, 2, 3, 4}, cc, yc, 0.6));

    // ---- Section 6: commodities / index / private markets ----
    CommodityCurve cmd = CommodityCurve::of(50, {0.25, 0.5, 1, 2}, {51, 52, 53.5, 55});
    p("cmd.roll", cmd.annualizedRollYield(0.25, 1));
    p("cmd.carry", cmd.impliedCarry(1, 0.02));
    std::vector<double> iw = index_construction::capWeights(
            {10, 20, 30}, {1000, 500, 200}, {0.9, 0.8, 1.0});
    p("idx.w0", iw[0]);
    p("idx.w2", iw[2]);
    p("idx.div", index_construction::adjustDivisor(1000, 50000, 52000));
    p("idx.turn", index_construction::turnover({0.5, 0.3, 0.2}, {0.4, 0.4, 0.2}));
    p("pm.irr", private_market_analytics::irr({-100, -50, 30, 40, 150}));
    p("pm.tvpi", private_market_analytics::tvpi(150, 120, 80));
    p("pm.kspme", private_market_analytics::ksPme({100, 50, 0, 0},
            {0, 0, 30, 40}, 150, {100, 105, 110, 120}));
    p("pm.geltner2", private_market_analytics::geltnerDesmooth(
            {0.02, 0.01, -0.005, 0.015}, 0.4)[2]);

    // ---- Section 7: risk / vol / backtest ----
    std::vector<double> sin500(500);
    for (int i = 0; i < 500; i++) sin500[i] = std::sin(i + 1.0) * 0.01;
    std::vector<double> expo = {1e6, 2e6, 1.5e6};
    var_engine::Matrix cov3 = {{0.04, 0.006, 0.012}, {0.006, 0.09, 0.009}, {0.012, 0.009, 0.0625}};
    var_engine::Matrix gam3 = {{0.01, 0, 0}, {0, 0.02, 0}, {0, 0, 0.005}};
    p("var.dn", var_engine::deltaNormalVar(expo, cov3, 0.99));
    p("var.dnes", var_engine::deltaNormalEs(expo, cov3, 0.99));
    p("var.dges", var_engine::deltaGammaEs(expo, gam3, cov3, 0.99));
    var_engine::Matrix fr(500, std::vector<double>(3));
    for (int i = 1; i <= 500; i++)
        for (int j = 1; j <= 3; j++)
            fr[i - 1][j - 1] = std::sin(static_cast<double>(i) * j) * 0.01;
    var_engine::VarResult hv = var_engine::historicalVar(expo, fr, 0.99);
    p("var.hist", hv.var);
    p("var.hist.es", hv.expectedShortfall);
    component_var::Allocation al = component_var::allocate({0.5, 0.3, 0.2}, cov3, 0.99);
    p("cvar.pvar", al.portfolioVar);
    p("cvar.c0", al.components[0]);
    p("cvar.c1", al.components[1]);
    p("cvar.c2", al.components[2]);
    covariance_shrinkage::Matrix lwR(60, std::vector<double>(3));
    for (int i = 1; i <= 60; i++)
        for (int j = 1; j <= 3; j++)
            lwR[i - 1][j - 1] = std::sin(static_cast<double>(i) * j) * 0.01;
    covariance_shrinkage::Result lw = covariance_shrinkage::ledoitWolf(lwR);
    p("lw.trace", lw.matrix[0][0] + lw.matrix[1][1] + lw.matrix[2][2]);
    p("lw.intensity", lw.intensity);
    std::vector<double> losses(200);
    for (int i = 0; i < 200; i++) losses[i] = std::fabs(std::sin(i + 1.0)) * 0.01;
    extreme_value_theory::GpdFit evt = extreme_value_theory::fitPot(losses, 0.9);
    p("evt.shape", evt.shape);
    p("evt.scale", evt.scale);
    p("frtb.lh", frtb_es::liquidityHorizonEs({1e5, 5e4, 3e4, 2e4, 1e4},
            {10, 20, 40, 60, 120}));
    var_backtest::VarBacktestResult vb = var_backtest::test(sin500, 0.008, 0.99);
    p("kup.stat", vb.kupiecStatistic);
    p("kup.pval", vb.kupiecPValue);
    p("pca.e1", Pca(cov3).eigenvalue(0));
    std::vector<double> o(10), h(10), l(10), c(10);
    for (int i = 1; i <= 10; i++) {
        o[i - 1] = 100 + i;
        c[i - 1] = o[i - 1] + std::sin(static_cast<double>(i));
        h[i - 1] = std::max(o[i - 1], c[i - 1]) + 1;
        l[i - 1] = std::min(o[i - 1], c[i - 1]) - 1;
    }
    p("rv.park", RangeVolatility::parkinson(h, l, 252));
    p("rv.gk", RangeVolatility::garmanKlass(o, h, l, c, 252));
    p("rv.rs", RangeVolatility::rogersSatchell(o, h, l, c, 252));
    p("rv.yz", RangeVolatility::yangZhang(o, h, l, c, 252));
    Garch11::Params gp = Garch11::fit(sin500);
    p("garch.omega", gp.omega);
    p("garch.alpha", gp.alpha);
    p("garch.beta", gp.beta);
    p("garch.fc1", Garch11::forecastVariance(sin500, gp, 1));
    p("ewma.vol", EwmaVolatility::riskMetrics().latestVol(sin500));
    p("ic.aic", InformationCriteria::aic(-1234.5, 3));
    p("ic.bic", InformationCriteria::bic(-1234.5, 3, 500));
    p("rm.sharpe", risk_metrics::sharpeRatio(sin500, 0.0, 252));
    p("rm.sortino", risk_metrics::sortinoRatio(sin500, 0.0, 252));
    std::vector<double> eq(100);
    double e = 100;
    for (int i = 0; i < 100; i++) {
        e *= 1 + sin500[i];
        eq[i] = e;
    }
    p("rm.maxdd", risk_metrics::maxDrawdown(eq));
    std::vector<Trade> trades = {
        {"X", 0, 5, 0, 0, 100, 105, 1, 500, 0.05, Trade::REASON_SIGNAL},
        {"X", 6, 8, 0, 0, 105, 103, 1, -200, -0.02, Trade::REASON_SIGNAL},
        {"X", 9, 15, 0, 0, 103, 106, 1, 300, 0.03, Trade::REASON_SIGNAL},
        {"X", 16, 18, 0, 0, 106, 105, 1, -100, -0.01, Trade::REASON_SIGNAL},
        {"X", 19, 30, 0, 0, 105, 112, 1, 700, 0.07, Trade::REASON_SIGNAL},
        {"X", 31, 33, 0, 0, 112, 111.5, 1, -50, -0.005, Trade::REASON_SIGNAL}};
    trade_analytics::Result ta = trade_analytics::analyze(trades);
    p("ta.winrate", ta.winRate);
    p("ta.expect", ta.expectancy);
    p("ta.payoff", ta.payoffRatio);
    p("ta.kelly", ta.kellyFraction);
    std::vector<purged_k_fold::Split> sp = purged_k_fold::splits(100, 5, 3, 2);
    const purged_k_fold::Split& s0 = sp[0];
    p("pkf.f0.from", s0.testFrom);
    p("pkf.f0.to", s0.testTo);
    p("pkf.f0.trainlen", static_cast<double>(s0.trainIndices.size()));
    long long sum = 0;
    for (int idx : s0.trainIndices) sum += idx;
    p("pkf.f0.trainsum", static_cast<double>(sum));
    p("sv.psr", sharpe_validation::probabilisticSharpe(0.1, 0.0, 252, -0.5, 4.0));
    p("sv.dsr", sharpe_validation::deflatedSharpe(0.1,
            {0.02, 0.05, 0.1, 0.03, -0.01, 0.07}, 252, -0.5, 4.0));
    p("kelly", position_sizing::kellyFraction(0.08, 0.04));
    std::vector<double> v(100);
    for (int i = 1; i <= 100; i++) v[i - 1] = 100 + 5 * std::sin(static_cast<double>(i));
    p("ind.sma", Indicators::sma(v, 20)[99]);
    p("ind.ema", Indicators::ema(v, 20)[99]);
    p("ind.rsi", Indicators::rsi(v, 14)[99]);
    p("ind.macd", Indicators::macd(v, 12, 26, 9).line[99]);
    p("ind.boll.up", Indicators::bollinger(v, 20, 2).upper[99]);

    // ---- Section 8: fx / crb / alpha / microstructure (Phase 3) ----
    std::vector<double> fixPrices = {100.1, 100.3, 100.2, 100.5};
    std::vector<double> fixSizes = {10.0, 20.0, 15.0, 5.0};
    p("fx.twap", fixing_risk::windowTwap(fixPrices));
    p("fx.vwap", fixing_risk::windowVwap(fixPrices, fixSizes));
    p("fx.trackerr", fixing_risk::trackingErrorStd(0.002, 5.0));
    p("fx.partrate", fixing_risk::participationRate(50000, 400000));

    std::vector<double> hedgeExposures = {1000000.0, -500000.0};
    hedge_optimizer::Matrix hedgeCov = {{0.04, 0.01}, {0.01, 0.09}};
    hedge_optimizer::Matrix hedgeLoadings = {{1.0, 0.5}, {0.3, 1.0}};
    std::vector<double> hedgeCost = {200.0, 300.0};
    std::vector<double> hedge0 = hedge_optimizer::hedge(hedgeExposures, hedgeCov, hedgeLoadings, hedgeCost, 0.0);
    p("crb.hedge0.h0", hedge0[0]);
    p("crb.hedge0.h1", hedge0[1]);
    std::vector<double> hedgeL1 = hedge_optimizer::hedge(hedgeExposures, hedgeCov, hedgeLoadings, hedgeCost, 400.0);
    p("crb.hedgeL1.h0", hedgeL1[0]);
    p("crb.hedgeL1.h1", hedgeL1[1]);
    skewed_quoter::Quote skq = skewed_quoter::quote(100.0, 5.0, 300000, 1000000, 0.6);
    p("crb.quote.bid", skq.bid);
    p("crb.quote.ask", skq.ask);
    p("crb.quote.skew", skq.skewBps);

    std::vector<double> alphaScores = {1.2, -0.5, 0.3, 2.1, -1.8, 0.9};
    std::vector<double> alphaW = portfolio_construction::zScoreWeights(alphaScores, 1.0, 0.3);
    p("alpha.zw0", alphaW[0]);
    p("alpha.zw3", alphaW[3]);

    TradeClassifier tc;
    tc.onQuote(99.0, 101.0);
    std::vector<double> tradePrices = {101.0, 100.5, 99.0, 100.0, 100.0, 102.0};
    double leeReadySum = 0;
    for (double tp : tradePrices) {
        leeReadySum += tc.classify(tp);
    }
    p("micro.leeready.sum", leeReadySum);

    Vpin vp(100, 3);
    vp.onTrade(60, true);
    vp.onTrade(40, false);
    vp.onTrade(30, true);
    vp.onTrade(70, false);
    vp.onTrade(90, true);
    vp.onTrade(10, true);
    p("micro.vpin.value", vp.vpin());

    HawkesIntensity hi(3.0, 0.2, 1'000'000'000LL);
    hi.onEvent(0LL);
    hi.onEvent(500'000'000LL);
    hi.onEvent(1'200'000'000LL);
    hi.onEvent(1'300'000'000LL);
    p("micro.hawkes.intensity", hi.intensity(2'000'000'000LL));
    p("micro.hawkes.burst", hi.burstScore(2'000'000'000LL));

    EwmaCovariance ec(2, 0.9);
    ec.onReturns({0.01, 0.02});
    ec.onReturns({-0.015, 0.005});
    ec.onReturns({0.02, -0.01});
    ec.onReturns({0.005, 0.015});
    p("micro.ewmacov.cov01", ec.covariance(0, 1));
    p("micro.ewmacov.corr01", ec.correlation(0, 1));

    AvellanedaStoikov avs(0.1, 1.5);
    p("micro.as.bid", avs.bidQuote(100.0, 500.0, 0.0004, 300.0));
    p("micro.as.ask", avs.askQuote(100.0, 500.0, 0.0004, 300.0));

    JumpRobustVolatility jrv(5'000'000'000LL);
    jrv.onReturn(0.001, 1'000'000'000LL);
    jrv.onReturn(-0.0008, 1'000'000'000LL);
    jrv.onReturn(0.0015, 1'000'000'000LL);
    jrv.onReturn(-0.002, 1'000'000'000LL);
    p("micro.jrv.vol", jrv.volPerSqrtSecond());
    p("micro.jrv.jumpfrac", jrv.jumpFraction());

    // ---- Section 9: execution ----
    std::vector<Slice> twap = twap_scheduler::schedule(1003, 10000, 4);
    p("exec.twap.first", static_cast<double>(twap[0].quantity));
    p("exec.twap.last", static_cast<double>(twap[3].quantity));

    std::vector<double> vwapProfile = {0.2, 0.3, 0.51};
    std::vector<Slice> vwap = vwap_scheduler::schedule(1000, vwapProfile, 9000);
    p("exec.vwap.q0", static_cast<double>(vwap[0].quantity));
    p("exec.vwap.q2", static_cast<double>(vwap[2].quantity));

    PovTracker pov(100000, 0.1, 0, 100000);
    pov.onMarketVolume(1000);
    std::int64_t povDue = pov.dueQuantity();
    pov.onExecuted(povDue);
    pov.onMarketVolume(1000);
    p("exec.pov.due", static_cast<double>(povDue));
    p("exec.pov.realized", pov.realizedParticipation());

    AlmgrenChriss::Params isParams(10000, 1.0, 5, 0.3, 0.1, 0.01, 0.5);
    std::vector<Slice> isSlices = implementation_shortfall_scheduler::schedule(isParams, 5000);
    p("exec.is.first", static_cast<double>(isSlices.front().quantity));
    p("exec.is.last", static_cast<double>(isSlices.back().quantity));

    order_placement_policy::PostRegion region =
            order_placement_policy::postRegion(0.01, 0.02, -0.005, 0.001);
    p("exec.opp.from", region.from);
    p("exec.opp.to", region.to);

    DarkPoolSimulator dp;
    dp.onQuote(99.99, 100.01);
    dp.submit(Side::SELL, 60, 0);
    dp.submit(Side::SELL, 60, 50);
    std::vector<DarkPoolSimulator::Fill> darkFills = dp.submit(Side::BUY, 100, 60);
    std::int64_t darkFilled = 0;
    for (const auto& f : darkFills) {
        darkFilled += f.quantity;
    }
    p("exec.dark.filled", static_cast<double>(darkFilled));
    p("exec.dark.restsell", static_cast<double>(dp.restingQty(Side::SELL)));

    Ucb1Selector ucb(3);
    ucb.select();
    ucb.record(0, 0.4);
    ucb.select();
    ucb.record(1, 0.9);
    ucb.select();
    ucb.record(2, 0.2);
    int ucbArm = ucb.select();
    p("exec.ucb.arm", static_cast<double>(ucbArm));

    BenchmarkExecutor be = BenchmarkExecutor::of(Side::BUY, 1000, BenchmarkExecutor::Benchmark::ARRIVAL_PRICE);
    double benchDrift = be.scheduleDrift(0.3, BenchmarkExecutor::MarketState::neutral(100.0, 0.3));
    p("exec.bench.drift", benchDrift);

    OrderThrottle throttle(10, 5);
    for (int i = 0; i < 5; i++) {
        throttle.tryAcquire(0LL);
    }
    throttle.tryAcquire(0LL);
    p("exec.throttle.nanosuntil", static_cast<double>(throttle.nanosUntilAvailable(0LL)));

    LastLookGate gate(0.0001);
    bool llgUp = gate.accept(true, 1.2000, 1.2002);
    bool llgDown = gate.accept(true, 1.2000, 1.1998);
    p("exec.llg.up", llgUp ? 1.0 : 0.0);
    p("exec.llg.down", llgDown ? 1.0 : 0.0);

    VenueScorecard sc(2);
    sc.onFill(0, 1'000'000LL, true, 100.0, 0LL);
    sc.onMid(100.05, 100'000'000LL);
    sc.onFill(0, 1'200'000LL, true, 100.05, 100'000'000LL);
    sc.onMid(100.02, 200'000'000LL);
    p("exec.venue.markout", sc.postFillMarkout(0));

    return 0;
}
