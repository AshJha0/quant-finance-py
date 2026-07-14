import com.quantfinlib.pricing.BlackScholes;
import com.quantfinlib.pricing.BlackScholes.OptionType;
import com.quantfinlib.pricing.Black76;
import com.quantfinlib.pricing.DigitalOption;
import com.quantfinlib.pricing.BarrierOption;
import com.quantfinlib.pricing.TouchOption;
import com.quantfinlib.pricing.AsianOption;
import com.quantfinlib.pricing.StructuredNotes;
import com.quantfinlib.pricing.ExchangeOption;
import com.quantfinlib.pricing.QuantoOption;
import com.quantfinlib.pricing.VarianceSwap;
import com.quantfinlib.rates.YieldCurve;
import com.quantfinlib.rates.BondPricer;
import com.quantfinlib.rates.NelsonSiegel;
import com.quantfinlib.rates.Svensson;
import com.quantfinlib.rates.SwapPricer;
import com.quantfinlib.rates.ShortRateModels;
import com.quantfinlib.credit.CreditCurve;
import com.quantfinlib.credit.CdsPricer;
import com.quantfinlib.credit.CreditSpreads;
import com.quantfinlib.credit.CvaApproximator;
import com.quantfinlib.commodities.CommodityCurve;
import com.quantfinlib.markets.IndexConstruction;
import com.quantfinlib.markets.PrivateMarketAnalytics;
import com.quantfinlib.risk.VarEngine;
import com.quantfinlib.risk.ComponentVar;
import com.quantfinlib.risk.CovarianceShrinkage;
import com.quantfinlib.risk.ExtremeValueTheory;
import com.quantfinlib.risk.FrtbEs;
import com.quantfinlib.risk.VarBacktest;
import com.quantfinlib.risk.Pca;
import com.quantfinlib.risk.RiskMetrics;
import com.quantfinlib.volatility.RangeVolatility;
import com.quantfinlib.volatility.Garch11;
import com.quantfinlib.volatility.EwmaVolatility;
import com.quantfinlib.volatility.InformationCriteria;
import com.quantfinlib.backtest.Trade;
import com.quantfinlib.backtest.TradeAnalytics;
import com.quantfinlib.backtest.validation.PurgedKFold;
import com.quantfinlib.backtest.validation.SharpeValidation;
import com.quantfinlib.backtest.portfolio.PositionSizing;
import com.quantfinlib.indicators.Indicators;

import java.util.List;
import java.util.Locale;

public final class Probe {
    static void p(String label, double v) {
        System.out.printf(Locale.ROOT, "%s=%.15e%n", label, v);
    }

    public static void main(String[] args) {
        // ---- Section 1: Black-Scholes ----
        double S = 100, K = 105, r = 0.03, q = 0.01, sig = 0.25, T = 0.75;
        double call = BlackScholes.price(OptionType.CALL, S, K, r, q, sig, T);
        p("bs.call", call);
        p("bs.put", BlackScholes.price(OptionType.PUT, S, K, r, q, sig, T));
        p("bs.delta.call", BlackScholes.delta(OptionType.CALL, S, K, r, q, sig, T));
        p("bs.delta.put", BlackScholes.delta(OptionType.PUT, S, K, r, q, sig, T));
        p("bs.gamma", BlackScholes.gamma(S, K, r, q, sig, T));
        p("bs.vega", BlackScholes.vega(S, K, r, q, sig, T));
        p("bs.theta.call", BlackScholes.theta(OptionType.CALL, S, K, r, q, sig, T));
        p("bs.rho.call", BlackScholes.rho(OptionType.CALL, S, K, r, q, sig, T));
        p("bs.rho.put", BlackScholes.rho(OptionType.PUT, S, K, r, q, sig, T));
        p("bs.iv.roundtrip", BlackScholes.impliedVol(OptionType.CALL, call, S, K, r, q, T));

        // ---- Section 2: Black-76 ----
        p("b76.call", Black76.price(OptionType.CALL, 100, 95, 0.02, 0.3, 0.5));
        p("b76.put", Black76.price(OptionType.PUT, 100, 95, 0.02, 0.3, 0.5));

        // ---- Section 3: exotics & structured ----
        p("dig.cash.call", DigitalOption.cashOrNothing(OptionType.CALL, S, K, r, q, sig, T, 10));
        p("bar.doc", BarrierOption.downAndOutCall(100, 105, 85, r, q, sig, T));
        p("touch.one", TouchOption.oneTouch(100, 90, r, q, sig, T, 10));
        p("asian.geo", AsianOption.geometricPrice(OptionType.CALL, 100, 100, r, q, sig, 1.0, 12));
        p("asian.tw", AsianOption.arithmeticPrice(OptionType.CALL, 100, 100, r, q, sig, 1.0, 12));
        p("sn.rc", StructuredNotes.reverseConvertible(1000, 0.09, 100, 95, r, q, sig, 1.0));
        p("sn.rc.delta", StructuredNotes.reverseConvertibleDelta(1000, 100, 95, r, q, sig, 1.0));
        p("sn.cpn", StructuredNotes.capitalProtectedNote(1000, 0.9, 0.5, 100, r, q, sig, 1.0));
        p("sn.part", StructuredNotes.participationFor(1000, 0.9, 980, 100, r, q, sig, 1.0));
        p("sn.dc", StructuredNotes.discountCertificate(100, 110, r, q, sig, 1.0));
        p("sn.dc.delta", StructuredNotes.discountCertificateDelta(100, 110, r, q, sig, 1.0));
        p("ex.margrabe", ExchangeOption.margrabe(100, 95, 0.01, 0.02, 0.25, 0.2, 0.5, 1.0));
        p("ex.kirk", ExchangeOption.kirkSpreadCall(100, 95, 5, 0.02, 0.25, 0.2, 0.5, 1.0));
        p("q.fwd", QuantoOption.quantoForward(100, 0.03, 0.01, 0.25, 0.12, -0.3, 0.75));
        p("q.price", QuantoOption.price(OptionType.CALL, 100, 105, 0.03, 0.01, 0.25, 0.12, -0.3, 0.75));
        double[] vsK = {80, 90, 100, 110, 120};
        double[] vsP = {0.5, 1.5, 4.5, 10.5, 19.0};
        double[] vsC = {20.5, 11.5, 4.8, 1.6, 0.4};
        p("vs.fairvar", VarianceSwap.fairVariance(vsK, vsP, vsC, 100, 0.02, 0.5));
        p("vs.volstrike", VarianceSwap.volSwapStrike(0.05, 0.004));

        // ---- Section 4: rates ----
        YieldCurve yc = YieldCurve.bootstrapAnnualParSwaps(
                new int[]{1, 2, 3, 4, 5},
                new double[]{0.020, 0.023, 0.025, 0.027, 0.028});
        p("yc.zero1", yc.zeroRate(1));
        p("yc.zero3", yc.zeroRate(3));
        p("yc.zero5", yc.zeroRate(5));
        p("yc.fwd23", yc.forwardRate(2, 3));
        p("yc.df5", yc.discountFactor(5));
        double bp = BondPricer.priceFromYield(100, 0.05, 2, 10, 0.04);
        p("bond.price", bp);
        p("bond.ytm", BondPricer.yieldToMaturity(bp, 100, 0.05, 2, 10));
        double[] fitT = {0.25, 0.5, 1, 2, 3, 5, 7, 10};
        double[] fitZ = {0.020, 0.021, 0.0225, 0.024, 0.025, 0.0265, 0.027, 0.0275};
        NelsonSiegel.Fit ns = NelsonSiegel.fit(fitT, fitZ);
        p("ns.b0", ns.b0());
        p("ns.rmse", ns.rmse());
        p("ns.z5", ns.zeroRate(5));
        Svensson.Fit sv = Svensson.fit(fitT, fitZ);
        p("sv.b0", sv.b0());
        p("sv.rmse", sv.rmse());
        p("sv.z5", sv.zeroRate(5));
        p("swap.par5", SwapPricer.parRate(yc, 5));
        p("swap.annuity5", SwapPricer.annuity(yc, 5));
        p("swap.dv01", SwapPricer.dv01(yc, 5, 0.025));
        p("srm.vasicek", ShortRateModels.vasicekBond(0.02, 0.5, 0.03, 0.01, 5));
        p("srm.cir", ShortRateModels.cirBond(0.02, 0.5, 0.03, 0.05, 5));
        p("srm.hw", ShortRateModels.hullWhiteBond(yc, 1.0, 4.0, 0.025, 0.1, 0.01));
        p("srm.feller", ShortRateModels.cirFeller(0.5, 0.03, 0.05));

        // ---- Section 5: credit ----
        CreditCurve cc = CreditCurve.bootstrap(new int[]{1, 3, 5},
                new double[]{0.010, 0.012, 0.015}, 0.4, yc);
        p("cc.s1", cc.survivalProbability(1));
        p("cc.s3", cc.survivalProbability(3));
        p("cc.s5", cc.survivalProbability(5));
        p("cc.h2", cc.hazard(2));
        p("cds.par5", CdsPricer.parSpread(cc, yc, 5));
        p("cds.upfront", CdsPricer.upfront(cc, yc, 0.01, 5));
        double zp = CreditSpreads.priceWithZSpread(100, 0.05, 2, 5, yc, 0.012);
        p("zs.price", zp);
        p("zs.roundtrip", CreditSpreads.zSpread(zp, 100, 0.05, 2, 5, yc));
        p("cva", CvaApproximator.cva(new double[]{1e6, 9e5, 7e5, 4e5},
                new double[]{1, 2, 3, 4}, cc, yc, 0.6));

        // ---- Section 6: commodities / index / private markets ----
        CommodityCurve cmd = CommodityCurve.of(50,
                new double[]{0.25, 0.5, 1, 2}, new double[]{51, 52, 53.5, 55});
        p("cmd.roll", cmd.annualizedRollYield(0.25, 1));
        p("cmd.carry", cmd.impliedCarry(1, 0.02));
        double[] iw = IndexConstruction.capWeights(new double[]{10, 20, 30},
                new double[]{1000, 500, 200}, new double[]{0.9, 0.8, 1.0});
        p("idx.w0", iw[0]);
        p("idx.w2", iw[2]);
        p("idx.div", IndexConstruction.adjustDivisor(1000, 50000, 52000));
        p("idx.turn", IndexConstruction.turnover(new double[]{0.5, 0.3, 0.2},
                new double[]{0.4, 0.4, 0.2}));
        p("pm.irr", PrivateMarketAnalytics.irr(new double[]{-100, -50, 30, 40, 150}));
        p("pm.tvpi", PrivateMarketAnalytics.tvpi(150, 120, 80));
        p("pm.kspme", PrivateMarketAnalytics.ksPme(new double[]{100, 50, 0, 0},
                new double[]{0, 0, 30, 40}, 150, new double[]{100, 105, 110, 120}));
        p("pm.geltner2", PrivateMarketAnalytics.geltnerDesmooth(
                new double[]{0.02, 0.01, -0.005, 0.015}, 0.4)[2]);

        // ---- Section 7: risk / vol / backtest ----
        double[] sin500 = new double[500];
        for (int i = 0; i < 500; i++) sin500[i] = Math.sin(i + 1) * 0.01;
        double[] expo = {1e6, 2e6, 1.5e6};
        double[][] cov3 = {{0.04, 0.006, 0.012}, {0.006, 0.09, 0.009}, {0.012, 0.009, 0.0625}};
        double[][] gam3 = {{0.01, 0, 0}, {0, 0.02, 0}, {0, 0, 0.005}};
        p("var.dn", VarEngine.deltaNormalVar(expo, cov3, 0.99));
        p("var.dnes", VarEngine.deltaNormalEs(expo, cov3, 0.99));
        p("var.dges", VarEngine.deltaGammaEs(expo, gam3, cov3, 0.99));
        double[][] fr = new double[500][3];
        for (int i = 1; i <= 500; i++)
            for (int j = 1; j <= 3; j++)
                fr[i - 1][j - 1] = Math.sin((double) i * j) * 0.01;
        VarEngine.VarResult hv = VarEngine.historicalVar(expo, fr, 0.99);
        p("var.hist", hv.var());
        p("var.hist.es", hv.expectedShortfall());
        ComponentVar.Allocation al = ComponentVar.allocate(
                new double[]{0.5, 0.3, 0.2}, cov3, 0.99);
        p("cvar.pvar", al.portfolioVar());
        p("cvar.c0", al.components()[0]);
        p("cvar.c1", al.components()[1]);
        p("cvar.c2", al.components()[2]);
        double[][] lwR = new double[60][3];
        for (int i = 1; i <= 60; i++)
            for (int j = 1; j <= 3; j++)
                lwR[i - 1][j - 1] = Math.sin((double) i * j) * 0.01;
        CovarianceShrinkage.Result lw = CovarianceShrinkage.ledoitWolf(lwR);
        p("lw.trace", lw.matrix()[0][0] + lw.matrix()[1][1] + lw.matrix()[2][2]);
        p("lw.intensity", lw.intensity());
        double[] losses = new double[200];
        for (int i = 0; i < 200; i++) losses[i] = Math.abs(Math.sin(i + 1)) * 0.01;
        ExtremeValueTheory.GpdFit evt = ExtremeValueTheory.fitPot(losses, 0.9);
        p("evt.shape", evt.shape());
        p("evt.scale", evt.scale());
        p("frtb.lh", FrtbEs.liquidityHorizonEs(new double[]{1e5, 5e4, 3e4, 2e4, 1e4},
                new int[]{10, 20, 40, 60, 120}));
        VarBacktest.VarBacktestResult vb = VarBacktest.test(sin500, 0.008, 0.99);
        p("kup.stat", vb.kupiecStatistic());
        p("kup.pval", vb.kupiecPValue());
        p("pca.e1", new Pca(cov3).eigenvalue(0));
        double[] o = new double[10], h = new double[10], l = new double[10], c = new double[10];
        for (int i = 1; i <= 10; i++) {
            o[i - 1] = 100 + i;
            c[i - 1] = o[i - 1] + Math.sin(i);
            h[i - 1] = Math.max(o[i - 1], c[i - 1]) + 1;
            l[i - 1] = Math.min(o[i - 1], c[i - 1]) - 1;
        }
        p("rv.park", RangeVolatility.parkinson(h, l, 252));
        p("rv.gk", RangeVolatility.garmanKlass(o, h, l, c, 252));
        p("rv.rs", RangeVolatility.rogersSatchell(o, h, l, c, 252));
        p("rv.yz", RangeVolatility.yangZhang(o, h, l, c, 252));
        Garch11.Params gp = Garch11.fit(sin500);
        p("garch.omega", gp.omega());
        p("garch.alpha", gp.alpha());
        p("garch.beta", gp.beta());
        p("garch.fc1", Garch11.forecastVariance(sin500, gp, 1));
        p("ewma.vol", EwmaVolatility.riskMetrics().latestVol(sin500));
        p("ic.aic", InformationCriteria.aic(-1234.5, 3));
        p("ic.bic", InformationCriteria.bic(-1234.5, 3, 500));
        p("rm.sharpe", RiskMetrics.sharpeRatio(sin500, 0.0, 252));
        p("rm.sortino", RiskMetrics.sortinoRatio(sin500, 0.0, 252));
        double[] eq = new double[100];
        double e = 100;
        for (int i = 0; i < 100; i++) {
            e *= 1 + sin500[i];
            eq[i] = e;
        }
        p("rm.maxdd", RiskMetrics.maxDrawdown(eq));
        List<Trade> trades = List.of(
                new Trade("X", 0, 5, 0, 0, 100, 105, 1, 500, 0.05, Trade.REASON_SIGNAL),
                new Trade("X", 6, 8, 0, 0, 105, 103, 1, -200, -0.02, Trade.REASON_SIGNAL),
                new Trade("X", 9, 15, 0, 0, 103, 106, 1, 300, 0.03, Trade.REASON_SIGNAL),
                new Trade("X", 16, 18, 0, 0, 106, 105, 1, -100, -0.01, Trade.REASON_SIGNAL),
                new Trade("X", 19, 30, 0, 0, 105, 112, 1, 700, 0.07, Trade.REASON_SIGNAL),
                new Trade("X", 31, 33, 0, 0, 112, 111.5, 1, -50, -0.005, Trade.REASON_SIGNAL));
        TradeAnalytics.Result ta = TradeAnalytics.analyze(trades);
        p("ta.winrate", ta.winRate());
        p("ta.expect", ta.expectancy());
        p("ta.payoff", ta.payoffRatio());
        p("ta.kelly", ta.kellyFraction());
        List<PurgedKFold.Split> sp = PurgedKFold.splits(100, 5, 3, 2);
        PurgedKFold.Split s0 = sp.get(0);
        p("pkf.f0.from", s0.testFrom());
        p("pkf.f0.to", s0.testTo());
        p("pkf.f0.trainlen", s0.trainIndices().length);
        long sum = 0;
        for (int idx : s0.trainIndices()) sum += idx;
        p("pkf.f0.trainsum", sum);
        p("sv.psr", SharpeValidation.probabilisticSharpe(0.1, 0.0, 252, -0.5, 4.0));
        p("sv.dsr", SharpeValidation.deflatedSharpe(0.1,
                new double[]{0.02, 0.05, 0.1, 0.03, -0.01, 0.07}, 252, -0.5, 4.0));
        p("kelly", PositionSizing.kellyFraction(0.08, 0.04));
        double[] v = new double[100];
        for (int i = 1; i <= 100; i++) v[i - 1] = 100 + 5 * Math.sin(i);
        p("ind.sma", Indicators.sma(v, 20)[99]);
        p("ind.ema", Indicators.ema(v, 20)[99]);
        p("ind.rsi", Indicators.rsi(v, 14)[99]);
        p("ind.macd", Indicators.macd(v, 12, 26, 9).line()[99]);
        p("ind.boll.up", Indicators.bollinger(v, 20, 2).upper()[99]);
    }
}
