# Architecture

Package map and dependency structure of `quantfinlib` (223 modules
under `src/quantfinlib/`, 26 packages). Every arrow in the diagram was
derived from the actual `from quantfinlib.<pkg> import ...` statements
in the source tree (`grep -rn "quantfinlib\." src/`), not from intent;
the module counts exclude `__init__.py`.

## Layering

The library is a strict DAG, four layers deep. Nothing imports upward,
and the only "hub" below the research layer is `microstructure`, whose
`execution.py` (`Side`, `Execution`) is the shared order/fill vocabulary
that the wire-format, trading and regulatory packages all speak.

| Layer | Packages | Depends on |
|---|---|---|
| 0 -- primitives | `util` (`math_utils`), `sim` (`_java_random`, `MonteCarloSimulator`), `persist` (`Checkpoint`), `data` | `util` only (`persist` and `data` are leaves) |
| 1 -- models | `pricing`, `rates`, `credit`, `commodities`, `markets`, `risk`, `volatility`, `indicators`, `ml` | `util`; `pricing` also uses `sim` (the SABR calibration RNG); `credit` uses `rates` |
| 2 -- desk engines | `microstructure`, `fx`, `crb`, `backtest`, `screener`, `execution` | layer 0-1; `backtest` and `execution` use `microstructure`; `crb` uses `pricing` + `risk`; `fx` uses `rates` + `persist` |
| 3 -- research & plumbing | `alpha`, `dsl`, `trading`, `fix`, `sbe`, `marketdata`, `regulatory` | `alpha` sits on `backtest` + `risk` + `microstructure`; `dsl` on `backtest`; the wire/venue packages only need `microstructure.execution.Side` |

## Dependency diagram

```mermaid
flowchart TB
    subgraph L0["Layer 0 - primitives"]
        util["util (1)"]
        sim["sim (3)"]
        persist["persist (1)"]
        data["data (9)"]
    end
    subgraph L1["Layer 1 - models"]
        pricing["pricing (22)"]
        rates["rates (8)"]
        credit["credit (4)"]
        commodities["commodities (1)"]
        markets["markets (2)"]
        risk["risk (15)"]
        volatility["volatility (9)"]
        indicators["indicators (2)"]
        ml["ml (6)"]
    end
    subgraph L2["Layer 2 - desk engines"]
        micro["microstructure (24)"]
        fx["fx (12)"]
        crb["crb (9)"]
        backtest["backtest (35)"]
        screener["screener (7)"]
        execution["execution (21)"]
    end
    subgraph L3["Layer 3 - research and plumbing"]
        alpha["alpha (12)"]
        dsl["dsl (3)"]
        trading["trading (3)"]
        fix["fix (5)"]
        sbe["sbe (3)"]
        marketdata["marketdata (3)"]
        regulatory["regulatory (3)"]
    end

    sim --> util
    pricing --> util
    pricing --> sim
    rates --> util
    credit --> rates
    risk --> util
    volatility --> util
    indicators --> util
    ml --> util

    micro --> util
    micro --> data
    micro --> persist
    fx --> util
    fx --> rates
    fx --> persist
    crb --> util
    crb --> pricing
    crb --> risk
    backtest --> util
    backtest --> data
    backtest --> indicators
    backtest --> micro
    screener --> util
    screener --> data
    screener --> indicators
    execution --> util
    execution --> persist
    execution --> micro

    alpha --> util
    alpha --> data
    alpha --> risk
    alpha --> backtest
    alpha --> micro
    dsl --> data
    dsl --> backtest
    trading --> micro
    fix --> micro
    sbe --> micro
    marketdata --> micro
    regulatory --> micro
```

`commodities` and `markets` import nothing from the rest of the library
(pure closed forms over NumPy), which is why they have no outgoing
arrows.

## Package map

Module counts in parentheses; the class named is the one to read first.

**Layer 0**
- `util` (1) -- `math_utils`: 27 public numerical primitives (mean/variance two-pass, `percentile`, `cholesky` with the Java pivot rule, Acklam `norm_inv`, A&S `norm_cdf`, Lanczos `log_gamma`, Lentz `regularized_incomplete_beta`, `t_cdf`, `pair_sort`). The `not (x > 0)` NaN-rejecting gate idiom lives here and is copied everywhere.
- `sim` (3) -- `_java_random.SplittableRandom64` (bit-exact `java.util.SplittableRandom`), `MonteCarloSimulator`, `SimulationResult`.
- `persist` (1) -- `Checkpoint`: byte-identical to the Java file format; `LpScorecard`, `EwmaCovariance`, `VolumeCurve` snapshot into it.
- `data` (9) -- `Bar`, `BarSeries` (structure-of-arrays, read-only views, strictly increasing timestamps), `csv_bar_loader`, `series_aligner`, `corporate_actions`, `point_in_time_universe`, QFLT tick reader/writer.

**Layer 1**
- `pricing` (22) -- `BlackScholes` (+`Greeks`, `implied_vol`), `Black76`, `HigherOrderGreeks`, `IncrementalGreeks`, `BinomialTree`, `DigitalOption`, `BarrierOption`, `TouchOption`, `VannaVolga`, `ExchangeOption` (Margrabe/Kirk), `QuantoOption`, `VarianceSwap`, `AsianOption`, `StructuredNotes`, `Autocallable`, `Heston`, `SabrModel`, `VolSurface`, `ForwardCurve`, `DividendSchedule`, `FairValueEngine`, `TriangularArbitrage`.
- `rates` (8) -- `YieldCurve`, `BondPricer`, `NelsonSiegel`, `Svensson`, `SwapPricer`, short-rate models (Vasicek/CIR/Hull-White), `KeyRateDurations`, `rates_options` (swaptions/caps).
- `credit` (4) -- `CreditCurve`, `CdsPricer`, `credit_spreads` (Z-spread), `CvaApproximator`.
- `commodities` (1) -- `CommodityCurve`.
- `markets` (2) -- `IndexConstruction`, `PrivateMarketAnalytics`.
- `risk` (15) -- `risk_metrics`, `var_engine`, `component_var`, `covariance_shrinkage` (Ledoit-Wolf), `extreme_value_theory`, `stress_tester`, `frtb_es`, `pnl_attribution`, `var_backtest`, `pca`, `gaussian_copula`, `dependence`, `concentration_risk`, `counterparty_exposure_tracker`, `settlement_risk_analyzer`.
- `volatility` (9) -- `EwmaVolatility`, `Garch11`, `GjrGarch11`, `Egarch11`, `HarRv`, `volatility_index`, `range_volatility`, `information_criteria`, `volatility_decomposition`.
- `indicators` (2) -- `Indicators` (batch), `streaming_indicators`.
- `ml` (6) -- `GradientBoostedRegressor`, `regime_detector` (HMM `RegimeModel`), `anomaly_detector` (quote stuffing, price spikes), `VolatilityForecaster`, `MarketImpactPredictor`, `IntradayLiquidityForecaster`.

**Layer 2**
- `microstructure` (24) -- `execution` (`Side`, `Execution`: the shared vocabulary), `AlmgrenChriss`, `KylesLambda`, `OrnsteinUhlenbeck`, `VarianceRatio`, `LeadLagEstimator`, `TransactionCostAnalyzer`, `MarketImpactModel`, `VolumeCurve`/`VolatilityCurve`/`DayTypeProfiles`, `QueueModel`, `QueuePositionEstimator`, `FillProbabilityModel`, `TradeClassifier` (Lee-Ready), `FlowSignals`, `Vpin`, `HawkesIntensity`, `EwmaCovariance`, `AvellanedaStoikov`, `SpreadForecaster`, `HiddenLiquidityDetector`, `LiquidityMeasures`, `jump_robust_volatility`.
- `fx` (12) -- `CurrencyPair`, `BusinessCalendar`, `SwapPointsCurve`, `FxSwap`, `Ndf`, `FxVolSurface`, `FxTierBook`, `AggregatedBook`, `LpScorecard`, `LpRouter`, `SyntheticCross`, `FixingRisk`.
- `crb` (9) -- `FactorRegistry`, `CentralRiskBook`, `SkewedQuoter`, `InternalizationEngine`, `HedgeOptimizer`, `CrbHedgeUniverse`, `CrbAutoHedger`, `CrbRouter`, `CrbPnlLedger`.
- `backtest` (35) -- core: `Backtester`, `BacktestConfig`, `TradingStrategy`, `Signal`, `Trade`, `TradeCostModel`, `execution_models`, `ExecutionAwareBacktester`, `ParentOrder`, `TradeAnalytics`, `PerformanceAnalytics`, `DrawdownAnalytics`, `BenchmarkComparison`; `strategies/` (SMA/EMA cross, RSI, MACD, Bollinger); `validation/` (`PurgedKFold`, `OverfitProbability`, `BlockBootstrap`, `SharpeValidation`, `MonteCarloTradeShuffle`, `WalkForwardAnalyzer`, `GridSearchOptimizer`, `ParameterGrid`); `portfolio/` (`PositionSizing`, `PortfolioOptimizer`, `RiskParityOptimizer`, `BlackLitterman`, `ConstrainedPortfolioOptimizer`, `PortfolioBacktester`, `PortfolioStrategy`, `CrossSectionalMomentum`).
- `screener` (7) -- `StockScreener`, `StockSnapshot`, `Fundamentals`, filters, `RankingEngine`.
- `execution` (21) -- `twap_scheduler`, `vwap_scheduler`, `PovTracker`, `implementation_shortfall_scheduler` (Almgren-Chriss trajectory), `wmr_fixing_scheduler`, `HftSor`, `AdaptiveSor`, `Ucb1Selector`, `DarkPoolSimulator`, `BenchmarkExecutor`, `SpreadExecutionAlgo`, `FuturesRollAlgo`, `IcebergOrder`, `LiquiditySeekingAlgo`, `MidPegTracker`, `order_placement_policy`, `AntiGamingJitter`, `PortfolioExecutor`, `Slice`, `VenueQuote`, `VenueScorecard`.

**Layer 3**
- `alpha` (12) -- `AlphaContext`, `AlphaFactor` + `factors`, `SignalEvaluator`, `AlphaValidation`, `AlphaBacktester`, `PortfolioConstruction`, `AlphaReport`, `AlphaEnsemble`, `OnlineAlphaLearner`, `FamaMacBeth`, `CalendarAnomalies`.
- `dsl` (3) -- `StrategyBuilder`, `Rule`, `rules` (the no-lookahead builder).
- `trading` (3) -- `OrderThrottle`, `LastLookGate`, `PaperTradingGateway`.
- `fix` (5) -- `FixMessage`, `fix_parse`, `FixDecoder`, `FixOrderEncoder`, `FixExecReportView`.
- `sbe` (3) -- order/quote/trade flyweights (byte-exact).
- `marketdata` (3) -- `itch_codec` (`ItchView` + encoders), `L3BookBuilder`, `Nbbo`.
- `regulatory` (3) -- `BestExecutionAnalyzer`, `fix_analyzer` (`FixImpactReport`), `market_quality_metrics` (quoted/effective/realized spread, price impact).

## Data flow through a typical study

```mermaid
flowchart LR
    csv["CSV / tick file"] --> loader["data.csv_bar_loader\n(sort, dedupe, epoch-seconds heuristic)"]
    loader --> series["data.BarSeries\n(strict timestamps)"]
    series --> ind["indicators.Indicators"]
    ind --> strat["backtest.strategies.*"]
    strat --> bt["backtest.Backtester\n(fills at close, SL/TP intrabar)"]
    bt --> ta["backtest.TradeAnalytics /\nPerformanceAnalytics"]
    bt --> val["backtest.validation\n(PurgedKFold, CSCV, DSR)"]
    ta --> rm["risk.risk_metrics\n(VaR/ES, gated)"]
    series --> vol["volatility.Garch11 / EwmaVolatility"]
    vol --> ve["risk.var_engine\n(delta-normal / MC / delta-gamma / historical)"]
    ve --> frtb["risk.frtb_es\n(LH cascade, MAR33.6)"]
    ve --> vb["risk.var_backtest\n(Kupiec, Christoffersen, traffic light)"]
```

## Invariants the layering enforces

1. **Validation happens at the boundary.** Every public function in
   layers 0-1 validates its own inputs (finite, non-empty, sorted where
   order matters) and raises `ValueError`; layer 2-3 code may assume
   clean numbers once a layer-1 call has returned.
2. **`microstructure.execution.Side` is the only shared enum** across
   the wire-format, trading, execution and regulatory packages; no
   package defines a second buy/sell type.
3. **RNG discipline.** `sim._java_random` is the one bit-exact Java
   stream; `pricing.sabr_model` imports it rather than carrying a copy.
   Everything else uses `numpy.random.default_rng(seed)` and pins
   distribution properties, stated per docstring.
4. **`persist.Checkpoint` is a leaf.** Classes opt in (`LpScorecard`,
   `EwmaCovariance`, `VolumeCurve`); the checkpoint module never imports
   the classes it serialises.
5. **No threads, no sockets.** Nothing under `src/` spawns a thread or
   opens a network connection; the live layer is Java/C++ territory.
