# quant-finance-py

Python/NumPy port of [Quant-Finance-Library](https://github.com/AshJha0/Quant-Finance-Library)
(the Java reference implementation, v1.17.0). NumPy is the only runtime
dependency -- no scipy, no pandas.

The port is algorithm-faithful: classes keep their Java names
(methods become snake_case), array sweeps are vectorized where safe,
but every scalar algorithm (Acklam norm_inv, Lanczos log_gamma, Lentz
incomplete beta, Cholesky pivot rules, GARCH grids, curve bootstraps)
is transcribed from the Java source so the Java test suite's
hand-pinned values transfer at the same tolerances.

**Validation contract.** Every public entry point in `util`, `pricing`
(Black-Scholes, Heston, SABR), `rates` (curve factories), `credit`,
`risk` (`risk_metrics`, `var_engine`, `var_backtest`, `frtb_es`, EVT,
shrinkage), `volatility` (GARCH/GJR/EGARCH fits) and `data`
(`BarSeries`) rejects NaN, infinity, empty input and out-of-range
confidence levels with `ValueError` -- never by returning 0, NaN, or a
solver bound. Solvers are bracket-checked and never return an endpoint:
an unattainable implied vol is `NaN`, an expired option or a NaN price
raises. Timestamps in a `BarSeries` must be strictly increasing. Where
a Java test pinned a seeded RNG stream, the port either reproduces the
stream bit-exactly (`sim/_java_random.py`, used by `MonteCarloSimulator`
and `SabrModel.calibrate`) or pins the distribution property instead and
says so in the docstring. See `docs/LEARN.md` Section 1 and the
[Real-world usage notes](#real-world-usage-notes) below.

Honesty note: this port is the *algorithms* lane. The Java (and
upcoming C++) siblings own the low-latency story -- Python is for
research, teaching, and cross-checking numbers, not the hot path.

## What's ported

| Package | Contents |
|---|---|
| `util` | `math_utils`: 27 public numerical primitives (mean/variance, percentile, Cholesky, Acklam `norm_inv`, A&S `norm_cdf`, Lanczos `log_gamma`, Lentz incomplete beta, `t_cdf`, ...) |
| `data` | `Bar` (validates high >= low, as the Java record does), `BarSeries` (strictly increasing timestamps), CSV loader (European-decimal regression pinned), series aligner, corporate actions, point-in-time universe, QFLT tick files |
| `pricing` | Black-Scholes (all greeks incl. the vol = 0 limits, implied vol), Black-76, higher-order greeks, binomial tree, digitals, barriers, touches, vanna-volga, Margrabe/Kirk, quanto, variance swap, Asian (Kemna-Vorst + Turnbull-Wakeman), structured notes, autocallable, Heston (little-trap form, pinned to Broadie-Kaya and Andersen reference prices), SABR, vol surface, forward curve, dividends, fair value, triangular arbitrage |
| `rates` | YieldCurve bootstrap (validated pillars), bond pricer (bracket-checked YTM), Nelson-Siegel, Svensson, swap pricer (annuity/par/DV01), Vasicek/CIR/Hull-White, key-rate durations, swaptions/caps |
| `credit` | Credit curve (hazard bootstrap), CDS pricer (incl. the sub-grid maturity gate), Z-spread, CVA |
| `commodities` | Futures curve: roll yield, implied carry, contango/backwardation |
| `markets` | Index construction (divisor continuity), private markets (IRR/TVPI/KS-PME/Geltner) |
| `risk` | Four VaR flavors + ES, component VaR (Euler), Ledoit-Wolf shrinkage, EVT, stress + reverse stress, FRTB ES (MAR33.5 cascade, MAR33.6 stress calibration), P&L attribution (KS), VaR backtests (Kupiec/Christoffersen/traffic light), PCA, Gaussian/t copulas, concentration, counterparty PFE, settlement risk |
| `volatility` | EWMA, GARCH/GJR/EGARCH, HAR-RV, model-free vol index, range estimators (Parkinson/GK/RS/Yang-Zhang), AIC/BIC, vol decomposition |
| `indicators` | Batch + streaming indicator set (SMA/EMA/RSI/MACD/Bollinger/ATR/VWAP...) |
| `backtest` | Full strategy engine (Backtester, 5 strategies, cost models, execution models with the cash-conservation contract, walk-forward with warm folds, grid search) + trade/performance/drawdown analytics; `validation/` purged K-fold, CSCV overfit probability, block bootstrap, deflated Sharpe, Monte Carlo trade reshuffle; `portfolio/` position sizing, optimizers, portfolio backtester, cross-sectional momentum |
| `fx` | Currency-pair conventions + calendars, swap points, FX swaps/NDFs, delta-quoted vol surface (premium-adjusted), tier book, LP scorecard/router |
| `crb` | Central risk book: factor-space netting, skewed quoting, internalization, L1 hedge optimizer, auto-hedger, router, P&L ledger |
| `alpha` | Research pipeline: context/factors/evaluator/validation/backtester/construction/report, ensemble, online learner, Fama-MacBeth, calendar anomalies |
| `microstructure` | Almgren-Chriss, Kyle's lambda, OU, variance ratio, TCA, impact models, seasonality curves (per-bucket seeding), queue/fill models, Lee-Ready, VPIN, Hawkes, EWMA covariance, Avellaneda-Stoikov (with a `degraded()` feed flag) |
| `execution` | TWAP/VWAP/POV/implementation-shortfall schedulers, smart + adaptive order routing, dark-pool MEQ simulation, BenchmarkExecutor, spread/roll/iceberg algos, PortfolioExecutor |
| `trading` | Order throttle, last-look gate, paper trading gateway |
| `fix` / `marketdata` / `sbe` | FIX codec + framer + encoder/view; ITCH codec, L3 book builder (exact queue position), NBBO; SBE byte-exact flyweights |
| `persist` | `Checkpoint` -- file format transcribed from Java's `Checkpoint` (round-trip tested in Python; no Java-written golden file is tracked in this repo), wired into LpScorecard, EwmaCovariance and VolumeCurve |
| `screener` / `ml` / `sim` / `dsl` / `regulatory` | Screener; GBDT, HMM regime detector, anomaly detection; Monte Carlo with a bit-exact port of Java's `SplittableRandom`; the no-lookahead strategy builder; regulatory reports |

223 modules - 1027 tests, all green.

**Cross-port verified**: an identical probe battery runs against Java,
C++, and Python (156 labels) and diffs to zero mismatches -- see
`tools/crossport/` and `tests/test_crossport_compare.py`, which pins
that the documented invocation works.

### Phases

| Phase | Scope | Status |
|---|---|---|
| 1-3 | `util`, `data`, `pricing`, `rates`, `credit`, `commodities`, `markets`, `risk`, `volatility`, `indicators`, `backtest`, `fx`, `crb`, `alpha`, `microstructure` | done |
| 4 | `fix/`, `marketdata/`, `sbe/`, `persist/` | done |
| 5 | `execution/`, `trading/`, `screener/`, `ml/`, `sim/`, `dsl/`, `regulatory/`, the data loaders | done |
| -- | The threaded live layer (FIX session threads, market-data bus, WebSocket feeds, dashboards) | deliberately not ported -- Java/C++ territory |

**New to the codebase?** Start with [docs/LEARN.md](docs/LEARN.md) --
a guided tour with run-verified snippets and five worked experiments
(including re-running the cross-port probe). Then
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (package map and dependency
diagram) and [docs/COOKBOOK.md](docs/COOKBOOK.md) (runnable recipes,
every one executed against this checkout). The Java repo's teaching
material (its LEARN.md exercises, COOKBOOK.md recipes and DIAGRAMS.md,
counts as of the v1.17.0 tag) transfers 1:1 -- only syntax differs.
Documentation site: <https://ashjha0.github.io/quant-finance-py/> -- see
[docs/GITHUB_PAGES.md](docs/GITHUB_PAGES.md) for how it is published.

## Install and test

```bash
python3 -m pip install -e .[dev]
python3 -m pytest -q
```

Requires Python >= 3.11 and NumPy >= 1.26 (the suite is verified on
3.11; nothing in the code needs a 3.12 feature). Without installing,
`PYTHONPATH=src python3 ...` works from the repo root, and `pytest` picks
up `src/` through `pyproject.toml`.

## Real-world usage notes

**Units, stated once.** Rates, carry/dividend yields and volatilities
are annualized decimals (`0.05`, not `5`); time is in years as a float;
theta is per year (divide by 365 for per-calendar-day); vega and rho are
per 1.00 change (divide by 100 for per-point). Returns are simple
periodic returns unless a function says `log_returns`. VaR and ES are
POSITIVE loss numbers -- fractions in `risk_metrics`, currency units in
`var_engine`. Timestamps are epoch milliseconds (`int64`); the CSV
loader promotes epoch-seconds files and parses ISO dates. There are no
day-count or holiday calendars outside `fx` -- `pricing`/`rates`/`credit`
take year fractions you have already computed.

**Validation policy.** Inputs are gated, not repaired: NaN, infinity,
empty series, unsorted/duplicate timestamps, sub-year or unsorted curve
pillars, confidence levels outside `(0, 1)` (or `(0.5, 1)` for
`var_engine`), and expired options passed to `implied_vol` all raise
`ValueError` with the offending value in the message. Nothing is
clamped before it is checked, so a bad input can never surface as
"zero risk". The exceptions, both documented in their docstrings:
`AvellanedaStoikov` treats a non-finite variance/horizon as 0 and sets
`degraded()`; `KylesLambda` skips non-finite samples. `ValueError` is the
only exception type the library's gates raise (`RuntimeError` for the
Java `IllegalStateException` cases, e.g. an empty `BarSeries.Builder`).

**Conventions to know before trusting a number.**
`risk_metrics.conditional_var` averages returns at or below an
interpolated percentile; `var_engine.tail` uses the `ceil(c*n)-1` order
statistic -- at small `n` they differ by one order statistic. `frtb_es.
stress_calibrated_es` implements MAR33.6 as `ES_RS x max(1, ES_FC/ES_RC)`
with arguments in that order. CSCV counts an exact-median rank as
overfit (Bailey et al. count strictly below). The FRTB module is
"styled after MAR33, not certified".

**Out of scope.** Multi-curve (OIS) discounting, day-count and stub
conventions, ISDA standard-model CDS, Student-t GARCH likelihoods and
standard errors, Heston calibration and the QE simulation scheme,
shifted SABR, Harrell-Davis quantiles, short-side or multi-instrument
backtesting, exact-binomial Kupiec at small `n`, NMRF/DRC/PLA capital
add-ons, and every threaded/live component. Those are desk-specific or
belong to the Java/C++ siblings; the docstrings say so where it
matters.

## References

Grouped by package. Each entry is the primary source the module
docstring names or the formula it transcribes.

**Numerical primitives (`util/math_utils`, `sim`)**
- Abramowitz, M. & Stegun, I. A. (1964). *Handbook of Mathematical Functions*, S26.2.17 (normal CDF). NBS Applied Mathematics Series 55.
- Acklam, P. J. (2003). "An algorithm for computing the inverse normal cumulative distribution function" (web note; the coefficients used in `norm_inv`).
- Lanczos, C. (1964). "A precision approximation of the gamma function." *J. SIAM Numer. Anal. B* 1(1), 86-96.
- Lentz, W. J. (1976). "Generating Bessel functions in Mie scattering calculations using continued fractions." *Applied Optics* 15(3), 668-671 (modified Lentz, incomplete beta); Press et al. (2007) *Numerical Recipes*, 3rd ed., S6.4.
- Marsaglia, G. & Bray, T. A. (1964). "A convenient method for generating normal variables." *SIAM Review* 6(3), 260-264 (polar method in `sim`).
- Steele, G. L., Lea, D. & Flood, C. H. (2014). "Fast splittable pseudorandom number generators." *OOPSLA 2014*, ACM SIGPLAN Notices 49(10), 453-472 (SplitMix64 / `java.util.SplittableRandom`).

**Option pricing (`pricing`)**
- Black, F. & Scholes, M. (1973). "The pricing of options and corporate liabilities." *J. Political Economy* 81(3), 637-654.
- Merton, R. C. (1973). "Theory of rational option pricing." *Bell J. Economics and Management Science* 4(1), 141-183 (continuous dividend yield; down-and-out barrier).
- Black, F. (1976). "The pricing of commodity contracts." *J. Financial Economics* 3(1-2), 167-179 (Black-76, caps/swaptions).
- Garman, M. B. & Kohlhagen, S. W. (1983). "Foreign currency option values." *J. International Money and Finance* 2(3), 231-237.
- Cox, J. C., Ross, S. A. & Rubinstein, M. (1979). "Option pricing: a simplified approach." *J. Financial Economics* 7(3), 229-263 (binomial tree).
- Reiner, E. & Rubinstein, M. (1991). "Breaking down the barriers." *Risk* 4(8), 28-35; Rubinstein, M. & Reiner, E. (1991). "Unscrambling the binary code." *Risk* 4(9), 75-83 (barriers, digitals, touches).
- Kemna, A. G. Z. & Vorst, A. C. F. (1990). "A pricing method for options based on average asset values." *J. Banking & Finance* 14(1), 113-129.
- Turnbull, S. M. & Wakeman, L. M. (1991). "A quick algorithm for pricing European average options." *J. Financial and Quantitative Analysis* 26(3), 377-389.
- Margrabe, W. (1978). "The value of an option to exchange one asset for another." *J. Finance* 33(1), 177-186.
- Kirk, E. (1995). "Correlation in the energy markets." In *Managing Energy Price Risk*, Risk Publications, 71-78.
- Castagna, A. & Mercurio, F. (2007). "The vanna-volga method for implied volatilities." *Risk* 20(1), 106-111.
- Heston, S. L. (1993). "A closed-form solution for options with stochastic volatility with applications to bond and currency options." *Review of Financial Studies* 6(2), 327-343.
- Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). "The little Heston trap." *Wilmott Magazine* Jan 2007, 83-92.
- Lord, R. & Kahl, C. (2010). "Complex logarithms in Heston-like models." *Mathematical Finance* 20(4), 671-694.
- Broadie, M. & Kaya, O. (2006). "Exact simulation of stochastic volatility and other affine jump diffusion processes." *Operations Research* 54(2), 217-231; Andersen, L. (2008). "Simple and efficient simulation of the Heston stochastic volatility model." *J. Computational Finance* 11(3), 1-42 (the reference prices pinned in `tests/test_pricing_market_risk.py`).
- Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E. (2002). "Managing smile risk." *Wilmott Magazine* Sept 2002, 84-108 (SABR).
- Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999). "More than you ever wanted to know about volatility swaps." Goldman Sachs Quantitative Strategies Research Notes (variance swap).
- Cboe (2019). *Cboe Volatility Index (VIX) White Paper* (model-free index in `volatility/volatility_index`).
- Reiner, E. (1992). "Quanto mechanics." *Risk* 5(3), 59-63 (quanto).

**Rates (`rates`)**
- Nelson, C. R. & Siegel, A. F. (1987). "Parsimonious modeling of yield curves." *J. Business* 60(4), 473-489.
- Svensson, L. E. O. (1994). "Estimating and interpreting forward interest rates: Sweden 1992-1994." NBER Working Paper 4871.
- Vasicek, O. (1977). "An equilibrium characterization of the term structure." *J. Financial Economics* 5(2), 177-188.
- Cox, J. C., Ingersoll, J. E. & Ross, S. A. (1985). "A theory of the term structure of interest rates." *Econometrica* 53(2), 385-407.
- Hull, J. & White, A. (1990). "Pricing interest-rate-derivative securities." *Review of Financial Studies* 3(4), 573-592.
- Ho, T. S. Y. (1992). "Key rate durations: measures of interest rate risks." *J. Fixed Income* 2(2), 29-44.
- Litterman, R. & Scheinkman, J. (1991). "Common factors affecting bond returns." *J. Fixed Income* 1(1), 54-61 (PCA of curves).

**Credit (`credit`)**
- O'Kane, D. & Turnbull, S. (2003). "Valuation of credit default swaps." Lehman Brothers Quantitative Credit Research Quarterly, Q1/Q2 2003 (piecewise-constant hazard bootstrap, accrual-on-default).
- Jarrow, R. A. & Turnbull, S. M. (1995). "Pricing derivatives on financial securities subject to credit risk." *J. Finance* 50(1), 53-85 (reduced-form intensity).
- Gregory, J. (2015). *The xVA Challenge*, 3rd ed., Wiley (CVA approximation, PFE).

**Risk (`risk`)**
- Jorion, P. (2006). *Value at Risk*, 3rd ed., McGraw-Hill (VaR flavours; component VaR).
- Garman, M. B. (1996). "Improving on VaR." *Risk* 9(5), 61-63 (marginal/component VaR); Tasche, D. (2000). "Risk contributions and performance measurement." Working paper, TU Muenchen (Euler allocation).
- Zangari, P. (1996). "A VaR methodology for portfolios that include options." *RiskMetrics Monitor* Q1 1996 (delta-gamma / Cornish-Fisher); Cornish, E. A. & Fisher, R. A. (1938). "Moments and cumulants in the specification of distributions." *Revue de l'Institut International de Statistique* 5(4), 307-320.
- Acerbi, C. & Tasche, D. (2002). "On the coherence of expected shortfall." *J. Banking & Finance* 26(7), 1487-1503.
- Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *J. Multivariate Analysis* 88(2), 365-411.
- Pickands, J. (1975). "Statistical inference using extreme order statistics." *Annals of Statistics* 3(1), 119-131; Balkema, A. A. & de Haan, L. (1974). "Residual life time at great age." *Annals of Probability* 2(5), 792-804 (POT/GPD).
- Hosking, J. R. M. & Wallis, J. R. (1987). "Parameter and quantile estimation for the generalized Pareto distribution." *Technometrics* 29(3), 339-349 (PWM fit).
- McNeil, A. J. & Frey, R. (2000). "Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach." *J. Empirical Finance* 7(3-4), 271-300 (GPD VaR/ES formulas).
- Kupiec, P. H. (1995). "Techniques for verifying the accuracy of risk measurement models." *J. Derivatives* 3(2), 73-84.
- Christoffersen, P. F. (1998). "Evaluating interval forecasts." *International Economic Review* 39(4), 841-862.
- Basel Committee on Banking Supervision (1996). *Supervisory framework for the use of "backtesting" in conjunction with the internal models approach* (traffic-light zones).
- Basel Committee on Banking Supervision (2019). *Minimum capital requirements for market risk* (d457), MAR33 (FRTB IMA ES, liquidity horizons, stress calibration, P&L attribution); the stress-calibration formula first appears in BCBS (2013) *Fundamental review of the trading book: a revised market risk framework* (bcbs265).
- Li, D. X. (2000). "On default correlation: a copula function approach." *J. Fixed Income* 9(4), 43-54 (Gaussian copula); Demarta, S. & McNeil, A. J. (2005). "The t copula and related copulas." *International Statistical Review* 73(1), 111-129.
- Kendall, M. G. (1938). "A new measure of rank correlation." *Biometrika* 30(1-2), 81-93; Spearman, C. (1904). "The proof and measurement of association between two things." *Am. J. Psychology* 15, 72-101 (rank dependence).
- Herfindahl, O. C. (1950) / Hirschman, A. O. (1945). *National Power and the Structure of Foreign Trade* (HHI concentration).

**Volatility (`volatility`)**
- Engle, R. F. (1982). "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." *Econometrica* 50(4), 987-1007; Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." *J. Econometrics* 31(3), 307-327.
- Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). "On the relation between the expected value and the volatility of the nominal excess return on stocks." *J. Finance* 48(5), 1779-1801 (GJR).
- Nelson, D. B. (1991). "Conditional heteroskedasticity in asset returns: a new approach." *Econometrica* 59(2), 347-370 (EGARCH).
- Engle, R. F. & Mezrich, J. (1996). "GARCH for groups." *Risk* 9(8), 36-40 (variance targeting).
- J.P. Morgan/Reuters (1996). *RiskMetrics -- Technical Document*, 4th ed. (EWMA lambda = 0.94).
- Corsi, F. (2009). "A simple approximate long-memory model of realized volatility." *J. Financial Econometrics* 7(2), 174-196 (HAR-RV).
- Parkinson, M. (1980). "The extreme value method for estimating the variance of the rate of return." *J. Business* 53(1), 61-65.
- Garman, M. B. & Klass, M. J. (1980). "On the estimation of security price volatilities from historical data." *J. Business* 53(1), 67-78.
- Rogers, L. C. G. & Satchell, S. E. (1991). "Estimating variance from high, low and closing prices." *Annals of Applied Probability* 1(4), 504-512.
- Yang, D. & Zhang, Q. (2000). "Drift-independent volatility estimation based on high, low, open, and close prices." *J. Business* 73(3), 477-491.
- Akaike, H. (1974). "A new look at the statistical model identification." *IEEE Trans. Automatic Control* 19(6), 716-723; Schwarz, G. (1978). "Estimating the dimension of a model." *Annals of Statistics* 6(2), 461-464.
- Barndorff-Nielsen, O. E. & Shephard, N. (2004). "Power and bipower variation with stochastic volatility and jumps." *J. Financial Econometrics* 2(1), 1-37 (`microstructure/jump_robust_volatility`).

**Microstructure / execution (`microstructure`, `execution`)**
- Almgren, R. & Chriss, N. (2000). "Optimal execution of portfolio transactions." *J. Risk* 3(2), 5-39.
- Almgren, R., Thum, C., Hauptmann, E. & Li, H. (2005). "Direct estimation of equity market impact." *Risk* 18(7), 58-62 (square-root law calibration); Gatheral, J. (2010). "No-dynamic-arbitrage and market impact." *Quantitative Finance* 10(7), 749-759.
- Kyle, A. S. (1985). "Continuous auctions and insider trading." *Econometrica* 53(6), 1315-1335.
- Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit order book." *Quantitative Finance* 8(3), 217-224.
- Easley, D., Lopez de Prado, M. & O'Hara, M. (2012). "Flow toxicity and liquidity in a high-frequency world." *Review of Financial Studies* 25(5), 1457-1493 (VPIN).
- Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually exciting point processes." *Biometrika* 58(1), 83-90; Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015). "Hawkes processes in finance." *Market Microstructure and Liquidity* 1(1), 1550005.
- Lee, C. M. C. & Ready, M. J. (1991). "Inferring trade direction from intraday data." *J. Finance* 46(2), 733-746.
- Lo, A. W. & MacKinlay, A. C. (1988). "Stock market prices do not follow random walks: evidence from a simple specification test." *Review of Financial Studies* 1(1), 41-66 (variance ratio).
- Uhlenbeck, G. E. & Ornstein, L. S. (1930). "On the theory of the Brownian motion." *Physical Review* 36(5), 823-841.
- Perold, A. F. (1988). "The implementation shortfall: paper versus reality." *J. Portfolio Management* 14(3), 4-9 (IS benchmark/TCA).
- Kissell, R. & Glantz, M. (2003). *Optimal Trading Strategies*, AMACOM (TWAP/VWAP/POV scheduling, volume curves).

**Backtest validation & portfolio (`backtest`, `alpha`)**
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Wiley, ch. 7 (purged K-fold, embargo), ch. 11-12.
- Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2017). "The probability of backtest overfitting." *J. Computational Finance* 20(4), 39-69 (CSCV; SSRN 2014).
- Bailey, D. H. & Lopez de Prado, M. (2012). "The Sharpe ratio efficient frontier." *J. Risk* 15(2), 3-44 (PSR, MinTRL).
- Bailey, D. H. & Lopez de Prado, M. (2014). "The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality." *J. Portfolio Management* 40(5), 94-107.
- Politis, D. N. & Romano, J. P. (1994). "The stationary bootstrap." *J. American Statistical Association* 89(428), 1303-1313; Kuensch, H. R. (1989). "The jackknife and the bootstrap for general stationary observations." *Annals of Statistics* 17(3), 1217-1241.
- Kelly, J. L. (1956). "A new interpretation of information rate." *Bell System Technical Journal* 35(4), 917-926.
- Markowitz, H. (1952). "Portfolio selection." *J. Finance* 7(1), 77-91.
- Black, F. & Litterman, R. (1992). "Global portfolio optimization." *Financial Analysts Journal* 48(5), 28-43.
- Maillard, S., Roncalli, T. & Teiletche, J. (2010). "The properties of equally weighted risk contribution portfolios." *J. Portfolio Management* 36(4), 60-70 (risk parity).
- Jegadeesh, N. & Titman, S. (1993). "Returns to buying winners and selling losers." *J. Finance* 48(1), 65-91 (cross-sectional momentum).
- Fama, E. F. & MacBeth, J. D. (1973). "Risk, return, and equilibrium: empirical tests." *J. Political Economy* 81(3), 607-636.
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems* (RSI, ATR, ADX, Parabolic SAR); Bollinger, J. (2001). *Bollinger on Bollinger Bands*; Appel, G. (2005). *Technical Analysis: Power Tools for Active Investors* (MACD).

**Markets (`markets`)**
- Kaplan, S. N. & Schoar, A. (2005). "Private equity performance: returns, persistence, and capital flows." *J. Finance* 60(4), 1791-1823 (KS-PME).
- Geltner, D. (1991). "Smoothing in appraisal-based returns." *J. Real Estate Finance and Economics* 4(3), 327-345 (desmoothing).

**ML (`ml`)**
- Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition." *Proc. IEEE* 77(2), 257-286 (Baum-Welch/Viterbi regime detector).
- Friedman, J. H. (2001). "Greedy function approximation: a gradient boosting machine." *Annals of Statistics* 29(5), 1189-1232.
- Rousseeuw, P. J. & Croux, C. (1993). "Alternatives to the median absolute deviation." *J. American Statistical Association* 88(424), 1273-1283 (robust z-score anomaly detection).

## License

MIT License, Copyright (c) 2026 Ashish Jha — see [LICENSE](LICENSE).
