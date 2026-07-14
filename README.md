# quant-finance-py

Python/NumPy port of [Quant-Finance-Library](https://github.com/AshJha0/Quant-Finance-Library)
(the Java reference implementation, v1.17.0). NumPy is the only runtime
dependency -- no scipy, no pandas.

The port is algorithm-faithful: classes keep their Java names
(methods become snake_case), array sweeps are vectorized where safe,
but every scalar algorithm (Acklam norm_inv, Lanczos log_gamma, Lentz
incomplete beta, Cholesky pivot rules, GARCH grids, curve bootstraps)
is transcribed from the Java source so the Java test suite's
hand-pinned values transfer at the same tolerances. NaN-rejecting
gates raise ValueError; solvers are bracket-checked and never return
an endpoint. Where a Java test pinned a seeded RNG stream, the port
pins the distribution property instead and says so in the docstring.

Honesty note: this port is the *algorithms* lane. The Java (and
upcoming C++) siblings own the low-latency story -- Python is for
research, teaching, and cross-checking numbers, not the hot path.

## What's ported (Phases 1-3)

| Package | Contents |
|---|---|
| `util` | math_utils: 28 numerical primitives |
| `data` | Bar (validates high >= low, as the Java record does) |
| `pricing` | Black-Scholes (all greeks, implied vol), Black-76, higher-order greeks, binomial tree, digitals, barriers, touches, vanna-volga, Margrabe/Kirk, quanto, variance swap, Asian (Kemna-Vorst + Turnbull-Wakeman), structured notes, autocallable, Heston, SABR, vol surface, forward curve, dividends, fair value, triangular arbitrage |
| `rates` | YieldCurve bootstrap, bond pricer (bracket-checked YTM), Nelson-Siegel, Svensson, swap pricer (annuity/par/DV01), Vasicek/CIR/Hull-White, key-rate durations, swaptions/caps |
| `credit` | Credit curve (hazard bootstrap), CDS pricer (incl. the sub-grid maturity gate), Z-spread, CVA |
| `commodities` | Futures curve: roll yield, implied carry, contango/backwardation |
| `markets` | Index construction (divisor continuity), private markets (IRR/TVPI/KS-PME/Geltner) |
| `risk` | Four VaR flavors + ES, component VaR (Euler), Ledoit-Wolf shrinkage, EVT, stress + reverse stress, FRTB ES, P&L attribution (KS), VaR backtests, PCA, Gaussian/t copulas, concentration, counterparty PFE, settlement risk |
| `volatility` | EWMA, GARCH/GJR/EGARCH, HAR-RV, model-free vol index, range estimators (Parkinson/GK/RS/Yang-Zhang), AIC/BIC, vol decomposition |
| `indicators` | Batch + streaming indicator set (SMA/EMA/RSI/MACD/Bollinger/ATR/VWAP...) |
| `backtest` | Full strategy engine (Backtester, 5 strategies, cost models, execution models with the cash-conservation contract, walk-forward with warm folds, grid search) + trade/performance/drawdown analytics; `validation/` purged K-fold, CSCV overfit probability, block bootstrap, deflated Sharpe, Monte Carlo trade reshuffle; `portfolio/` position sizing, optimizers, portfolio backtester, cross-sectional momentum |
| `fx` | Currency-pair conventions + calendars, swap points, FX swaps/NDFs, delta-quoted vol surface (premium-adjusted), tier book, LP scorecard/router |
| `crb` | Central risk book: factor-space netting, skewed quoting, internalization, L1 hedge optimizer, auto-hedger, router, P&L ledger |
| `alpha` | Research pipeline: context/factors/evaluator/validation/backtester/construction/report, ensemble, online learner, Fama-MacBeth, calendar anomalies |
| `microstructure` | Almgren-Chriss, Kyle's lambda, OU, variance ratio, TCA, impact models, seasonality curves (per-bucket seeding), queue/fill models, Lee-Ready, VPIN, Hawkes, EWMA covariance, Avellaneda-Stoikov |

223 modules - 926 tests, all green.

Phase 5 (done): `execution/` -- TWAP/VWAP/POV/implementation-shortfall
schedulers, smart + adaptive order routing, dark-pool MEQ simulation,
BenchmarkExecutor, spread/roll/iceberg algos, PortfolioExecutor;
`trading/` -- throttle, last-look gate, paper gateway; `screener/`,
`ml/` (GBDT, HMM regime detector, anomaly detection), `sim/` (Monte
Carlo with a bit-exact port of Java's SplittableRandom), `dsl/`
(the no-lookahead strategy builder), `regulatory/`, and the data
loaders (CSV with the European-decimal regression pinned, corporate
actions, point-in-time universe).

**Cross-port verified**: an identical probe battery runs against Java,
C++, and Python (138 labels) and diffs to zero mismatches -- see
`tools/crossport/`.

Phase 4 (done): `fix/` codec + framer + encoder/view, `marketdata/`
ITCH + L3 book (exact queue position) + NBBO, `sbe/` byte-exact
flyweights, QFLT tick files, and `persist/Checkpoint` whose file
format is byte-identical to Java's (a checkpoint written by one
language restores in the other -- verified against a Java-written
golden file), wired into LpScorecard, EwmaCovariance and VolumeCurve.

**New to the codebase?** Start with [docs/LEARN.md](docs/LEARN.md) --
a guided tour with run-verified snippets, five worked experiments
(including re-running the cross-port probe), and pointers into the
Java repo's teaching material (1000 exercises, 300 recipes,
100 diagrams -- the concepts transfer 1:1, only syntax differs).

Remaining (deliberate): the threaded live layer (FIX session threads,
market-data bus, WebSocket feeds, dashboards) -- Java/C++ territory;
Python owns research, teaching, and cross-checking.

## Install and test

```bash
python -m pip install -e .[dev]
python -m pytest
```

Requires Python >= 3.12 and NumPy >= 1.26.
