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

## Phase 1 (current): the quant core

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
| `backtest` | Trade analytics, performance/drawdown analytics, benchmark comparison; `validation/` purged K-fold, CSCV overfit probability, block bootstrap, deflated Sharpe, Monte Carlo trade reshuffle; `portfolio/` position sizing, optimizers (MV/risk-parity/Black-Litterman/constrained) |

81 modules - 419 tests, all green.

Phase 2 (planned): the strategy engine, execution models and
microstructure analytics. Phase 3: the teaching docs adapted to Python.

## Install and test

```bash
python -m pip install -e .[dev]
python -m pytest
```

Requires Python >= 3.12 and NumPy >= 1.26.
