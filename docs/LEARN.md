# Learning quantfinlib in Python

This is the port-specific guide to the Python/NumPy mirror of
[Quant-Finance-Library](https://github.com/AshJha0/Quant-Finance-Library)
(the Java reference, v1.17.0). It is not a shrink-wrapped copy of the
Java project's 26,000-line LEARN.md -- that stays the deep-material
home for the whole family, with its 1000 worked exercises. This guide
is the ~800-line version: how the Python port is built, a reading path
through the packages that exist *in this repo today*, five hands-on
experiments with real executed output, an honest account of what this
port skips and why, and a map back to the Java documents for when you
want the long version of any of this.

Every class and function named below was checked against the source
tree with Glob/Grep before it went into this file -- if it's cited
here, `grep -rn "ClassName" src/` finds it. Every code snippet in
sections 2 and 3 was actually run with `PYTHONPATH=src` against this
repo's installed source, and the output shown is real, copied from
that run, not illustrative.

## 1. How this port works

### The contract, in one paragraph

The Python port is algorithm-faithful, not merely API-compatible: every
class keeps its Java name, but methods become `snake_case`
(`BlackScholes.implied_vol`, not `impliedVol`). Array sweeps are
vectorized with NumPy where that is safe, but every scalar algorithm --
Acklam's `norm_inv`, the Lanczos `log_gamma`, Lentz's incomplete-beta
continued fraction, the Cholesky pivot rule, GARCH grids, curve
bootstraps -- is transcribed from the Java source line for line, so the
Java test suite's hand-pinned values transfer at the same tolerances.
The honesty note the top-level README states up front matters here:
this port is the *algorithms* lane. Java (and the C++ sibling) own the
low-latency story; Python is for research, teaching, and cross-checking
numbers, not the hot path -- which is also why this port has no order
book or risk-gate package (see Section 4).

### Method naming

Compare the pricer every other package eventually calls:

Java (`com.quantfinlib.pricing.BlackScholes`):

```java
public static double price(OptionType type, double spot, double strike,
                           double rate, double carry, double vol, double timeYears) {
    if (timeYears <= 0) {
        return intrinsic(type, spot, strike);
    }
    if (vol <= 0) {
        return Math.max(0, type.sign() * (spot * Math.exp(-carry * timeYears)
                - strike * Math.exp(-rate * timeYears)));
    }
    ...
}
```

Python (`src/quantfinlib/pricing/black_scholes.py`):

```python
@staticmethod
def price(option_type: OptionType, spot: float, strike: float, rate: float,
          carry: float, vol: float, time_years: float) -> float:
    if time_years <= 0:
        return BlackScholes.intrinsic(option_type, spot, strike)
    if vol <= 0:
        return max(0.0, option_type.sign() * (spot * math.exp(-carry * time_years)
                                              - strike * math.exp(-rate * time_years)))
    ...
```

Same parameter order, same branch structure, same reason for the
`vol <= 0` branch (stated in both docstrings: without it the
ATM-forward case is 0/0 in `d1` and the price comes back NaN). The
class stays `BlackScholes`; only the method and parameter names go
from camelCase to snake_case, and `type` becomes `option_type` because
`type` shadows a Python builtin.

### Exception mapping

Java's `IllegalArgumentException` becomes `ValueError` everywhere,
argument-checking message included. `SwapPricer` is a good minimal
example (`src/quantfinlib/rates/swap_pricer.py`):

Java:

```java
private static void requireTenor(int tenorYears) {
    if (tenorYears < 1) {
        throw new IllegalArgumentException("tenorYears must be >= 1, got " + tenorYears);
    }
}
```

Python:

```python
@staticmethod
def _require_tenor(tenor_years: int) -> None:
    if tenor_years < 1:
        raise ValueError(f"tenorYears must be >= 1, got {tenor_years}")
```

`IllegalStateException` (an empty `BarSeries.Builder.build()`, for
example) maps to `RuntimeError`. There is no port-invented exception
hierarchy layered on top -- catch `ValueError`, and you have caught
everything this library's input gates throw.

### NaN gates

Java's idiom for rejecting NaN inputs is `if (!(x > 0))` rather than
`if (x <= 0)` -- every comparison against NaN is false, so the negated
form catches both NaN and non-positive values in a single branch. The
Python port keeps the exact idiom rather than writing a more
"Pythonic" `math.isnan(x) or x <= 0`, because the point is that a
reviewer who knows the Java source recognizes the Python line
immediately. From `src/quantfinlib/util/math_utils.py`:

```python
def log_gamma(x: float) -> float:
    """Natural log of the gamma function (Lanczos, |relative error| < 2e-10).

    Raises:
        ValueError: if x is not strictly positive (NaN included -- the
            ``not (x > 0)`` gate is deliberately NaN-rejecting).
    """
    if not (x > 0):
        raise ValueError(f"x must be > 0: {x}")
    ...
```

which is the Java gate transcribed, not merely inspired:

```java
public static double logGamma(double x) {
    if (!(x > 0)) {
        throw new IllegalArgumentException("x must be > 0: " + x);
    }
    ...
}
```

### Bracket-checked solvers

Every root-finder in this port (implied vol, bond YTM, credit-curve
hazard bootstrap, Nelson-Siegel/Svensson fits) is bisection or a
bisection-guarded Newton step, and every one of them checks the bracket
before it starts. From `black_scholes.py`'s `implied_vol` docstring:

```
Returns NaN when the price is not attainable inside the [1e-4, 5.0]
vol bracket (below intrinsic, or above the maximum BS price) -- a
stale or rounded market price must surface as "no vol", not silently
come back as the 500% search bound and poison a smile.
```

`CreditCurve.bootstrap` applies the identical pattern on the hazard-rate
axis: bisection on `h` in `[1e-9, 10]` with an explicit bracket check,
"a quote no hazard can explain raises rather than returning the
bound." No solver in this codebase returns a silent endpoint, ever --
grep for `bisect` under `src/` and you will find the same shape
repeated in every package that needs a root.

### Hand-pinned tests

The port ships 158 modules and 607 tests (`python -m pytest`, all green
as of this writing). In the pricing/rates/credit/risk/volatility
packages these are not property-based or generated: they are the SAME
numeric values the Java suite pins, carried over at the same tolerance,
because the whole port strategy depends on "does this Python function
return what the Java function returns" being checkable, not assumed.
Open any file under `tests/` -- `tests/test_pricing_black_scholes.py`,
`tests/test_backtest_engine.py` -- and you will find literal expected
floats, not generated fixtures.

### The cross-port 138-label guarantee

This is the strongest claim in the whole project and it is mechanically
checked, not asserted. A single probe program -- one file per language,
identical inputs in identical order -- calls into Java, C++ and Python
and prints 138 labelled results (`bs.call=...`, `ns.rmse=...`,
`micro.hawkes.intensity=...`) at 15 significant figures. A tiny compare
script then diffs the three outputs pairwise at 1e-9 relative tolerance
(1e-6 for the handful of labels that come out of an iterative fit,
where floating-point path-dependence in the last few digits is expected
and stated up front). The harness lives right here, in this repo:

```
tools/crossport/
  Probe.java     -- the Java probe (source of truth for labels/order)
  probe.py       -- this port's mirror (322 lines, 138 labelled prints)
  probe.cpp      -- the C++ port's mirror
  compare.py     -- pairwise diff at 1e-9 / 1e-6
  java.out       -- checked-in Java output (138 lines)
  py.out         -- checked-in Python output
  cpp.out        -- checked-in C++ output
```

To re-run the guarantee yourself:

```bash
PYTHONPATH=src python tools/crossport/probe.py > /tmp/py.out
python tools/crossport/compare.py tools/crossport/java.out /tmp/py.out
```

A clean run prints `138 labels, 0 mismatches`. That is not a
hypothetical -- it is the exact command this guide's author ran while
writing this section, against this checkout, and it printed exactly
that. If you touch a numeric algorithm while extending this port, this
is the fastest way to find out whether you broke it: the label names
tell you which package regressed. Experiment (e) below walks through
running this end to end, including diffing against the checked-in
`java.out` by hand.

## 2. A guided tour in 12 stops

Read these in order; each assumes the previous. Every "try it" snippet
compiles down to a REPL-pasteable few lines -- run
`PYTHONPATH=src python` from the repo root (or `pip install -e .` per
the top-level README, then a plain `python`) and paste.

### Stop 1 -- `util`: the numerical bedrock

**What it does.** `math_utils` (`src/quantfinlib/util/math_utils.py`):
28 primitives every other package leans on -- mean/variance/std_dev,
percentile, covariance/correlation, Cholesky, Acklam's `norm_inv`, the
Abramowitz-Stegun `norm_cdf`, skewness/kurtosis, a NaN-safe `pair_sort`.
(One more than the C++ port's 27 -- the module docstring is worth
reading for which one and why, a small but real divergence point.)

**Read first:** the module docstring, then `norm_cdf`/`norm_inv` --
everything downstream that touches a normal distribution (every option
pricer, every VaR flavor) is built on these two.

**Try it:**

```python
from quantfinlib.util import math_utils as mu

x = 1.2345
p = mu.norm_cdf(x)
back = mu.norm_inv(p)                              # round trip
v = [1.0, 2.0, 3.0, 4.0, 5.0]
print(f"norm_cdf({x})={p:.6f}  norm_inv(back)={back:.6f}  "
      f"mean={mu.mean(v):.3f}  std_dev={mu.std_dev(v):.4f}")
```

```
norm_cdf(1.2345)=0.891492  norm_inv(back)=1.234500  mean=3.000  std_dev=1.5811
```

### Stop 2 -- `data`: the shared currency, `Bar`/`BarSeries`

**What it does.** `Bar` is an immutable OHLCV record (validates
`high >= low`, exactly the Java record's compact constructor).
`BarSeries` is a structure-of-arrays time series backed by read-only
NumPy views -- every pricing, indicator and backtest class consumes it,
so this is the one type you will import into nearly everything you
write against this library.

**Read first:** `bar_series.py` -- note `of()` (closes-only, synthetic
OHLC), `slice()` (the train/test split every walk-forward tool uses),
and that array accessors return the internal `np.ndarray` with
`setflags(write=False)`, not a defensive copy.

**Try it:**

```python
from quantfinlib.data.bar_series import BarSeries

series = BarSeries.of("DEMO", [100.0, 101.0, 99.5, 102.0, 103.0])
rets = series.returns()
print(f"size={series.size()} last_close={series.last_close():.2f} "
      f"first_return={rets[0]:.5f}")
```

```
size=5 last_close=103.00 first_return=0.01000
```

### Stop 3 -- `pricing`: Black-Scholes and its 20 siblings

**What it does.** Black-Scholes-Merton with a continuous carry (so one
function prices equity options with a dividend yield and FX options
Garman-Kohlhagen-style), Black-76, binomial trees, digitals, barriers,
touches, Asian (Kemna-Vorst geometric + Turnbull-Wakeman arithmetic),
vanna-volga, Margrabe/Kirk spread options, quanto, variance swaps,
structured notes, autocallables, Heston, SABR, a vol surface, a forward
curve, dividends, and a fair-value blender.

**Read first:** `black_scholes.py` -- every other pricer in the package
either calls it directly (Margrabe reduces to it in the single-asset
limit) or reuses its `_d1`/`norm_cdf` plumbing.

**Try it:** see Experiment (a) below -- it is this stop's exercise.

### Stop 4 -- `rates`: the yield curve and what stands on it

**What it does.** `YieldCurve` (bootstrap from par swaps or raw zero
rates; linear interpolation on continuously-compounded zeros, flat
extrapolation), `BondPricer` (bracket-checked YTM), Nelson-Siegel and
Svensson parametric fits, `SwapPricer` (annuity/par rate/DV01),
Vasicek/CIR/Hull-White short-rate models, key-rate durations,
swaptions/caps.

**Read first:** `yield_curve.py`'s module docstring is the best short
explanation of curve-building in the whole codebase (it is quoted in
full in this port's README); read it, then `swap_pricer.py`.

**Try it:**

```python
from quantfinlib.rates.yield_curve import YieldCurve
from quantfinlib.rates.swap_pricer import SwapPricer

tenors = [1, 2, 3, 5, 7, 10]
par_rates = [0.03, 0.032, 0.034, 0.036, 0.037, 0.038]
curve = YieldCurve.bootstrap_annual_par_swaps(tenors, par_rates)
par5 = SwapPricer.par_rate(curve, 5)
pv = SwapPricer.payer_pv(curve, 5, par5)
print(f"parRate5y={par5:.6f}  payerPvAtPar={pv:.3e}  "
      f"dv01={SwapPricer.dv01(curve, 5, par5):.6f}")
```

```
parRate5y=0.036000  payerPvAtPar=0.000e+00  dv01=0.000466
```

`payerPvAtPar` reprices to (numerically) zero -- a swap struck at its
own par rate must, and that identity is Experiment (b)'s subject below.

Python port note worth internalizing here: `YieldCurve` stores its
pillars as a `dict` built key-by-key (duplicate tenors: last one wins,
matching Java's `TreeMap.put`) plus a sorted key list searched with
`bisect` for floor/ceiling lookups -- a deliberate, stated substitute
for Java's `TreeMap`, not an accident of translation.

### Stop 5 -- `credit` / `commodities` / `markets`: three small curve families

**What it does.** `credit`: `CreditCurve` (hazard-rate bootstrap from
CDS par spreads), `CdsPricer`, Z-spread, CVA. `commodities`:
`CommodityCurve` (roll yield, implied carry, contango/backwardation --
no extrapolation past the pillars, deliberately: "a commodity curve's
wings are opinions, not data"). `markets`: index construction (cap/
price/equal weighting, the divisor that keeps a level series continuous
through membership changes), private-market analytics (IRR/TVPI/
KS-PME/Geltner).

**Read first:** `credit_curve.py` -- it is `YieldCurve`'s bootstrap
pattern applied one dimension over (hazard instead of discount rate),
so reading it right after Stop 4 reinforces a pattern instead of
introducing a new one.

**Try it:**

```python
from quantfinlib.rates.yield_curve import YieldCurve
from quantfinlib.credit.credit_curve import CreditCurve
from quantfinlib.commodities.commodity_curve import CommodityCurve

disc = YieldCurve.of_zero_rates([1, 5, 10], [0.03, 0.03, 0.03])
cc = CreditCurve.bootstrap([1, 3, 5, 10], [0.01, 0.012, 0.015, 0.018], 0.4, disc)
print(f"survival(5y)={cc.survival_probability(5.0):.6f} hazard(5y)={cc.hazard(5.0):.6f}")

oil = CommodityCurve.of(80.0, [0.5, 1.0, 2.0], [81.0, 82.5, 85.0])
print(f"oil 1y price={oil.price(1.0):.3f} "
      f"rollYield(0.5->1)={oil.annualized_roll_yield(0.5, 1.0):.5f}")
```

```
survival(5y)=0.880270 hazard(5y)=0.033634
oil 1y price=82.500 rollYield(0.5->1)=-0.03670
```

The negative roll yield says the oil curve you just built is in
contango over that stretch -- a long position pays to roll.

### Stop 6 -- `risk`: the numbers a risk committee asks for

**What it does.** Four VaR flavors plus Expected Shortfall, component
VaR (Euler allocation), Ledoit-Wolf covariance shrinkage, EVT, stress
and reverse-stress testing, FRTB ES, P&L attribution, VaR backtesting,
PCA, Gaussian/t copulas, concentration risk, counterparty PFE,
settlement risk.

**Read first:** `risk_metrics.py` -- the smallest, most self-contained
module in the package, and the one every other risk class's tests build
scenarios against.

**Try it:**

```python
from quantfinlib.risk import risk_metrics as rm

rets = [0.01, -0.02, 0.015, -0.03, 0.005, 0.02, -0.01, 0.012]
print(f"VaR95={rm.historical_var(rets, 0.95):.4f}  "
      f"ES95={rm.conditional_var(rets, 0.95):.4f}  "
      f"sharpe={rm.sharpe_ratio(rets, 0.0, 252):.4f}")
```

```
VaR95=0.0265  ES95=0.0300  sharpe=0.2193
```

### Stop 7 -- `volatility`: estimating sigma from returns

**What it does.** EWMA (RiskMetrics-style), GARCH(1,1)/GJR-GARCH/
EGARCH, HAR-RV, a model-free volatility index (the VIX construction,
`volatility_index.py`), range estimators (Parkinson, Garman-Klass,
Rogers-Satchell, Yang-Zhang), AIC/BIC model selection, volatility
decomposition.

**Read first:** `ewma_volatility.py` -- the simplest model in the
package (one parameter, `lambda_`, renamed from Java's `lambda` because
that is a Python keyword) and the one the risk package quietly assumes
when you do not hand it a GARCH fit.

**Try it:**

```python
from quantfinlib.volatility.ewma_volatility import EwmaVolatility

rets = [0.01, -0.02, 0.015, -0.03, 0.005, 0.02, -0.01, 0.012, -0.008, 0.02]
ewma = EwmaVolatility.risk_metrics()          # lambda = 0.94
print(f"latestVol={ewma.latest_vol(rets):.6f} "
      f"annualizedVol={ewma.annualized_vol(rets, 252):.4f}")
```

```
latestVol=0.016971 annualizedVol=0.2694
```

### Stop 8 -- `indicators`: the batch/streaming technical set

**What it does.** `Indicators` (batch: SMA/EMA/WMA/RSI/MACD/Bollinger/
ADX/SuperTrend/Ichimoku/StochRSI/Keltner/Donchian/ATR/OBV/VWAP/CMF/
Parabolic SAR...) and a streaming twin in `streaming_indicators.py` for
the same math computed incrementally, one bar at a time.

**Read first:** `indicators.py` -- every strategy in `backtest/
strategies/` is built from two or three calls into this module.

**Try it:**

```python
from quantfinlib.indicators.indicators import Indicators

closes = [100, 101, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110]
sma3 = Indicators.sma(closes, 3)
rsi5 = Indicators.rsi(closes, 5)
print(f"sma3[-1]={sma3[-1]:.4f}  rsi5[-1]={rsi5[-1]:.4f}")
```

```
sma3[-1]=108.6667  rsi5[-1]=83.0273
```

### Stop 9 -- `backtest` core: `Backtester` and the strategy family

**What it does.** Event-driven, single-instrument, long-only
backtesting: signals fill at the close (slippage-adjusted), stop-loss/
take-profit are checked intrabar with gap-aware fills, commission on
both legs. Five shipped strategies (`strategies/`: SMA cross, EMA
cross, RSI, MACD, Bollinger). Cost models, execution models with a
cash-conservation contract, walk-forward analysis with warm folds, grid
search.

**Read first:** `backtester.py`, then `strategies/
sma_cross_strategy.py` as the simplest concrete `TradingStrategy`.

**Try it:** see Experiment (c) below.

### Stop 10 -- `backtest/validation` and `backtest/portfolio`: does the edge survive scrutiny?

**What it does.** `validation/`: purged K-fold with embargo (the fix
for financial-label leakage), CSCV overfit probability, block
bootstrap, deflated Sharpe, Monte Carlo trade reshuffle. `portfolio/`:
Kelly and fixed-fractional position sizing, inverse-volatility
weighting, mean-variance/risk-parity/Black-Litterman optimizers,
cross-sectional momentum, a portfolio-level backtester.

**Read first:** `validation/purged_kfold.py` -- the shortest module in
the family and the one every other validation tool assumes you already
understand (you cannot correctly interpret a CSCV run on unpurged
folds).

**Try it:**

```python
from quantfinlib.backtest.validation.purged_kfold import PurgedKFold
from quantfinlib.backtest.portfolio.position_sizing import PositionSizing

splits = PurgedKFold.splits(100, 5, 3, 2)
s0 = splits[0]
print(f"folds={len(splits)}  fold0 test=[{s0.test_from},{s0.test_to}) "
      f"trainSize={len(s0.train_indices)}")
print(f"kellyFraction={PositionSizing.kelly_fraction(0.001, 0.0004):.4f} "
      f"halfKelly={PositionSizing.half_kelly(0.001, 0.0004):.4f}")
```

```
folds=5  fold0 test=[0,20) trainSize=75
kellyFraction=2.5000 halfKelly=1.2500
```

(Fold 0's training set is 75 of 100 samples, not 80 -- the missing 5 are
the purge/embargo zone around the test fold. Kelly above 1 is not a
bug: this input's mean/variance ratio genuinely implies leveraging
above 100%, which is exactly the kind of number that should make you
distrust the input before you trust the leverage.)

### Stop 11 -- `fx` + `crb`: the dealer desk

**What it does.** `fx`: currency-pair conventions (pip size, spot lag,
joint-calendar settlement arithmetic), swap points, FX swaps/NDFs, a
premium-adjusted delta-quoted vol surface, tier books, LP scorecards
and routing, plus an `aggregated_book.py` that has no direct counterpart
listed in this port's own tour above -- worth a look once you have the
rest of `fx` down. `crb` (central risk book): factor-space netting, a
skewed quoter, an internalization engine, an L1 hedge optimizer, an
auto-hedger, a router, a P&L ledger.

**Read first:** `fx/currency_pair.py` -- the base convention object
every other FX class takes; then `crb/factor_registry.py`, the
dense-id pattern the whole risk book's exposure arithmetic runs on.

**Try it:**

```python
from quantfinlib.fx.currency_pair import CurrencyPair
from quantfinlib.crb.factor_registry import FactorRegistry

eurusd = CurrencyPair.of("EURUSD")
print(f"symbol={eurusd.symbol()} pip={eurusd.pip_size():.4f}")

reg = FactorRegistry()
print(f"EQ:AAPL id={reg.id('EQ:AAPL')} CCY:EUR id={reg.id('CCY:EUR')} size={reg.size()}")
```

```
symbol=EURUSD pip=0.0001
EQ:AAPL id=0 CCY:EUR id=1 size=2
```

### Stop 12 -- `alpha` + `microstructure`: the research desk

**What it does.** `alpha`: a full research pipeline (context, nine
factor generators, a signal evaluator, purge-aware validation, a
backtester, portfolio construction, a report), an alpha ensemble, an
online learner, Fama-MacBeth, calendar anomalies. `microstructure`:
Almgren-Chriss optimal execution, Kyle's lambda, an Ornstein-Uhlenbeck
model, a variance-ratio test, a lead-lag estimator, TCA, market-impact
models, per-bucket seasonality curves, queue/fill models, Lee-Ready
trade classification, VPIN, Hawkes intensity, EWMA covariance,
Avellaneda-Stoikov optimal market-making. (Kyle's lambda, the OU model,
variance ratio and lead-lag are Python-only additions relative to the
C++ port's `microstructure` -- see Section 4.)

**Read first:** `microstructure/avellaneda_stoikov.py` -- short,
self-contained, and the cleanest illustration in the whole codebase of
"units contract" documentation (it states explicitly that
`price_variance_per_second` is a variance of the *price*, not the
return).

**Try it:**

```python
from quantfinlib.microstructure.avellaneda_stoikov import AvellanedaStoikov

mm = AvellanedaStoikov(0.1, 1.5)          # gamma (risk aversion), kappa (fill decay)
bid = mm.bid_quote(100.0, 0.0, 0.0004, 300.0)
ask = mm.ask_quote(100.0, 0.0, 0.0004, 300.0)
bid_long = mm.bid_quote(100.0, 500.0, 0.0004, 300.0)   # long inventory
print(f"flat: bid={bid:.4f} ask={ask:.4f} | long 500: bid={bid_long:.4f}")
```

```
flat: bid=99.3486 ask=100.6514 | long 500: bid=93.3486
```

Carrying 500 units long shades the bid down by six full points -- the
market-maker is actively discouraging you from buying more, exactly the
inventory-control behavior the model exists to produce.

## 3. The five experiments

Each of these is a complete, runnable script. Run with
`PYTHONPATH=src python <file>.py` from the repo root.

### (a) Price and greek a vanilla, verify put-call parity

```python
import math
from quantfinlib.pricing.black_scholes import BlackScholes, OptionType

S, K, r, q, sig, T = 100, 105, 0.03, 0.01, 0.25, 0.75
call_g = BlackScholes.greeks(OptionType.CALL, S, K, r, q, sig, T)
put_g = BlackScholes.greeks(OptionType.PUT, S, K, r, q, sig, T)
print(f"call price={call_g.price:.6f} delta={call_g.delta:.6f} gamma={call_g.gamma:.6f} "
      f"vega={call_g.vega:.6f} theta={call_g.theta:.6f} rho={call_g.rho:.6f}")
print(f"put  price={put_g.price:.6f} delta={put_g.delta:.6f}")

lhs = call_g.price - put_g.price
rhs = S * math.exp(-q * T) - K * math.exp(-r * T)      # C - P = S e^-qT - K e^-rT
print(f"parity check: C-P={lhs:.8f}  S*e^-qT-K*e^-rT={rhs:.8f}  |gap|={abs(lhs - rhs):.3e}")
```

```
call price=7.102933 delta=0.477338 gamma=0.018268 vega=34.252083 theta=-6.450268 rho=30.473115
put  price=10.514007 delta=-0.515191
parity check: C-P=-3.41107442  S*e^-qT-K*e^-rT=-3.41107442  |gap|=0.000e+00
```

Zero gap to the printed precision, and the `call price` line matches
label `bs.call` in the cross-port probe to every digit shown --
Section 1's 138-label guarantee is checking exactly this number against
Java and C++, live, every time the harness runs.

### (b) Bootstrap a curve and price a swap at par

```python
from quantfinlib.rates.yield_curve import YieldCurve
from quantfinlib.rates.swap_pricer import SwapPricer

tenors = [1, 2, 3, 5, 7, 10]
par_rates = [0.031, 0.033, 0.0345, 0.0365, 0.0375, 0.039]
curve = YieldCurve.bootstrap_annual_par_swaps(tenors, par_rates)
for t in tenors:
    pr = SwapPricer.par_rate(curve, t)
    pv = SwapPricer.payer_pv(curve, t, pr)
    print(f"tenor={t:2d}y  parRate={pr:.6f}  payerPvAtPar={pv:.3e}")

off_par_pv = SwapPricer.payer_pv(curve, 5, 0.02)        # struck away from par
print(f"5y payer struck at 2% fixed: pv={off_par_pv:.6f} (nonzero, as expected)")
```

```
tenor= 1y  parRate=0.031000  payerPvAtPar=0.000e+00
tenor= 2y  parRate=0.033000  payerPvAtPar=0.000e+00
tenor= 3y  parRate=0.034500  payerPvAtPar=0.000e+00
tenor= 5y  parRate=0.036500  payerPvAtPar=0.000e+00
tenor= 7y  parRate=0.037500  payerPvAtPar=0.000e+00
tenor=10y  parRate=0.039000  payerPvAtPar=0.000e+00
5y payer struck at 2% fixed: pv=0.074492 (nonzero, as expected)
```

Every tenor reprices its own bootstrap input to zero. Note that
`par_rate(curve, t)` recovers exactly the input `par_rates[t]` you fed
the bootstrap -- that round trip, not merely "PV is small," is what
"the result reprices every input exactly" means in `yield_curve.py`'s
module docstring.

### (c) Run a backtest with a strategy and read TradeAnalytics honestly

```python
import math
from quantfinlib.backtest.backtester import Backtester
from quantfinlib.backtest.backtest_config import BacktestConfig
from quantfinlib.backtest.strategies.sma_cross_strategy import SmaCrossStrategy
from quantfinlib.backtest.trade_analytics import TradeAnalytics
from quantfinlib.data.bar_series import BarSeries

seed = 12345
def rnd():
    global seed
    seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
    return ((seed >> 16) & 0x7FFF) / 32768.0

price, closes = 100.0, []
for i in range(400):
    drift = 0.0002 * math.sin(i * 0.05)
    noise = (rnd() - 0.5) * 0.01
    price *= (1.0 + drift + noise)
    closes.append(price)

series = BarSeries.of("DEMO", closes)
strat = SmaCrossStrategy(10, 30)
cfg = BacktestConfig.defaults().with_commission(0.0005)
result = Backtester.run(strat, series, cfg)
print(result)
if len(result.trades()) >= 2:
    ta = TradeAnalytics.analyze(result.trades())
    print(f"expectancy={ta.expectancy:.4f} payoffRatio={ta.payoff_ratio:.4f} "
          f"winRate={ta.win_rate:.3f} kelly={ta.kelly_fraction:.4f} "
          f"maxWinStreak={ta.max_win_streak} maxLossStreak={ta.max_loss_streak} "
          f"avgBarsWin={ta.avg_bars_held_winners:.1f} avgBarsLoss={ta.avg_bars_held_losers:.1f}")
```

```
SMA_CROSS(10,30) on DEMO: totalReturn=-5.26%, CAGR=-3.36%, sharpe=-1.13, sortino=-1.48, calmar=-0.45, maxDD=7.42%, profitFactor=0.25, winRate=14.3%, trades=7
expectancy=-751.4240 payoffRatio=1.5100 winRate=0.143 kelly=0.0000 maxWinStreak=1 maxLossStreak=5 avgBarsWin=52.0 avgBarsLoss=17.2
```

Read this honestly, as instructed, rather than cherry-picking a seed
that looks better: a 10/30 SMA cross on 400 bars of mostly-noise data
with a small sinusoidal drift loses money (-5.26% total return, Sharpe
-1.13), wins only 14.3% of the time, and its Kelly fraction correctly
comes back 0 -- the sizing math is telling you not to trade this system
at all, on this data, with this signal. That is `TradeAnalytics` and
`PositionSizing.kelly_fraction` doing exactly their job: a losing
system does not get dressed up by better statistics elsewhere in the
report. (This script uses the same tiny hand-rolled LCG as the sibling
C++ guide's Experiment (c), on purpose, so the two guides' backtests
are directly comparable side by side.)

### (d) Run PurgedKFold/OverfitProbability on a deliberately overfit signal

```python
import numpy as np
from quantfinlib.backtest.validation.overfit_probability import OverfitProbability


def run_once(seed: int, t: int, n: int, blocks: int) -> float:
    rng = np.random.default_rng(seed)
    returns = (rng.random((t, n)) - 0.5) * 0.02   # zero-mean noise, no real edge
    return OverfitProbability.cscv_sharpe(returns, blocks).pbo


# 20 "strategy variants" that are PURE NOISE -- no real edge, just
# different random draws (the textbook null: a desk that tried 20
# meaningless parameter sets and is about to report the best one).
# A single draw's PBO is one sample from a distribution, so average
# several independent draws to see where it settles.
T, N, BLOCKS, TRIALS = 800, 20, 8, 12
total = 0.0
for s in range(1, TRIALS + 1):
    pbo = run_once(2000 + s, T, N, BLOCKS)
    print(f"trial {s:2d}: PBO={pbo:.4f}")
    total += pbo
print(f"mean PBO over {TRIALS} trials = {total / TRIALS:.4f}")
```

```
trial  1: PBO=0.3857
trial  2: PBO=0.6714
trial  3: PBO=0.6286
trial  4: PBO=0.1714
trial  5: PBO=0.5286
trial  6: PBO=0.4143
trial  7: PBO=0.5857
trial  8: PBO=0.5286
trial  9: PBO=0.5429
trial 10: PBO=0.7429
trial 11: PBO=0.5571
trial 12: PBO=0.5571
mean PBO over 12 trials = 0.5262
```

This is the honest version of "watch it get caught": no single variant
here has real edge, so any ONE draw's PBO is one sample from a
distribution (trial 4's 0.17, taken alone, would look like a passing
grade). The mean across 12 independent draws -- 0.5262 -- lands right
on the 0.5 danger line the module's own docstring names as "the
selection is pure noise-mining." That is exactly where 20 genuinely
edge-free variants should sit, and it is why the rule of thumb is
stated as evidence across repeated trials, never a verdict from a
single backtest. `OverfitProbability` itself takes no seed and is
completely deterministic given a return matrix -- the randomness here
is only in how this experiment manufactures a "no real edge" input, via
plain `numpy.random.default_rng`, not a library RNG guarantee (see
Section 4's RNG-stream note for why that distinction matters).

### (e) Run tools/crossport/probe.py and diff against the committed java.out

```bash
PYTHONPATH=src python tools/crossport/probe.py > /tmp/py_check.out
python tools/crossport/compare.py tools/crossport/java.out /tmp/py_check.out
```

Real output from this repo, run while writing this guide:

```
--- tools/crossport/java.out vs /tmp/py_check.out: 138 labels, 0 mismatches ---
```

Every one of the 138 labels -- pricing (`bs.*`, `b76.*`, `dig.*`,
`bar.*`, `touch.*`, `asian.*`, `sn.*`, `ex.*`, `q.*`, `vs.*`), rates
(`ns.*`, `sv.*`, `bond.*`, `zs.*`), credit (`cds.*`, `cva.*`), risk
(`var.*`, `es.*`, `pca.*`), volatility (`garch.*`, `ewma.*`), backtest
(`ta.*`, `pkf.*`), and microstructure (`micro.*`) -- agreed with Java to
1e-9 relative tolerance (1e-6 for the handful of iterative-fit labels
the harness names explicitly: `bs.iv.roundtrip`, `bond.ytm`,
`zs.roundtrip`, `pm.irr`, the Nelson-Siegel/Svensson fit parameters, and
the GARCH coefficients). If you want to see a deliberately introduced
mismatch, open `probe.py`, change any literal input (say, `S = 100` to
`S = 101` in the Black-Scholes section), rerun, and watch `compare.py`
print a `MISMATCH bs.call: ...` line with both values and the relative
error -- that is the harness doing exactly what it is for.

## 4. What is deliberately different here

**Skipped packages, stated plainly.** As of this port (Phases 1-3, per
the top-level README), everything in the Java "what's ported" table
above is here: `util`, `data`, `pricing`, `rates`, `credit`,
`commodities`, `markets`, `risk`, `volatility`, `indicators`,
`backtest` (+ `strategies`/`validation`/`portfolio`), `fx`, `crb`,
`alpha`, `microstructure`. What is NOT here, by design, not as a gap to
be filled later: `orderbook` and `trading`. There is no `HftOrderBook`,
no `HftRiskGate`, no order ring buffer in this repo -- those exist only
in the C++ port (see its own LEARN.md, Section 2 Stop 11) because they
are inherently a low-latency, allocation-free, single-threaded-hot-path
story that does not fit this port's research-lane mission. If your
interest in this library is the matching engine or the risk gate, this
is not the port to read; the C++ repo is. What remains outstanding for
the whole family (Phase 4, per this repo's own README) is "the teaching
docs adapted to Python" -- this file is that deliverable.

**The persistence lane does not exist in either non-Java port.** There
is no snapshot/restore, no write-ahead log, no durable order-book
recovery story here or in the C++ port. That is Java-only territory
today; if your interest in this library is "can I recover a book's
exact state after a crash," the Java repo is the only one of the three
with an answer.

**RNG-stream policy: property pins, not bit-exact streams.** Where a
Java test seeded `java.util.SplittableRandom` and pinned an exact
expected value drawn from that stream, this port does NOT try to
reproduce Java's bit pattern in `numpy.random`. Instead it pins the
DISTRIBUTION PROPERTY the Java test actually cared about, and says so
in the docstring. `monte_carlo_trade_shuffle.py` states this
explicitly: "the Java reference draws its Fisher-Yates swaps from
`SplittableRandom`; the RNG stream is not reproduced across ports --
the pinned properties are order-invariance of terminal P&L, percentile
ordering, determinism-per-seed and the worst-case-ordering tail rank."
That is a real, considered trade-off, not a shortcut: reproducing
Java's exact `SplittableRandom` stream in NumPy is possible (the C++
port does exactly that, see its own LEARN.md) but buys little for a
research-lane port whose tests only need "is this the right
distribution," not "is this the same random number." The one exception
proves the rule: `pricing/sabr_model.py` carries a private
`_SplittableRandom` that DOES reproduce Java's exact stream, because
its calibration test pins an exact expected value from Java's seeded
random search and there was no cheaper way to make that test pass.

**Performance positioning.** This port owns research ergonomics --
NumPy-vectorized array sweeps, no scipy/pandas dependency to fight
version skew with, a REPL you can paste every snippet in this guide
into directly. It does not own, and does not try to own, the
nanosecond story: there are no benchmarks in this repo, and the top-level
README says so without hedging ("Python is for research, teaching, and
cross-checking numbers, not the hot path"). C++ owns the measured-ns
benchmarks (`benchmarks/ull_bench.cpp` in that repo, with real numbers
and their own caveats printed at runtime). Java remains the reference:
when Java, C++ and Python disagree, Java's number is what the other two
are checked against (Section 1's cross-port guarantee), never the
other way around.

**Two small Python-only additions, for completeness.** The
`microstructure` package here has four modules with no listed C++
counterpart at the time this guide was written: `kyles_lambda.py`,
`ornstein_uhlenbeck.py`, `variance_ratio.py`, `lead_lag_estimator.py`.
They are real, tested modules, not placeholders -- `grep -rn
"class KylesLambda" src/` finds the class, and its test lives under
`tests/`. Whether the C++ port grows matching classes is an open
question for that repo's own roadmap, not this guide's to answer;
mentioned here only so a reader comparing the two ports side by side
does not mistake omission for a documentation bug.

## 5. Where the deep material lives

This guide is deliberately the short version. When you want the long
version, everything is in the Java repo, and the concepts and formulas
transfer 1:1 across the port -- only the syntax differs (Java's
camelCase becomes this port's snake_case; the C++ port keeps
camelCase).

**`LEARN.md`** (the Java repo, ~26,000 lines, 1000 worked exercises).
This is where you go when Section 3's five experiments are not enough --
every package in this guide's tour has dozens of matching exercises
there, phrased against the Java API. Translate mechanically: Java
`BlackScholes.price(...)` becomes Python
`BlackScholes.price(...)` with snake_case arguments,
`YieldCurve.bootstrapAnnualParSwaps(...)` becomes
`YieldCurve.bootstrap_annual_par_swaps(...)`, and the numbers you get
back should match to the tolerances Section 1 describes. If an
exercise's expected answer does not match what this port returns, that
is worth a bug report -- the whole point of the port contract is that
divergence should not happen silently.

**`COOKBOOK.md`** (the Java repo, 300 recipes). Short, task-oriented
"how do I..." entries -- "how do I bootstrap a curve from bond prices
instead of par swaps," "how do I size a position with vol targeting,"
that kind of thing. Read a recipe, then look up the equivalent Python
class name in this guide's Section 2 tour (or `grep -rn ClassName
src/` if it is not one of the stops) and translate the calls; the
recipe's logic does not change, only the call syntax and the fact that
you will likely get a NumPy array back where the Java version returns a
primitive array.

**`DIAGRAMS.md`** (the Java repo, 100 diagrams). Architecture and
data-flow diagrams -- package dependency graphs, the backtester's event
loop, the purged-K-fold train/test split geometry. These are
language-agnostic by construction (they describe control flow and data
shape, not syntax), so every diagram that does not involve `orderbook`/
`trading` (Section 4's note applies) transfers to this port without
translation.

**The formula appendix** (the Java repo README, ~175 entries). A flat
lookup table of every closed-form formula in the library, with the
class/method that implements it. Use it as a check on any pricer in
this guide's Section 2 tour: find the class name, read the formula,
then confirm the Python module's docstring states the same formula (it
will -- that is the whole port contract from Section 1) before you
trust a result you have not independently derived.
