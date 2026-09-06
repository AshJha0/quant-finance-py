# Cookbook

Task-oriented recipes for `quantfinlib`. Every snippet below was executed
against this checkout (`PYTHONPATH=src python3 recipe.py` from the repo
root) and the output shown is what it printed -- the generator that
builds this file refuses to emit a recipe whose script fails. Numbers
that depend on `numpy.random.default_rng(seed)` are stable across NumPy
versions for a fixed seed; nothing here depends on wall-clock time.

26 recipes. Units and conventions: see README
"Real-world usage notes". Package layout: `docs/ARCHITECTURE.md`.

## Contents

**Pricing**
1. Price a vanilla option, read every Greek, round-trip the implied vol
2. Tell a stale quote from an expired option in the implied-vol solver
3. Greeks in the deterministic (vol = 0) and expired (T = 0) limits
4. Compare a Heston price with Black-Scholes and check the Feller condition
5. Fit a SABR smile to market quotes and interpolate a strike

**Rates and credit**
6. Bootstrap a zero curve from par swaps; discount factors, forwards, DV01
7. Price a bond, solve its yield, and get duration/convexity
8. Bootstrap a credit curve from CDS spreads and price a CDS

**Risk**
9. Historical, parametric and conditional VaR on a return series (and what a NaN does)
10. Portfolio VaR four ways from factor exposures
11. Allocate VaR to positions (Euler component VaR) and shrink a noisy covariance
12. Backtest a VaR model: Kupiec, Christoffersen, Basel traffic light
13. FRTB ES: liquidity-horizon cascade and MAR33.6 stress calibration
14. Tail risk beyond the sample: peaks-over-threshold GPD

**Volatility**
15. Fit GARCH(1,1), forecast variance, and compare with EWMA
16. Range-based volatility from OHLC bars

**Data and backtesting**
17. Load OHLCV from CSV into a BarSeries and compute indicators
18. Run a strategy backtest and read the metrics honestly
19. Purged K-fold splits and the probability of backtest overfitting (CSCV)
20. Deflate a Sharpe ratio for the number of trials
21. Size positions: Kelly, risk parity and max-Sharpe weights

**Microstructure and execution**
22. Schedule an execution: Almgren-Chriss trajectory vs TWAP and VWAP slices
23. Classify trades (Lee-Ready) and watch flow toxicity (VPIN)
24. Quote as a market maker with Avellaneda-Stoikov and detect a dead vol feed

**Simulation and persistence**
25. Monte Carlo a portfolio with the bit-exact Java RNG and checkpoint a running estimator

**Verification**
26. Re-run the cross-port guarantee

## Pricing

### 1. Price a vanilla option, read every Greek, round-trip the implied vol

`carry` is the dividend yield (equities) or the foreign rate (FX). Theta is
per year, vega/rho per 1.00 change. The implied vol must reproduce the
input vol to 1e-8 -- if it does not, something upstream is wrong.

```python
from quantfinlib.pricing import BlackScholes, OptionType

S, K, r, q, vol, T = 100.0, 105.0, 0.03, 0.01, 0.25, 0.75
g = BlackScholes.greeks(OptionType.CALL, S, K, r, q, vol, T)
print(f"price={g.price:.6f} delta={g.delta:.6f} gamma={g.gamma:.6f}")
print(f"vega/pt={g.vega / 100:.6f} theta/day={g.theta / 365:.6f} rho/bp={g.rho / 10_000:.6f}")
iv = BlackScholes.implied_vol(OptionType.CALL, g.price, S, K, r, q, T)
print(f"implied vol={iv:.10f}  |iv - vol|={abs(iv - vol):.2e}")
```

```
price=7.102933 delta=0.477338 gamma=0.018268
vega/pt=0.342521 theta/day=-0.017672 rho/bp=0.003047
implied vol=0.2500000000  |iv - vol|=7.22e-16
```

### 2. Tell a stale quote from an expired option in the implied-vol solver

An unattainable price (below intrinsic, above the no-arbitrage cap) comes
back as `NaN` -- "no vol". A NaN price or a zero time-to-expiry is an
input error and raises. Neither ever returns a search bound.

```python
import math
from quantfinlib.pricing import BlackScholes, OptionType

below_intrinsic = BlackScholes.implied_vol(OptionType.CALL, 0.5, 120, 100, 0.03, 0.0, 1.0)
print("below intrinsic ->", below_intrinsic)
for label, kwargs in (("NaN price", dict(price=math.nan, t=1.0)),
                      ("expired", dict(price=5.0, t=0.0))):
    try:
        BlackScholes.implied_vol(OptionType.CALL, kwargs["price"], 100, 100, 0.03, 0.0, kwargs["t"])
    except ValueError as e:
        print(f"{label} -> ValueError: {e}")
```

```
below intrinsic -> nan
NaN price -> ValueError: market_price must be finite and >= 0, got nan
expired -> ValueError: time_years must be > 0 for an implied vol
```

### 3. Greeks in the deterministic (vol = 0) and expired (T = 0) limits

Both limits are closed-form and consistent with `price`: at `vol = 0`
delta is the discounted step `e^{-qT} 1{F > K}`, gamma and vega vanish,
theta and rho are the carry terms only. Handy for sanity-checking a
surface's wings or an `IncrementalGreeks` ladder.

```python
from quantfinlib.pricing import BlackScholes, OptionType

for S in (90.0, 100.0, 110.0):
    g = BlackScholes.greeks(OptionType.CALL, S, 100.0, 0.03, 0.01, 0.0, 1.0)
    print(f"S={S:5.1f} vol=0: price={g.price:9.6f} delta={g.delta:.6f} gamma={g.gamma} "
          f"theta={g.theta:9.6f} rho={g.rho:10.6f}")
g = BlackScholes.greeks(OptionType.PUT, 90.0, 100.0, 0.03, 0.01, 0.2, 0.0)
print(f"T=0 put: price={g.price} delta={g.delta} (intrinsic, step delta)")
```

```
S= 90.0 vol=0: price= 0.000000 delta=0.000000 gamma=0.0 theta= 0.000000 rho=  0.000000
S=100.0 vol=0: price= 1.960430 delta=0.990050 gamma=0.0 theta=-1.921287 rho= 97.044553
S=110.0 vol=0: price=11.860928 delta=0.990050 gamma=0.0 theta=-1.822282 rho= 97.044553
T=0 put: price=10.0 delta=-1.0 (intrinsic, step delta)
```

### 4. Compare a Heston price with Black-Scholes and check the Feller condition

`HestonParams.feller()` tells you whether `2 kappa theta >= sigma_v^2`; the
semi-analytic `call` is fine either way (little-trap form), but the
Monte Carlo scheme is biased when Feller fails. Both reference prices
below are published values (Broadie-Kaya 2006; Andersen 2008 Case I).

```python
from quantfinlib.pricing import BlackScholes, Heston, HestonParams, OptionType

bk = HestonParams(kappa=6.21, theta=0.019, sigma_v=0.61, rho=-0.7, v0=0.010201)
print(f"Broadie-Kaya case: feller={bk.feller():.3f}  call={Heston.call(100, 100, 0.0319, 0.0, 1.0, bk):.4f} (paper: 6.8061)")
an = HestonParams(kappa=0.5, theta=0.04, sigma_v=1.0, rho=-0.9, v0=0.04)
print(f"Andersen case I:   feller={an.feller():.3f}  call={Heston.call(100, 100, 0.0, 0.0, 10.0, an):.4f} (paper: 13.0847)")
bs = BlackScholes.price(OptionType.CALL, 100, 100, 0.0, 0.0, 0.2, 10.0)
print(f"Black-Scholes at sqrt(v0)=20%: {bs:.4f}  (skew/vol-of-vol worth {bs - Heston.call(100, 100, 0.0, 0.0, 10.0, an):.4f})")
```

```
Broadie-Kaya case: feller=0.634  call=6.8061 (paper: 6.8061)
Andersen case I:   feller=0.040  call=13.0847 (paper: 13.0847)
Black-Scholes at sqrt(v0)=20%: 24.8170  (skew/vol-of-vol worth 11.7324)
```

### 5. Fit a SABR smile to market quotes and interpolate a strike

Beta is fixed by convention (0.5 for rates, 1.0 for FX/equities);
`calibrate` returns `(alpha, rho, nu)` plus the fit RMSE in vol points.
The calibration is a seeded random search using the bit-exact
`SplittableRandom` port, so the numbers match the Java library exactly.

```python
from quantfinlib.pricing import SabrModel

f, t, beta = 100.0, 1.0, 1.0
strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
market = [0.28, 0.24, 0.21, 0.20, 0.205]
p = SabrModel.calibrate(f, t, beta, strikes, market)
print(f"alpha={p.alpha:.5f} rho={p.rho:.4f} nu={p.nu:.4f} rmse={p.rmse * 100:.3f} vol pts")
for k in (85.0, 95.0, 105.0):
    print(f"  K={k:5.1f}  sigma={SabrModel.implied_vol(f, k, t, p.alpha, beta, p.rho, p.nu):.4f}")
```

```
alpha=0.19916 rho=-0.3735 nu=1.0860 rmse=0.048 vol pts
  K= 85.0  sigma=0.2589
  K= 95.0  sigma=0.2235
  K=105.0  sigma=0.2024
```

## Rates and credit

### 6. Bootstrap a zero curve from par swaps; discount factors, forwards, DV01

Pillars must be strictly ascending whole years >= 1 (the annual-leg
bootstrap has no meaning for 0.5y); the curve reprices every input par
rate exactly. `SwapPricer.dv01` is a parallel zero-curve bump.

```python
from quantfinlib.rates import YieldCurve, SwapPricer

curve = YieldCurve.bootstrap_annual_par_swaps([1, 2, 3, 5, 7, 10],
                                              [0.031, 0.033, 0.0345, 0.0365, 0.0375, 0.039])
for t in (1, 2, 5, 10):
    print(f"{t:2d}y zero={curve.zero_rate(t):.6f} df={curve.discount_factor(t):.6f} "
          f"par={SwapPricer.par_rate(curve, t):.6f}")
print(f"fwd 2y->5y = {curve.forward_rate(2, 5):.6f}")
print(f"5y payer swap at 3.65%: pv={SwapPricer.payer_pv(curve, 5, 0.0365):.2e}  dv01={SwapPricer.dv01(curve, 5, 0.0365):.6f}")
try:
    YieldCurve.bootstrap_annual_par_swaps([2, 1, 3], [0.03, 0.031, 0.032])
except ValueError as e:
    print("unsorted pillars ->", e)
```

```
 1y zero=0.030529 df=0.969932 par=0.031000
 2y zero=0.032499 df=0.937069 par=0.033000
 5y zero=0.036013 df=0.835215 par=0.036500
10y zero=0.038647 df=0.679453 par=0.039000
fwd 2y->5y = 0.038356
5y payer swap at 3.65%: pv=-3.13e-17  dv01=0.000466
unsorted pillars -> tenors must be strictly ascending, got 1.0 after 2.0
```

### 7. Price a bond, solve its yield, and get duration/convexity

`yield_to_maturity` is a bracket-checked bisection: a price no yield can
reproduce raises rather than returning the bracket edge.

```python
from quantfinlib.rates import BondPricer

face, coupon, freq, years = 100.0, 0.05, 2, 10.0
price = BondPricer.price_from_yield(face, coupon, freq, years, 0.045)
ytm = BondPricer.yield_to_maturity(price, face, coupon, freq, years)
print(f"price@4.5%={price:.6f}  ytm(price)={ytm:.8f}")
print(f"macaulay={BondPricer.macaulay_duration(face, coupon, freq, years, ytm):.4f}y "
      f"modified={BondPricer.modified_duration(face, coupon, freq, years, ytm):.4f} "
      f"convexity={BondPricer.convexity(face, coupon, freq, years, ytm):.4f} "
      f"dv01={BondPricer.dv01(face, coupon, freq, years, ytm):.6f}")
```

```
price@4.5%=103.990928  ytm(price)=0.04500000
macaulay=8.0356y modified=7.8587 convexity=74.5506 dv01=0.081724
```

### 8. Bootstrap a credit curve from CDS spreads and price a CDS

Hazard rates are piecewise-constant between integer-year pillars; the
premium leg accrues on default at the half-period rule (O'Kane-Turnbull).
The par spread recovered at a pillar equals the input spread.

```python
from quantfinlib.rates import YieldCurve
from quantfinlib.credit import CreditCurve, CdsPricer

disc = YieldCurve.of_zero_rates([1, 5, 10], [0.03, 0.03, 0.03])
cc = CreditCurve.bootstrap([1, 3, 5, 10], [0.0100, 0.0120, 0.0150, 0.0180], 0.4, disc)
for t in (1.0, 3.0, 5.0, 10.0):
    print(f"{t:4.1f}y survival={cc.survival_probability(t):.6f} hazard={cc.hazard(t):.6f}")
print(f"5y par spread={CdsPricer.par_spread(cc, disc, 5.0) * 1e4:.2f} bp (input 150)")
print(f"5y risky annuity={CdsPricer.risky_annuity(cc, disc, 5.0):.6f}")
print(f"upfront to BUY 5y protection at a 100bp running coupon: {CdsPricer.upfront(cc, disc, 0.0100, 5.0):.6f} per unit notional")
```

```
 1.0y survival=0.983471 hazard=0.016667
 3.0y survival=0.941521 hazard=0.021796
 5.0y survival=0.880270 hazard=0.033634
10.0y survival=0.732175 hazard=0.036842
5y par spread=150.00 bp (input 150)
5y risky annuity=4.389750
upfront to BUY 5y protection at a 100bp running coupon: 0.021949 per unit notional
```

## Risk

### 9. Historical, parametric and conditional VaR on a return series (and what a NaN does)

All three are POSITIVE loss fractions. A NaN anywhere in the series
raises -- it must never come back as "VaR = 0" -- and so does an empty
series or a confidence outside (0, 1).

```python
import math
import numpy as np
from quantfinlib.risk import risk_metrics as rm

rng = np.random.default_rng(1)
rets = rng.standard_t(4, 500) * 0.008
for c in (0.95, 0.99):
    print(f"c={c}: hist={rm.historical_var(rets, c):.5f} param={rm.parametric_var(rets, c):.5f} "
          f"ES={rm.conditional_var(rets, c):.5f}")
print(f"sharpe={rm.sharpe_ratio(rets, 0.0, 252):.4f} sortino={rm.sortino_ratio(rets, 0.0, 252):.4f} "
      f"maxDD(equity)={rm.max_drawdown(np.cumprod(1 + rets)):.4f}")
poisoned = rets.copy(); poisoned[100] = math.nan
try:
    rm.historical_var(poisoned, 0.99)
except ValueError as e:
    print("NaN in series ->", e)
```

```
c=0.95: hist=0.01732 param=0.01972 ES=0.02716
c=0.99: hist=0.03216 param=0.02777 ES=0.04476
sharpe=-0.3978 sortino=-0.5592 maxDD(equity)=0.3232
NaN in series -> returns must be finite (NaN/Inf found)
```

### 10. Portfolio VaR four ways from factor exposures

Exposures are currency P&L per unit factor move; covariance and factor
returns are per horizon; `gamma` is the n x n second-derivative matrix.
Delta-gamma tilts the quantile for a short-gamma book; historical replays
real rows. Confidence must be in (0.5, 1).

```python
import numpy as np
from quantfinlib.risk import var_engine as ve

exposures = [1_000_000.0, -400_000.0]                      # $ per 1.00 factor move
cov = [[0.0004, 0.00012], [0.00012, 0.0009]]               # 1-day factor covariance
gamma = [[-50_000_000.0, 0.0], [0.0, 0.0]]                 # short gamma in factor 1 (n x n)
c = 0.99
print(f"delta-normal VaR={ve.delta_normal_var(exposures, cov, c):,.0f}  ES={ve.delta_normal_es(exposures, cov, c):,.0f}")
mc = ve.monte_carlo_var(exposures, cov, c, scenarios=200_000, seed=7)
print(f"monte carlo  VaR={mc.var:,.0f}  ES={mc.expected_shortfall:,.0f}")
print(f"delta-gamma  VaR={ve.delta_gamma_var(exposures, gamma, cov, c):,.0f}  ES={ve.delta_gamma_es(exposures, gamma, cov, c):,.0f}")
hist = np.random.default_rng(3).multivariate_normal([0, 0], cov, size=1000)
h = ve.historical_var(exposures, hist, c)
print(f"historical   VaR={h.var:,.0f}  ES={h.expected_shortfall:,.0f}")
```

```
delta-normal VaR=49,240  ES=56,412
monte carlo  VaR=49,076  ES=56,164
delta-gamma  VaR=99,387  ES=120,241
historical   VaR=48,179  ES=54,866
```

### 11. Allocate VaR to positions (Euler component VaR) and shrink a noisy covariance

Component VaRs sum to the portfolio VaR by construction; a negative
component is a hedge. Ledoit-Wolf shrinks toward `mu * I` and reports the
shrinkage intensity it chose.

```python
import numpy as np
from quantfinlib.risk import component_var, covariance_shrinkage

rng = np.random.default_rng(11)
n_assets, n_obs = 5, 60                                   # T barely above N: sample cov is noisy
true_cov = np.full((n_assets, n_assets), 0.3) * 0.0004 + np.eye(n_assets) * 0.0007
returns = rng.multivariate_normal(np.zeros(n_assets), true_cov, size=n_obs)
lw = covariance_shrinkage.ledoit_wolf(returns)
print(f"shrinkage intensity={lw.intensity:.3f} target mu={lw.target:.6f}  "
      f"cond(sample)={np.linalg.cond(np.cov(returns.T)):.1f} cond(shrunk)={np.linalg.cond(lw.matrix):.1f}")
w = [0.3, 0.3, 0.2, 0.1, 0.1]
a = component_var.allocate(w, lw.matrix, 0.99)
print(f"portfolio VaR={a.portfolio_var:.5f}  sum(components)={a.components.sum():.5f}")
print("components:", np.round(a.components, 5))
```

```
shrinkage intensity=0.327 target mu=0.000690  cond(sample)=4.9 cond(shrunk)=2.8
portfolio VaR=0.03456  sum(components)=0.03456
components: [0.01264 0.01127 0.005   0.00356 0.0021 ]
```

### 12. Backtest a VaR model: Kupiec, Christoffersen, Basel traffic light

Feed realized returns and the VaR forecast per day (or a scalar). A NaN
return raises rather than silently counting as a non-exception. The
traffic light applies the 250-day / 99% zones.

```python
import numpy as np
from quantfinlib.risk import var_backtest, frtb_es

rng = np.random.default_rng(5)
rets = rng.standard_normal(250) * 0.01
for label, var in (("honest 99% VaR", 2.326 * 0.01), ("too-tight VaR", 1.5 * 0.01)):
    r = var_backtest.test(rets, var, 0.99)
    print(f"{label}: exceptions={r.exceptions} (expected {r.expected_exceptions:.1f}) "
          f"kupiec LR={r.kupiec_statistic:.3f} p={r.kupiec_p_value:.3f} "
          f"christoffersen p={r.independence_p_value:.3f} passes(5%)={r.passes(0.05)} "
          f"light={frtb_es.TrafficLight.of(r.exceptions).value}")
```

```
honest 99% VaR: exceptions=2 (expected 2.5) kupiec LR=0.108 p=0.742 christoffersen p=0.857 passes(5%)=True light=GREEN
too-tight VaR: exceptions=12 (expected 2.5) kupiec LR=19.016 p=0.000 christoffersen p=0.270 passes(5%)=False light=RED
```

### 13. FRTB ES: liquidity-horizon cascade and MAR33.6 stress calibration

`es975` is the tail mean at 97.5%; the cascade squares the horizon
increments; the stress calibration is `ES_RS x max(1, ES_FC / ES_RC)`
with the arguments in regulation order (full-current, reduced-current,
reduced-stressed).

```python
import numpy as np
from quantfinlib.risk import frtb_es

losses = np.random.default_rng(9).standard_normal(2000) * 1e6
es_full = frtb_es.es975(losses)
print(f"ES97.5 (10-day, full set)={es_full:,.0f}")
cascade = frtb_es.liquidity_horizon_es([es_full, 0.6 * es_full, 0.3 * es_full], [10, 20, 60])
print(f"LH cascade (10/20/60)={cascade:,.0f}")
print(f"stress-calibrated: {frtb_es.stress_calibrated_es(es_full, 0.8 * es_full, 1.7 * es_full):,.0f} "
      f"(ratio 1.25 x stressed 1.7)")
print(f"floor engaged:     {frtb_es.stress_calibrated_es(es_full, 1.2 * es_full, 1.7 * es_full):,.0f} "
      f"(full < reduced -> ratio floored at 1)")
```

```
ES97.5 (10-day, full set)=2,239,630
LH cascade (10/20/60)=2,937,247
stress-calibrated: 4,759,213 (ratio 1.25 x stressed 1.7)
floor engaged:     3,807,370 (full < reduced -> ratio floored at 1)
```

### 14. Tail risk beyond the sample: peaks-over-threshold GPD

`fit_pot` fits a generalized Pareto to exceedances above a quantile
(Hosking-Wallis PWM); `var`/`expected_shortfall` then extrapolate to
confidence levels the sample barely reaches. The PWM fit is biased for
shape > 0.5, which the docstring states.

```python
import numpy as np
from quantfinlib.risk import extreme_value_theory as evt
from quantfinlib.risk import risk_metrics as rm

losses = -np.random.default_rng(21).standard_t(3, 2000) * 0.01      # positive = loss
fit = evt.fit_pot(losses, 0.95)
print(f"threshold={fit.threshold:.5f} shape(xi)={fit.shape:.3f} scale={fit.scale:.5f} exceedances={fit.exceedances}")
for p in (0.99, 0.999):
    print(f"p={p}: GPD VaR={fit.var(p):.5f} ES={fit.expected_shortfall(p):.5f}  "
          f"empirical VaR={rm.historical_var(-losses, p):.5f}")
```

```
threshold=0.02425 shape(xi)=0.232 scale=0.01243 exceedances=100
p=0.99: GPD VaR=0.04849 ES=0.07197  empirical VaR=0.05010
p=0.999: GPD VaR=0.10337 ES=0.14338  empirical VaR=0.10051
```

## Volatility

### 15. Fit GARCH(1,1), forecast variance, and compare with EWMA

`Garch11.fit` needs >= 100 finite returns with non-zero variance (it
raises otherwise -- a fit that never happened must not look like one).
Variance targeting pins omega to the sample variance.

```python
import math
import numpy as np
from quantfinlib.volatility import Garch11, EwmaVolatility

rng = np.random.default_rng(2024)
omega, alpha, beta = 2e-6, 0.08, 0.90
h, r = omega / (1 - alpha - beta), np.empty(3000)
for t in range(3000):
    r[t] = math.sqrt(h) * rng.standard_normal()
    h = omega + alpha * r[t] ** 2 + beta * h
fit = Garch11.fit(r)
print(f"alpha={fit.alpha:.4f} beta={fit.beta:.4f} persistence={fit.persistence():.4f} "
      f"uncond vol={math.sqrt(fit.unconditional_variance() * 252):.4f}")
for k in (1, 5, 20):
    print(f"  {k:2d}-day-ahead vol forecast={math.sqrt(Garch11.forecast_variance(r, fit, k) * 252):.4f}")
print(f"EWMA(0.94) latest annualized vol={EwmaVolatility.risk_metrics().annualized_vol(r, 252):.4f}")
try:
    Garch11.fit(np.full(200, 0.01))
except ValueError as e:
    print("constant series ->", e)
```

```
alpha=0.0632 beta=0.9202 persistence=0.9834 uncond vol=0.1462
   1-day-ahead vol forecast=0.1362
   5-day-ahead vol forecast=0.1368
  20-day-ahead vol forecast=0.1390
EWMA(0.94) latest annualized vol=0.1301
constant series -> returns have zero sample variance; GARCH likelihood undefined
```

### 16. Range-based volatility from OHLC bars

Parkinson uses high/low only; Garman-Klass adds open/close; Rogers-Satchell
is drift-independent; Yang-Zhang adds the overnight gap. All return an
annualized vol given `periods_per_year`. The bars below are simulated
at 20% vol with no overnight gap, so every estimator should land near
0.20 with less sampling noise than close-to-close.

```python
import numpy as np
from quantfinlib.volatility import RangeVolatility

rng = np.random.default_rng(8)
n, steps, sigma_daily = 250, 390, 0.20 / np.sqrt(252)          # one path per minute of a 6.5h day
path = 100 * np.exp(np.cumsum(rng.standard_normal(n * steps) * sigma_daily / np.sqrt(steps)))
days = path.reshape(n, steps)                                     # one continuous path, no overnight gap
open_, high, low, close = days[:, 0], days.max(axis=1), days.min(axis=1), days[:, -1]
print(f"parkinson      ={RangeVolatility.parkinson(high, low, 252):.4f}")
print(f"garman-klass   ={RangeVolatility.garman_klass(open_, high, low, close, 252):.4f}")
print(f"rogers-satchell={RangeVolatility.rogers_satchell(open_, high, low, close, 252):.4f}")
print(f"yang-zhang     ={RangeVolatility.yang_zhang(open_, high, low, close, 252):.4f}")
print(f"close-to-close ={np.std(np.diff(np.log(close)), ddof=1) * np.sqrt(252):.4f}")
```

```
parkinson      =0.1998
garman-klass   =0.1965
rogers-satchell=0.1959
yang-zhang     =0.1982
close-to-close =0.2086
```

## Data and backtesting

### 17. Load OHLCV from CSV into a BarSeries and compute indicators

The loader handles US and European decimals, ISO dates and epoch
seconds/millis, sorts rows by time, and refuses duplicate timestamps.
`BarSeries` array accessors are read-only NumPy views.

```python
import os, tempfile
from quantfinlib.data import csv_bar_loader
from quantfinlib.indicators import Indicators

rows = ["timestamp,open,high,low,close,volume"]
px = 100.0
for d in range(1, 41):
    px *= 1 + (0.004 if d % 7 else -0.02)
    rows.append(f"2024-01-{d:02d}" if d <= 31 else f"2024-02-{d - 31:02d}")
    rows[-1] += f",{px:.2f},{px * 1.01:.2f},{px * 0.99:.2f},{px:.2f},{1000 + d}"
path = os.path.join(tempfile.mkdtemp(), "demo.csv")
with open(path, "w") as f:
    f.write("\n".join(rows) + "\n")
series = csv_bar_loader.load(path, "DEMO")
print(f"bars={series.size()} first_ts={series.timestamp(0)} last_close={series.last_close():.2f} "
      f"closes writeable? {series.closes().flags.writeable}")
print(f"SMA(10)[-1]={Indicators.sma(series.closes(), 10)[-1]:.4f}  RSI(14)[-1]={Indicators.rsi(series.closes(), 14)[-1]:.2f}  "
      f"ATR(14)[-1]={Indicators.atr(series.highs(), series.lows(), series.closes(), 14)[-1]:.4f}")
```

```
bars=40 first_ts=1704067200000 last_close=103.95 closes writeable? False
SMA(10)[-1]=103.0910  RSI(14)[-1]=59.54  ATR(14)[-1]=2.1755
```

### 18. Run a strategy backtest and read the metrics honestly

Signals fill at the bar's close with proportional slippage/commission;
stop-loss and take-profit are checked intrabar (stop first). The result
carries the equity curve, the trade list and `PerformanceMetrics`.

```python
import numpy as np
from quantfinlib.backtest import Backtester, BacktestConfig, TradeAnalytics
from quantfinlib.backtest.strategies import SmaCrossStrategy
from quantfinlib.data import BarSeries

rng = np.random.default_rng(42)
closes = 100 * np.exp(np.cumsum(rng.standard_normal(600) * 0.01 + 0.0003))
series = BarSeries.of("DEMO", closes)
cfg = BacktestConfig.defaults().with_commission(0.0005).with_stop_loss(0.05)
res = Backtester.run(SmaCrossStrategy(20, 50), series, cfg)
m = res.metrics()
print(f"{res.strategy_name()} trades={m.trade_count} totalReturn={m.total_return:.2%} CAGR={m.cagr:.2%} "
      f"sharpe={m.sharpe_ratio:.2f} maxDD={m.max_drawdown:.2%} winRate={m.win_rate:.1%} profitFactor={m.profit_factor:.2f}")
ta = TradeAnalytics.analyze(res.trades())
print(f"expectancy={ta.expectancy:.2f} payoff={ta.payoff_ratio:.2f} kelly={ta.kelly_fraction:.3f} "
      f"maxLossStreak={ta.max_loss_streak}")
buy_and_hold = closes[-1] / closes[0] - 1
print(f"buy-and-hold over the same bars: {buy_and_hold:.2%}")
```

```
SMA_CROSS(20,50) trades=7 totalReturn=-10.49% CAGR=-4.55% sharpe=-0.50 maxDD=14.15% winRate=14.3% profitFactor=0.36
expectancy=-1498.09 payoff=2.16 kelly=0.000 maxLossStreak=4
buy-and-hold over the same bars: 1.99%
```

### 19. Purged K-fold splits and the probability of backtest overfitting (CSCV)

`PurgedKFold.splits(n, folds, purge, embargo)` removes the leakage zone
around each test fold. `OverfitProbability.cscv_sharpe` on a matrix of
strategy-variant returns gives PBO: near 0.5 or above means the
selection is noise-mining. A single draw is one sample from a wide
distribution -- average several before reading the number.

```python
import numpy as np
from quantfinlib.backtest.validation import PurgedKFold, OverfitProbability

for s in PurgedKFold.splits(120, 4, 3, 2)[:2]:
    print(f"test=[{s.test_from},{s.test_to}) train_size={len(s.train_indices)}")
pbo_noise, pbo_edge = [], []
for seed in range(10):                                            # one draw is one sample; average them
    noise = np.random.default_rng(seed).standard_normal((600, 16)) * 0.01   # 16 variants, no edge
    edge = noise.copy(); edge[:, 0] += 0.004                       # variant 0 has real edge
    pbo_noise.append(OverfitProbability.cscv_sharpe(noise, 8).pbo)
    pbo_edge.append(OverfitProbability.cscv_sharpe(edge, 8).pbo)
print(f"mean PBO over 10 draws: no edge={np.mean(pbo_noise):.3f} (range {min(pbo_noise):.2f}-{max(pbo_noise):.2f})  "
      f"one real edge={np.mean(pbo_edge):.3f}")
```

```
test=[0,30) train_size=85
test=[30,60) train_size=82
mean PBO over 10 draws: no edge=0.494 (range 0.09-0.70)  one real edge=0.000
```

### 20. Deflate a Sharpe ratio for the number of trials

PSR is the probability the true Sharpe exceeds a benchmark given the
sample's length, skew and kurtosis; DSR raises the benchmark to the
expected maximum over the trials you ran. MinTRL says how long a track
record you need.

```python
import numpy as np
from quantfinlib.backtest.validation import SharpeValidation
from quantfinlib.util import math_utils as mu

rng = np.random.default_rng(3)
rets = rng.standard_normal(500) * 0.01 + 0.0006
sr = mu.mean(rets) / mu.std_dev(rets)                              # per-period Sharpe
sk, ku = mu.skewness(rets), mu.kurtosis(rets)
print(f"sharpe/period={sr:.4f} skew={sk:.3f} kurtosis={ku:.3f}")
print(f"PSR(SR* = 0)={SharpeValidation.probabilistic_sharpe(sr, 0.0, 500, sk, ku):.4f}")
trials = list(rng.standard_normal(50) * 0.03)
print(f"DSR after 50 trials={SharpeValidation.deflated_sharpe(sr, trials, 500, sk, ku):.4f}")
print(f"MinTRL to beat 0 at 95%={SharpeValidation.min_track_record_length(sr, 0.0, sk, ku, 0.95):.0f} periods")
```

```
sharpe/period=0.1148 skew=-0.067 kurtosis=3.128
PSR(SR* = 0)=0.9945
DSR after 50 trials=0.9133
MinTRL to beat 0 at 95%=209 periods
```

### 21. Size positions: Kelly, risk parity and max-Sharpe weights

`PositionSizing.kelly_fraction(mean, variance)` is the full-Kelly leverage
(often > 1 -- distrust the inputs before you trust the leverage).
Risk parity equalizes risk contributions; `PortfolioOptimizer` is a
seeded derivative-free search over the simplex.

```python
import numpy as np
from quantfinlib.backtest.portfolio import PositionSizing, RiskParityOptimizer, PortfolioOptimizer

mu_ = np.array([0.08, 0.05, 0.03])
cov = np.array([[0.04, 0.006, 0.002], [0.006, 0.01, 0.001], [0.002, 0.001, 0.0025]])
print(f"kelly(mean 5bp/day, var 4bp^2)={PositionSizing.kelly_fraction(0.0005, 0.0004):.3f} "
      f"half-kelly={PositionSizing.half_kelly(0.0005, 0.0004):.3f}")
rp = RiskParityOptimizer.equal_risk_contribution(mu_, cov)
print("risk parity weights:", np.round(rp.weights, 4), " risk contributions:",
      np.round(RiskParityOptimizer.risk_contributions(rp.weights, cov), 4))
ms = PortfolioOptimizer(mu_, cov).max_sharpe(0.02)
print(f"max-sharpe weights: {np.round(ms.weights, 4)} exp.return={ms.expected_return:.4f} vol={ms.volatility:.4f}")
```

```
kelly(mean 5bp/day, var 4bp^2)=1.250 half-kelly=0.625
risk parity weights: [0.1393 0.2786 0.5821]  risk contributions: [0.3333 0.3333 0.3333]
max-sharpe weights: [0.194  0.3881 0.4179] exp.return=0.0475 vol=0.0707
```

## Microstructure and execution

### 22. Schedule an execution: Almgren-Chriss trajectory vs TWAP and VWAP slices

`AlmgrenChriss.optimal_trajectory` trades the expected cost of impact
against the variance of slow execution: more risk aversion front-loads
the schedule, paying more impact for less timing risk. The schedulers
turn a quantity and a duration into `Slice(offset_millis, quantity)` lists.

```python
from quantfinlib.microstructure import AlmgrenChriss
from quantfinlib.execution import twap_scheduler, vwap_scheduler

# volatility in $/share/sqrt(year) (30% on a $5 stock), impacts in $/share per share traded
p = AlmgrenChriss.Params(total_shares=100_000, horizon=1.0, intervals=8, volatility=1.5,
                         temporary_impact=1e-6, permanent_impact=5e-7, risk_aversion=1e-6)
opt, twap = AlmgrenChriss.optimal_trajectory(p), AlmgrenChriss.twap(p)
print(f"AC   kappa={opt.kappa:.3f} expected cost=${opt.expected_cost:,.0f} cost stdev=${opt.cost_variance ** 0.5:,.0f}")
print(f"TWAP kappa={twap.kappa:.3f} expected cost=${twap.expected_cost:,.0f} cost stdev=${twap.cost_variance ** 0.5:,.0f}")
print("AC trades per interval (front-loaded):", [int(x) for x in opt.trades])
print("TWAP slices:", [(s.offset_millis, s.quantity) for s in twap_scheduler.schedule(100_000, 3_600_000, 4)])
print("VWAP slices:", [(s.offset_millis, s.quantity) for s in
                       vwap_scheduler.schedule(100_000, [0.4, 0.2, 0.15, 0.25], 3_600_000)])
```

```
AC   kappa=1.522 expected cost=$12,950 cost stdev=$66,779
TWAP kappa=0.000 expected cost=$12,188 cost stdev=$78,437
AC trades per interval (front-loaded): [19237, 16306, 13967, 12135, 10743, 9741, 9093, 8774]
TWAP slices: [(0, 25000), (900000, 25000), (1800000, 25000), (2700000, 25000)]
VWAP slices: [(0, 40000), (900000, 20000), (1800000, 15000), (2700000, 25000)]
```

### 23. Classify trades (Lee-Ready) and watch flow toxicity (VPIN)

`TradeClassifier` needs the prevailing quote before each trade; `Vpin`
fills fixed-volume buckets and is NaN until the first bucket completes.
Fractional bucket sizes (FX/crypto notionals) complete exactly.

```python
import numpy as np
from quantfinlib.microstructure import TradeClassifier, Vpin

tc, vpin = TradeClassifier(), Vpin(bucket_volume=5_000, window=10)
rng = np.random.default_rng(6)
mid = 100.0
for i in range(400):
    tc.on_quote(mid - 0.01, mid + 0.01)
    informed = i >= 250                                   # one-sided selling in the last 150 trades
    price = mid - 0.01 if (informed or rng.random() < 0.5) else mid + 0.01
    vpin.on_trade(200, tc.is_buy_aggressor(price))
    if i in (49, 249, 399):
        print(f"trade {i + 1:3d}: buckets={vpin.buckets_completed():2d} ready={vpin.ready()} vpin={vpin.vpin():.3f}")
    mid += rng.standard_normal() * 0.002
```

```
trade  50: buckets= 2 ready=False vpin=0.120
trade 250: buckets=10 ready=True vpin=0.152
trade 400: buckets=10 ready=True vpin=0.672
```

### 24. Quote as a market maker with Avellaneda-Stoikov and detect a dead vol feed

Reservation price shades against inventory; the half-spread never drops
below the liquidity floor. A NaN/inf variance is treated as zero (the
quote goes unshaded) and `degraded()` flips so you can pull quotes.

```python
import math
from quantfinlib.microstructure import AvellanedaStoikov

mm = AvellanedaStoikov(gamma=0.1, kappa=1.5)
for inv in (0.0, 300.0, -300.0):
    print(f"inventory={inv:6.0f}: bid={mm.bid_quote(100.0, inv, 0.0004, 300.0):.4f} "
          f"ask={mm.ask_quote(100.0, inv, 0.0004, 300.0):.4f} degraded={mm.degraded()}")
bid = mm.bid_quote(100.0, 300.0, math.nan, 300.0)
print(f"NaN variance: bid={bid:.4f} (unshaded floor quote) degraded={mm.degraded()}")
```

```
inventory=     0: bid=99.3486 ask=100.6514 degraded=False
inventory=   300: bid=95.7486 ask=97.0514 degraded=False
inventory=  -300: bid=102.9486 ask=104.2514 degraded=False
NaN variance: bid=99.3546 (unshaded floor quote) degraded=True
```

## Simulation and persistence

### 25. Monte Carlo a portfolio with the bit-exact Java RNG and checkpoint a running estimator

`MonteCarloSimulator(seed)` reproduces Java's `SplittableRandom` stream,
so `SimulationResult` numbers match the Java library per seed.
`Checkpoint` writes the Java file format; `EwmaCovariance` restores to
the same state.

```python
import os, tempfile
import numpy as np
from quantfinlib.sim import MonteCarloSimulator
from quantfinlib.microstructure import EwmaCovariance
from quantfinlib.persist import Checkpoint

res = MonteCarloSimulator(seed=7).simulate(1_000_000.0, 0.07, 0.18, horizon_days=252, simulations=20_000)
print(f"expected={res.expected_value():,.0f} median={res.median_value():,.0f} "
      f"VaR95={res.value_at_risk(0.95):.2%} of initial CVaR95={res.conditional_value_at_risk(0.95):.2%} "
      f"P(loss)={res.probability_of_loss():.3f}")

cov = EwmaCovariance(2, lam=0.94)
for r in np.random.default_rng(1).multivariate_normal([0, 0], [[1e-4, 4e-5], [4e-5, 2e-4]], size=100):
    cov.on_returns(r)
path = os.path.join(tempfile.mkdtemp(), "state.ckpt")
Checkpoint.writer(path).section("ewma", cov.write_state).close()
restored = EwmaCovariance(2, lam=0.94)
Checkpoint.reader(path).section("ewma", restored.read_state)
print(f"corr before={cov.correlation(0, 1):.6f} after restore={restored.correlation(0, 1):.6f} "
      f"samples={restored.samples()} bytes={os.path.getsize(path)}")
```

```
expected=1,071,460 median=1,053,125 VaR95=21.31% of initial CVaR95=27.17% P(loss)=0.386
corr before=-0.117781 after restore=-0.117781 samples=100 bytes=55
```

## Verification

### 26. Re-run the cross-port guarantee

The probe prints 156 labelled numbers; `compare.py` diffs them against
the checked-in Java output at 1e-9 (1e-6 for iterative fits). Both
repo-relative and absolute paths work.

```python
import subprocess, sys, tempfile, os
out = os.path.join(tempfile.mkdtemp(), "py_check.out")
with open(out, "w") as f:
    subprocess.run([sys.executable, "tools/crossport/probe.py"], stdout=f, check=True,
                   env={**os.environ, "PYTHONPATH": "src"})
print(subprocess.run([sys.executable, "tools/crossport/compare.py", "tools/crossport/java.out", out],
                     capture_output=True, text=True, check=True).stdout.strip())
```

```
--- tools/crossport/java.out vs /tmp/tmpfv1qybms/py_check.out: 156 labels, 0 mismatches ---
```
