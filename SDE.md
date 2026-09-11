# SDE forecasting benchmarks

The SDE models reuse the existing forecasting and walk-forward validation
interfaces. They use the target ticker only; related tickers remain available
to GP/VAR in the same validation experiment. No Heston model is included.

```python
from stock_api import predict

result = predict("AMZN", model="gbm_zero_drift", seed=42, n_paths=10_000)
result.plot()
point = result.forecasts[-1]
print(point.mean_price, point.median_price, point.price_std)
print(point.lower_95, point.upper_95, point.probability_up)
result.terminal_returns      # simulated terminal simple returns S_T/S_0 - 1
result.terminal_log_returns  # simulated terminal log returns log(S_T/S_0)
```

Use `gbm_estimated_drift` or `ou_returns` to change models. All three generate
Monte Carlo ensembles (10,000 paths by default). Price means, medians, standard
deviations, probabilities and quantiles use their exact conditional lognormal
marginals to avoid Monte Carlo noise in scores. The simulated paths follow that
same distribution; they are not artificial perturbations of a point forecast.
`predicted_close` remains the **median**, consistent with the existing GP/VAR
interface. Future means and medians are both exposed. Intervals condition on
estimated parameters and exclude parameter-estimation uncertainty.

## Time and drift conventions

One unit of model time is **one trading session**. Training returns are daily
close-to-close log returns. Forecast transitions use `dt=1` per forecast session,
without annualization and without extra Brownian increments on weekends.
A seven-calendar-day validation horizon thus usually simulates four or five
trading-session increments, not seven. `horizon_unit="trading"` requests seven
increments instead.

For geometric Brownian motion, `dS = mu*S*dt + sigma*S*dW`, the exact transition
is `S_next = S * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*Z)`. Production GBM
forecasts **do not use Euler–Maruyama**.

- `gbm_zero_drift`: sets **mu=0**, meaning zero expected simple/arithmetic return.
  Volatility is the training log-return sample standard deviation (`ddof=1`).
  The expected price equals the last observed price, but the median is
  `S0*exp(-sigma²*t/2)` and `P(S_t>S0)` is slightly below 0.5. This Itô correction
  explains a weak downward median forecast; it is not a learned bearish signal.
- `gbm_estimated_drift`: estimates `mu = mean(training log returns) + sigma²/2`.
  Consequently its expected log return per step equals the training mean log
  return. Substituting that mean directly for mu would subtract the variance
  correction twice and is deliberately avoided.
- `last_price` is retained permanently: its point forecast equals day zero
  exactly. A separate zero-mean-log-return random walk would have a flat median,
  but positive expected arithmetic return; it is not identical to mu=0 GBM.

For GBM, `mu` is per trading day and `sigma` per square root of a trading day.
Every estimate is computed afresh within each training fold. Volatility and
expected drift are not tuned on the scored future prices. Drift estimates from
short histories can be very noisy, and plug-in predictive intervals may
under-cover when drift uncertainty is large.

## Generic scalar Itô solver

```python
from stock_api.sde import euler_maruyama, simulate_gbm, ItoProcess

paths = euler_maruyama(
    100.0,
    drift=lambda x, t: 0.001*x,
    diffusion=lambda x, t: 0.02*x,
    steps=700, dt=0.01, n_paths=10_000, seed=42,
)
# paths.shape == (10000, 701); first column is the initial state.

exact = simulate_gbm(100, mu=0.001, sigma=0.02,
                     steps=7, dt=1, n_paths=10_000, seed=42)
```

The generic solver implements `X_next = X + drift(X,t)*dt + diffusion(X,t)*sqrt(dt)*Z`.
It vectorizes across paths; coefficients may return scalars or vectors.
Alternatively subclass `ItoProcess`, implement `drift(self,x,t)` and
`diffusion(self,x,t)`, then call `.simulate(x0, ...)`.

Pass a seed or an existing NumPy `Generator`, not both. `normal_draws=` can
supply coupled standard-normal increments for convergence tests. Invalid steps,
dt, or nonfinite states fail explicitly. Generic EM does not clip negative
states: it does not automatically preserve positivity for multiplicative SDEs.
Use the exact simulator for GBM. Smaller dt is a numerical accuracy choice, not
an opportunity to infer extra observations from daily training data.

For future non-Gaussian SDEs, wrap the simulator in the existing
`ModelSpec.fit_predict` interface and return empirical path medians/quantiles,
optionally `probability_up`, `mean_price`, `median_price`, and `price_std`.
The validation interface does not require a Gaussian distribution. The current
GBM/OU wrappers use Gaussian log marginals because they are exact for these models.

## OU-style daily returns

`ou_returns` fits the discrete transition of a continuous OU return state:
`r_next = mu_r + phi*(r - mu_r) + epsilon`, with `phi=exp(-theta)` at dt=1.
OLS on consecutive **training** log returns estimates phi and the intercept;
residual variance estimates transition innovation variance. When stable,
`theta=-log(phi)` and the continuous diffusion is
`sqrt(innovation_variance*2*theta/(1-phi²))`.

It simulates these returns using their exact Gaussian transition and compounds
prices with `S_next=S*exp(r_next)`. A two-state covariance recursion for the
current return and cumulative log price preserves cross-step correlations and
provides exact conditional marginal moments.

The following fixed, training-only checks trigger an explicit fallback to IID
Gaussian returns with the training mean and variance:

- Fewer than 20 training returns, or effectively constant return predictors.
- Estimated phi outside `(0, 0.98)`, including negative autocorrelation, which a
  scalar continuous-time OU cannot represent.
- Implied equilibrium more than three sample return standard deviations from
  the training return mean.

Fallback reasons are stored in per-fold `diagnostics` and forecast warnings.
The IID fallback has the same log-return marginals as estimated-drift GBM;
identical forecasts in such folds are expected. These safeguards are fixed
before validation, not selected by validation performance.

## Validation and return diagnostics

```python
from stock_api import ValidationConfig, builtin_models, validate

registry = builtin_models(n_paths=10_000)
names = ["last_price", "gp", "multitask_gp", "var",
         "gbm_zero_drift", "gbm_estimated_drift", "ou_returns"]
run = validate("AMZN", related_tickers=["GOOGL", "AAPL"],
               config=ValidationConfig(reference_date="2026-09-01",
                                       train_days=30, horizon=7, seed=42),
               models={name:registry[name] for name in names})
print(run.summary())
```

Existing metrics remain. `terminal_direction_accuracy` explicitly names the
terminal-fold sign score; `direction_accuracy` is a compatibility alias.
Flat point forecasts are N/A and excluded from its denominator, reported as
`directional_folds`. For persistence this is zero directional folds, not a 0%
hit rate. Both aliases are NaN in summary tables/blank in CSV for that case.

Additional summary fields:

- `terminal_log_return_mae`: mean absolute error of terminal log returns.
- `terminal_log_return_mae_pct`: the above times 100 (log-return percentage
  points, not dollars and not exactly simple-return percentage points).
- `terminal_brier_score`: mean `(P(S_T>S_0) - observed_up)^2`, lower is better.
  A constant 0.5 event probability scores 0.25. Missing probabilities remain N/A.
- `mean_probability_up` and `realized_up_frequency`: coarse calibration checks.

`run.metrics` records predicted and realized terminal log returns, individual
absolute errors, terminal event probabilities and Brier scores. `run.predictions`
retains daily price means, medians, standard deviations and probabilities when
provided. `run.distributions` stores 10,000 simulated terminal log returns per
SDE fold; use `expm1` for simple returns. Model diagnostics include OU fallbacks.
Older saved runs can still be loaded; terminal errors and direction semantics
are updated from stored observations, and unavailable historical probabilities
remain N/A. They are not guessed from point forecasts.

There are **12 terminal outcomes**, not 56 independent weekly experiments.
Daily errors within each path are correlated. `coverage_95` averages the fold
coverage fractions and Brier averages terminal events. Neither gives a reliable
calibration guarantee from only 12 folds. Cross-lookback comparisons reuse the
same outcomes, so they are also dependent. No significance claims or model
tuning on these folds are made; retain a separate final test period.

## Reproduce the four-lookback experiment

```bash
python examples/sde_comparison.py --reference-date 2026-09-01
```

This downloads one common adjusted-price snapshot and scores all seven models
on 30, 60, 90 and 180 **calendar-day** training windows, twelve monthly folds,
a seven-calendar-day horizon, and seed 42. JSON records (including terminal
ensembles), complete summary CSVs, fold metrics and plots are saved in a new
output directory. Replay with `--prices path/to/prices.csv`; use `--output` to
name a new directory. Existing directories are refused to preserve prior runs.
The snapshot may differ from older Yahoo downloads due to retrospective
adjustments, so all seven models are rerun on the same snapshot for this exercise.
