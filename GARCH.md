# Regression–GJR-GARCH

`regression_gjr_garch` fits a return regression and asymmetric conditional
variance jointly by maximum likelihood, using standardized Student-t innovations.
Its dependency is included in the standard installation and `environment.yml`:

```bash
python -m pip install .
```

```python
from stock_api import predict

result = predict("AMZN", model="regression_gjr_garch", lookback=252,
                 horizon=7, seed=42, n_paths=10000)
result.plot()
```

The public Python/HTTP model uses an intercept and the previous target return.
It supports model comparisons, saved plots, validation and the historical runner.
It is included in the default validation, backtest and historical suites.
Validation defaults to 90 calendar days of training; new historical configurations
default to 60/90/180/365-day windows so GJR has enough observations. Explicit
shorter windows remain available and fail if they cannot meet the model's
31-return minimum. Saved configurations retain their recorded settings.
The public `lookback` counts trading sessions; historical
runner lookbacks count calendar days. Leave `inference=None`: this model uses
joint MLE, and does not implement the GP posterior samplers.

## Model and uncertainty

For daily log return `r[t]`, the regression and innovation equations are:

```text
r[t] = c + phi*r[t-1] + b' x[t] + epsilon[t]
epsilon[t] = sqrt(h[t]) * z[t]
h[t] = omega + alpha*epsilon[t-1]^2
       + gamma*I(epsilon[t-1] < 0)*epsilon[t-1]^2 + beta*h[t-1]
z[t] ~ Student-t(nu) * sqrt((nu-2)/nu)
```

`x[t]` is optional. A positive `gamma` makes negative residuals increase the
next variance more than equal positive residuals. Estimation uses percent log
returns internally for numerical scaling; the external return API uses
dimensionless log returns. Diagnostics expose both the fitted percent-unit
parameters and regression coefficients in external input units.

Forecast paths recursively update return, residual and variance every session.
Medians, direction probabilities and intervals come from these paths.
`sigma_1/2/3` use the corresponding equal-tail probability levels; they do not
assume a Gaussian predictive distribution. `log_std` is the sample standard
deviation in log-price space. Simulation conditions on the MLE parameters and
does **not** incorporate parameter-estimation uncertainty. In contrast, the
archived Bayesian return-GP forecasts integrate parameter and latent uncertainty.

Student-t log returns have no finite exponential moments, so `mean_price` and
`price_std` are `None`. Sample price averages are not reported as population
expectations. Quantiles remain defined. Numerical overflow fails explicitly;
the implementation does not clip tails. Fitted curves start at the third price
observation because of the AR lag. Their bands use conditional Student-t
quantiles with parameters estimated on the whole training window; these are
descriptive fits, not historical out-of-sample forecasts.

The implementation requires at least 31 nondegenerate returns and rejects
collinear regressors, failed optimization and unstable AR estimates. Near-unit
variance persistence and degrees of freedom near two generate diagnostic
warnings. Numerical convergence alone does not establish parameter precision.

## Adding external regressors

Use the exported lower-level fit API with explicitly aligned arrays:

```python
import numpy as np
from stock_api import fit_regression_gjr

# prices: N positive closes; X_train: (N-1, k); X_future: (steps, k).
# X_train[i] must have been available before the return prices[i] -> prices[i+1].
fit = fit_regression_gjr(np.diff(np.log(prices)), regressors=X_train)
return_paths = fit.simulate_returns(
    steps, n_paths=10000, seed=42, future_regressors=X_future
)
price_paths = prices[-1] * np.exp(return_paths.cumsum(axis=1))
```

Scaling is estimated from training regression rows only. Future inputs are
mandatory when external regressors were fitted. Supply a deterministic known
or forecast path of shape `(steps, k)`, or joint scenarios of shape
`(n_paths, steps, k)`. A deterministic path omits predictor uncertainty.
The caller is responsible for information availability, publication lags and
scenario dependence. Never supply realized future stock prices as predictors.
`related_tickers` does not build these features automatically. The direct
`stock_api.garch.regression_gjr_garch` adapter also accepts `regressors` and
`future_regressors` when price forecasts are needed.

## Amazon comparison

See the [findings and recommendation](reports/GJR_AMZN_COMPARISON.md) and
[generated experiment report](historical_validation/AMZN/gjr-2025-2026-v1/REPORT.md).
The comparison covers September 2025–August 2026 using the original frozen
adjusted-close snapshot and saved GP predictions. It contains no external
regressors. Reproduce or resume with the same environment and implementation:

```bash
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 \
  .conda-env/bin/python examples/gjr_amzn_comparison.py
```

The script checks record and snapshot checksums, preserves the original archive,
and writes forecasts, provenance, scores, paired comparisons and a chart to a
separate experiment. Changed model code, dependencies or comparison settings
require a new output directory via `--output`.

For a small fresh download/backtest:

```bash
python -m stock_api.backtest AMZN --lookback 252 --folds 5 \
  --models last_price regression_gjr_garch
```

Tests in `tests/test_garch.py` cover the reference-library variance calculation,
recursive simulation, regressor recovery and explicit future inputs, rejection
of failed fits, public serialization, historical exclusion of future data, and
deterministic validation/backtests. GARCH tests require the standard project
dependencies and run as part of the full suite.
