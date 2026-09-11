# Stock movement API

A Python package that downloads daily stock prices and forecasts a target
stock, optionally using related stocks. It also includes an optional FastAPI
HTTP service. The original `example.py`, `utils.py`,
and `predict_stock.py` remain standalone plotting/mean/linear-fit experiments.
The API lives in `stock_api/` and does not import their executable demo code.

## Install and use in Python / IPython

From the repository root, install with pip (using the included `setup.py` and
`pyproject.toml` metadata):

```bash
python -m pip install .
```

For development, use `python -m pip install -e .`. After installation, you can
import the package from any directory in Python, IPython, or a notebook:

```python
from stock_api import predict

result = predict("AMZN")  # next seven calendar days, single-ticker GP
print(result.direction, result.predicted_change_pct)

# Jointly fit Amazon, Google, and Apple to forecast Amazon.
result = predict("AMZN", related_tickers=["GOOGL", "AAPL"], model="multitask_gp")
for point in result.forecasts:
    print(point.date, point.predicted_close, point.lower_95, point.upper_95)

# All request options are keyword arguments.
result = predict("AMZN", related_tickers=["GOOGL", "AAPL"],
                 model="var", horizon=7, horizon_unit="trading", lookback=120)

result.model_dump()       # ordinary Python dictionary
result.model_dump_json()  # JSON string
```

For a table in a notebook:

```python
import pandas as pd
table = pd.DataFrame([point.model_dump() for point in result.forecasts])
```

### Plot the fit, training data, and forecast

Install the optional plotting dependency with `python -m pip install '.[plots]'`
from the repository (the conda environment already includes it). Plotting is
explicit and never happens just because you call `predict`:

```python
result = predict("AMZN", related_tickers=["GOOGL", "AAPL"])
result.plot()  # all training data, historical target fit, forecast, 1/2/3σ bands

# Zoom to the seven calendar days before the origin and seven days after it.
result.plot(history_days=7, forecast_days=7)

# Target only, with 1σ and 2σ error bars instead of shaded bands.
result.plot(history_days=30, show_related=False, sigmas=(1, 2),
            uncertainty="errorbars")

# Return a Matplotlib Axes without displaying; customize or save it.
ax = result.plot(show=False, show_fit=False, sigmas=(1, 2, 3))
ax.figure.savefig("forecast.png", dpi=150)
```

`history_days=None` (default) shows the entire training window;
`forecast_days=None` shows the entire stored forecast. Both limits count
**calendar days** relative to `as_of`, inclusive of the boundary. They only
crop the display: they do not refit, download data, or extend the forecast.
Use `predict(horizon=..., horizon_unit=...)` to change the forecast itself.
`forecast_days=0` hides future points, `sigmas=()` hides uncertainty, and
`show_fit=False` hides the historical fit and its bands. You can supply `ax=`
for an existing subplot or `figsize=(12, 6)` for a new figure.

The default plot includes every training ticker. Related tickers are rebased
to the target's last training price for comparison; their original adjusted
prices remain in `result.training_data`, a list of dated `closes` dictionaries.
The displayed fit and uncertainty are for the target only.

`result.fitted` and `result.forecasts` both expose `predicted_close`, `log_std`,
and `sigma_1`, `sigma_2`, `sigma_3` intervals with `.lower` and `.upper` values.
For example, `result.forecasts[0].sigma_2.lower` gives the first forecast's
lower 2σ bound. Each interval is `exp(log_mean ± n * log_std)`: Gaussian sigma
levels in **log-price space**, approximately 68.27%, 95.45%, and 99.73% under
the model. They are asymmetric in price units; the existing `lower_95` and
`upper_95` forecast fields still use 1.96σ rather than 2σ.

For the GP, the historical fit is the posterior conditioned on the entire
training window, with observation noise included in its intervals. For VAR,
it is an in-sample one-step conditional fit with residual innovation variance;
the first six prices lack sufficient lags and have no fitted point. For the
random walk, the fitted value is the previous close with one-step return
variance; the first price has no fitted point. All training observations are
still plotted. These historical fits are **not held-out forecasts**, and the
bands exclude parameter-estimation uncertainty. No empirical coverage guarantee
is implied. Serialized results retain training data and fitted intervals, so
`ForecastResponse.model_validate_json(saved_json).plot()` also works offline.

`help(predict)` documents the parameters and exceptions. Calls run directly in
the current process; no server or command-line invocation is needed. Importing
the package does not download data or fit a model. Prediction calls need internet.
The base installation does not require FastAPI or Uvicorn.

Invalid arguments raise `pydantic.ValidationError` (a `ValueError`). You can
handle download/data/model failures using public exception classes:

```python
from stock_api import DataError, ProviderError, ForecastError

try:
    result = predict("AMZN")
except (DataError, ProviderError, ForecastError) as exc:
    print(f"Could not forecast: {exc}")
```

## Conda installation and optional HTTP service

From the repository root:

```bash
conda env create -f environment.yml
conda activate stock-api
uvicorn stock_api.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for interactive requests and response schemas.
`GET /health` checks that the service is running (it does not check Yahoo).

The conda environment includes the Python API, HTTP service, tests, and plotting dependencies for the
original scripts. An alternative on Python 3.11–3.13:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plots,http]'
```

### Python choice

**Python 3.13** is the recommended standard CPython build: a practical balance
of scientific-package compatibility and support lifetime, rather than a claim
that one interpreter makes a forecasting model more accurate. Python 3.13 has
scheduled support through October 2029, versus October 2028 for 3.12
([Python version status](https://devguide.python.org/versions/)).
SciPy distributes Python 3.13 wheels ([SciPy](https://pypi.org/project/scipy/)),
and it is within PyTorch's supported Python range
([PyTorch installation](https://pytorch.org/get-started/locally/)).
That leaves room for neural models and GPyTorch later. The current implementation
uses NumPy/SciPy on CPU, without requiring a large deep-learning installation.
Dependency ranges are compatibility constraints, not an exact reproducibility
lock; export a resolved environment for reproducible experiments.

For an existing base installation, `python -m pip install '.[http]'` from the
repository installs the HTTP extras. Both Python and HTTP entry points use the
same forecasting implementation and return the same fields.

## HTTP requests

A ticker alone uses the GP with seven calendar days and 120 training sessions:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"AMZN"}'
```

Jointly fit Google, Apple, and Amazon to predict Amazon:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"AMZN","related_tickers":["GOOGL","AAPL"],"model":"multitask_gp","horizon":7,"horizon_unit":"calendar","lookback":120}'
```

Use `"model":"var"` with the same related tickers for autoregression. Use
`"model":"last_price"` and omit related tickers for the baseline.

| Field | Default | Meaning |
|---|---|---|
| `ticker` | required | US-equity Yahoo symbol; normalized to uppercase |
| `related_tickers` | `[]` | Up to 3 auxiliary stocks; duplicates/target removed |
| `model` | `multitask_gp` | `gp`, `multitask_gp`, `var`, `last_price` (alias: `random_walk`), `gbm_zero_drift`, `gbm_estimated_drift`, `ou_returns` |
| `horizon` | `7` | 1–30 calendar days or trading sessions |
| `horizon_unit` | `calendar` | `calendar` or `trading` |
| `lookback` | `120` | 60–252 completed market sessions |

The response includes:

- `as_of`, `last_close`, and `training_observations`: the forecast origin and data size.
- `forecasts`: session dates, `predicted_close`, `lower_95`, `upper_95`,
  `predicted_change_pct`, and `probability_up` relative to the last close.
- `direction` and `predicted_change_pct`: movement at the final forecast session.
- `diagnostics`: GP fit parameters/convergence or VAR settings, plus `warnings`.

Prices are **adjusted closing prices**, using explicit `auto_adjust=True`
([yfinance download reference](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)).
`predicted_close` is the median of a lognormal forecast, not its arithmetic mean.
Intervals are pointwise 95% model prediction intervals; `probability_up` is a
model-derived probability, not an empirically calibrated success rate.

### Dates and data

The origin is the latest completed XNYS session at request time. Today's bar is
excluded until the market closes. A seven-calendar-day forecast runs from that
origin to origin + 7 days, emitting only exchange sessions. Thus it usually has
4–5 predictions; weekends and holidays have no fabricated prices. With
`horizon_unit=trading`, it emits exactly seven sessions. A calendar horizon with
no sessions returns 422. The XNYS calendar is also suitable for the common US
Nasdaq stocks in the examples; foreign exchanges and 24/7 assets are unsupported.

Every requested stock must have valid positive closes for the entire shared
training window, including the latest completed session. Missing/stale data is
rejected rather than forward-filled. Yahoo can delay the latest daily bar;
retry later in that case. Downloads use a 15-second provider request timeout;
Yahoo may internally make several requests. There is no persistence or cache,
and each request downloads and refits.

| HTTP status | Meaning |
|---|---|
| 422 | Invalid parameters, missing/stale prices, or no forecast sessions |
| 502 | Provider failed or returned no usable response |
| 503 | Another forecast is running in this worker; retry after 5 seconds |
| 500 | Numerical model fit failed |

## Models

The suite also includes `gbm_zero_drift`, `gbm_estimated_drift`, and `ou_returns`.
They generate predictive ensembles with configurable `seed` and `n_paths`, and
use exact transitions rather than Euler–Maruyama. A separate reusable generic
Itô solver is provided for future nonlinear models. See [SDE models and benchmarks](SDE.md)
for drift conventions, simulation examples, and the four-lookback experiment.

**Last price (`last_price`):** every future point, including day 7, equals the
last observed close (day 0), exactly. `random_walk` remains an alias. Log-return
variance estimated from the training history grows linearly with the horizon.
This is a useful, deliberately difficult-to-beat stock-price baseline.

**Single-ticker GP (`gp`):** fits a Matérn 3/2 GP to the target's log-price
history alone, estimating its timescale, amplitude, and observation noise.
Use `predict("AMZN", model="gp")` without `related_tickers`. It provides the
same historical fit, uncertainty intervals, and `result.plot()` as other models.
In a multi-model validation run, it ignores auxiliary ticker columns so it can
be compared directly with `multitask_gp` on identical data and folds.

**Multitask GP (`multitask_gp`):** jointly fits independent Matérn 3/2 GPs to
all ticker log-price histories. It optimizes one shared timescale and separate
amplitude and observation-noise parameters for every ticker by summing their
log marginal likelihoods. The covariance within ticker i is
`a_i² (1 + sqrt(3)|t-s|/ell) exp(-sqrt(3)|t-s|/ell)`; cross-ticker covariance is
zero. Other tickers constrain `ell`, not the target's price through regression
or cross-ticker correlation. The target posterior uses its own observations
and the jointly estimated timescale. Training-only standardization is undone
in outputs. Diagnostics include `length_scale`, `time_unit`, and per-ticker
`amplitudes` / `observation_noises` in log-price units (target first).
The stationary prior tends toward the training log-price mean far into the
future. Saved GP v1 experiments used the earlier correlated-output model;
new validation runs record GP version 2. Keep distinct names when comparing them.

**VAR:** a five-lag vector autoregression on scaled log returns, with an
unpenalized intercept and ridge penalty 10. All tickers' future returns are
forecast recursively together. A companion-state covariance propagates residual
innovation uncertainty into cumulative target log prices, including cross-step
correlations. Lags and regularization are fixed defaults, not tuned on test data.

Both advanced models estimate transformations from the supplied history only.
Intervals condition on fitted parameters and omit parameter-estimation and
regime-change uncertainty. No predictive advantage is assumed.

## Test and evaluate

For configurable monthly/weekly walk-forward validation, custom stochastic
models and sentiment regressors, fold-error plots, and saved experiment
comparisons, see [the validation guide](VALIDATION.md):

```python
from stock_api import ValidationConfig, validate

run = validate("AMZN", related_tickers=["GOOGL", "AAPL"],
               config=ValidationConfig(reference_date="2026-09-01",
                                       months=12, train_days=30, horizon=7))
print(run.summary())
run.plot()
run.save("validation_runs/amazon.json")
```

The reference date is an exclusive cutoff. The default training and prediction
windows count calendar days, with only exchange sessions scored. `run.rerun(...)`
evaluates new models on the saved snapshot for a matching comparison later.

```bash
python -m pytest -q
python -m stock_api.backtest AMZN --related GOOGL AAPL --lookback 120 --horizon 7 --folds 5
```

Tests use deterministic synthetic prices and a mocked provider; no internet is
needed. They check API validation/errors, holidays and completed-session logic,
GP cross-ticker influence, constant series, and chronological evaluation windows.

The evaluation command needs internet and compares all seven models on successive,
disjoint held-out windows. It refits on only the preceding `lookback` sessions and
reports price MAE, final-session direction accuracy, and observed 95% interval
coverage. Its horizon is always **trading sessions**. Five folds are only a smoke
test; use more history and separate validation/test periods before choosing
models or related tickers. Yahoo's retrospectively adjusted historical prices
are not a point-in-time data archive, and this evaluation is not a trading P&L
simulation (no fees, execution, or corporate-action timing model).

The service is intended for local experiments. Fits are bounded to four stocks,
252 observations, and one concurrent forecast per worker. Larger datasets or
production use warrant a job queue, data cache, and sparse/variational GPs.
