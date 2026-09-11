# Walk-forward model validation

Use `validate` from Python or IPython. It trains fresh models on past data,
forecasts a held-out future block, scores it, and repeats across calendar folds.
The existing `stock_api.backtest` command remains a small legacy evaluation;
this framework adds dated features, model plugins, monthly/weekly schedules,
plots, and persistent experiment comparisons.

## Monthly example

```python
from stock_api import ValidationConfig, validate

config = ValidationConfig(
    reference_date="2026-09-01",  # exclusive cutoff; configurable
    months=12,
    train_days=30,
    horizon=7,
    train_unit="calendar",
    horizon_unit="calendar",
    frequency="monthly",
    seed=42,
)

run = validate(
    "AMZN",
    related_tickers=["GOOGL", "AAPL"],
    config=config,
    models=["last_price", "gp", "multitask_gp", "var"],  # omit to run all built-ins
)
print(run.summary())
run.plot()  # normalized MAE (% of each fold's average observed price)
run.save("validation_runs/amazon-v1.json")
```

The examples use the historical cutoff `2026-09-01`. The downloader rejects
experiments requiring future observations. For offline experiments,
pass your own `prices` DataFrame. There is no download if prices are supplied.

The evaluation year is `[reference_date - 12 months, reference_date)`. Each
monthly forecast origin is the first calendar day of that month. For the
example above there are twelve forecast origins, September 1, 2025 through
August 1, 2026. Each uses the **preceding** 30 calendar days and predicts the
first seven calendar days of the origin month. The February 1 fold trains on
January 2–31 and scores February 1–7 (market sessions only). A 30-day window
is intentionally fixed length; it does not mean a whole calendar month.

A fold trained on August 2026 and forecasting September 2026 would need a
cutoff after the scored September dates. This framework never scores dates at
or beyond the specified cutoff. Mid-month reference dates are allowed; only
origins within the evaluation period with a complete forecast window are used.
`run.folds` records every exact training and test date for inspection. Plot
labels are **forecast origins**, so the February point measures the fit trained
on January's history.

Use `train_unit="trading"` or `horizon_unit="trading"` for counts of actual
XNYS market sessions. `frequency="weekly"` uses Mondays as origins. Holidays
and weekends have no fabricated prices. The default minimum is ten training
observations; insufficient windows fail explicitly. Incomplete forecast windows
at the cutoff are excluded. All requested models must complete all scheduled
folds; a failure stops the experiment with the model name and fold date rather
than silently comparing different subsets. Long forecast horizons can cause
overlapping test windows; the resulting fold errors are not independent.

## Scores and plots

`run.summary()` returns a pandas table, one row per model:

| Column | Interpretation |
|---|---|
| `mean_mae` | Average of fold mean absolute price errors; **equal weight per fold** |
| `pooled_mae` | MAE across all forecast observations; longer folds get more weight |
| `mean_rmse`, `pooled_rmse` | Analogous root mean square price errors |
| `mean_nmae_pct`, `mean_nrmse_pct` | Mean of each fold's MAE/RMSE divided by that fold's mean observed price, multiplied by 100 |
| `pooled_nmae_pct`, `pooled_nrmse_pct` | Pooled MAE/RMSE divided by the mean observed price over all scored forecast points, multiplied by 100 |
| `mean_mape_pct` | Mean fold absolute percentage error |
| `terminal_direction_accuracy` | Correct terminal movement sign among nonzero directional forecasts; flat forecasts are N/A |
| `direction_accuracy` | Compatibility alias of `terminal_direction_accuracy` |
| `directional_folds` | Number of non-flat forecasts used in direction accuracy |
| `terminal_log_return_mae`, `terminal_log_return_mae_pct` | Terminal log-return MAE as a fraction and multiplied by 100 |
| `terminal_brier_score` | Mean squared error of terminal up-event probability; N/A if probabilities were not supplied |
| `coverage_95` | Mean fold coverage of supplied 95% intervals; missing for models without intervals |
| `folds`, `observations` | Number of scored folds and forecast points |

Lower MAE/RMSE/MAPE indicates smaller price errors. Compare to the random-walk
baseline; low price error alone does not establish trading value. Models return
price medians or point forecasts; the framework scores those directly without
assuming all models use Gaussian distributions.

```python
run.plot(metric="nrmse_pct")
ax = run.plot(metric="nmae_pct", show=False)
ax.figure.savefig("errors-by-month.png", dpi=150)
run.summary().to_csv("average-errors.csv")

import pandas as pd
fold_metrics = pd.DataFrame(run.metrics)
individual_predictions = pd.DataFrame(run.predictions)
```

Normalized MAE is `100 * mean(abs(predicted - actual)) / mean(actual)` for
each fold's held-out forecast window. For example, a $6 MAE on an average
observed price of $240 is **2.5%**. Normalized RMSE uses the same denominator.
These scores are unchanged by a uniform rescaling of prices and are more useful
than raw dollar errors when comparing stocks with different share prices.
They do not adjust for differences in volatility or forecasting difficulty.

This differs from MAPE, which divides each individual error by its own observed
price before averaging. The denominator is used only for scoring, never for
training. `mean_actual_price`, `nmae_pct`, and `nrmse_pct` are stored per fold.
The default plot now shows normalized MAE; `metric="mae"` retains the raw-price
view. Loading older JSON records automatically derives the new metrics from
stored predictions without refitting or modifying the original file.

Plots require the existing `plots` extra (`pip install '.[plots]'`). The `gp`
model fits only the target ticker with a Matérn 3/2 kernel; its timescale,
amplitude, and noise are estimated from the target alone. It ignores related
tickers in validation and uses the same dates as the other models. Built-in
GP evaluation uses actual elapsed calendar time as its kernel coordinate,
with a shared Matérn 3/2 timescale and a separate amplitude and noise for each
independent ticker process. Other tickers constrain the timescale only. VAR uses all ticker
returns; the random walk uses only the target. The built-ins ignore additional
non-price features; supply a custom model to use sentiment or other regressors.

## Adding a model

A model implements `fit_predict(train, future, *, rng)`. Wrap a **factory** in
`ModelSpec` so each fold starts from fresh, unfitted state:

```python
import numpy as np
import pandas as pd
from stock_api import ModelSpec, validate

class StochasticDrift:
    def fit_predict(self, train, future, *, rng):
        # Only training prices and available historical features are supplied.
        log_prices = np.log(train.prices.iloc[:, 0].to_numpy())
        returns = np.diff(log_prices)
        samples = log_prices[-1] + np.cumsum(
            rng.normal(returns.mean(), max(returns.std(), 1e-6),
                       size=(2000, len(future))), axis=1)
        paths = np.exp(samples)
        return pd.DataFrame({
            "predicted_close": np.median(paths, axis=0),
            "lower_95": np.quantile(paths, 0.025, axis=0),
            "upper_95": np.quantile(paths, 0.975, axis=0),
        }, index=future.index)

models = {"stochastic-drift-v1": ModelSpec(StochasticDrift, version="1")}
run = validate("AMZN", config=config, prices=prices, models=models)
```

To evaluate built-ins and new models together, start with
`models = builtin_models()` (imported from `stock_api`) and add your named
`ModelSpec` entries to that dictionary before calling `validate`.

`train` is a `TrainingData` instance with:

- `prices`: daily adjusted closes, target column first, then related tickers.
- `features`: numerical regressors available on each training date, including
  `time` (calendar days since 1970-01-01).

`future` is a DataFrame indexed by the dates to predict. By default its only
column is `time`; it contains **no future target or related ticker prices**.
The output must have exactly the same index and a positive finite
`predicted_close` column. Optional `lower_95` and `upper_95` columns must both
be supplied and contain the point forecast. Samples, distribution families,
internal model structure and fitting procedures are otherwise model-specific.

The supplied `rng` has a deterministic seed for each fold, shared across model
runs with the same configuration. Use it for stochastic predictions and derive
seeds from it for third-party estimators. Do not use global unseeded randomness.
Fit scalers, feature selection, imputation parameters and hyperparameters only
on `train`; tuning needs an inner chronological validation loop. A fresh model
is created per fold, but the framework cannot prevent custom code from consulting
external data, mutable global state, or future information in a closure.

Factories may accept parameters through a closure. Record those settings in
`ModelSpec(..., version="2", parameters={"lags": 3})`, and use a distinct model
name such as `sentiment-v2` when comparing versions. Version and parameters are
experiment metadata; the factory itself must apply the parameters. A best-effort
source hash is also saved, but does not replace explicit model versioning.

## Sentiment and other dated regressors

```python
features = pd.DataFrame(
    {"sentiment": sentiment_values},
    index=pd.to_datetime(sentiment_availability_dates),
)
run = validate("AMZN", related_tickers=["GOOGL", "AAPL"], config=config,
               prices=prices, features=features, models=my_sentiment_models)
```

`prices` must cover every required exchange session and ticker. Input indexes
must be unique, timezone-naive daily dates; numeric values must be finite and
prices positive. Features are indexed by **availability date**, not the news
article's event date. Aggregate intraday features into values actually available
by that day's close; information arriving after close belongs to a later date.
Feature alignment forward-fills past values only, including weekend releases;
no backward filling is performed. Missing initial values cause an error.

Future realized sentiment is not handed to a model. The model must forecast it,
use past lagged values, or explicitly assume it stays constant. Only features
known in advance, such as a scheduled calendar event, should be listed in
`known_future_features=["scheduled_event"]`; those columns become available in
`future`. Declaring a feature known in advance is your responsibility and does
not make future observed data safe. Any upstream sentiment model or revised
financial data must also respect point-in-time availability.

## Save today, add a model tomorrow

```python
from stock_api import ValidationResult, compare_results

old = ValidationResult.load("validation_runs/amazon-v1.json")
# Reuse the exact saved data; no new Yahoo download or old-model fitting.
new = old.rerun({"stochastic-drift-v1": ModelSpec(StochasticDrift, version="1")})
new.save("validation_runs/amazon-drift.json")

comparison = compare_results(old, new)
print(comparison.summary().sort_values("mean_mae"))
comparison.plot()
comparison.save("validation_runs/amazon-comparison.json")
```

JSON records store full predictions and observations, per-fold metrics, exact
fold dates, configuration, seeds, model metadata, dependency versions and the
price/aligned-feature snapshot. They contain no pickled executable model code.
A model-independent experiment fingerprint covers data, configuration and folds.
Comparison requires matching fingerprints and distinct model names; it refuses
to silently compare new downloads, different windows, seeds or tickers. Saved
files are never overwritten. Replaying a snapshot requires the new model code
and a compatible calendar/dependency environment; the recorded fingerprint will
reveal changed fold/data semantics. Keep environment locks and explicit model
versions for long-term reproducibility.

## Runnable demonstration

From the repository root after installing the package and plotting extra:

```bash
# Observed Yahoo data, all seven built-ins, 12 monthly folds:
python examples/validation_experiment.py --reference-date 2026-09-01

# The same historical cutoff, using explicitly synthetic demonstration data:
python examples/validation_experiment.py --reference-date 2026-09-01 --synthetic
```

Each execution saves a uniquely named JSON record, summary CSV, and PNG under
`validation_runs/`. Adjust `--ticker`, `--related`, `--models`, `--train-days`,
`--horizon`, `--months`, `--seed`, and `--output` as needed.

Historical Yahoo prices are retrospectively adjusted, not a point-in-time archive.
This is a forecasting evaluation, not a transaction-cost or execution simulator.
Repeatedly choosing the best model on these folds turns them into a validation
set; reserve a separate untouched period for final performance claims.

For exact-transition GBM, OU returns, Monte Carlo ensembles, and the generic
Euler–Maruyama solver, see [SDE.md](SDE.md). These models are included when
`models` is omitted. Terminal probabilities supplied by models also produce
Brier scores; the 56 daily observations in the example remain correlated within
12 fold trajectories, not 56 independent terminal events.
