# Historical walk-forward validation

This runner evaluates existing models and saves ticker-specific reliability evidence.
It does not trade, allocate capital, select a winning model, or tune inference against
realized outcomes. The full AMZN experiment is prepared but has not been started or
scheduled by Codex.

## Frozen first experiment

Configuration: [configs/historical-amzn-v1.json](configs/historical-amzn-v1.json).
Target AMZN, auxiliaries GOOGL/AAPL, January 1, 2016 through September 1, 2026
(exclusive), weekly origins, seven-calendar-day horizons, and the frozen
30/60/90/180-calendar-day lookbacks. All nine existing models are included.
There are **556 origins × 4 lookbacks × 9 models = 20,016 units** and **96 planned
model/lookback/loss comparisons** against persistence (8 × 4 × 3).

An origin is **before the opening of the first exchange session of the week**.
Training closes satisfy `date < origin`; forecast closes satisfy
`origin <= date < origin + horizon_days`. The first observation at the origin is
therefore a forecast target, not a training input. The starting price is the last
training close. Exchange holidays do not create duplicate origins. Holiday shifts
can cause adjacent forecast episodes to overlap; statistical comparisons resample
blocks of episodes and do not count individual forecast-path points as independent.
The first partial week of a custom date range starts at its first eligible session.

All model inputs are copied from the current training window. There is no global
normalization, auxiliary future-price input, forward/backward filling, or feature
engineering. Missing training data fail that unit explicitly; missing held-out
prices preserve the forecast but mark the unit unscorable. Missing auxiliary prices
affect multivariate models, not target-only models. Data availability never moves
an origin retrospectively. The saved data-quality metadata identifies usable dates
and missing counts rather than concealing gaps by dropping folds.

**Data limitation:** Yahoo's current adjusted-close histories are retrospectively
adjusted and may be revised. They are not point-in-time provider vintages. The runner
enforces timestamp isolation on the supplied snapshot, but cannot prove those exact
adjusted values were available historically. The default experiment is explicitly
labeled `retrospective_adjusted`: development evidence, not a fully point-in-time
claim. A user-audited point-in-time dataset can be supplied with `prices=` or
`--prices-csv` and the corresponding provenance setting. The software cannot certify
that external provenance by itself. No full run was launched to work around this
limitation.

## Python interface and ticker portability

Install the optional dependencies with `python -m pip install -e '.[historical]'`.
The existing `.conda-env` is ready; it now includes the lightweight `threadpoolctl`
dependency for bounded BLAS execution.

```python
from stock_api import HistoricalConfig, run_historical_validation

settings = HistoricalConfig(
    target_ticker="NVDA",
    auxiliary_tickers=["MSFT", "AAPL"],
    start_date="2018-01-01",
    end_date="2026-09-01",
    lookbacks=[30, 60, 90, 180],
    forecast_horizon_days=7,
    origin_frequency="weekly",  # also monthly or daily
)
# This call starts numerical work; it is an example, not executed during setup.
run_historical_validation(settings, output="historical_validation/NVDA/experiment-v1", workers=1)
```

Alternatively pass these settings directly as keyword arguments alongside `output`.
Targets, auxiliaries, exchange calendar, dates, horizons, models, and lookbacks are
configuration, not AMZN-specific model assumptions. The default exchange calendar
is XNYS; use an appropriate supported calendar for other markets. Prices must be
positive daily closes in their quote currency. Normalized errors are invariant to
price-unit scaling; GP priors adapt using the fold's own target/ticker scatter.

For a protected chronological development boundary, configure either/both:

```json
{"development_end": "2024-01-01", "untouched_holdout_start": "2024-01-01"}
```

Dates are exclusive boundaries. The earlier applicable boundary restricts downloads,
saved snapshots, origins and scored horizons. No final holdout has been designated
for the frozen 2016–2026 experiment; once inspected, that period is development data.

## Inference modes

`production_validation` is the historical runner. Gaussian price GPs use the existing
exact Gaussian likelihood and dynesty hyperparameter marginalization. Return GPs
use the existing joint latent sampler with centered/noncentered interweaving and
conditional mean-function integration. Length priors remain log-uniform [0.5, 300]
in trading-session units, with the existing model-specific amplitude and signed
level priors. Positive parameters remain in log coordinates. The historical path
never invokes the old bounded point optimizer, including as an initialization or
predictive comparator. Ordinary forecast APIs and existing diagnostic defaults are
preserved.

Frozen defaults: 150 live points, dlogz=0.1, maximum 200,000 nested likelihood calls;
four latent chains, 1,500 warmup iterations and 5,000 retained draws per chain;
2,000 posterior predictive paths per unit. Profiles/multistarts are disabled.
A return-GP fit requires maximum split R-hat <1.05 and minimum autocorrelation
ESS >100. The nested flag records whether the call budget was reached. These are
the existing safeguards, not proof of convergence or predictive-tail accuracy.
Failures are **not automatically retuned or given a different prior**.

`diagnostic` is the separate `diagnose` command below. It enables the existing
profiles, multistart fits, corner plots, latent plots and detailed posterior checks
for an explicitly selected saved fold. It uses the historical unit's deterministic
seed and the same training snapshot/priors; legacy optimizer comparisons remain
available only inside this diagnostic mode.

Non-GP models retain their existing conditional/plug-in inference. They propagate
innovation uncertainty but do not gain a new parameter posterior. The report makes
this distinction explicit. The `last_price` point forecast is exactly flat and
abstains on direction; its existing random-walk uncertainty supplies a 0.5 event
probability and probabilistic null for comparison. `gbm_zero_drift` means zero
arithmetic-price drift, which has a negative half-variance log drift.

Return-mixture price moments may not exist; those path means are stored as null
with a reason. Medians, return statistics, quantiles and finite predictive samples
are retained. Population zero-mean volatility-GP median/P(up) are exactly persistence/
0.5, rather than finite-Monte-Carlo apparent alpha. Other Monte Carlo quantities
have sampling error. No unstable density estimate is forced onto mixtures.

## Unattended full run and resume

**These commands start the full experiment. They have not been executed.**
Run from any ordinary terminal; no Codex, OpenAI API key or agent is needed.
The launcher uses one worker and one BLAS thread. It appends stdout/stderr to a
persistent log, disconnects stdin, and survives a terminal hangup with `nohup`.

```bash
cd /Users/newen/github/stocks && mkdir -p historical_validation/AMZN/2016-2026-v1 && nohup /bin/bash scripts/run_historical.sh configs/historical-amzn-v1.json historical_validation/AMZN/2016-2026-v1 1 >> historical_validation/AMZN/2016-2026-v1/run.log 2>&1 < /dev/null &
```

**Resume uses the identical command.** It skips every valid saved successful unit
and, by default, every recorded failed attempt. A second concurrent runner is
rejected by an OS file lock, including when launched from another terminal.
A killed worker's uncommitted unit remains pending. The last completed unit is
never dependent on an in-memory aggregate.

The first run downloads and atomically freezes the price snapshot and environment/
source fingerprints. Subsequent runs require the same configuration, snapshot,
Python/package versions, inference source hashes, and git commit. Finish commits
and environment changes **before** starting the experiment. After interruption,
restore that environment/code instead of bypassing the identity checks. Use a new
output directory for an explicit clean restart or a changed experiment; successful
forecasts are never silently replaced. Failed retries preserve the previous attempt
under `failures/`; invoke them explicitly:

```bash
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python -u -m stock_api.historical run \
  --config configs/historical-amzn-v1.json --output historical_validation/AMZN/2016-2026-v1 \
  --workers 1 --retry-failures --report
```

With unchanged deterministic seeds, retrying an actual convergence failure is not
expected to fix it. This option primarily recovers transient errors. Changing
inference settings requires a new experiment and disclosure, not relabeling the
same historical results as untouched evidence.

### Lightweight progress

```bash
cd /Users/newen/github/stocks && .conda-env/bin/python -m stock_api.historical status --output historical_validation/AMZN/2016-2026-v1
```

This reads only the manifest and heartbeat, not all forecast distributions. A
heartbeat every 30 seconds and every unit reports completed/total, current unit(s),
failures, elapsed time and completion fraction. Inspect the timestamp for freshness;
a completed fraction of 1 includes failed attempts. `--verify` additionally scans
and checksums all saved records. Before initial download completes there may not yet
be a manifest; the persistent log is then the source of startup status:

```bash
tail -n 20 /Users/newen/github/stocks/historical_validation/AMZN/2016-2026-v1/run.log
```

### Report generation after inference

The launcher automatically generates the report after the runner finishes. To
regenerate it later or view a partial run, without any fitting or downloads:

```bash
cd /Users/newen/github/stocks && MPLBACKEND=Agg .conda-env/bin/python -m stock_api.historical report --output historical_validation/AMZN/2016-2026-v1
```

Open `historical_validation/AMZN/2016-2026-v1/index.html` in a browser. Partial reports
show pending units separately from failures and successful fits.

### Scheduling for 2:30 AM

The host has `/usr/bin/at`, but its shipped `com.apple.atrun` service configuration
has `Disabled = true`. No service was enabled and no job was submitted. An `at`
example, **only if the host's scheduler is operational**, is:

```bash
printf '%s\n' 'cd /Users/newen/github/stocks && mkdir -p historical_validation/AMZN/2016-2026-v1 && /bin/bash scripts/run_historical.sh configs/historical-amzn-v1.json historical_validation/AMZN/2016-2026-v1 1 >> historical_validation/AMZN/2016-2026-v1/run.log 2>&1' | /usr/bin/at 02:30
```

Confirm the date/time that `at` echoes; if today's 02:30 has passed, select the next
intended date. `/usr/bin/atq` lists queued jobs. Having `at` installed alone does not
prove its service is running.

The `nohup` launch above is the fallback to run manually at the intended time.
For a `nohup` process that waits until the **next local 02:30** without using `at`:

```bash
cd /Users/newen/github/stocks && mkdir -p historical_validation/AMZN/2016-2026-v1 && nohup .conda-env/bin/python scripts/start_at_local_time.py --time 02:30 --config configs/historical-amzn-v1.json --output historical_validation/AMZN/2016-2026-v1 >> historical_validation/AMZN/2016-2026-v1/run.log 2>&1 < /dev/null &
```

This is an example only and has not been started. `nohup` does not survive a reboot
or guarantee execution while the computer sleeps; neither launcher manages power
settings. Numerical work can substantially outlast a Codex usage window and does
not consume Codex tokens. Full-budget MCMC on 90/180-day windows can be much slower
than a 30-day smoke fit; do not extrapolate a completion date from the short smoke.

## Saved artifacts and reproducible analysis

Each `records/ORIGIN__MODEL__LOOKBACK.json` is an atomic, checksummed unit containing
identity/version/seed, training and horizon dates, training checksum, starting and
realized prices, actual terminal return/direction, path means/medians where defined,
terminal expected/median return and sigma, price and return quantiles for
50/68/90/95%, P(up), 2,000 terminal predictive samples, posterior hyperparameter
summaries, component activity/edge masses, convergence checks, warnings/fallbacks,
errors and runtime. Failed inference forecasts remain inspectable but are excluded
from successful-fit metrics. Original posterior chains are not stored at scale;
selected detailed diagnostics can reproduce the fold using the frozen snapshot.

Reports use only those records and the frozen manifest. Outputs include:

- `summaries/overall.csv`, `yearly/metrics.csv`: path and terminal losses, normalized
  errors, paired skill, calibration, failure/pending counts and runtime.
- `summaries/forecast-scores.csv`: paired per-origin losses, standardized errors,
  PIT, direction abstentions, confidence and retrospective regime labels.
- `summaries/paired-comparisons.csv`: circular moving-block bootstrap mean loss
  difference, 95% interval, descriptive standardized effect and centered-bootstrap
  p-value. Blocks span eight weeks on the full origin grid, preserving missing units.
  Fewer than max(20, two blocks) successful paired origins yields no inference.
  Holm adjustment accounts for all 96 planned tests, treating unavailable tests
  conservatively. The effect denominator is the sample SD of paired differences,
  not an assumption that episodes are independent.
- `summaries/cumulative-loss.csv` and full-period/annual plots: model minus persistence
  losses; negative is better. Gaps remain visible when pairs are unavailable.
- `calibration/reliability-bins.csv`: decile probability bins merged to minimum
  counts, mean model probability, empirical positive frequency, realized return,
  error and sparse-bin flag. ECE is explicitly a bin-dependent descriptive measure;
  fitted logistic calibration intercept/slope are omitted when not identifiable.
- `calibration/confidence-bins.csv`: confidence quantiles retaining ties; directional
  count/accuracy, standardized error, signed realized return, Brier and coverage.
- `calibration/coverage-strata.csv`: 50/68/90/95% terminal/path coverage by year,
  predicted sigma, realized volatility, confidence, trailing volatility and return
  environment. Regime/quantile thresholds are post-hoc summaries only.
- PIT histograms and time series; standardized residual histograms for mixtures and
  Normal QQ-style diagnostics for conditional Gaussian forecasts. Aggregate PIT
  moments, extreme-tail frequency, standardized-error mean and variance are saved.
- `summaries/disagreement.csv` and `disagreement-bins.csv`: return/probability/risk
  disagreement versus subsequent error, Brier and realized volatility. Bin analyses
  use complete model panels, with available/expected model counts retained for all
  panels so failure-induced changes in composition are visible.
- `summaries/episodes.csv`: strongest positive/negative paired-loss episodes and
  their record identifiers. `inference-details.json` includes a selected-fold
  diagnostic command, statuses, hyperparameters, failures and runtime evidence.
- `summaries/ticker-reliability.json`: common cross-ticker schema preserving all
  accuracy, calibration, risk, confidence, regime and disagreement components.
  There is deliberately no arbitrary combined confidence or allocation score.

The main coverage column is **terminal-return coverage**, whereas the existing
benchmark's path-wise coverage is retained as `path_coverage_95`. Return errors are
dimensionless log returns; multiply by 100 for log-return percentage points.
Price losses use quote currency; normalized errors use percentages. Point losses
score medians to retain the existing benchmark convention; expected returns are
saved separately. Analytic negative log scores are recorded only for conditional
Gaussian terminal log-return models (and can be negative because densities can
exceed one). Non-Gaussian GP mixtures use PIT/quantile diagnostics instead.

## Selected detailed GP diagnosis

```bash
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python -m stock_api.historical diagnose \
  --output historical_validation/AMZN/2016-2026-v1 \
  --origin 2020-03-02 --lookback 90 --model heteroskedastic_gp_returns
```

This explicitly reruns only the requested fold using existing detailed diagnostics;
it is expensive and not part of report generation. `--model` can be repeated for
other GP models. The output is `diagnostics/TICKER/ORIGIN/lookback_N/MODEL/` under
the experiment root, and existing detailed diagnostics support their own resume.

## Verification performed in this implementation session

The full suite passed **155 tests**. Twenty-two focused historical tests cover generic
tickers, holiday origins, holdout exclusion, no future leakage, deterministic seeds,
worker-count equivalence, failure retries, partial/checksum/atomic persistence,
concurrent-run locks, report-only regeneration, normalized scale invariance,
production GP bypass of old optimizers, diagnostic bridging and paired statistics.
Zero-variance distributions omit continuous-uniform PIT, log density and standardized
residual scores where these are undefined; their event probabilities and interval
coverage remain available.

The two-origin, all-nine-model smoke uses existing saved real AMZN data and reduced
sampling budgets: 18 atomic units, 12 successful and six explicitly failed
convergence checks. Those settings are confined to the smoke and are not the full
experiment settings. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python examples/historical_smoke.py
```

A separate **one-origin** smoke exercises the frozen production sampling budgets:

```bash
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python examples/historical_smoke.py --production
```

All nine models passed their configured checks in the production-budget smoke.
Browse its [report/gallery](historical_validation/smoke/AMZN-production-one-origin-v2/index.html)
or the [reduced-budget failure-handling smoke](historical_validation/smoke/AMZN-two-origins-v2/index.html).

Both commands reuse saved real data without downloading; they resume their own
small directories under `historical_validation/smoke/`. No full-interval inference
or scheduling command was executed during implementation.
