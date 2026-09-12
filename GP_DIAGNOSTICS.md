# GP inference diagnostics on a saved fold

Completed experiment: [written findings](diagnostics/AMZN/2025-11-01/REPORT.md),
[plot gallery](diagnostics/AMZN/2025-11-01/index.html), and
[artifact verification](diagnostics/AMZN/2025-11-01/verification.json).
Refresh the report and verify existing samples without fitting or sampling:

```bash
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python examples/gp_diagnostic_report.py
```

This is an opt-in inference experiment, not a validation run. Existing production
optimizers, priors/regularizers, and saved benchmark scores are unchanged.

```bash
python -m pip install -e '.[diagnostics]'
OPENBLAS_NUM_THREADS=1 python examples/gp_diagnostics.py \
  --origin 2025-11-01 --lookbacks 30 60 90 180 --output diagnostics
```

The default source is `validation_runs/return-gp-2026-09-01-benchmarks`.
No download or backtest is performed. The origin is selected because the old
volatility estimates approach the two-day bound. It is one of the existing
September-2026-cutoff folds; all four lookbacks use exactly that same origin.

Use the saved validation interface from Python:

```python
from stock_api import ValidationResult, DiagnosticConfig, PriorConfig

run = ValidationResult.load("validation_runs/return-gp-2026-09-01-benchmarks/lookback-60.json")
settings = DiagnosticConfig(
    profile_surfaces=True, posterior_sampling=True, multistart=True,
    latent_plots=True, corner_plots=True, starts=30,
    priors=PriorConfig(length=(0.5, 300),
                       return_mean_amplitude_ratio=(1e-3, 10)),
)
root = run.diagnose("2025-11-01", config=settings,
                    models=["gp", "multitask_gp", "volatility_gp_returns",
                            "heteroskedastic_gp_returns"], output="my-diagnostics")
```

`diagnose_fold(run, origin, ...)` is the equivalent standalone function.
`enabled=False` disables the diagnostic call. Sampling, profiles/surfaces,
multistart and optional plot groups can be independently disabled.
The CLI accepts `--config SETTINGS.json` using `DiagnosticConfig`'s JSON schema.
Existing directories are refused unless `resume=True` / `--resume` is passed.
Resume checks the configuration, snapshot identity and training-data checksum
before reusing results. Completed sampling/profile artifacts survive a later
plotting interruption. This does not change ordinary `predict()` behavior.

## Parameterization and priors

The audit found that all four existing GP-style models **already** optimize
positive covariance parameters as logarithms: lengths, amplitudes, white noise,
and variational precision. Signed latent means remain signed. Diagnostic
coordinates follow that convention; CSV summaries and plot axes show physical
values. Files explicitly named `coordinates` retain internal coordinates.

| Parameter | Explicit diagnostic prior |
|---|---|
| Any length, trading sessions | log-uniform [0.5, 300] |
| Ordinary GP amplitude / training standard deviation | log-uniform [0.001, 100] |
| Ordinary white-noise amplitude / training standard deviation | log-uniform [0.0001, 10] |
| Return mean amplitude / training return standard deviation | log-uniform [0.001, 10] |
| Latent log-variance amplitude | log-uniform [0.01, 3], dimensionless |
| Latent log-variance mean | Normal centered on log training return variance, SD 2, truncated at ±5 SD |

The positive prior ranges, signed-level SD and truncation width are configurable in
`PriorConfig`. Numerical jitter is inherited from the production model.
The log-variance amplitude prior spans nearly constant volatility through
order-unity fluctuations in **log variance**, rather than sharing a dollar or
return-amplitude prior. A one-SD log-variance change of 3 corresponds to a
volatility multiplier `exp(3/2)`.

The existing ordinary GPs actually fit **log prices**, not raw USD prices.
Their scale and reported amplitudes are therefore in log-price units:
`A / std(log(S_train))`. Calling those amplitudes dollars would be incorrect.
For return models the mean-amplitude prior uses `std(r_train)` even though the
existing variational code internally divides returns by their RMS. The adapter
converts the prior to that internal scale; it never centers zero-mean returns.
All empirical scales, centers, prior levels and fits use only training data.

The old validation path uses **calendar-day** coordinates for ordinary price
GPs. The diagnostic experiment uses trading-session coordinates for every model
so the stated length prior has a consistent interpretation. The red marker is
the existing optimizer run on those trading coordinates. The old calendar-day
fit is saved separately in `old-optimizer.json`; its lengths should not be
compared numerically to trading-day lengths. Return-model time units are unchanged.

## What each inference method means

For `gp` and `multitask_gp`, dynesty samples the exact analytic Gaussian marginal
likelihood with explicit prior transforms. Static nested sampling defaults to
150 live points and `dlogz=0.1`. Samples, weights, logZ, estimated logZ error,
likelihood calls and weighted effective sample size are retained. Hitting the
call budget is a failure of the stopping check, not successful convergence.
The evidence is the density of observed log prices conditional on the existing
training-derived centering and scale. Do not compare single-stock and multitask
logZ as a Bayes factor: their likelihoods contain different observed tickers.

For both return GPs, posterior sampling **does not use the ELBO**. It targets the
joint Bayesian latent model under the specified priors:

```
g = m_g + L_g z,  z ~ Normal(0, I)
y | g, theta_mu ~ Normal(0, K_mu + diag(exp(g)))
```

The mean function is integrated out analytically conditional on `g` (or is
identically zero in the volatility-only model). Elliptical slice sampling
updates `z`; bounded univariate slice updates sample the log hyperparameters
and signed level, including its Normal prior density. No hyperparameter MAP
penalties enter this sampler. Conditional Gaussian draws recover `f` for latent
plots and predictions. This is proper latent-variable MCMC, not an exact
closed-form heteroskedastic marginal likelihood and not a variational evidence
hyperposterior. There is no returned heteroskedastic logZ.

Defaults are four independently initialized chains, 1,000 warmup iterations,
and 2,000 retained draws per chain. Split R-hat, autocorrelation ESS and trace
plots are saved. `converged` requires all hyperparameter split R-hats <1.05 and
ESS >100; this is a diagnostic threshold, not a proof of convergence. Inspect
traces and joint structure as well as scalar thresholds. References:
[elliptical slice sampling](https://proceedings.mlr.press/v9/murray10a.html) and
[dynesty sampling/results](https://dynesty.readthedocs.io/en/stable/quickstart.html).

## Profiles, multistart and plot interpretation

Profiles use 41 evenly spaced **log-coordinate** values over each full positive
prior range. Signed `m_g` is profiled on a linear grid. At each point, two starts
reoptimize the remaining free parameters, including variational means and
covariance coordinates for return GPs. Convergence status and all fitted
hyperparameters are retained per grid point.

Ordinary-GP profiles show relative log marginal likelihood. Return-GP profiles
show relative **unregularized variational ELBO**, with the original penalties
subtracted exactly. The latter is an evidence lower bound, not the likelihood
used by the Bayesian sampler. Thus disagreement can arise from both posterior
volume effects and the variational approximation. Profiles omit the signed-level
prior density and distinguish objective information from prior influence.

The 25×25 2D grids are explicitly labeled **conditional slices**. Other
hyperparameters and, for return GPs, the whitened variational mean/precision
coordinates are fixed at the best profile solution. They are not profiled
marginals. Full 1D reoptimization and Bayesian contours complement those slices.
Colors show relative log likelihood/ELBO, clipped at −30 for readability; the CSV
retains untruncated numerical values. Black contours enclose approximately
68/95% posterior mass in bins of log-positive coordinates. Markers indicate
old optimizer, best profile solution, and posterior median.

Multistart runs 30 broadly prior-drawn starts for the old objective/bounds and
another 30 with the same starts for the broad-domain likelihood/ELBO objective.
Starts outside old bounds are projected into the old feasible region; both raw
and feasible initial values are saved. Old regularized and new unregularized
objective values have separate baselines. Red crosses identify unsuccessful
optimizations; distinct numerical endpoints on a flat ridge are not necessarily
distinct statistical modes.

Corner plots use physical units on logarithmic axes for positive parameters,
weighted histograms, median, and 68/95% intervals. MCMC trace plots retain chain
identity. Forecast plots compare hyperparameter-marginalized predictions with
the old point fit; held-out prices are drawn only after inference. Latent plots
show training returns, posterior mean-return function and volatility, 68/95%
credible bands, forecast shading, and separately marked held-out returns.

An amplitude below `0.1 * training scatter` is the default negligible-mean
threshold. The posterior probability of that event is reported. A mean process
is marked active only if this probability is below 5%; otherwise its length
must not be interpreted as a measured timescale. Ordinary GP amplitude
inactivity is also reported relative to its own training log-price scale.
A sub-day length indicates unresolved daily correlation, and a length approaching
300 sessions can describe an effectively constant process in these windows.

As in the return-model implementation, uncertainty in log variance produces
very heavy price tails. Price moments may not exist in the population; medians,
quantiles and log-return diagnostics are used for the predictive comparison.

## Saved outputs

Each `diagnostics/AMZN/2025-11-01/lookback_N/MODEL/` contains:

- `config.json`, training-only checksum, `training.csv`, `held-out.csv`;
- old optimizer (and calendar-day comparator where relevant);
- multistart endpoints and status; profile/surface CSVs and plots;
- physical posterior samples/weights, internal samples and latent states;
- all-parameter posterior summary, sampling status and evidence where available;
- corner/chain plots, marginalized forecast draws/plot, and latent plots;
- a completion marker for safe resumption.

The experiment report links these primary visual artifacts. This machinery
examines one origin; it does not establish performance over other origins or
license a large backtest before the inference limitations are understood.

## Extending poorly mixed chains

```python
from stock_api.diagnostics.refine import refine_posterior
refine_posterior("diagnostics/AMZN/2025-11-01/lookback_60/heteroskedastic_gp_returns",
                 warmup=1500, draws=5000, interweave=True)
```

This keeps the original samples and profiles intact and writes new results into
`refined/`. Centered/noncentered interweaving adds hyperparameter slice updates
that hold physical `g` fixed. Their target includes its Gaussian prior density
(and the Cholesky log determinant), followed by rewhitening. Both update blocks
target the same posterior; this is a mixing improvement, not a different model
or a replacement likelihood. Set `DiagnosticConfig(interweave=True)` to use
it in an initial run. Refinement uses independent starts and seed 143.
