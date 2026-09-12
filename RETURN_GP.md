# Latent Gaussian processes for daily log returns

Both models use the existing Python, HTTP, plotting, and walk-forward validation
interfaces, with NumPy/SciPy only. No installation changes are needed.

```python
from stock_api import predict

result = predict("AMZN", model=["last_price", "gbm_zero_drift",
                               "volatility_gp_returns", "heteroskedastic_gp_returns"],
                 lookback=120, horizon=7, seed=42, n_paths=10_000)
result.plot(history_days=30, sigmas=(2,))
model = result["heteroskedastic_gp_returns"]
print(model.diagnostics["ell_mu"], model.diagnostics["ell_sigma"])
print(model.diagnostics["expected_terminal_log_return"])
print(model.forecasts[-1].probability_up)
terminal_log_returns = model.terminal_log_returns
```

These are target-only models. Multivariate models in a comparison can still use
`related_tickers`. As before, `predict(lookback=...)` counts trading sessions;
`ValidationConfig(train_days=...)` defaults to calendar days.

## Models and time units

For adjacent completed trading closes, `r[t] = log(S[t]/S[t-1])`.
One time step is one trading session, including a Friday-to-Monday return.
Kernels use session indices, so all fitted lengths are in **trading days**.

- `volatility_gp_returns`: `r[t] | g[t] ~ Normal(0, exp(g[t]))`.
- `heteroskedastic_gp_returns`: `r[t] | f[t],g[t] ~ Normal(f[t], exp(g[t]))`.

Both latent processes have Matérn-3/2 covariance
`A² (1 + sqrt(3)*distance/ell) exp(-sqrt(3)*distance/ell)`.
The volatility GP has a learned constant mean `m_g`; the mean-return GP has
prior mean zero. Its amplitude is regularized toward zero. `ell_mu` and
`ell_sigma` are separately optimized, with neither tied to the other.

**Zero log return is different from zero arithmetic-price drift.** The volatility
model has exactly the persistence median price and `P(up)=0.5`, whatever its
volatility forecast. The existing `gbm_zero_drift` instead has zero arithmetic
GBM drift, so its expected log return is `-sigma²/2` per session. Neither model
is changed to hide this distinction. Validation continues to score **median**
prices, not arithmetic price means.

## Joint variational inference

The implementation follows the collapsed Gaussian variational approximation of
[Lázaro-Gredilla and Titsias (2011)](https://icml.cc/2011/papers/456_icmlpaper.pdf).
It optimizes a common heteroskedastic likelihood, not a separate regression of
squared residuals. For scaled returns `y`,

```
q(f,g) = q(f) q(g)
q(g) = Normal(m, V)
R = diag(exp(m - diag(V)/2))
ELBO = log Normal(y | 0, K_f + R) - trace(V)/4 - KL(q(g) || p(g))
```

The zero-mean model sets `K_f=0` and has no inferred `f`. For the joint model,
`q(f)` is optimized analytically: mean `K_f (K_f+R)^-1 y`, covariance
`K_f - K_f (K_f+R)^-1 K_f`. The log-variance posterior has a **full covariance**
`V=(K_g^-1+diag(lambda))^-1`, parameterized using positive `lambda`. Its mean is
whitened as `m=m_g+L_g u`. The numerical optimization learns `u`, `lambda`,
`m_g`, and both kernel hyperparameter sets jointly. Analytic derivatives are
checked against finite differences in tests.

This is Gaussian mean-field **approximate** inference. It omits posterior
coupling between `f` and `g`, non-Gaussian latent-posterior structure, and
hyperparameter uncertainty. Hyperparameters maximize a regularized ELBO;
they are not integrated out in a fully Bayesian treatment.

Returns are divided by their training-only RMS, with no centering and a floor
of `1e-6`. All output returns are transformed back to original units. No data
outside a fold enter the RMS, likelihood, initialization, or optimization.
The fixed regularization penalties in scaled coordinates are:

```
0.5*A_sigma² + 0.5*(m_g/2)²
+ 0.5*((log(ell_sigma)-log(20))/1.5)²
+ [0.5*(A_mu/0.3)² + 0.5*((log(ell_mu)-log(20))/1.5)²]  # joint model only
```

Lengths are bounded to `[2,252]` sessions, volatility amplitude to `[0.03,3]`,
mean amplitude to `[0.01,2]` scaled-return units, and scaled `m_g` to `[-6,3]`.
Covariances have `1e-6` diagonal jitter in scaled latent units. Whitened latent
means are bounded to `[-8,8]`, and log variational precision to `[-12,6]`.
Two fixed initializations are tried; the converged solution with the better
training objective is used. Convergence requires optimizer success and a
projected-gradient maximum below `0.02`. There is no silent fallback: a failed
fit raises a controlled error, and validation identifies its model and fold.
The helper supports 8–512 daily training returns. These fixed settings were
chosen using synthetic checks, before inspecting the historical results.

Small fitted amplitudes mean a timescale may be effectively unidentified. In
particular, a reported length near the regularization center of 20 sessions
with amplitude at its floor is **not evidence** of a 20-day stock cycle.

## Predictive distributions and plots

For each path, jointly sample latent functions at future sessions, then daily
conditionally Gaussian returns, and reconstruct prices as
`S[h] = S[0] * exp(sum(r[1:h+1]))`. Temporal GP covariance is retained; these
are not independently sampled price marginals. The default is 10,000 paths.
Antithetic return-noise pairs reduce simulation error (5,000 latent-volatility
draws for 10,000 paths); paths are therefore not independent Monte Carlo draws.
A fixed seed reproduces the ensemble. Forecast medians use exact log-return
symmetry; up probabilities integrate Gaussian return noise conditional on
sampled volatility. The zero-mean model reports `P(up)=0.5` exactly.

Forecast points expose `mean_price`, `median_price`, `price_std`, `lower_95`,
`upper_95`, and `probability_up`. The full terminal log-return samples remain
in `terminal_log_returns`. Diagnostics expose `expected_terminal_log_return`
and `terminal_log_return_sigma` in **dimensionless log-return units**. The
latter uses the posterior predictive log-return variance, including mean-GP
uncertainty and innovation noise. `ell_mu`, `ell_sigma`, amplitudes, original-unit
log-variance mean, optimizer status, projected gradient, and training latent
means/expected variances are also retained in diagnostics and saved runs.

The 95% intervals are empirical price quantiles. Plot `sigma_1/2/3` bands use
quantiles at `Phi(-n)` and `Phi(n)` (68.27%, 95.45%, 99.73% central probability),
**not** Gaussian log-price error bars. The historical fit is a one-step return
fit converted from each observed previous close, conditioned on the entire
training window. It is an in-sample fit, not a backtest.

**Heavy-tail limitation:** Gaussian log-variance uncertainty makes returns a
lognormal variance mixture. Its return variance is finite, but after exponentiating
the return the population price mean and variance are infinite. Consequently,
`mean_price` and `price_std` are explicitly finite-ensemble Monte Carlo summaries,
not estimates of finite population moments. They can depend strongly on path
count and rare draws. Medians, quantiles, up probabilities, and log-return moments
remain well-defined. The implementation does not truncate posterior tails to
manufacture finite price moments; numerical path overflow fails explicitly.

## Historical experiment

```bash
OPENBLAS_NUM_THREADS=1 python examples/sde_comparison.py \
  --reference-date 2026-09-01 \
  --lookbacks 30 60 90 180 \
  --prices validation_runs/sde-2026-09-01-benchmarks/prices.csv \
  --output validation_runs/my-return-gp-run \
  --models last_price gp multitask_gp var gbm_zero_drift \
           gbm_estimated_drift ou_returns volatility_gp_returns heteroskedastic_gp_returns
```

The output directory must be new. Single-thread BLAS is useful for these small
matrices and makes timing more predictable; it does not change the model.
The script saves the input snapshot, immutable JSON runs (including paths and
latent diagnostics), full summary CSVs, per-fold scores, per-fold diagnostic
CSVs, and error plots. All existing scoring is preserved. Summary aliases
`terminal_return_mae` and `brier_score` mean the existing
`terminal_log_return_mae` and `terminal_brier_score`; the return MAE is **log**
return, not simple return. The compact report multiplies return MAE by 100.

See [the saved comparison and diagnostic report](validation_runs/return-gp-2026-09-01-benchmarks/REPORT.md).
There are 12 folds with 56 prices per model; within-fold prices are correlated,
and lookbacks reuse the same targets. Neither 56 points nor 48 fold/lookback
combinations constitute independent trials. These results support model
development, not a claim of predictive alpha. Adjusted historical closes also
remain a retrospectively adjusted snapshot, not a point-in-time archive.
