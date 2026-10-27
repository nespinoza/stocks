# Architecture inspection and decisions made before implementation

The input is the **22-name recommendation table** in
`reports/monthly-stock-screen-2026-10-18/REPORT.md`, not its 100-name table or scores.
The parser will require exactly 22 distinct symbols. The existing cached Yahoo
OHLC, adjusted closes, dividends and splits cover these names and SPY.

## Existing GP implementation

- `stock_api/heteroskedastic.py`: `_kernel` is Matérn 3/2; `_objective` implements
  the Gaussian return likelihood with two latent processes, mean `f` and log
  variance `g`. The variational approximation is `q(f)q(g)`, with `q(f)` collapsed.
- `fit_latent_returns` scales returns using training-only RMS and optimizes a
  regularized ELBO with two L-BFGS-B starts and analytic gradients. Mean amplitude
  is strongly shrunk toward zero. It rejects unsuccessful fits. Kernel amplitudes,
  length scales and the log-variance level are fixed within a forecast and refitted
  each rebalance; latent mean and log variance vary through future time.
- `_future` returns the full future means and covariance matrices for both latent
  processes. `_root`, `_return_draws` and `paths_from_returns` implement Gaussian
  latent draws, conditional Gaussian innovations and cumulative log-price paths.
  Future days are correlated through latent kernels; scaling a daily return by
  the horizon would lose that dependence.
- `heteroskedastic_gp_returns` estimates drift and volatility.
  `volatility_gp_returns` fixes latent drift and its uncertainty to zero while
  estimating volatility. Both use the same fitting and prediction machinery.
- `stock_api/diagnostics/{problems,sampling,predictive}.py` also supports Bayesian
  MCMC. This experiment uses the standard variational machinery, as in the recent
  22-ticker GP ranking; it does not integrate hyperparameter uncertainty.
- `stock_api/models.py::multitask_gp` fits separate log-price GPs with a shared
  length scale and zero cross-ticker covariance. It is not an existing joint
  heteroskedastic return model and cannot supply the required dependence.
- `stock_api/data.py` supplies the XNYS calendar and adjusted-close downloader.
  We reuse the calendar and already-downloaded files, checking their recorded
  SHA256 hashes rather than downloading a new vintage.

## Forward distribution and cross-asset dependence

The portfolio adapter will call `fit_latent_returns`, `_future` and `_root`
directly to retain latent drift and innovation uncertainty separately. Each
ticker's univariate GP is unchanged. Dependence is added through correlated
Gaussian simulation drivers, without fitting a new multi-output GP.

For the fitted training window, standardized residuals are
`(r - E[f]) / sqrt(E[conditional variance])`. Their sample correlation is shrunk
50% toward the identity. This PSD matrix couples daily innovations and latent
drift Gaussian drivers. Its elementwise square (also PSD, with unit diagonal)
couples latent log-volatility drivers; this gives nonnegative common volatility
dependence. Temporal covariance for each latent GP is retained using its own
matrix root. The three driver families are independent of one another.

This is a documented dependence approximation, not a jointly inferred posterior.
The residuals are in-sample smoothed residuals, but every observation is available
before the forecast. Strong fixed shrinkage addresses the short 30-observation,
22-variable correlation estimate. No future observations enter this estimate.

Antithetic paths preserve the analytical monthly mean. The finite Monte Carlo
terminal covariance is normalized to the GP's analytical marginal variances:
`scale² * (sum(f_cov) + sum(exp(g_mean + diag(g_cov)/2)))`. Latent-drift variance
and innovation variance are saved separately. The empirical correlation of the
joint paths supplies off-diagonal dependence. Monte Carlo marginal rescaling
and its size are recorded.

## Necessary return-moment convention

The existing latent lognormal variance mixture has finite log-return moments but
may have infinite exponential moments. A finite Monte Carlo arithmetic-return
mean would therefore not be a valid population expectation. We will **optimize
monthly log-return means and covariance**, explicitly defining the objective's
asset return vector in those units. This is a mean–variance proxy for allocation,
not the exact expected simple return or exact expected log return of the
buy-and-hold portfolio. Realized dollar performance uses exact adjusted-price
ratios. Portfolio log-return diagnostic samples use `log(sum(w*exp(asset log R)))`,
computed stably; no arithmetic population moment is asserted and no tails are clipped.

## Dates, execution and accounting

Twelve-month return interval: October 17, 2025–October 16, 2026, with October 16,
2025 as the opening price anchor. Initial training ends April 16, 2026. Validation
starts April 17, 2026. Rebalance on the first session on/after the 17th of each
month, April through September, and mark the terminal portfolio on October 16.

Forecasts use closes through the previous session. At the next session's opening
print, holdings are marked at that opening price, the optimizer uses these actual
drifted pre-trade weights, and trades are modeled immediately after the open at
the published Open plus the proportional cost allowance. This is an opening-price
execution proxy, not a claim that an order submitted after the print could fill
exactly at the auction price; intraday fill data are unavailable. No same-day high,
low or close enters optimization. Adjusted Open = Open × Adj Close / Close, the
same corporate-action factor as the total-return close series.

The GP forecasts close-to-close returns through the session before the next
execution (through October 16 for the last holding). These are a proxy for the
roughly equal-length open-to-open holding returns. Both correctly aligned forecast
target returns and actual open-to-open holding returns will be saved, so this
execution-clock approximation is visible. Daily valuation uses adjusted closes.

Costs are 10 bp times actual dollars purchased **plus** sold, including initial
investment. Solve `C = rate * sum(abs((V-C)*w - old_asset_dollars))` so trades,
fees and the $2,000 cash budget reconcile exactly. Fully invest remaining wealth;
assume fractional shares, automatic dividend reinvestment and no extra commissions.
Record fees as cash outflow. There is no terminal liquidation or sale fee: ending
values are marked holdings. Gross performance replays the same selected weights
without debiting fees. The optimizer's turnover penalty affects choices but is
not separately deducted from wealth.

## Frozen choices and controls

Primary lookback is **30 daily returns (31 closes)**, unlike the prior experiment's
30 calendar days. Primary risk aversion is fixed at 10; six initial months are
too short to select it reliably. Initial history supplies the first GP windows,
training diagnostics and the historical-covariance benchmark. There is no selection
on validation performance. Sensitivities are the full small grid of lookbacks
30/60, lambda 3/10/30, and costs 0/10/25 bp, all declared before running.

Required controls: monthly equal weight, shrinkage historical minimum variance
(up to 126 past sessions), identical GP covariance with drift set to zero
(minimum variance plus costs), and SPY buy-and-hold. Additional diagnostics:
the existing independently fitted volatility-only GP and maximum predicted mean.
The identical-covariance zero-drift control isolates the effect of expected drift;
the native volatility-only GP changes the fitted volatility model as well.

**This is a retrospective conditional backtest.** The 22 stocks were screened
using information through October 18, 2026, including the validation period.
There is no claim of independent prospective stock-selection validation.
