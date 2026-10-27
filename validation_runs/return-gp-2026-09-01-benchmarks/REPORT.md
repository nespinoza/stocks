# AMZN latent return-GP benchmark

Cutoff **2026-09-01**, 12 monthly origins from 2025-09-01 through 2026-08-01, 7 calendar-day horizon (available XNYS sessions), AMZN target, GOOGL/AAPL available for multivariate models. Seed 42; 10,000 Monte Carlo paths. All nine models use the original scoring of median prices. Every model has 12 folds and 56 scored prices.

For each lookback, the experiment identity (configuration, price snapshot, folds and seed) and **all existing seven-model summary metrics were checked against the saved SDE benchmark and match exactly**. No benchmark scores or definitions were changed. The old result files remain intact.

MAE is the equal-fold mean in adjusted-price dollars. NMAE normalizes each fold by its mean observed price; terminal error below is `100 * mean(abs(predicted_log_return - actual_log_return))`, in log-return percentage points. Coverage is the equal-fold mean of pointwise 95% coverage. Brier uses one terminal up/down event per fold. All detailed and pooled metrics are in the CSVs/JSON.

## 30-calendar-day lookback

| Model | MAE $ | NMAE % | Terminal error (log pp) | 95% coverage | Brier |
|---|---:|---:|---:|---:|---:|
| last_price | 6.006 | 2.485 | 3.477 | 84.17% | 0.2500 |
| gp | 9.546 | 3.805 | 5.161 | 69.17% | 0.3816 |
| multitask_gp | 9.410 | 3.772 | 5.123 | 67.50% | 0.3698 |
| var | 8.945 | 3.651 | 4.736 | 65.83% | 0.2712 |
| gbm_zero_drift | 6.094 | 2.519 | 3.548 | 84.17% | 0.2550 |
| gbm_estimated_drift | 6.437 | 2.694 | 4.265 | 84.17% | 0.2468 |
| ou_returns | 14.007 | 5.435 | 7.986 | 77.92% | 0.2346 |
| volatility_gp_returns | 6.006 | 2.485 | 3.477 | 84.17% | 0.2500 |
| heteroskedastic_gp_returns | 5.642 | 2.347 | 3.272 | 84.17% | 0.2326 |

[Full summary](lookback-30.csv), [fold scores](folds-30.csv), [latent diagnostics](diagnostics-30.csv), [error plot](lookback-30.png), [immutable run](lookback-30.json).

## 60-calendar-day lookback

| Model | MAE $ | NMAE % | Terminal error (log pp) | 95% coverage | Brier |
|---|---:|---:|---:|---:|---:|
| last_price | 6.006 | 2.485 | 3.477 | 88.33% | 0.2500 |
| gp | 9.270 | 3.664 | 4.950 | 85.00% | 0.3317 |
| multitask_gp | 8.611 | 3.428 | 4.699 | 83.33% | 0.3303 |
| var | 8.026 | 3.337 | 4.641 | 76.67% | 0.3842 |
| gbm_zero_drift | 6.072 | 2.511 | 3.531 | 90.00% | 0.2546 |
| gbm_estimated_drift | 6.615 | 2.761 | 4.059 | 88.33% | 0.3089 |
| ou_returns | 6.328 | 2.630 | 3.881 | 88.33% | 0.2791 |
| volatility_gp_returns | 6.006 | 2.485 | 3.477 | 90.00% | 0.2500 |
| heteroskedastic_gp_returns | 5.839 | 2.425 | 3.377 | 90.00% | 0.2436 |

[Full summary](lookback-60.csv), [fold scores](folds-60.csv), [latent diagnostics](diagnostics-60.csv), [error plot](lookback-60.png), [immutable run](lookback-60.json).

## 90-calendar-day lookback

| Model | MAE $ | NMAE % | Terminal error (log pp) | 95% coverage | Brier |
|---|---:|---:|---:|---:|---:|
| last_price | 6.006 | 2.485 | 3.477 | 91.67% | 0.2500 |
| gp | 7.989 | 3.247 | 4.699 | 88.33% | 0.3029 |
| multitask_gp | 7.588 | 3.079 | 4.451 | 90.00% | 0.2967 |
| var | 6.634 | 2.748 | 3.709 | 86.67% | 0.2725 |
| gbm_zero_drift | 6.067 | 2.509 | 3.527 | 91.67% | 0.2545 |
| gbm_estimated_drift | 6.152 | 2.549 | 3.636 | 88.33% | 0.2549 |
| ou_returns | 5.748 | 2.383 | 3.389 | 90.00% | 0.2334 |
| volatility_gp_returns | 6.006 | 2.485 | 3.477 | 90.00% | 0.2500 |
| heteroskedastic_gp_returns | 5.915 | 2.454 | 3.429 | 90.00% | 0.2481 |

[Full summary](lookback-90.csv), [fold scores](folds-90.csv), [latent diagnostics](diagnostics-90.csv), [error plot](lookback-90.png), [immutable run](lookback-90.json).

## 180-calendar-day lookback

| Model | MAE $ | NMAE % | Terminal error (log pp) | 95% coverage | Brier |
|---|---:|---:|---:|---:|---:|
| last_price | 6.006 | 2.485 | 3.477 | 90.00% | 0.2500 |
| gp | 6.942 | 2.830 | 4.220 | 96.67% | 0.2638 |
| multitask_gp | 6.951 | 2.829 | 4.215 | 96.67% | 0.2714 |
| var | 7.748 | 3.161 | 4.492 | 83.33% | 0.2567 |
| gbm_zero_drift | 6.065 | 2.509 | 3.530 | 91.67% | 0.2546 |
| gbm_estimated_drift | 6.060 | 2.510 | 3.542 | 90.00% | 0.2447 |
| ou_returns | 6.123 | 2.524 | 3.634 | 93.33% | 0.2242 |
| volatility_gp_returns | 6.006 | 2.485 | 3.477 | 90.00% | 0.2500 |
| heteroskedastic_gp_returns | 5.907 | 2.448 | 3.420 | 90.00% | 0.2455 |

[Full summary](lookback-180.csv), [fold scores](folds-180.csv), [latent diagnostics](diagnostics-180.csv), [error plot](lookback-180.png), [immutable run](lookback-180.json).

## Timescales and uncertainty diagnostics

Lengths are trading sessions; entries are median [minimum, maximum] across folds. V denotes `volatility_gp_returns`, H denotes `heteroskedastic_gp_returns`.

| Lookback | V: ell_sigma | H: ell_mu | H: ell_sigma | H: mean amplitude at floor |
|---|---:|---:|---:|---:|
| 30 | 19.96 [2.10, 19.99] | 20.00 [6.75, 22.05] | 19.96 [2.10, 19.99] | 10/12 |
| 60 | 13.58 [2.00, 19.96] | 19.97 [15.33, 20.04] | 19.74 [2.00, 19.96] | 9/12 |
| 90 | 9.85 [2.00, 19.80] | 19.95 [7.54, 20.12] | 10.40 [2.00, 19.89] | 9/12 |
| 180 | 2.08 [2.00, 19.19] | 19.93 [3.85, 20.28] | 2.08 [2.00, 19.86] | 9/12 |

| Lookback | V: mean terminal sigma (log pp) | H: mean terminal sigma (log pp) | H: mean P(up) | Realized up frequency |
|---|---:|---:|---:|---:|
| 30 | 5.267 | 5.295 | 0.5215 | 0.75 |
| 60 | 4.945 | 5.021 | 0.5132 | 0.75 |
| 90 | 4.856 | 4.951 | 0.5069 | 0.75 |
| 180 | 4.670 | 4.822 | 0.5080 | 0.75 |

V always predicts P(up)=0.5. The H mean probabilities are also near 0.5, whereas 9/12 held-out terminal returns are positive. The [reliability table](reliability.csv) groups probabilities into fixed bins; 12 events do not support a reliable fitted calibration slope. [Diagnostic summary](diagnostic-summary.csv) retains amplitude ranges, boundary counts, predicted sigma and expected-return magnitudes. Per-fold diagnostics retain original-unit log-variance levels, forecast origins, realized returns, and complete convergence fields.

All **96 new-model fits converged**, with both starts converging in every fit; no fold was dropped and no fallback was used. Maximum retained projected gradient was below 0.001. Mean amplitudes are at their lower bound in 37/48 joint fits. Their near-20-day lengths are therefore usually weakly identified and heavily influenced by the fixed regularization, not evidence for a market cycle. The joint volatility length reaches its 2-session lower bound in 0/2/3/6 folds for 30/60/90/180-day lookbacks. The processes can and do infer different lengths, but a numerical difference is not evidence that both lengths are identified.

## Interpretation

1. **Volatility only:** point errors exactly equal persistence, by construction. Brier is 0.25, tying persistence and modestly below zero-arithmetic-drift GBM (about 0.255). This Brier difference is the zero-log-mean convention, not learned directional information. Coverage improves over persistence only at 60 days (90% versus 88.33%), is worse at 90 days, and ties at 30/180 days. It ties zero-drift GBM coverage at 30/60 days and is worse at 90/180 days.

2. **Adding a mean GP:** reduces both price and terminal-log-return MAE versus V/persistence at every lookback. Price-MAE reductions are about 6.07%, 2.79%, 1.52%, and 1.65%. It also improves Brier over V at every lookback, but its pointwise interval coverage is identical to V here. Against OU it improves both point-error measures at 30/60/180 days; OU wins at 90 days. H beats OU Brier at 30/60 days; OU wins at 90/180 days. H interval coverage is closer to 95% than OU at 30/60 days, ties at 90 days, and is worse at 180 days.

3. **The gains are concentrated:** the 2026-05-01 fold supplies most of the joint model's improvement. Excluding that fold reverses the average price-MAE advantage at 60, 90 and 180 days. This is a post-hoc sensitivity check, not a revised benchmark. [Gain concentration](gain-concentration.csv) retains the calculation.

4. **No extreme mean forecast was found:** the largest absolute expected terminal log return is +2.624 log percentage points for origin 2026-05-01 with the 30-day lookback (realized +2.279 log pp, predicted sigma 4.536 log pp). The maxima at 60/90/180 days are +1.556/+1.225/+0.652 log pp, also for May. No expected terminal return exceeds its predictive standard deviation. The May 30-day forecast is worth inspecting because of its influence, but its magnitude is not implausible relative to its uncertainty. No runaway drift appeared in the short windows. Small mean/volatility amplitudes, short volatility lengths and concentration in one favorable fold still warrant caution about overfitting and weak identification. Tiny signed means from the positive amplitude floor count as directional forecasts under the unchanged scorer; their direction accuracy should not be read as economically meaningful signal.

5. **Intervals remain undercovered:** both new models cover 84.17%, 90%, 90%, and 90% of prices by the existing equal-fold metric. Neither is consistently better calibrated. The final trajectory sigma can be large (up to 16.38 log pp in a 30-day fit); this is visible in per-fold diagnostics and does not imply a large mean forecast.

## Approximation and limitations

Both latent functions are inferred together using a collapsed Gaussian mean-field variational bound in SciPy; independent Matérn-3/2 hyperparameters maximize a fixed regularized training-only objective. Predictive sampling retains temporal latent GP covariance but omits hyperparameter uncertainty and posterior dependence between f and g. See [inference details and usage](../../RETURN_GP.md) and [Lázaro-Gredilla and Titsias (2011)](https://icml.cc/2011/papers/456_icmlpaper.pdf).

Gaussian log-variance mixing has **no finite population price mean or variance** after exponentiating returns. Reported price mean/std are finite Monte Carlo summaries, while quantiles, medians, probabilities and return-space moments are well-defined. No posterior tails were clipped to hide this issue.

These are model-development results, **not predictive alpha**. The 56 scored prices are correlated within 12 trajectories; lookbacks reuse those same trajectories. Hyperparameters were not tuned to their final targets. Prices are the same retrospectively adjusted snapshot used before, not a point-in-time data archive.

Validation: the full 118-test suite passed; two additional empirical-band adapter checks were then added and the 12-test return-GP suite passed. Tests cover gradients, synthetic latent recovery, zero-mean shrinkage, independent lengths, reproducibility, path reconstruction, valid quantiles/probabilities, controlled failure, historical no-leakage, saved diagnostics, and public forecast/fit bands. Existing dependency deprecation warnings remain. [Runtime versions](runtime.json) and [benchmark identity audit](benchmark-audit.json) are saved alongside the results.
