# Amazon: regression–GJR-GARCH versus heteroskedastic GP

The implemented GJR model is a useful, inexpensive regression baseline, but
this comparison does **not establish it as more accurate overall** than the
heteroskedastic return GP. With a year of training history, GJR slightly improves
price error and interval scoring; the GP produces better direction probabilities.
Both remain close to the last-price baseline. I would keep both models for the
next regressor experiment and use the 365-calendar-day GJR configuration as the
initial GJR setting, rather than the poorly estimated short-window versions.

## Period, data and experiment

The requested period is **September 2025–August 2026**: 52 weekly forecast
origins from September 2, 2025 through August 24, 2026, with the existing
seven-calendar-day horizon and observed exchange sessions only. Each new fit
uses prices available before its forecast origin. Outcomes run through
August 31, 2026.

The experiment reuses the repository's frozen AMZN adjusted closes and archived
Bayesian heteroskedastic-GP forecasts. There was no fresh price download and no
GP refit, so changing provider history cannot explain differences. The original
archive is preserved. The [generated report](../historical_validation/AMZN/gjr-2025-2026-v1/REPORT.md)
links the checksummed input records, configuration, individual scores and plots.

The new model jointly estimates an intercept, AR(1) return coefficient,
GJR-GARCH(1,1) variance and Student-t degrees of freedom. The implementation
supports external regressors, but this experiment uses **only AMZN's lagged
return**. Both models use 2,000 simulated predictive paths. GJR conditions on
MLE parameters; the saved GP integrates parameter and latent uncertainty with
MCMC. Consequently this is a comparison of available forecasting configurations,
not an isolated test of volatility equations.

Before fitting GJR, the primary comparison was fixed as **GJR with 365 calendar
days of history against GP with 30 calendar days**, the previously identified
reliable GP setting. The GJR fits use 249–250 price observations; the GP uses
18–20. Equal-window comparisons at 60, 90 and 180 calendar days and a 730-day
GJR sensitivity were also specified. A 30-calendar-day GJR fit is unsupported:
it cannot meet the minimum 31 returns needed by this implementation.

## Primary results

All three configurations below succeeded on all 52 origins. Lower MAE, Brier
and weighted interval score (WIS) are better. MAE averages each origin's daily
median-price errors. WIS scores the terminal log-return distribution at four
central interval levels, balancing width and missed outcomes.

| Measure | GJR, 365 days | Heteroskedastic GP, 30 days | Last price, 90 days |
| --- | ---: | ---: | ---: |
| Price MAE, dollars | 5.7594 | 5.8633 | 5.8025 |
| Terminal return MAE, log percentage points | 3.6627 | 3.6364 | 3.6595 |
| Direction Brier score | 0.2532 | 0.2452 | 0.2500 |
| WIS, log percentage points | 2.0763 | 2.2237 | 2.0809 |
| Terminal 95% interval coverage | 88.5% | 90.4% | 92.3% |

GJR's price MAE is **1.8% lower** and its WIS **6.6% lower** than the GP's.
Its intervals are narrower, but cover fewer outcomes: 46 of 52 versus 47 of 52
for the GP. Neither achieved nominal 95% coverage in this sample. The GP has
better direction probabilities and slightly lower terminal return error.
GJR's Brier score is also worse than the constant 0.5 probability baseline.
Its improvement over last price is only $0.043 per predicted close; WIS is
effectively tied with that baseline.

Eight-week circular moving-block bootstrap intervals account for local temporal
dependence. The paired price-MAE difference (GJR minus GP) is −$0.104, with
95% interval [−$0.301, +$0.093]. The WIS difference is −0.147 log percentage
points, interval [−0.263, −0.060]. That interval is marginal: after Holm
correction over the 100 specified contrasts, its p-value is 0.425. **None of
the primary differences is significant after correction.** The comparison
against last price likewise provides no statistically supported improvement.

## Equal-history comparisons and fit reliability

These comparisons use only dates where both methods succeeded. This matters
especially for the 180-day GP: comparing its 24 successful forecasts with all
52 GJR forecasts would confound performance with date selection.

| Training window, calendar days | Paired origins | GJR price MAE | GP price MAE | GP failed fits |
| --- | ---: | ---: | ---: | ---: |
| 60 | 51 | $6.7015 | $5.7765 | 1/52 |
| 90 | 44 | $6.0198 | $5.7794 | 8/52 |
| 180 | 24 | $6.6907 | $6.6385 | 28/52 |

On equal short histories, the GP generally performs better when it fits.
The 60-day GJR price error is worse even after the specified correction
(adjusted p ≈ 0.050, at the bootstrap resolution limit). The larger-history
GJR advantage in the primary table should therefore not be described as
unqualified superiority of the GJR model family.

All **260 GJR fits** across five windows converged, with no fallback. However,
18 of 52 fits at 60 days and 12 at 90 days have variance persistence at least
0.995; one 60-day fit has Student-t degrees of freedom at or below 2.1.
None of the 365- or 730-day fits triggers those thresholds. This supports using
more history for estimation, while convergence alone cannot establish stability.
The 730-day sensitivity has the lowest aggregate WIS (2.0649), but worse price
MAE and Brier score than 365 days; it is not a new confirmed winner.

The median recorded runtime for 365-day GJR is about **0.020 seconds per
forecast**, versus **78.1 seconds** in the archived 30-day GP run. These are
different executions and inference methods, not a controlled hardware benchmark.
They do show that GJR is practical for repeated regressor experiments here.

## Implications for the next experiment

Retain GJR365, GP30 and last price as reference configurations. The saved
historical last-price model already includes random-walk uncertainty bands;
it is a meaningful distributional baseline as well as a point forecast.
Adding predictors should be evaluated against the same model without those
predictors, on identical forecast origins, with their publication lags enforced.
The [new regressor API](../GARCH.md#adding-external-regressors) requires explicit
future paths or scenarios and estimates scaling from training observations only.

This year was already examined in earlier repository analyses, so these are
retrospective development results. One stock and 52 weekly episodes cannot
establish generalization. Adjusted closes are not a point-in-time corporate-action
archive, simulation adds Monte Carlo error, and GJR currently omits parameter
uncertainty. A subsequent period reserved before choosing regressors and training
windows is needed to assess whether either model adds predictive value.

Implementation and usage: [GARCH.md](../GARCH.md). Full numerical results:
[summary](../historical_validation/AMZN/gjr-2025-2026-v1/summary.csv),
[paired tests](../historical_validation/AMZN/gjr-2025-2026-v1/paired-comparisons.csv),
[per-origin scores](../historical_validation/AMZN/gjr-2025-2026-v1/scores.csv).

Validation: the complete repository test suite passes **193 tests** with the
optional GARCH dependency installed. The new checks include the GJR variance
recursion, regressor recovery, chronological isolation, and deterministic
forecasts and replay. The historical run completed 520 new records: 260 GJR
fits and 260 matching last-price checks, all successful. The comparison tables
use the archived 90-day last-price forecasts as their common baseline.
