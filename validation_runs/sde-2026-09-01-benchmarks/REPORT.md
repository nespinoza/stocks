# SDE benchmark results

AMZN target; GOOGL and AAPL available to the joint GP and VAR. Reference cutoff: **2026-09-01**. Twelve monthly forecast origins, September 2025–August 2026. Forecast: first seven calendar days of each month, **56 scored trading-session prices** in each experiment. Lookbacks are calendar days; SDE increments are trading days. Seed 42; 10,000 paths per SDE fold.

All models use the same downloaded adjusted-close snapshot: March 5, 2025–August 7, 2026 (359 sessions). All model parameters use only the corresponding training window. No parameters were tuned using the scored targets. Point forecasts are medians, except persistence which is exactly the last price. Full statistics are in [comparison.csv](comparison.csv); JSON files preserve folds, predictions, diagnostics, and terminal ensembles.

Price errors are dollars per adjusted AMZN share. NMAE and MAPE are percentages. Terminal MAE is absolute log-return error multiplied by 100 (log-return percentage points). Direction is terminal sign accuracy among non-flat forecasts. Coverage is the mean fold fraction inside nominal 95% price intervals. Brier is for the terminal up event; lower is better.

## 30-day lookback

| Model | MAE $ | RMSE $ | NMAE % | MAPE % | Terminal MAE | Direction % | Coverage % | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| last_price | 6.006 | 6.970 | 2.485 | 2.499 | 3.477 | N/A | 84.167 | 0.250 |
| gp | 9.546 | 10.484 | 3.805 | 3.818 | 5.161 | 50.000 | 69.167 | 0.382 |
| multitask_gp | 9.410 | 10.244 | 3.772 | 3.787 | 5.123 | 50.000 | 67.500 | 0.370 |
| var | 8.945 | 10.014 | 3.651 | 3.668 | 4.736 | 66.667 | 65.833 | 0.271 |
| gbm_zero_drift | 6.094 | 7.037 | 2.519 | 2.533 | 3.548 | 25.000 | 84.167 | 0.255 |
| gbm_estimated_drift | 6.437 | 7.485 | 2.694 | 2.711 | 4.265 | 50.000 | 84.167 | 0.247 |
| ou_returns | 14.007 | 15.990 | 5.435 | 5.473 | 7.986 | 50.000 | 77.917 | 0.235 |

OU diagnostics: {'fewer_than_20_returns': 5, 'ar_coefficient_outside_stable_ou_range': 4, 'OU fit': 3}.

[Plot](lookback-30.png) · [Full summary](lookback-30.csv) · [Saved experiment](lookback-30.json)

## 60-day lookback

| Model | MAE $ | RMSE $ | NMAE % | MAPE % | Terminal MAE | Direction % | Coverage % | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| last_price | 6.006 | 6.970 | 2.485 | 2.499 | 3.477 | N/A | 88.333 | 0.250 |
| gp | 9.270 | 10.072 | 3.664 | 3.675 | 4.950 | 66.667 | 85.000 | 0.332 |
| multitask_gp | 8.611 | 9.382 | 3.428 | 3.443 | 4.699 | 58.333 | 83.333 | 0.330 |
| var | 8.026 | 9.273 | 3.337 | 3.352 | 4.641 | 41.667 | 76.667 | 0.384 |
| gbm_zero_drift | 6.072 | 7.020 | 2.511 | 2.524 | 3.531 | 25.000 | 90.000 | 0.255 |
| gbm_estimated_drift | 6.615 | 7.692 | 2.761 | 2.776 | 4.059 | 41.667 | 88.333 | 0.309 |
| ou_returns | 6.328 | 7.305 | 2.630 | 2.647 | 3.881 | 41.667 | 88.333 | 0.279 |

OU diagnostics: {'OU fit': 9, 'ar_coefficient_outside_stable_ou_range': 3}.

[Plot](lookback-60.png) · [Full summary](lookback-60.csv) · [Saved experiment](lookback-60.json)

## 90-day lookback

| Model | MAE $ | RMSE $ | NMAE % | MAPE % | Terminal MAE | Direction % | Coverage % | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| last_price | 6.006 | 6.970 | 2.485 | 2.499 | 3.477 | N/A | 91.667 | 0.250 |
| gp | 7.989 | 8.975 | 3.247 | 3.257 | 4.699 | 58.333 | 88.333 | 0.303 |
| multitask_gp | 7.588 | 8.571 | 3.079 | 3.091 | 4.451 | 50.000 | 90.000 | 0.297 |
| var | 6.634 | 7.461 | 2.748 | 2.758 | 3.709 | 50.000 | 86.667 | 0.273 |
| gbm_zero_drift | 6.067 | 7.019 | 2.509 | 2.522 | 3.527 | 25.000 | 91.667 | 0.254 |
| gbm_estimated_drift | 6.152 | 7.105 | 2.549 | 2.565 | 3.636 | 50.000 | 88.333 | 0.255 |
| ou_returns | 5.748 | 6.666 | 2.383 | 2.399 | 3.389 | 66.667 | 90.000 | 0.233 |

OU diagnostics: {'OU fit': 8, 'ar_coefficient_outside_stable_ou_range': 4}.

[Plot](lookback-90.png) · [Full summary](lookback-90.csv) · [Saved experiment](lookback-90.json)

## 180-day lookback

| Model | MAE $ | RMSE $ | NMAE % | MAPE % | Terminal MAE | Direction % | Coverage % | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| last_price | 6.006 | 6.970 | 2.485 | 2.499 | 3.477 | N/A | 90.000 | 0.250 |
| gp | 6.942 | 7.842 | 2.830 | 2.836 | 4.220 | 58.333 | 96.667 | 0.264 |
| multitask_gp | 6.951 | 7.923 | 2.829 | 2.842 | 4.215 | 50.000 | 96.667 | 0.271 |
| var | 7.748 | 8.805 | 3.161 | 3.182 | 4.492 | 41.667 | 83.333 | 0.257 |
| gbm_zero_drift | 6.065 | 7.021 | 2.509 | 2.522 | 3.530 | 25.000 | 91.667 | 0.255 |
| gbm_estimated_drift | 6.060 | 7.029 | 2.510 | 2.527 | 3.542 | 58.333 | 90.000 | 0.245 |
| ou_returns | 6.123 | 6.955 | 2.524 | 2.543 | 3.634 | 58.333 | 93.333 | 0.224 |

OU diagnostics: {'ar_coefficient_outside_stable_ou_range': 5, 'OU fit': 7}.

[Plot](lookback-180.png) · [Full summary](lookback-180.csv) · [Saved experiment](lookback-180.json)

## Interpretation

- Best observed SDE configuration: OU returns with 90 days of training. Price MAE $5.748 versus persistence $6.006 (about 4.3% lower); terminal log-return MAE 3.389 versus 3.477 log-return percentage points. This is an observed comparison, not a selected production default.
- Within SDEs, zero-drift GBM has the lowest price MAE at 30 and 60 days; OU wins at 90; estimated-drift GBM wins narrowly at 180. Zero-drift GBM wins terminal-return MAE at 30, 60 and 180 days; OU wins at 90.
- Estimated GBM drift worsens price MAE at 30/60/90 days and improves it by only about $0.005 per share at 180 days; normalized error at 180 is still slightly worse. Estimated drift worsens terminal-return MAE at every lookback relative to zero drift. It does not beat persistence in any of these runs.
- OU at 90 days is the only SDE/lookback combination here beating persistence on both price and terminal-return MAE. Zero-drift GBM has a flat expected price but a slightly declining median due to the Ito correction, explaining its small price-error difference from persistence and mechanical downward median direction.
- SDE coverage ranges from 77.9% to 93.3%, below nominal 95%. Plug-in estimates omit parameter uncertainty. OU/180 has the smallest terminal Brier score (0.224); persistence is 0.250. Nine of twelve outcomes are up, while model average probabilities are generally around 0.49–0.56. Twelve outcomes are far too few to establish calibration.
- A conspicuous short-window OU failure occurs at the August 2026 origin: terminal log return forecast +47.10% versus +1.06% observed (approximately +60% versus +1.1% in simple price returns). The stable AR coefficient is about 0.806, but estimated equilibrium log return is 0.0355/day. Its fold MAE is about $95.96, and none of the fold observations fall inside its intervals. Thus a formally stable AR coefficient does not guarantee a useful short-window OU estimate. No threshold was changed after inspecting this target. The model should remain an experimental benchmark, particularly at short lookbacks.
- The same OU fold has a very good up-event Brier score despite its huge magnitude error: event scoring and price scoring measure different things. Fallbacks are explicit (9/12 OU folds at 30 days, 3/12 at 60, 4/12 at 90, 5/12 at 180). IID fallback forecasts can coincide with estimated-drift GBM.
- Persistence terminal direction is N/A, not a failure rate: it makes no nonzero directional forecast. Its probabilistic random-walk intervals and 0.5 up probabilities are retained for uncertainty comparison.
- The 56 daily scores are correlated within 12 trajectories; the four lookbacks reuse the same outcomes. These are not 224 independent tests, and no significance claim follows from the small improvements. Yahoo retrospective adjustment is not point-in-time data; no trading costs/execution are modeled.

## Reproduction

```bash
python examples/sde_comparison.py --reference-date 2026-09-01 \
  --prices validation_runs/sde-2026-09-01-benchmarks/prices.csv \
  --output validation_runs/sde-replay
```

Input CSV SHA-256: `8867cde2297b6cba1f5c16d9090dcc65fcb57a588701f36fac0077e82d7dcf52`.

Implementation and conventions: [SDE.md](../../SDE.md). Verification: 93 automated tests passed, including exact GBM moments, EM convergence, active OU covariance propagation, seed reproducibility and held-out-target isolation.
