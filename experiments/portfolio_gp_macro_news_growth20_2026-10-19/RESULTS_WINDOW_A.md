| Strategy | Net return | Ending $ | Sharpe | Max drawdown | Turnover | Costs |
| --- | --- | --- | --- | --- | --- | --- |
| GP only | 154.24% | 5,084.73 | 1.04 | -39.71% | 72.1% | $122.53 |
| GP + Macro | 144.68% | 4,893.60 | 0.99 | -39.13% | 75.7% | $124.38 |
| GP + News (deterministic) | 199.94% | 5,998.84 | 1.16 | -36.57% | 74.2% | $131.05 |
| GP + Macro + News (deterministic) | 170.50% | 5,410.00 | 1.06 | -37.96% | 76.7% | $125.79 |
| Equal weight | 88.03% | 3,760.67 | 0.68 | -55.11% | 11.4% | $15.49 |
| SPY | 43.47% | 2,869.34 | 0.55 | -33.72% | 2.3% | $2.00 |

# Window A: 18-stock diagnostic; SAIH and FATN unavailable

43 walk-forward holding periods. 18 assets; each strategy separately starts at $2,000. Evaluation dates: 2019-07-17–2023-01-31. The two windows are independent and must not be compounded together. Window A ends with an 11-session partial holding. No LLM was evaluated.

The requested universe is exactly: MXL, DELL, AXTI, BAND, SAIH, SILC, PENG, MU, AEHR, RNG, QMCO, MRVL, AMD, FATN, INTC, HPE, SIMO, OKTA, STX, FTNT. The label “top growing” is the user's selection criterion, not an independently established ranking. This is a hindsight-selected October 2026 universe, not a point-in-time screen. Selecting recent winners can strongly inflate retrospective performance. No historical profits validate the selection process.

FATN started trading April 8, 2025. SAIH's vendor history begins June 28, 2021, including its SPAC predecessor; the operating-company combination closed April 29, 2022. **Window A cannot be run on the full 20.** The earlier-window diagnostic uses the other 18 supplied names, explicitly excluding SAIH and FATN without substitutes. Window B uses all 20, but common training starts April 8, 2025, not January. Evaluation still starts July 17, 2025. The first numerical macro/news correction activates only after three completed GP forecast-error intervals. The July allocation has no augmentation. Missing prior distributions for SAIH/FATN remain exactly weak priors; no GP temporal state crosses the gap.

The supplied list supersedes the earlier screen's >$10 restriction: FATN closes at $5.00. Names are retained exactly as requested. A >$10 requirement throughout history is not applied. PENG includes the same issuer's pre-October-2024 SGH history. Vendor adjusted prices account for splits/dividends; raw cached actions and checksums are saved. There are zero missing required price observations in the declared usable windows, with no pre-listing filling or synthetic returns.

**No LLM was run.** News variants use the previous experiment's deterministic FOMC/CAMEO mechanisms, not sentiment, market surprises, or LLM interpretations. Numerical macro/news caches were reused in a separate copy; all GP joint distributions were recomputed for the changed asset universe. Estimated drift, 30 trading days, 10,000 paths, lambda=10, 10 bp per dollar bought or sold, $2,000 and a 20% allocation cap are unchanged. No parameter was chosen using these returns.


## Return forecasts and incremental information

- GP + Macro: net return difference vs GP -9.56 percentage points; gross difference -9.46 points. CRPS difference +0.001088, 95% block interval [-0.000650, +0.003906].
- GP + News (deterministic): net return difference vs GP +45.71 percentage points; gross difference +47.43 points. CRPS difference +0.000152, 95% block interval [-0.000792, +0.001111].
- GP + Macro + News (deterministic): net return difference vs GP +16.26 percentage points; gross difference +17.33 points. CRPS difference +0.001322, 95% block interval [-0.000614, +0.004321].

| strategy | mae | rmse | crps | coverage90 | coverage68 | correlation | spearman | direction_accuracy | rms_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GP only | 0.13181 | 0.18781 | 0.09705 | 0.85917 | 0.66150 | -0.05855 | -0.00775 | 0.49871 | 1.32807 |
| GP + Macro | 0.13337 | 0.18949 | 0.09814 | 0.85788 | 0.64858 | -0.04833 | 0.03593 | 0.51550 | 1.33666 |
| GP + News (deterministic) | 0.13269 | 0.18786 | 0.09720 | 0.86047 | 0.65762 | -0.01095 | 0.05518 | 0.53101 | 1.32665 |
| GP + Macro + News (deterministic) | 0.13429 | 0.18967 | 0.09837 | 0.85788 | 0.64212 | -0.01975 | 0.06943 | 0.53618 | 1.33658 |

Errors and scores are dimensionless monthly log returns. Coverage columns target 90% and 68%; cross-sectional Spearman is averaged across months. Paired inference resamples entire market months with three-month circular blocks, 2,000 draws. Intervals are exploratory and unadjusted for multiple comparisons, selection and research reuse.

| strategy | baseline | difference | lower95 | upper95 |
| --- | --- | --- | --- | --- |
| GP + Macro | GP only | 0.00109 | -0.00065 | 0.00391 |
| GP + Macro + News (deterministic) | GP only | 0.00132 | -0.00061 | 0.00432 |
| GP + Macro placebo | GP only | 0.00112 | -0.00028 | 0.00346 |
| GP + News (deterministic) | GP only | 0.00015 | -0.00079 | 0.00111 |
| GP + News placebo | GP only | 0.00010 | -0.00090 | 0.00097 |
| GP + Macro + News (deterministic) | GP + Macro | 0.00023 | -0.00031 | 0.00077 |

Paired cross-sectional rank-correlation differences (positive is better; same block procedure):

| strategy | baseline | difference | lower95 | upper95 |
| --- | --- | --- | --- | --- |
| GP + Macro | GP only | 0.04368 | -0.05348 | 0.13749 |
| GP + Macro + News (deterministic) | GP only | 0.07718 | 0.00240 | 0.15480 |
| GP + Macro placebo | GP only | 0.02568 | -0.06538 | 0.11650 |
| GP + News (deterministic) | GP only | 0.06293 | 0.00177 | 0.13023 |
| GP + News placebo | GP only | 0.02592 | -0.05381 | 0.10498 |
| GP + Macro + News (deterministic) | GP + Macro | 0.03350 | -0.01585 | 0.08875 |

Portfolio monthly differences versus GP, with matching block intervals:

| window | strategy | mean_monthly_difference | lower95 | upper95 |
| --- | --- | --- | --- | --- |
| A | Equal weight | -0.00623 | -0.01873 | 0.00618 |
| A | GP + Macro | -0.00088 | -0.00601 | 0.00311 |
| A | GP + Macro + News (deterministic) | 0.00154 | -0.00469 | 0.00719 |
| A | GP + Macro placebo | 0.00194 | -0.00138 | 0.00559 |
| A | GP + News (deterministic) | 0.00405 | 0.00035 | 0.00852 |
| A | GP + News placebo | 0.00057 | -0.00319 | 0.00463 |
| A | SPY | -0.01757 | -0.03818 | 0.00100 |

## Risk, allocations and attribution

Price-only realized daily squared-return sum / predicted innovation variance sum is 0.960; one is the aggregate target. Monthly standardized error RMS is 1.328. GP and augmented models use identical conditional covariance, so any apparent coverage change comes from the mean. Parameter uncertainty in the macro/news adjustment is not fully propagated. Portfolio predictive returns use log-sum-exp of simulated asset returns and weights; close-origin forecasts still approximate next-open execution.

| window | strategy | rms_z | coverage90 |
| --- | --- | --- | --- |
| A | GP + Macro | 2.10940 | 0.67442 |
| A | GP + Macro + News (deterministic) | 2.05481 | 0.62791 |
| A | GP + News (deterministic) | 2.01317 | 0.69767 |
| A | GP only | 2.07366 | 0.69767 |

| strategy | mean_cap_count | mean_effective_holdings | mean_monthly_turnover | best_month | worst_month | positive_month_fraction | net_annualized_volatility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GP only | 2.06977 | 6.77643 | 0.72054 | 0.43689 | -0.35545 | 0.62791 | 0.29507 |
| GP + Macro | 2.23256 | 6.64811 | 0.75677 | 0.43448 | -0.35290 | 0.65116 | 0.29919 |
| GP + News (deterministic) | 2.41860 | 6.62337 | 0.74231 | 0.43495 | -0.34624 | 0.65116 | 0.30785 |
| GP + Macro + News (deterministic) | 2.34884 | 6.55497 | 0.76664 | 0.43141 | -0.34725 | 0.65116 | 0.30978 |
| Equal weight | 0.00000 | 18.00000 | 0.11358 | 0.41650 | -0.36746 | 0.67442 | 0.35707 |
| SPY | 0.00000 | 1.00000 | 0.02323 | 0.17147 | -0.27182 | 0.67442 | 0.23587 |

Allocation caps apply at rebalance, not to weights drifting during the month. Largest monthly differences (joint macro+news minus GP): 2020-05-18: -7.83 pp, 2021-10-18: +3.94 pp, 2022-09-19: -3.65 pp. Largest asset-level dollar P&L differences: AEHR: $+303.93, FTNT: $-255.73, RNG: $+212.39, MU: $-211.26, INTC: $+207.31. Dollar attribution reflects evolving capital as well as weights, and is descriptive rather than a causal signal attribution. Exact holding-period contribution and fee identities are checked.

## Macroeconomic sensitivities

| window | factor | extreme | ticker | coefficient | lower95 | upper95 |
| --- | --- | --- | --- | --- | --- | --- |
| A | inflation | highest | FTNT | 0.04538 | -0.03727 | 0.12803 |
| A | inflation | lowest | AEHR | -0.15154 | -0.32816 | 0.02508 |
| A | inflation_expectations | highest | QMCO | 0.14152 | -0.21520 | 0.49825 |
| A | inflation_expectations | lowest | BAND | -0.11016 | -0.43077 | 0.21045 |
| A | oil | highest | AEHR | 0.24135 | -0.42732 | 0.91003 |
| A | oil | lowest | RNG | -0.23718 | -0.72320 | 0.24884 |
| A | rates | highest | HPE | 0.03055 | -0.08532 | 0.14642 |
| A | rates | lowest | OKTA | -0.15982 | -0.33864 | 0.01899 |
| A | unemployment | highest | FTNT | 0.03275 | 0.00694 | 0.05857 |
| A | unemployment | lowest | AEHR | -0.02697 | -0.08214 | 0.02820 |
| A | volatility | highest | QMCO | 0.09909 | -0.35488 | 0.55305 |
| A | volatility | lowest | PENG | -0.14159 | -0.43874 | 0.15557 |

These are full-window descriptive coefficients with conditional Gaussian 95% intervals. 1/108 exclude zero before multiplicity adjustment. Rates, inflation and unemployment units are one percentage-point change; oil is one asinh(WTI/50) unit and volatility one log-VIX unit. Coefficients control for SPY and other factors; monetary-policy changes are only proxied by yields, growth by unemployment. Coefficient extremes do not establish reliable causal sensitivity. Ridge intervals omit model misspecification and are not HAC inference. Rolling coefficients, uncertainty and all asset rows are saved.

Latest available incremental news sensitivities (not LLM estimates, surprises, or causal geopolitical/trade betas):

| factor | extreme | ticker | coefficient | lower95 | upper95 |
| --- | --- | --- | --- | --- | --- |
| conflict_attention | highest | AEHR | 0.02710 | -0.08572 | 0.13993 |
| conflict_attention | lowest | OKTA | -0.01200 | -0.07922 | 0.05522 |
| growth_weakness | highest | DELL | 0.00193 | -0.03468 | 0.03853 |
| growth_weakness | lowest | BAND | -0.05460 | -0.16733 | 0.05813 |
| inflation_attention | highest | AEHR | 0.03147 | -0.08522 | 0.14815 |
| inflation_attention | lowest | RNG | -0.01352 | -0.07952 | 0.05249 |
| policy_direction | highest | AXTI | 0.01409 | -0.06668 | 0.09485 |
| policy_direction | lowest | FTNT | -0.01142 | -0.04459 | 0.02175 |
| trade_attention | highest | MXL | 0.00000 | -0.05935 | 0.05935 |
| trade_attention | lowest | MXL | 0.00000 | -0.05935 | 0.05935 |

## Transfer, controls and events

| Strategy | Net return | Ending $ | Sharpe | Max drawdown | Turnover | Costs |
| --- | --- | --- | --- | --- | --- | --- |
| GP + Macro placebo | 175.60% | 5,512.05 | 1.11 | -37.19% | 80.0% | $137.35 |
| GP + News placebo | 159.18% | 5,183.67 | 1.03 | -40.05% | 79.7% | $137.24 |

Ex-post conditional exposure calibration uses realized factors for diagnostics only; these realized factors never enter a forward allocation:

| window | variant | conditional_mae | conditional_rmse | coverage95 |
| --- | --- | --- | --- | --- |
| A | weak_prior | 0.12816 | 0.24322 | 0.97804 |

Descriptive major-event abnormal returns, market model fitted on up to 60 prior sessions ending five sessions before the event; target event day through five following sessions:

| event | ticker | abnormal_log_return | lower95 | upper95 |
| --- | --- | --- | --- | --- |
| Russia invasion of Ukraine | BAND | -0.48820 | -0.66119 | -0.31522 |
| Russia invasion of Ukraine | DELL | -0.11685 | -0.18529 | -0.04841 |
| WHO pandemic characterization | QMCO | -0.92007 | -1.16571 | -0.67442 |
| WHO pandemic characterization | AXTI | -0.30941 | -0.50442 | -0.11441 |

The event reactions do not demonstrate advance anticipation. Scheduled FOMC statements are included by calendar rule. Named-event findings are post-event diagnostics; the initial 2025 tariff announcement precedes the full-20 common price history and is not estimable on all 20. Missing/short event windows are recorded in `exposures/event_exclusions.csv`. No hindsight narrative is used as a trading input.

## Monthly returns

| execution_date | GP only | GP + Macro | GP + News (deterministic) | GP + Macro + News (deterministic) | Equal weight | SPY |
| --- | --- | --- | --- | --- | --- | --- |
| 2019-07-17 | 1.07% | 0.86% | 1.37% | 1.30% | 0.33% | -2.62% |
| 2019-08-19 | 3.96% | 6.35% | 3.07% | 5.01% | 2.89% | 2.65% |
| 2019-09-17 | 8.44% | 7.16% | 8.27% | 6.85% | 0.50% | 0.37% |
| 2019-10-17 | 11.11% | 10.91% | 9.11% | 7.97% | 7.46% | 3.95% |
| 2019-11-18 | 0.96% | 1.43% | 0.03% | 1.47% | 3.87% | 2.69% |
| 2019-12-17 | 8.88% | 9.16% | 9.22% | 9.59% | 8.90% | 4.19% |
| 2020-01-17 | 3.46% | 3.37% | 3.35% | 2.96% | -0.95% | 1.45% |
| 2020-02-18 | -35.54% | -35.29% | -34.62% | -34.73% | -36.75% | -27.18% |
| 2020-03-17 | 43.69% | 43.45% | 43.49% | 43.14% | 41.65% | 17.15% |
| 2020-04-17 | 15.11% | 14.75% | 14.91% | 14.62% | 8.91% | 2.69% |
| 2020-05-18 | 8.37% | 0.72% | 9.07% | 0.54% | 9.54% | 7.17% |
| 2020-06-17 | 1.90% | 2.17% | 1.24% | 1.56% | 1.73% | 2.94% |
| 2020-07-17 | -2.69% | -1.73% | -2.54% | -1.25% | 3.54% | 4.99% |
| 2020-08-17 | -3.77% | -3.54% | -2.70% | -2.70% | -3.29% | -1.30% |
| 2020-09-17 | 9.97% | 10.10% | 9.87% | 10.07% | 10.05% | 4.94% |
| 2020-10-19 | -2.06% | -1.78% | -2.30% | -1.90% | 2.79% | 3.25% |
| 2020-11-17 | 17.55% | 17.65% | 17.85% | 17.70% | 18.07% | 3.33% |
| 2020-12-17 | 2.07% | 2.06% | 2.12% | 2.09% | 5.52% | 2.15% |
| 2021-01-19 | 10.65% | 10.58% | 11.53% | 11.44% | 13.38% | 3.19% |
| 2021-02-17 | -5.56% | -5.54% | -5.29% | -6.34% | -3.33% | 1.05% |
| 2021-03-17 | 6.25% | 6.23% | 6.27% | 6.31% | 1.73% | 5.85% |
| 2021-04-19 | -0.41% | -0.98% | -0.29% | -0.19% | -7.31% | -0.21% |
| 2021-05-17 | 7.14% | 5.20% | 6.08% | 3.74% | 7.91% | 1.51% |
| 2021-06-17 | -3.59% | -3.05% | -2.91% | -2.37% | -0.06% | 1.40% |
| 2021-07-19 | 10.56% | 12.40% | 12.88% | 13.88% | 4.84% | 4.24% |
| 2021-08-17 | 1.12% | 1.41% | -1.76% | -1.65% | 7.23% | 0.47% |
| 2021-09-17 | 11.56% | 11.43% | 10.93% | 11.19% | 4.95% | -0.21% |
| 2021-10-18 | 3.96% | 3.51% | 9.17% | 7.89% | 11.70% | 5.64% |
| 2021-11-17 | -2.67% | -1.80% | -1.58% | -1.43% | -6.68% | -1.24% |
| 2021-12-17 | 8.36% | 7.91% | 9.54% | 8.82% | 3.01% | -0.39% |
| 2022-01-18 | -4.65% | -4.14% | -4.68% | -3.17% | -7.53% | -3.59% |
| 2022-02-17 | -10.32% | -10.71% | -11.69% | -11.65% | -14.82% | -2.17% |
| 2022-03-17 | -5.53% | -6.03% | -5.72% | -6.15% | -6.71% | 1.06% |
| 2022-04-18 | -0.26% | 3.55% | 3.22% | 3.27% | -8.36% | -6.93% |
| 2022-05-17 | -10.88% | -10.50% | -11.34% | -10.75% | -12.09% | -9.70% |
| 2022-06-17 | 6.19% | 6.80% | 6.84% | 7.52% | 5.36% | 6.26% |
| 2022-07-18 | 4.29% | 4.22% | 9.54% | 6.07% | 14.36% | 9.66% |
| 2022-08-17 | -16.07% | -16.01% | -15.49% | -15.48% | -17.86% | -9.88% |
| 2022-09-19 | -6.29% | -9.55% | -7.57% | -9.95% | -14.08% | -4.77% |
| 2022-10-17 | 14.30% | 15.33% | 15.65% | 16.98% | 18.89% | 7.27% |
| 2022-11-17 | -0.18% | -0.69% | 0.18% | 0.09% | 1.77% | -1.34% |
| 2022-12-19 | 8.57% | 10.24% | 10.27% | 11.61% | 7.54% | 3.91% |
| 2023-01-17 | 2.42% | 0.04% | 4.27% | 4.07% | 6.06% | 2.01% |

Full gross/net metrics, annualization, daily equity, weights, costs and paired controls are saved in CSVs. Sharpe uses zero risk-free rate. Annualization is descriptive, especially for the six-month reset subset. See [architecture](ARCHITECTURE.md), [sources](DATA_SOURCES.md), and [main report](RESULTS.md).
