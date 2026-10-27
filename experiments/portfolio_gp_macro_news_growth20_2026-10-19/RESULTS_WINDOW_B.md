| Strategy | Net return | Ending $ | Sharpe | Max drawdown | Turnover | Costs |
| --- | --- | --- | --- | --- | --- | --- |
| GP only | 475.47% | 11,509.38 | 3.35 | -14.00% | 88.6% | $63.24 |
| GP + Macro | 476.43% | 11,528.52 | 3.35 | -14.05% | 88.9% | $63.60 |
| GP + News (deterministic) | 474.55% | 11,491.07 | 3.21 | -14.68% | 97.3% | $70.43 |
| GP + Macro + News (deterministic) | 483.03% | 11,660.60 | 3.24 | -14.62% | 96.9% | $70.78 |
| Equal weight | 335.85% | 8,716.97 | 2.77 | -22.27% | 21.5% | $13.31 |
| SPY | 23.77% | 2,475.50 | 1.44 | -8.88% | 6.7% | $2.00 |

# Window B: exact 20-stock primary replication; shortened initial training

15 walk-forward holding periods. 20 assets; each strategy separately starts at $2,000. Evaluation dates: 2025-07-17–2026-10-16. The two windows are independent and must not be compounded together. Window A ends with an 11-session partial holding. No LLM was evaluated.

The requested universe is exactly: MXL, DELL, AXTI, BAND, SAIH, SILC, PENG, MU, AEHR, RNG, QMCO, MRVL, AMD, FATN, INTC, HPE, SIMO, OKTA, STX, FTNT. The label “top growing” is the user's selection criterion, not an independently established ranking. This is a hindsight-selected October 2026 universe, not a point-in-time screen. Selecting recent winners can strongly inflate retrospective performance. No historical profits validate the selection process.

FATN started trading April 8, 2025. SAIH's vendor history begins June 28, 2021, including its SPAC predecessor; the operating-company combination closed April 29, 2022. **Window A cannot be run on the full 20.** The earlier-window diagnostic uses the other 18 supplied names, explicitly excluding SAIH and FATN without substitutes. Window B uses all 20, but common training starts April 8, 2025, not January. Evaluation still starts July 17, 2025. The first numerical macro/news correction activates only after three completed GP forecast-error intervals. The July allocation has no augmentation. Missing prior distributions for SAIH/FATN remain exactly weak priors; no GP temporal state crosses the gap.

The supplied list supersedes the earlier screen's >$10 restriction: FATN closes at $5.00. Names are retained exactly as requested. A >$10 requirement throughout history is not applied. PENG includes the same issuer's pre-October-2024 SGH history. Vendor adjusted prices account for splits/dividends; raw cached actions and checksums are saved. There are zero missing required price observations in the declared usable windows, with no pre-listing filling or synthetic returns.

**No LLM was run.** News variants use the previous experiment's deterministic FOMC/CAMEO mechanisms, not sentiment, market surprises, or LLM interpretations. Numerical macro/news caches were reused in a separate copy; all GP joint distributions were recomputed for the changed asset universe. Estimated drift, 30 trading days, 10,000 paths, lambda=10, 10 bp per dollar bought or sold, $2,000 and a 20% allocation cap are unchanged. No parameter was chosen using these returns.


## Return forecasts and incremental information

- GP + Macro: net return difference vs GP +0.96 percentage points; gross difference +0.99 points. CRPS difference +0.000304, 95% block interval [-0.000169, +0.000780].
- GP + News (deterministic): net return difference vs GP -0.92 percentage points; gross difference -0.17 points. CRPS difference -0.000091, 95% block interval [-0.001989, +0.001954].
- GP + Macro + News (deterministic): net return difference vs GP +7.56 percentage points; gross difference +8.40 points. CRPS difference +0.000135, 95% block interval [-0.001578, +0.001848].

| strategy | mae | rmse | crps | coverage90 | coverage68 | correlation | spearman | direction_accuracy | rms_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GP only | 0.17813 | 0.25650 | 0.13290 | 0.86333 | 0.69333 | 0.11822 | 0.03669 | 0.50667 | 1.18874 |
| GP + Macro | 0.17854 | 0.25717 | 0.13320 | 0.86000 | 0.69667 | 0.10656 | 0.00812 | 0.50000 | 1.19251 |
| GP + News (deterministic) | 0.17993 | 0.25615 | 0.13281 | 0.86667 | 0.67667 | 0.10704 | 0.04722 | 0.55000 | 1.18022 |
| GP + Macro + News (deterministic) | 0.17990 | 0.25671 | 0.13303 | 0.86667 | 0.68000 | 0.10047 | 0.04812 | 0.52000 | 1.18470 |

Errors and scores are dimensionless monthly log returns. Coverage columns target 90% and 68%; cross-sectional Spearman is averaged across months. Paired inference resamples entire market months with three-month circular blocks, 2,000 draws. Intervals are exploratory and unadjusted for multiple comparisons, selection and research reuse.

| strategy | baseline | difference | lower95 | upper95 |
| --- | --- | --- | --- | --- |
| GP + Macro | GP only | 0.00030 | -0.00017 | 0.00078 |
| GP + Macro + News (deterministic) | GP only | 0.00013 | -0.00158 | 0.00185 |
| GP + Macro placebo | GP only | -0.00004 | -0.00032 | 0.00020 |
| GP + Macro transfer | GP only | 0.00017 | -0.00029 | 0.00062 |
| GP + News (deterministic) | GP only | -0.00009 | -0.00199 | 0.00195 |
| GP + News placebo | GP only | -0.00037 | -0.00309 | 0.00219 |
| GP + Macro + News (deterministic) | GP + Macro | -0.00017 | -0.00178 | 0.00153 |
| GP + Macro transfer | GP + Macro | -0.00013 | -0.00029 | 0.00001 |

Paired cross-sectional rank-correlation differences (positive is better; same block procedure):

| strategy | baseline | difference | lower95 | upper95 |
| --- | --- | --- | --- | --- |
| GP + Macro | GP only | -0.02857 | -0.09715 | 0.03850 |
| GP + Macro + News (deterministic) | GP only | 0.01143 | -0.06397 | 0.08822 |
| GP + Macro placebo | GP only | 0.06967 | -0.02428 | 0.15621 |
| GP + Macro transfer | GP only | -0.01915 | -0.07661 | 0.03629 |
| GP + News (deterministic) | GP only | 0.01053 | -0.07580 | 0.09935 |
| GP + News placebo | GP only | -0.03930 | -0.15539 | 0.07571 |
| GP + Macro + News (deterministic) | GP + Macro | 0.04000 | -0.03449 | 0.14296 |
| GP + Macro transfer | GP + Macro | 0.00942 | -0.00872 | 0.02938 |

Portfolio monthly differences versus GP, with matching block intervals:

| window | strategy | mean_monthly_difference | lower95 | upper95 |
| --- | --- | --- | --- | --- |
| B | Equal weight | -0.02317 | -0.06076 | 0.01109 |
| B | GP + Macro | 0.00006 | -0.00249 | 0.00300 |
| B | GP + Macro + News (deterministic) | -0.00011 | -0.01463 | 0.01319 |
| B | GP + Macro placebo | 0.00111 | -0.00178 | 0.00529 |
| B | GP + Macro transfer | 0.00028 | -0.00244 | 0.00358 |
| B | GP + News (deterministic) | -0.00165 | -0.01578 | 0.01221 |
| B | GP + News placebo | -0.00380 | -0.01896 | 0.00890 |
| B | SPY | -0.12669 | -0.26142 | -0.02913 |

## Risk, allocations and attribution

Price-only realized daily squared-return sum / predicted innovation variance sum is 1.095; one is the aggregate target. Monthly standardized error RMS is 1.189. GP and augmented models use identical conditional covariance, so any apparent coverage change comes from the mean. Parameter uncertainty in the macro/news adjustment is not fully propagated. Portfolio predictive returns use log-sum-exp of simulated asset returns and weights; close-origin forecasts still approximate next-open execution.

| window | strategy | rms_z | coverage90 |
| --- | --- | --- | --- |
| B | GP + Macro | 1.80889 | 0.73333 |
| B | GP + Macro + News (deterministic) | 1.61623 | 0.73333 |
| B | GP + News (deterministic) | 1.59222 | 0.73333 |
| B | GP only | 1.81646 | 0.73333 |

| strategy | mean_cap_count | mean_effective_holdings | mean_monthly_turnover | best_month | worst_month | positive_month_fraction | net_annualized_volatility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GP only | 1.60000 | 7.65678 | 0.88612 | 0.89328 | -0.03624 | 0.73333 | 0.44664 |
| GP + Macro | 1.40000 | 7.68069 | 0.88872 | 0.88858 | -0.03736 | 0.73333 | 0.44648 |
| GP + News (deterministic) | 1.13333 | 7.86125 | 0.97290 | 0.86544 | -0.04145 | 0.86667 | 0.46881 |
| GP + Macro + News (deterministic) | 1.13333 | 7.81524 | 0.96942 | 0.87921 | -0.04118 | 0.86667 | 0.46764 |
| Equal weight | 0.00000 | 20.00000 | 0.21455 | 0.63022 | -0.12610 | 0.66667 | 0.46262 |
| SPY | 0.00000 | 1.00000 | 0.06660 | 0.05306 | -0.01679 | 0.73333 | 0.12298 |

Allocation caps apply at rebalance, not to weights drifting during the month. Largest monthly differences (joint macro+news minus GP): 2026-07-17: -4.40 pp, 2026-02-17: +4.37 pp, 2025-12-17: +3.64 pp. Largest asset-level dollar P&L differences: SAIH: $-321.46, AXTI: $+284.76, DELL: $+280.28, FTNT: $-230.15, MXL: $+196.88. Dollar attribution reflects evolving capital as well as weights, and is descriptive rather than a causal signal attribution. Exact holding-period contribution and fee identities are checked.

## Macroeconomic sensitivities

| window | factor | extreme | ticker | coefficient | lower95 | upper95 |
| --- | --- | --- | --- | --- | --- | --- |
| B | inflation | highest | MXL | 0.21325 | -0.15989 | 0.58639 |
| B | inflation | lowest | RNG | -0.06615 | -0.21675 | 0.08445 |
| B | inflation_expectations | highest | AXTI | 0.06734 | -0.56087 | 0.69556 |
| B | inflation_expectations | lowest | QMCO | -0.04432 | -0.57775 | 0.48911 |
| B | oil | highest | AXTI | 0.81430 | -0.48024 | 2.10884 |
| B | oil | lowest | OKTA | -0.22095 | -0.70466 | 0.26276 |
| B | rates | highest | OKTA | 0.12959 | -0.08377 | 0.34295 |
| B | rates | lowest | AEHR | -0.20923 | -0.64249 | 0.22403 |
| B | unemployment | highest | AXTI | 0.09740 | -1.05506 | 1.24987 |
| B | unemployment | lowest | QMCO | -0.18729 | -1.16587 | 0.79130 |
| B | volatility | highest | MU | 0.14889 | -0.28823 | 0.58601 |
| B | volatility | lowest | PENG | -0.16980 | -0.66630 | 0.32670 |

These are full-window descriptive coefficients with conditional Gaussian 95% intervals. 0/120 exclude zero before multiplicity adjustment. Rates, inflation and unemployment units are one percentage-point change; oil is one asinh(WTI/50) unit and volatility one log-VIX unit. Coefficients control for SPY and other factors; monetary-policy changes are only proxied by yields, growth by unemployment. Coefficient extremes do not establish reliable causal sensitivity. Ridge intervals omit model misspecification and are not HAC inference. Rolling coefficients, uncertainty and all asset rows are saved.

Latest available incremental news sensitivities (not LLM estimates, surprises, or causal geopolitical/trade betas):

| factor | extreme | ticker | coefficient | lower95 | upper95 |
| --- | --- | --- | --- | --- | --- |
| conflict_attention | highest | QMCO | 0.05100 | -0.07688 | 0.17888 |
| conflict_attention | lowest | PENG | 0.01090 | -0.06797 | 0.08976 |
| growth_weakness | highest | AXTI | 0.03116 | -0.13848 | 0.20080 |
| growth_weakness | lowest | QMCO | -0.01727 | -0.15825 | 0.12371 |
| inflation_attention | highest | AXTI | 0.06329 | -0.09579 | 0.22237 |
| inflation_attention | lowest | PENG | -0.00728 | -0.08882 | 0.07425 |
| policy_direction | highest | DELL | 0.01471 | -0.04914 | 0.07856 |
| policy_direction | lowest | AXTI | -0.04569 | -0.21164 | 0.12026 |
| trade_attention | highest | MXL | 0.00000 | -0.15045 | 0.15045 |
| trade_attention | lowest | MXL | 0.00000 | -0.15045 | 0.15045 |

## Transfer, controls and events

| Strategy | Net return | Ending $ | Sharpe | Max drawdown | Turnover | Costs |
| --- | --- | --- | --- | --- | --- | --- |
| GP + Macro placebo | 482.99% | 11,659.72 | 3.36 | -14.02% | 87.6% | $63.55 |
| GP + News placebo | 452.65% | 11,053.06 | 3.16 | -14.36% | 98.5% | $70.07 |
| GP + Macro transfer | 478.01% | 11,560.12 | 3.36 | -14.02% | 88.8% | $63.67 |

Ex-post conditional exposure calibration uses realized factors for diagnostics only; these realized factors never enter a forward allocation:

| window | variant | conditional_mae | conditional_rmse | coverage95 |
| --- | --- | --- | --- | --- |
| B | transfer_prior | 0.17613 | 0.24849 | 0.86333 |
| B | weak_prior | 0.17851 | 0.24988 | 0.87000 |

Descriptive major-event abnormal returns, market model fitted on up to 60 prior sessions ending five sessions before the event; target event day through five following sessions:

| event | ticker |
| --- | --- |

The event reactions do not demonstrate advance anticipation. Scheduled FOMC statements are included by calendar rule. Named-event findings are post-event diagnostics; the initial 2025 tariff announcement precedes the full-20 common price history and is not estimable on all 20. Missing/short event windows are recorded in `exposures/event_exclusions.csv`. No hindsight narrative is used as a trading input.

## Monthly returns

| execution_date | GP only | GP + Macro | GP + News (deterministic) | GP + Macro + News (deterministic) | Equal weight | SPY |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-07-17 | 0.89% | 0.89% | 0.89% | 0.89% | -1.03% | 2.85% |
| 2025-08-18 | 11.28% | 11.28% | 11.28% | 11.28% | 14.06% | 2.67% |
| 2025-09-17 | 7.96% | 8.06% | 8.59% | 8.57% | 7.82% | 0.20% |
| 2025-10-17 | -0.11% | -0.19% | 0.27% | 0.22% | -2.59% | 1.55% |
| 2025-11-17 | 6.42% | 6.40% | 8.35% | 7.92% | 8.45% | 1.52% |
| 2025-12-17 | -3.51% | -3.40% | 0.30% | 0.13% | 5.14% | 0.53% |
| 2026-01-20 | 8.59% | 8.56% | 7.51% | 7.48% | 3.84% | -0.20% |
| 2026-02-17 | 6.66% | 6.71% | 11.15% | 11.03% | 10.31% | -1.14% |
| 2026-03-17 | 24.88% | 26.73% | 25.05% | 26.83% | 34.91% | 5.31% |
| 2026-04-17 | 89.33% | 88.86% | 86.54% | 87.92% | 63.02% | 4.77% |
| 2026-05-18 | 23.33% | 23.27% | 20.40% | 20.64% | 22.96% | 1.55% |
| 2026-06-17 | -0.73% | -0.97% | -1.69% | -1.76% | -12.61% | -0.97% |
| 2026-07-17 | 31.79% | 30.78% | 28.24% | 27.38% | 34.49% | 4.60% |
| 2026-08-17 | 8.71% | 8.70% | 6.63% | 7.27% | -2.64% | -1.68% |
| 2026-09-17 | -3.62% | -3.74% | -4.14% | -4.12% | -9.05% | 0.26% |

Full gross/net metrics, annualization, daily equity, weights, costs and paired controls are saved in CSVs. Sharpe uses zero risk-free rate. Annualization is descriptive, especially for the six-month reset subset. See [architecture](ARCHITECTURE.md), [sources](DATA_SOURCES.md), and [main report](RESULTS.md).
