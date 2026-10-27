| Strategy | Net return | Ending $ | Sharpe | Max drawdown | Turnover | Costs |
| --- | --- | --- | --- | --- | --- | --- |
| GP only | 12.30% | 2,245.90 | 0.82 | -10.52% | 92.9% | $29.66 |
| GP + Macro | 10.60% | 2,211.95 | 0.72 | -11.19% | 95.4% | $30.40 |
| GP + News (deterministic) | 7.28% | 2,145.65 | 0.49 | -10.61% | 106.3% | $33.58 |
| GP + Macro + News (deterministic) | 5.38% | 2,107.65 | 0.38 | -11.20% | 106.5% | $32.81 |
| Equal weight | 13.78% | 2,275.58 | 0.94 | -9.96% | 12.2% | $3.85 |
| SPY | 23.77% | 2,475.50 | 1.44 | -8.88% | 6.7% | $2.00 |

# Window B: July 17, 2025–October 16, 2026

15 walk-forward holding periods, starting after January–June warm-up. Each strategy starts independently with $2,000. All news variants here are **deterministic**, not LLM-based. The final Window A holding is a short January 2023 period; returns are not directly comparable to Window B's shorter evaluation. Mean turnover counts buys plus sells, includes the initial purchase, and can exceed 100%.

## Predictive information and portfolio results

- **GP + Macro:** net return 10.60%, a -1.70 percentage-point difference from GP; gross difference -1.68 points. MAE 0.06545, RMSE 0.08807, CRPS 0.04656. Paired CRPS difference +0.000111, 95% market-month block interval [-0.000065, +0.000364].
- **GP + News (deterministic):** net return 7.28%, a -5.01 percentage-point difference from GP; gross difference -4.86 points. MAE 0.06635, RMSE 0.08937, CRPS 0.04725. Paired CRPS difference +0.000808, 95% market-month block interval [+0.000390, +0.001210].
- **GP + Macro + News (deterministic):** net return 5.38%, a -6.91 percentage-point difference from GP; gross difference -6.79 points. MAE 0.06609, RMSE 0.08929, CRPS 0.04711. Paired CRPS difference +0.000662, 95% market-month block interval [+0.000152, +0.001216].

Price-only GP reference: MAE 0.06528, RMSE 0.08776, CRPS 0.04645. All metrics use identical stock-month observations within this window, in dimensionless monthly log-return units.

| strategy | mae | rmse | crps | coverage90 | coverage68 | correlation | spearman | direction_accuracy | rms_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GP only | 0.0653 | 0.0878 | 0.0464 | 0.8848 | 0.6848 | -0.0036 | 0.0960 | 0.5091 | 1.1187 |
| GP + Macro | 0.0654 | 0.0881 | 0.0466 | 0.8818 | 0.6848 | -0.0125 | 0.0513 | 0.4939 | 1.1204 |
| GP + News (deterministic) | 0.0663 | 0.0894 | 0.0473 | 0.8788 | 0.6667 | -0.0489 | 0.0153 | 0.5121 | 1.1337 |
| GP + Macro + News (deterministic) | 0.0661 | 0.0893 | 0.0471 | 0.8909 | 0.6667 | -0.0426 | 0.0227 | 0.4970 | 1.1303 |

Paired CRPS comparisons, with three-month circular blocks resampling entire market months, 2,000 repetitions. Negative means better. Intervals are pointwise, exploratory and not multiplicity-adjusted; they do not account for universe screening or the research choices already informed by earlier experiments.

| strategy | baseline | difference | lower95 | upper95 |
| --- | --- | --- | --- | --- |
| GP + Macro | GP only | +0.000111 | -0.000065 | +0.000364 |
| GP + Macro + News (deterministic) | GP only | +0.000662 | +0.000152 | +0.001216 |
| GP + Macro placebo | GP only | -0.000037 | -0.000163 | +0.000068 |
| GP + Macro transfer | GP only | +0.000060 | -0.000069 | +0.000231 |
| GP + News (deterministic) | GP only | +0.000808 | +0.000390 | +0.001210 |
| GP + News placebo | GP only | +0.000293 | -0.000323 | +0.000938 |
| GP + Macro + News (deterministic) | GP + Macro | +0.000551 | +0.000110 | +0.000995 |
| GP + Macro transfer | GP + Macro | -0.000051 | -0.000166 | +0.000016 |

Additional controls and transfer (no strategy was selected by validation profits):

| Strategy | Net return | Ending $ | Sharpe | Max drawdown | Turnover | Costs |
| --- | --- | --- | --- | --- | --- | --- |
| GP + Macro placebo | 11.90% | 2,237.94 | 0.80 | -10.11% | 91.8% | $29.04 |
| GP + News placebo | 9.28% | 2,185.67 | 0.60 | -12.00% | 106.7% | $34.95 |
| GP + Macro transfer | 10.95% | 2,218.98 | 0.74 | -11.14% | 95.4% | $30.38 |

Gross returns, daily volatility, annualized returns, both drawdowns, effective holdings, best/worst month and positive-month fractions are in `portfolios/window_B_performance.csv`. Daily curves are in `portfolios/daily.csv`; annualized return extrapolates using 252 trading sessions and Sharpe uses zero risk-free return. Transaction costs are charged once using the existing exact self-financing equation.

## Exposure sensitivities

Highest and lowest full-window ridge coefficients, with conditional Gaussian 95% intervals. These end-of-window descriptions are not retroactively used in earlier allocations. Rates/inflation/unemployment coefficients are monthly log return per one percentage-point change; oil is per unit of asinh(WTI/50), volatility per unit log VIX. All are conditional on SPY and the other factors. The complete 22 × 6 matrix and its uncertainty are in `exposures/window_B_exposures.csv`; the `variant` column separates rolling, full-window and news mappings.

| factor | extreme | ticker | coefficient | lower95 | upper95 | n_months |
| --- | --- | --- | --- | --- | --- | --- |
| inflation | highest | IONS | 0.08370 | -0.06802 | 0.23541 | 21 |
| inflation | lowest | LPLA | -0.04214 | -0.14428 | 0.06000 | 21 |
| inflation_expectations | highest | LVS | 0.00974 | -0.19759 | 0.21707 | 21 |
| inflation_expectations | lowest | UPS | -0.01949 | -0.17067 | 0.13169 | 21 |
| oil | highest | COKE | 0.15019 | -0.17984 | 0.48022 | 21 |
| oil | lowest | ALK | -0.23500 | -0.77084 | 0.30083 | 21 |
| rates | highest | COKE | 0.04030 | -0.10547 | 0.18608 | 21 |
| rates | lowest | IONS | -0.11876 | -0.31680 | 0.07929 | 21 |
| unemployment | highest | IONS | 0.07000 | -0.33158 | 0.47157 | 21 |
| unemployment | lowest | DLB | -0.08177 | -0.30628 | 0.14274 | 21 |
| volatility | highest | COKE | 0.09079 | -0.15171 | 0.33329 | 21 |
| volatility | lowest | ALK | -0.27545 | -0.66918 | 0.11827 | 21 |

This is an observational association model, not identification of monetary-policy, inflation or geopolitical causal shocks. Early exposures use few monthly observations and are prior-sensitive. Gaussian intervals are conditional on shrinkage and an approximate independent-error likelihood; they do not provide robust frequentist coverage under serial dependence. Growth is represented only by unemployment, not a complete growth factor. Trade/conflict effects are separate regularized news-feature associations, not reliable company-specific structural exposures.

Ex-post conditional exposure checks use realized macro changes only for scoring, never in a trading forecast:

| window | variant | conditional_mae | conditional_rmse | conditional_coverage95 |
| --- | --- | --- | --- | --- |
| B | transfer_prior | 0.06260 | 0.08443 | 0.95152 |
| B | weak_prior | 0.06335 | 0.08435 | 0.95455 |

## Month and stock contributions

Joint macro/news versus GP-only: the three largest absolute monthly return differences were period 6: +4.84 pp, period 13: +3.67 pp, period 14: +2.56 pp. Net effects include cost differences.

The largest absolute cumulative stock-dollar contribution differences were DLB: $-42.95, DUK: $-42.73, ROP: $+41.32, IONS: $-39.62, EHC: $-33.94. These compare each strategy's own changing capital base; they are an accounting decomposition, not causal attribution. Fees are separate. Full data are in `portfolios/asset_contributions.csv`.

Monthly net returns:

| execution_date | GP only | GP + Macro | GP + News (deterministic) | GP + Macro + News (deterministic) | Equal weight | SPY |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-07-17 | 2.28% | 2.60% | 2.81% | 2.64% | 2.29% | 2.85% |
| 2025-08-18 | -0.12% | -0.75% | 2.48% | 0.26% | 1.75% | 2.67% |
| 2025-09-17 | 2.02% | 1.71% | 1.86% | 2.21% | 1.19% | 0.20% |
| 2025-10-17 | -0.52% | 0.36% | -0.70% | -1.51% | 1.92% | 1.55% |
| 2025-11-17 | -0.42% | -0.60% | 0.91% | 0.60% | 3.61% | 1.52% |
| 2025-12-17 | 1.46% | 1.64% | -3.55% | -3.37% | -0.15% | 0.53% |
| 2026-01-20 | 7.30% | 7.39% | 6.33% | 6.73% | 4.94% | -0.20% |
| 2026-02-17 | -2.47% | -2.52% | -2.95% | -2.86% | -4.11% | -1.14% |
| 2026-03-17 | -2.90% | -3.20% | -3.19% | -3.54% | 2.81% | 5.31% |
| 2026-04-17 | -2.33% | -2.75% | -1.90% | -2.16% | -4.64% | 4.77% |
| 2026-05-18 | 2.34% | 2.35% | 2.00% | 2.46% | 4.77% | 1.55% |
| 2026-06-17 | 3.15% | 2.58% | 2.12% | 2.20% | 2.58% | -0.97% |
| 2026-07-17 | 5.91% | 5.45% | 2.68% | 2.24% | 2.54% | 4.60% |
| 2026-08-17 | -5.35% | -5.01% | -4.05% | -2.79% | -4.98% | -1.68% |
| 2026-09-17 | 2.06% | 1.64% | 2.84% | 2.77% | -0.82% | 0.26% |

All results use an October 2026 hindsight-selected universe. The April–October 2026 subperiod was already examined in prior research and remains explicitly exploratory.
