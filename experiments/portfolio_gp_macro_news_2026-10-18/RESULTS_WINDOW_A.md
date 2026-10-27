| Strategy | Net return | Ending $ | Sharpe | Max drawdown | Turnover | Costs |
| --- | --- | --- | --- | --- | --- | --- |
| GP only | -0.93% | 1,981.48 | 0.10 | -33.59% | 92.3% | $72.71 |
| GP + Macro | -1.98% | 1,960.45 | 0.09 | -33.99% | 95.4% | $77.05 |
| GP + News (deterministic) | -3.99% | 1,920.16 | 0.06 | -32.43% | 107.4% | $83.76 |
| GP + Macro + News (deterministic) | -4.84% | 1,903.17 | 0.05 | -33.87% | 106.2% | $83.74 |
| Equal weight | 46.04% | 2,920.77 | 0.56 | -38.86% | 7.3% | $6.96 |
| SPY | 43.47% | 2,869.34 | 0.55 | -33.72% | 2.3% | $2.00 |

# Window A: July 17, 2019–January 31, 2023

43 walk-forward holding periods, starting after January–June warm-up. Each strategy starts independently with $2,000. All news variants here are **deterministic**, not LLM-based. The final Window A holding is a short January 2023 period; returns are not directly comparable to Window B's shorter evaluation. Mean turnover counts buys plus sells, includes the initial purchase, and can exceed 100%.

## Predictive information and portfolio results

- **GP + Macro:** net return -1.98%, a -1.05 percentage-point difference from GP; gross difference -0.96 points. MAE 0.07478, RMSE 0.11039, CRPS 0.05607. Paired CRPS difference +0.000708, 95% market-month block interval [-0.000170, +0.002319].
- **GP + News (deterministic):** net return -3.99%, a -3.07 percentage-point difference from GP; gross difference -2.54 points. MAE 0.07483, RMSE 0.10966, CRPS 0.05601. Paired CRPS difference +0.000648, 95% market-month block interval [+0.000038, +0.001418].
- **GP + Macro + News (deterministic):** net return -4.84%, a -3.92 percentage-point difference from GP; gross difference -3.48 points. MAE 0.07533, RMSE 0.11114, CRPS 0.05652. Paired CRPS difference +0.001157, 95% market-month block interval [+0.000036, +0.002803].

Price-only GP reference: MAE 0.07403, RMSE 0.10860, CRPS 0.05536. All metrics use identical stock-month observations within this window, in dimensionless monthly log-return units.

| strategy | mae | rmse | crps | coverage90 | coverage68 | correlation | spearman | direction_accuracy | rms_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GP only | 0.0740 | 0.1086 | 0.0554 | 0.8911 | 0.6776 | -0.1289 | -0.0866 | 0.4651 | 1.5427 |
| GP + Macro | 0.0748 | 0.1104 | 0.0561 | 0.8890 | 0.6670 | -0.1575 | -0.0220 | 0.4725 | 1.5494 |
| GP + News (deterministic) | 0.0748 | 0.1097 | 0.0560 | 0.8901 | 0.6712 | -0.1527 | -0.0822 | 0.4577 | 1.5529 |
| GP + Macro + News (deterministic) | 0.0753 | 0.1111 | 0.0565 | 0.8869 | 0.6638 | -0.1702 | -0.0556 | 0.4641 | 1.5563 |

Paired CRPS comparisons, with three-month circular blocks resampling entire market months, 2,000 repetitions. Negative means better. Intervals are pointwise, exploratory and not multiplicity-adjusted; they do not account for universe screening or the research choices already informed by earlier experiments.

| strategy | baseline | difference | lower95 | upper95 |
| --- | --- | --- | --- | --- |
| GP + Macro | GP only | +0.000708 | -0.000170 | +0.002319 |
| GP + Macro + News (deterministic) | GP only | +0.001157 | +0.000036 | +0.002803 |
| GP + Macro placebo | GP only | +0.000224 | -0.000050 | +0.000537 |
| GP + News (deterministic) | GP only | +0.000648 | +0.000038 | +0.001418 |
| GP + News placebo | GP only | +0.000596 | -0.000178 | +0.001486 |
| GP + Macro + News (deterministic) | GP + Macro | +0.000449 | +0.000058 | +0.000961 |

Additional controls and transfer (no strategy was selected by validation profits):

| Strategy | Net return | Ending $ | Sharpe | Max drawdown | Turnover | Costs |
| --- | --- | --- | --- | --- | --- | --- |
| GP + Macro placebo | 7.19% | 2,143.87 | 0.20 | -34.36% | 108.5% | $89.69 |
| GP + News placebo | -0.97% | 1,980.58 | 0.10 | -31.92% | 111.5% | $87.39 |

Gross returns, daily volatility, annualized returns, both drawdowns, effective holdings, best/worst month and positive-month fractions are in `portfolios/window_A_performance.csv`. Daily curves are in `portfolios/daily.csv`; annualized return extrapolates using 252 trading sessions and Sharpe uses zero risk-free return. Transaction costs are charged once using the existing exact self-financing equation.

## Exposure sensitivities

Highest and lowest full-window ridge coefficients, with conditional Gaussian 95% intervals. These end-of-window descriptions are not retroactively used in earlier allocations. Rates/inflation/unemployment coefficients are monthly log return per one percentage-point change; oil is per unit of asinh(WTI/50), volatility per unit log VIX. All are conditional on SPY and the other factors. The complete 22 × 6 matrix and its uncertainty are in `exposures/window_A_exposures.csv`; the `variant` column separates rolling, full-window and news mappings.

| factor | extreme | ticker | coefficient | lower95 | upper95 | n_months |
| --- | --- | --- | --- | --- | --- | --- |
| inflation | highest | COKE | 0.03754 | -0.03140 | 0.10649 | 49 |
| inflation | lowest | LVS | -0.05541 | -0.14109 | 0.03028 | 49 |
| inflation_expectations | highest | LPLA | 0.04748 | -0.12903 | 0.22399 | 49 |
| inflation_expectations | lowest | O | -0.04618 | -0.16551 | 0.07315 | 49 |
| oil | highest | HBAN | 0.30305 | 0.02536 | 0.58074 | 49 |
| oil | lowest | BDX | -0.13715 | -0.29802 | 0.02373 | 49 |
| rates | highest | LPLA | 0.11002 | -0.02007 | 0.24011 | 49 |
| rates | lowest | UPS | -0.03481 | -0.12625 | 0.05662 | 49 |
| unemployment | highest | ROP | 0.01469 | -0.00199 | 0.03138 | 49 |
| unemployment | lowest | ALK | -0.01986 | -0.04924 | 0.00953 | 49 |
| volatility | highest | UPS | 0.05662 | -0.10127 | 0.21451 | 49 |
| volatility | lowest | LVS | -0.14751 | -0.37666 | 0.08163 | 49 |

This is an observational association model, not identification of monetary-policy, inflation or geopolitical causal shocks. Early exposures use few monthly observations and are prior-sensitive. Gaussian intervals are conditional on shrinkage and an approximate independent-error likelihood; they do not provide robust frequentist coverage under serial dependence. Growth is represented only by unemployment, not a complete growth factor. Trade/conflict effects are separate regularized news-feature associations, not reliable company-specific structural exposures.

Ex-post conditional exposure checks use realized macro changes only for scoring, never in a trading forecast:

| window | variant | conditional_mae | conditional_rmse | conditional_coverage95 |
| --- | --- | --- | --- | --- |
| A | weak_prior | 0.07318 | 0.13707 | 0.96512 |

## Month and stock contributions

Joint macro/news versus GP-only: the three largest absolute monthly return differences were period 32: +6.55 pp, period 33: +4.71 pp, period 28: +3.66 pp. Net effects include cost differences.

The largest absolute cumulative stock-dollar contribution differences were UPS: $-87.06, SRE: $-63.57, KNX: $+63.43, COKE: $+62.45, IONS: $-53.20. These compare each strategy's own changing capital base; they are an accounting decomposition, not causal attribution. Fees are separate. Full data are in `portfolios/asset_contributions.csv`.

Monthly net returns:

| execution_date | GP only | GP + Macro | GP + News (deterministic) | GP + Macro + News (deterministic) | Equal weight | SPY |
| --- | --- | --- | --- | --- | --- | --- |
| 2019-07-17 | -6.28% | -6.24% | -5.77% | -5.77% | -1.04% | -2.62% |
| 2019-08-19 | -2.08% | -2.53% | -2.38% | -3.08% | 2.86% | 2.65% |
| 2019-09-17 | 1.34% | 0.80% | 1.25% | 0.62% | -0.44% | 0.37% |
| 2019-10-17 | -0.56% | -0.92% | 0.03% | -0.44% | 2.60% | 3.95% |
| 2019-11-18 | 0.90% | 1.02% | 2.29% | 0.99% | 2.14% | 2.69% |
| 2019-12-17 | 1.79% | 1.94% | 1.79% | 1.97% | 3.48% | 4.19% |
| 2020-01-17 | -0.24% | -0.09% | -0.49% | 0.14% | 0.41% | 1.45% |
| 2020-02-18 | -24.32% | -24.19% | -24.38% | -24.37% | -30.58% | -27.18% |
| 2020-03-17 | 16.22% | 16.33% | 16.22% | 16.32% | 12.92% | 17.15% |
| 2020-04-17 | 0.98% | 0.81% | 0.99% | 0.85% | -1.21% | 2.69% |
| 2020-05-18 | 4.91% | 6.92% | 4.39% | 7.05% | 12.15% | 7.17% |
| 2020-06-17 | 4.06% | 4.33% | 2.22% | 2.26% | 1.09% | 2.94% |
| 2020-07-17 | -0.76% | 0.19% | -0.91% | -0.26% | 4.18% | 4.99% |
| 2020-08-17 | -5.34% | -4.75% | -5.19% | -4.28% | -2.23% | -1.30% |
| 2020-09-17 | 2.24% | 2.17% | 2.46% | 2.13% | 2.72% | 4.94% |
| 2020-10-19 | 1.08% | 1.37% | 0.23% | 0.60% | 8.62% | 3.25% |
| 2020-11-17 | 0.75% | 0.39% | 0.34% | 0.23% | 2.16% | 3.33% |
| 2020-12-17 | 0.09% | 0.26% | -0.20% | -0.10% | 3.24% | 2.15% |
| 2021-01-19 | 1.50% | 2.18% | 1.67% | 1.83% | 1.30% | 3.19% |
| 2021-02-17 | 4.61% | 5.20% | 5.64% | 5.35% | 5.33% | 1.05% |
| 2021-03-17 | 1.78% | 1.72% | 1.58% | 1.55% | 3.83% | 5.85% |
| 2021-04-19 | -1.07% | -0.83% | -0.58% | -0.86% | 0.15% | -0.21% |
| 2021-05-17 | -1.13% | 0.49% | -3.35% | -1.17% | -1.37% | 1.51% |
| 2021-06-17 | 0.04% | 3.10% | -0.31% | 2.97% | -1.87% | 1.40% |
| 2021-07-19 | 1.48% | 0.54% | 1.51% | 0.64% | 3.08% | 4.24% |
| 2021-08-17 | -5.57% | -5.49% | -5.46% | -5.62% | -1.69% | 0.47% |
| 2021-09-17 | -1.42% | -1.49% | -2.28% | -2.01% | -0.21% | -0.21% |
| 2021-10-18 | 2.98% | 3.36% | 6.41% | 6.63% | 3.32% | 5.64% |
| 2021-11-17 | -0.67% | 0.02% | -0.56% | 0.45% | 0.17% | -1.24% |
| 2021-12-17 | -0.78% | -1.70% | 0.45% | -0.90% | 3.98% | -0.39% |
| 2022-01-18 | -2.37% | -3.08% | -0.89% | -2.00% | -1.09% | -3.59% |
| 2022-02-17 | 1.83% | -0.01% | -3.55% | -4.73% | -1.02% | -2.17% |
| 2022-03-17 | 10.00% | 6.24% | 8.43% | 5.30% | 4.74% | 1.06% |
| 2022-04-18 | -5.21% | -5.05% | -5.49% | -5.46% | -4.33% | -6.93% |
| 2022-05-17 | -9.41% | -8.95% | -8.33% | -7.83% | -8.54% | -9.70% |
| 2022-06-17 | 4.17% | 3.61% | 2.93% | 2.05% | 5.83% | 6.26% |
| 2022-07-18 | 8.08% | 8.41% | 10.13% | 10.27% | 10.07% | 9.66% |
| 2022-08-17 | -5.27% | -5.91% | -5.72% | -5.85% | -6.42% | -9.88% |
| 2022-09-19 | -5.79% | -6.92% | -4.50% | -5.21% | -4.69% | -4.77% |
| 2022-10-17 | 7.17% | 4.77% | 5.98% | 4.74% | 6.88% | 7.27% |
| 2022-11-17 | -0.24% | -0.80% | -0.29% | -0.46% | 1.95% | -1.34% |
| 2022-12-19 | 5.11% | 6.17% | 6.03% | 6.15% | 4.37% | 3.91% |
| 2023-01-17 | 2.46% | 2.45% | 1.59% | 2.25% | 1.51% | 2.01% |

All results use an October 2026 hindsight-selected universe. No claim of prospective stock-selection validity follows from this earlier-window test.
