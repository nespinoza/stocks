# ETF risk engine: volatility forecasts and risk-based portfolios

Date: 2026-10-27. Universe: 41 liquid ETFs (`configs/etf_universe.json`): US
equity, sectors, countries, bonds and commodities, all trading since April 2007.
Data: Yahoo daily OHLCV/adjusted closes, retrieved 2026-10-27, checksums in
`run_info.json`. Reproduce with `scripts/download_etf_universe.py` and
`scripts/run_etf_risk_study.py`. The universe was chosen for coverage and
history, not performance, and avoids single-stock survivorship problems.
ETF survivorship is small but nonzero.

## Answers

1. **HAR-RV with a Garman–Klass proxy is the best volatility forecaster tested,
   and the only well-calibrated one.** It beats the repository's GJR-GARCH and the
   volatility GP on QLIKE. It should be the baseline any new volatility model must beat.
2. **Risk-only portfolios work as risk tools, not return engines.** They cut
   volatility and drawdown a lot. Their Sharpe ratios (0.74–0.83) are close to SPY
   (0.86) and below a plain 60/40 (1.00) over 2010–2026. No allocation here beats
   60/40 on risk-adjusted terms, so 60/40 is the benchmark to beat.
3. **Risk-model quality barely changes minimum-variance outcomes in this
   universe.** Realized volatilities are within 0.2 pp of each other. The
   difference is in *calibration*: PCA-5 under-predicts portfolio risk by 32% and the
   sample covariance by 8%. Ledoit–Wolf is the best calibrated (0.97).

## 1. Volatility forecast benchmark

Next-21-session realized variance (summed squared close-to-close log returns),
forecast at each month end from 2009-12-31 to 2026-08-31: 201 months × 41 ETFs.
Scored on the 8,069 rows where every model succeeded. Lower QLIKE is better. The
Diebold–Mariano test uses the monthly cross-sectional mean QLIKE against `har_gk`
(positive = worse than HAR).

| Model | Mean QLIKE | MSE (log var) | 95% coverage of 21-day return | Kupiec p | DM vs HAR-GK (p) | ETFs where it beats HAR-GK | Failed fits |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HAR, Garman–Klass proxy** | **0.285** | **0.454** | **95.0%** | **0.94** | — | — | 0 |
| HAR, squared-return proxy | 0.316 | 0.520 | 95.7% | 0.005 | +2.7 (0.007) | 5/41 | 0 |
| GJR-GARCH, 252 days (repo model) | 0.335 | 0.495 | 93.4% | <0.001 | +1.9 (0.066) | 3/41 | 171 |
| EWMA λ = 0.94 | 0.342 | 0.496 | 92.8% | <0.001 | +3.0 (0.003) | 4/41 | 0 |
| Volatility GP, 63 days (repo model) | 0.352 | 0.498 | 93.4% | <0.001 | +3.0 (0.003) | 5/41 | 1 |
| Rolling 63-day variance | 0.430 | 0.571 | 93.1% | <0.001 | +2.0 (0.053) | 0/41 | 0 |

- Intraday range information (Garman–Klass) is worth about 10% in QLIKE over squared
  returns. That is free information the GP and GJR ignore. Feeding a range-based
  proxy into the GP's likelihood is a cheap upgrade worth testing.
- GJR's 171 failed fits (2%) are concentrated in low-volatility bond/gold ETFs
  (LQD 17, GLD 14, TIP 14). They were recorded, not replaced.
- The volatility GP, the repository's main volatility model, ranks fifth of six
  here. Its edge on single-stock AMZN did not carry over to this broad panel.

## 2. Risk-based portfolios, 2010-01 to 2026-10

Monthly rebalancing: decide at the month-end close, execute at the next close,
5 bp per unit traded (`stock_api.engine`). No return forecasts are used anywhere.
Minimum-variance weights are capped at 25%. Covariances use 252 daily returns.
"HAR + LW corr" combines HAR volatilities with Ledoit–Wolf-shrunk correlations.

| Strategy | CAGR | Vol | Sharpe | Max DD | Sharpe 2010–17 | Sharpe 2018–26 | Turnover / month | Realized ÷ predicted vol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY | 14.0% | 17.0% | 0.86 | −33.7% | 0.95 | 0.81 | 0% | 1.04 |
| 60/40 SPY/IEF | 9.7% | 9.7% | **1.00** | −21.2% | 1.28 | 0.84 | 2% | 1.03 |
| Equal weight (41) | 8.7% | 13.4% | 0.69 | −30.2% | 0.66 | 0.71 | 3% | 1.06 |
| Inverse vol (HAR) | 5.6% | 7.5% | 0.76 | −16.6% | 0.82 | 0.73 | 11% | 1.06 |
| Risk parity (HAR + LW corr) | 4.1% | 5.0% | 0.83 | −15.2% | 1.05 | 0.74 | 10% | 1.05 |
| Risk parity (sample cov) | 4.1% | 5.2% | 0.80 | −14.7% | 1.03 | 0.72 | 6% | 1.05 |
| Min var (sample cov) | 2.6% | 3.6% | 0.74 | −15.0% | 1.24 | 0.47 | 8% | 1.08 |
| Min var (Ledoit–Wolf) | 2.8% | 3.6% | 0.80 | −14.8% | 1.28 | 0.54 | 7% | **0.97** |
| Min var (PCA-5) | 2.7% | 3.7% | 0.74 | −14.3% | 1.27 | 0.44 | 8% | 1.32 |
| Min var (HAR + LW corr) | 2.7% | **3.5%** | 0.79 | −14.7% | 1.27 | 0.52 | 22% | 1.16 |

Notes:
- Unlevered risk parity and minimum variance hold mostly bonds, so their low CAGR
  is by construction. The fair comparison is Sharpe, or a version scaled to a
  volatility target. Levering is the natural next experiment, with explicit
  financing costs.
- Every strategy's Sharpe fell from 2010–17 to 2018–26. The bond-heavy ones fell
  most (rising rates in 2022). One 16-year sample with two sub-periods is thin
  evidence for choosing among these.
- Minimum variance using HAR volatilities has 3× the turnover of LW for almost no
  volatility reduction. Fast-moving volatility forecasts are better used for
  scaling (inverse volatility / risk parity) than as optimizer inputs.
- With the predicted variance fixed at the decision date, realized/predicted
  ratios above 1 for optimized portfolios are the expected optimizer bias
  (minimum variance concentrates in assets whose risk is underestimated). LW is the
  only risk model that offsets it here.

## What this means for the agent

- Use **HAR-GK volatilities with Ledoit–Wolf correlations** as the default risk
  model, and **risk parity or 60/40-style allocations** as the base portfolio. Any
  signal (momentum, GP drift, news) should be judged by whether it improves on
  that base, net of costs, in a Black–Litterman tilt (next step, report §C3).
- The volatility GP and GJR stay available as candidates, but they now have a
  clear bar to clear: QLIKE below 0.285 on this panel.

## Caveats

- One sample period (2010–2026) dominated by a US equity bull market. ETF
  histories are current vendor-adjusted series, not point-in-time archives.
- Execution at the next close with 5 bp is realistic for these ETFs but ignores
  taxes and financing.
- The volatility-forecast target uses close-to-close returns, which favors
  models of close-to-close variance. HAR still won.
