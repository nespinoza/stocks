# Re-scoring the saved GP portfolio studies with `stock_api.evaluation`

Date: 2026-10-27. Nothing was refitted: this applies the new evaluation module
(`stock_api/evaluation/`) to the forecasts and daily equity curves already saved by
the three GP portfolio studies. Reproduce with
`.conda-env/bin/python scripts/rescore_saved_forecasts.py` (needs the local,
git-ignored price caches). Full tables: `tables.md` and the CSVs in this folder.

**Verdict: the saved results show no forecasting or allocation skill that survives
standard controls.** The GP drift behaves as a one-month momentum tilt. Its
apparent profits come from the universe and from that factor exposure, and every
"improvement" from macro/news is indistinguishable from its own placebo.

## 1. Signal quality (cross-sectional IC, Newey–West t-statistics, months as units)

| Study / window | Months | IC: GP drift | IC: trailing 30d return | IC: GP drift after removing trailing return | XS corr(GP drift, trailing 30d) |
|---|---:|---:|---:|---:|---:|
| 22-stock A (2019–23) | 43 | **−0.087 (t −2.6)** | −0.052 (t −1.4) | −0.041 (t −1.4) | 0.75 |
| 22-stock B (2025–26) | 15 | +0.096 (t 1.5) | +0.018 (t 0.2) | +0.030 (t 0.5) | 0.71 |
| growth-20 A | 43 | −0.008 (t −0.2) | −0.010 (t −0.3) | +0.040 (t 1.1) | 0.71 |
| growth-20 B | 15 | +0.037 (t 0.8) | −0.039 (t −0.6) | +0.120 (t 2.0, p 0.07) | 0.79 |

- The only statistically significant IC is **negative**: GP drift *predicted the
  wrong direction* in 22-stock window A.
- After removing the trailing-return component, nothing is significant. The
  growth-20 B residual (p = 0.07, 15 months, one of many tests) is the only hint
  worth re-testing prospectively.
- Macro/news strategy means: no IC is significant except 22-stock A (news and
  news-placebo, both **negative**) and growth-20 A joint (+0.069, p = 0.04, before
  any multiple-testing adjustment across about 40 rows).

## 2. Did macro/news improve probabilistic forecasts? (Diebold–Mariano on monthly mean CRPS)

Positive difference means the augmented model is *worse* than the price-only GP.

- 22-stock B: **news is worse than GP (p = 0.002)**, and macro+news is worse (p = 0.03).
- Every other comparison (18 of 20): p > 0.1, and placebos score like the real features.
- No augmentation is significantly better than the GP in any window.

## 3. Calibration of the GP's monthly return distributions

| Study / window | 90% interval failures (expected 10%) | Kupiec p | Tickers rejecting Christoffersen CC (90%) | Tickers rejecting Berkowitz (Gaussian PIT) |
|---|---:|---:|---:|---:|
| 22-stock A | 10.9% | 0.37 | 0/22 | 2/22 |
| 22-stock B | 11.5% | 0.37 | 0/22 | 2/22 |
| growth-20 A | **14.1%** | **<0.001** | 2/18 | **8/18** |
| growth-20 B | **13.7%** | **0.04** | 1/20 | 1/20 |

Volatility calibration is good for liquid large/mid caps (the 22-stock universe)
and **too narrow for small, volatile growth names**. The growth-20 B PIT histogram
piles up in the top decile (59 vs about 31 expected): upside jumps are under-predicted.
That supports adding jump/event variance (report §C1). The Berkowitz test uses
the moment-matched Gaussian PIT `Φ(z)`, an approximation to the mixture forecast.

## 4. Factor attribution (daily net returns on FF5 + UMD + STR, Newey–West t)

Factor data end 2026-08-31, so window B regressions stop there.

**GP active return over equal weight** (same universe, same costs):

| Study / window | Alpha, annualized | t | β UMD (12–1 momentum) | t | β STR (short-term reversal) | t |
|---|---:|---:|---:|---:|---:|---:|
| 22-stock A | **−9.4%** | −2.0 | +0.10 | 4.1 | **−0.14** | **−3.4** |
| 22-stock B | +1.0% | 0.1 | +0.04 | 1.4 | −0.06 | −1.9 |
| growth-20 A | +6.1% | 0.8 | +0.10 | 3.8 | **−0.09** | **−3.9** |
| growth-20 B | +41% | 1.65 | −0.64 | −4.6 | **−0.61** | **−3.0** |
| 22-stock Apr–Oct 2026 | +10% | 0.9 | +0.01 | 0.2 | +0.02 | 0.8 |

- The GP's tilt consistently loads **negatively on short-term reversal**, meaning it
  is long last month's winners. This independently confirms report §B1.
- No active alpha is significant at 5% after factor controls. The best case
  (growth-20 B, +41%/yr) has t = 1.65 on 14 months in a hindsight-selected universe.
- In growth-20 B **equal weight alone** has +85%/yr alpha (t 3.3) against FF factors.
  That is the universe-selection effect (report §B2) and dwarfs anything the model does.

## 5. Overfitting statistics (22-stock six-month study)

- **Deflated Sharpe ratio** of the primary GP (annualized Sharpe 0.82, 127 days)
  against its own 18-configuration sensitivity grid: **0.60** (significance needs
  ≥ 0.95). If 50 or 100 configurations were tried repository-wide, it drops to 0.57 or 0.55.
  A Sharpe of 0.82 over six months is not distinguishable from luck.
- **Probability of backtest overfitting (CSCV)** across that grid: **0.67**. Across
  the GP/macro/news/placebo variants it is 0.43–0.90. Picking the in-sample best
  variant is about as likely as not to give a below-median variant out of sample.

## Caveats

- Monthly units: 43 months (A) and 15 months (B). With 15 months, intervals are
  wide, and the HAC bandwidth rule is crude.
- Signals and calibration use the saved forecast-aligned close-to-close returns
  (`actual`), not the open-to-open holding returns.
- The deflated Sharpe trial variance comes from the saved grid only. The true
  number of configurations explored in this repository is larger and unrecorded.
