# Black–Litterman momentum tilts on the ETF risk-parity base

Date: 2026-10-27. Same 41-ETF universe, data and HAR-GK volatility forecasts as
`reports/etf-risk-study-2026-10-27`. Reproduce with
`scripts/run_etf_bl_study.py` after `scripts/run_etf_risk_study.py`. Outputs are
`signal_ic.csv`, `strategies.csv`, `attribution.csv`, `daily_net.csv` and
`run_info.json`.

## Answers

1. **Neither momentum signal has a significant cross-sectional IC on this
   panel.** Over 21 days, the 12-1 month IC is +0.012 (t 0.47) across all 41 ETFs
   and +0.034 (t 1.39) within the 30 equity ETFs. It was positive in 2010–17 and
   about zero afterwards.
2. **Black–Litterman tilts from 12-1 momentum raise the information ratio against the
   risk-parity base (0.44–0.56, t 1.9–2.5), but that is mostly added risk, not skill.**
   The tilts carry beta 1.02–1.30 to the base. After adjusting for it, alpha is
   +1.0 to +1.9%/yr with t 1.4–1.9, and it is negative against a static
   SPY/IEF/GLD/DBC mix. The best of the 12 tilt configurations has a deflated Sharpe
   ratio of 0.75, below the 0.95 needed. PBO across the grid is 0.23, so the ranking
   of configurations is fairly stable, but the best one is not significant.
3. **The GP-drift stand-in (1-month momentum) hurts.** Its IC is −0.017, its
   information ratio −0.10 to +0.07 and its beta-adjusted alpha about 0. At $1B of
   assets, impact costs push its Sharpe ratio negative (−0.20 to −0.46 without a
   turnover penalty). This matches the re-score of the saved GP studies: the GP's
   view is short-term momentum, and short-term momentum is not a source of return.
4. **Turnover control is worth more than any view.** The cost-penalized optimizer
   beat the unpenalized one in 5 of the 6 signal/IC pairs on information
   ratio, while cutting turnover by 40–80% (12-1, IC 0.05: 44% → 10% a month). A
   1–2% no-trade band on the base raised Sharpe from 0.83 to 0.87–0.88. Part of that
   gain is drift toward equities (beta-adjusted alpha +0.3 to +0.5%/yr, t 2.0).
5. **The flat 5 bp cost assumption is fine at $10M and badly wrong at $1B for
   high-turnover strategies.** With the spread + √impact model, the unpenalized
   1-month tilt costs 9.5%/yr at $1B against 0.9%/yr at 5 bp.

**Recommendation:** keep risk parity with a 1–2% band as the base. Keep the
Black–Litterman layer and the cost-penalized optimizer as the place where views
enter, but do not switch on a price-momentum view: none passes the bar. The layer
is now ready for news views (step 5). The test of any news signal is the same as
here: IC first, then beta-adjusted alpha against the base, net of the cost model,
deflated by the number of variants tried.

## Method

- **Signals** (from data through the decision close only):
  - 12-1 momentum = log return from t−252 to t−21, divided by the 252-day volatility × √231.
  - 1-month momentum = the last 21 days' log return / (volatility × √21). This
    stands in for the GP drift, which had Spearman 0.71–0.79 with 30-day momentum
    in the saved studies. Running the GP itself on 41 ETFs × 202 months was not
    worth the compute given that correlation.
  - Combined = the mean of the two cross-sectional z-scores.
  - Risk adjustment keeps bonds from always ranking last.
- **IC:** cross-sectional Spearman between the signal at month end t and the
  return from the execution close (t+1) to t+1+h, for h = 5, 21 and 63 sessions.
  Newey–West t-statistics, with at least h/21 lags for overlapping outcomes.
- **Prior:**
  - Risk model: HAR-GK volatility × Ledoit–Wolf correlation (monthly).
  - Base weights: equal risk contribution (ERC).
  - Implied returns: π = δΣw, with δ chosen so the base has a 0.5 annual Sharpe ratio.
  - Without views, the optimizer returns exactly the base (unit-tested).
- **Views:**
  - One absolute view per asset, Q = π + α, with Grinold α = IC · σ · z.
  - The ICs are **fixed in advance at 0.05 and 0.10, not fitted**, so there is no
    look-ahead in view strength.
  - View uncertainty uses He–Litterman Ω = diag(τΣ) (confidence 0.5), with τ = 0.05.
- **Optimizer:**
  - Maximizes μᵀw − δ/2 wᵀΣw − κ‖w − w_held‖₁: long-only and fully invested.
  - w_held are the drifted holdings at the decision close.
  - κ = 0 or 5 bp (the one-way cost). The L1 term creates a no-trade region and is
    solved exactly with buy/sell split variables.
- **Execution:** decide at the month-end close, trade at the next close, 5 bp per
  unit traded (`stock_api.engine`, now with policy callbacks, no-trade bands and a
  pluggable cost model).
- **Cost sensitivity** (`stock_api.costs`), recomputed from the same trades:
  - Formula: half-spread + 1.0 · σ_daily · √(trade / ADV).
  - ADV and σ come from the 63 days before execution.
  - Half-spreads are *assumed* by dollar-volume tier (0.5 / 1.5 / 3 / 6 bp), not measured.
- **Multiple testing:** the deflated Sharpe ratio of the best daily active return
  over the 12 tilt configurations (3 signals × 2 ICs × penalty on/off), and CSCV
  PBO with 16 blocks.

## Signal IC (21-session horizon)

| Signal | Universe | 2010–2026 IC (t) | 2010–17 IC (t) | 2018–26 IC (t) |
|---|---|---:|---:|---:|
| 12-1 momentum | all 41 | +0.012 (0.47) | +0.049 (1.22) | −0.018 (−0.60) |
| 12-1 momentum | equity 30 | +0.034 (1.39) | +0.063 (1.64) | +0.013 (0.43) |
| 1-month momentum (GP proxy) | all 41 | −0.017 (−0.82) | −0.052 (−1.59) | +0.018 (0.72) |
| 1-month momentum (GP proxy) | equity 30 | −0.007 (−0.35) | −0.048 (−1.44) | +0.029 (1.24) |
| Combined | all 41 | −0.003 (−0.17) | −0.006 (−0.16) | +0.004 (0.18) |

5- and 63-session horizons tell the same story (`signal_ic.csv`). Of 54
signal × horizon × universe × period cells, the largest |t| is 1.98 (1-month
momentum, 5 days, equities, 2018–26, p = 0.05). That is about what 54 tries
produce by chance.

## Strategies, 2010-01 to 2026-10

Active return, information ratio (IR) and t are against the risk-parity base,
from monthly sums of daily net-return differences. "β-adj. α" is the intercept of
monthly excess returns on the base's excess returns (Newey–West t).

| Strategy | Sharpe | Vol | Max DD | Turnover / mo | Active %/yr | IR (t) | β to base | β-adj. α %/yr (t) | Sharpe at $1B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Risk parity (base) | 0.83 | 5.0% | −15.2% | 10% | — | — | — | — | 0.81 |
| Risk parity, 1% band | 0.87 | 5.1% | −15.9% | 6% | +0.3 | 0.55 (2.1) | 1.00 | +0.3 (2.0) | 0.86 |
| Risk parity, 2% band | 0.88 | 5.2% | −15.7% | 5% | +0.4 | 0.43 (1.8) | 1.00 | +0.5 (2.0) | 0.87 |
| 60/40 SPY/IEF | 1.00 | 9.7% | −21.2% | 2% | +5.6 | 1.04 (5.1) | 1.49 | +4.3 (4.2) | 0.99 |
| BL 12-1, IC .05 | 0.83 | 6.5% | −12.4% | 44% | +1.3 | 0.44 (1.9) | 1.06 | +1.0 (1.5) | 0.67 |
| BL 12-1, IC .05, penalized | **0.92** | 6.0% | −9.9% | 10% | +1.4 | 0.50 (2.1) | 1.02 | +1.3 (1.9) | 0.87 |
| BL 12-1, IC .10 | 0.77 | 8.5% | −14.7% | 54% | +2.5 | 0.51 (2.3) | 1.30 | +1.5 (1.4) | 0.54 |
| BL 12-1, IC .10, penalized | 0.82 | 8.3% | −12.7% | 21% | +2.7 | 0.56 (2.5) | 1.25 | +1.9 (1.6) | 0.73 |
| BL 1-month, IC .05 | 0.61 | 6.2% | −20.2% | 120% | −0.3 | −0.10 (−0.4) | 1.02 | −0.4 (−0.6) | −0.20 |
| BL 1-month, IC .05, penalized | 0.69 | 5.7% | −17.2% | 41% | −0.2 | −0.07 (−0.3) | 0.96 | −0.0 (−0.1) | 0.31 |
| BL 1-month, IC .10 | 0.54 | 8.2% | −24.8% | 145% | +0.3 | 0.07 (0.3) | 1.16 | −0.2 (−0.1) | −0.46 |
| BL 1-month, IC .10, penalized | 0.56 | 7.9% | −23.9% | 86% | +0.3 | 0.06 (0.3) | 1.11 | −0.0 (−0.0) | −0.16 |
| BL combined, IC .05 | 0.77 | 6.4% | −15.2% | 84% | +0.8 | 0.25 (1.1) | 1.01 | +0.7 (1.0) | 0.30 |
| BL combined, IC .05, penalized | 0.84 | 6.1% | −13.2% | 20% | +1.0 | 0.35 (1.6) | 0.98 | +1.0 (1.6) | 0.71 |
| BL combined, IC .10 | 0.70 | 8.4% | −17.4% | 104% | +1.7 | 0.34 (1.5) | 1.19 | +1.0 (0.9) | 0.07 |
| BL combined, IC .10, penalized | 0.71 | 8.2% | −17.0% | 46% | +1.6 | 0.34 (1.5) | 1.17 | +1.1 (1.0) | 0.39 |

Best tilt (12-1, IC 0.10, penalized): deflated Sharpe ratio **0.75**. The
expected best of 12 null trials has a daily Sharpe of 0.022 against the observed
0.032. PBO = 0.23.

Against a static SPY/IEF/GLD/DBC regression, every strategy's alpha is ≤ 0. The
risk-parity base itself shows −1.2%/yr (t −3.0): spreading across 41 ETFs
(international equity, commodities) cost return relative to US-centric asset-class
betas in this period (`attribution.csv`). That is a fact about 2010–2026, not a
reason to drop diversification, but it means 60/40 is a hard benchmark.

## Caveats

- One 16-year sample, two sub-periods. 12-1 momentum's IC was positive in
  2010–17 and about zero in 2018–26. ETF momentum is a documented but weak premium.
- The view mapping (absolute views, Ω = diag(τΣ), α = IC·σ·z) gives larger α to
  high-volatility assets, so positive momentum in equities during a bull market
  raises equity weight. That is why the beta-adjusted alpha, not the raw active
  return, is the number to trust.
- The 1-month signal is a proxy for the GP drift, not the GP itself.
- Spread tiers are assumptions. There is no financing or tax modeling.
