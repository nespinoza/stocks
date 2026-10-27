# Repository review: stochastic models, news signals and portfolio studies

Reviewer: Claude (Opus 5.5), 2026-10-26. Scope: all code under `stock_api/`, `scripts/` and `tests/`; the top-level docs (`README.md`, `VALIDATION.md`, `HISTORICAL_VALIDATION.md`, `RETURN_GP.md`, `SDE.md`, `GARCH.md`); and every report under `reports/`, `validation_runs/` and `experiments/`. The test suite passes (216 tests, about 1,500 dependency deprecation warnings), so the problems below are about research design, not broken code. I wrote no new models and re-ran no experiments. I did run **one new diagnostic** on the saved forecast CSVs (§B1), because it explains most of the portfolio results.

---

## TL;DR

1. **The volatility side is sound; the "alpha" side is not.** The heteroskedastic GP gives reasonably calibrated volatility (90% coverage around 86–92%, realized-to-predicted variance around 1.0). Its expected-return ("drift") output, which drives every portfolio, is **mostly a trailing 30-day momentum score**. Its cross-sectional rank correlation with the past 30-day return is **0.71–0.79** in all four windows I checked. The GP adds no forecasting skill beyond that (§B1).
2. **Every positive portfolio result was produced on a universe chosen with hindsight.** The 22-stock list was chosen using an audit period (2024–Sep 2026) that contains the test windows, and its selection blocks (2017–2023) contain Window A. The growth-20 list is made of the year's winners. On a universe like that, a momentum tilt has to win. The one window with no recent-winner bias (22 names, 2019–2023) shows **GP −0.9% against equal weight +46%**. That is what a one-month momentum signal does in a short-term-reversal regime (§B2).
3. **The "news" signal was never news sentiment.** It was regex flags on FOMC statements, plus counts of conflict events from **one 15-minute GDELT file sampled twice a month** (about 15 events per sample). These are market-wide scalars with no company-level text, and they were mapped onto each stock through ridge regressions with 3–24 observations. The negative finding tells us nothing about LLM news sentiment (§B6, §C2).
4. **Several settings were tuned on almost no data and then "frozen" as if validated.** Examples: the 30-day lookback (picked from 12 AMZN folds), λ = 10 (gives about 95% monthly turnover), the macro mappings (switched on after 3 months), and placebos (a single permutation).
5. **The infrastructure is better than the science.** Codex built careful point-in-time machinery: ALFRED vintages, checksums, chronological guards and audits. The best next step is to point that machinery at the missing validation pieces (§D), not to build more models.

---

## (A) Changes I suggest to the repository

### A1. Hygiene and version control (do first)

| Problem | Evidence | Fix |
|---|---|---|
| Large uncommitted work | 20 modified tracked files; ~40 untracked code/test files (`stock_api/garch.py`, `portfolio.py`, `macro_news.py`, all `scripts/*`) | Commit the code now on a branch, in logical chunks. The core of the portfolio and news work exists only in your working tree. |
| ~2.2 GB of untracked artifacts | `historical_validation/` 1.0 GB, `reports/` 516 MB, `experiments/` 367 MB, `diagnostics/` 252 MB, `validation_runs/` 117 MB | Keep code and small `RESULTS.md` files in git. Move raw caches and predictive samples to an external store (DVC, git-lfs, or an `artifacts/` directory in `.gitignore`), keyed by the content hashes the manifests already record. |
| Stray files at root | `2016-01-*__*_gp_returns__*.json`, `check_progress.py`, `.DS_Store`, `stock_value.txt` | Delete or move them. Add `.DS_Store`, `*.json` at root, `__pycache__` and `MPLCONFIGDIR` artifacts to `.gitignore`. |
| Two environments | `.venv` (py3.11, 297 MB) and `.conda-env` (py3.13, 520 MB) | Pick one (the docs recommend 3.13). Add a lock file (`uv lock` or `conda-lock`). Delete `setup.py` (duplicates `pyproject.toml`), `build/` and `*.egg-info`. |
| Legacy demos | `example.py`, `utils.py`, `predict_stock.py` | Move them to `legacy/` or delete them. `utils.get_stock_data` is a third, unvalidated data path. |

### A2. Architecture: separate the library from the research

The codebase has grown one ~300-line script per experiment (`run_portfolio_gp_22.py`, `run_macro_news_study.py`, `report_*`, `audit_*`), and each one reimplements schedules, accounting and reporting. For an autonomous agent you need one pipeline with replaceable stages:

```
data/        point-in-time store (raw + adjusted OHLCV, corporate actions, universe membership, news) → parquet
signals/     price models (GP, GARCH, ...) and news/LLM features → standardized per-asset scores + uncertainty
risk/        covariance / volatility models (GARCH, factor model, Ledoit–Wolf, GP-vol)
portfolio/   construction (Black–Litterman, mean–variance, risk parity, constraints, turnover bands)
execution/   broker adapter (paper first, e.g. Alpaca/IBKR), order sizing, cost model
backtest/    ONE event-driven walk-forward engine used by every study
evaluation/  forecast scoring, portfolio metrics, statistical tests (§D)
research/    experiment configs + notebooks; no library logic
```

Specific code changes:

- **Replace the `(mean, variance, info)` tuple with magic dict keys** (`_fit`, `_statistics`, `_ensemble`, `_fit_statistics` in `stock_api/forecast.py:_fit_forecast`) with a typed `PredictiveDistribution` object that holds samples and/or quantiles plus metadata. Half of `forecast.py` is code that translates between the Gaussian and sample conventions.
- **Make the model interface return-space and multi-asset first.** Portfolios need a joint distribution of next-period returns. The price-level API (`predict("AMZN")` → 7-day prices) is a demo layer and should sit on top of that, not underneath it.
- **Move logic out of `scripts/`** into tested modules. The accounting in `run_portfolio_gp_22.py` and `run_macro_news_study.py` (fills, fees, drifted weights) should be one function used by both.
- **Readability.** `stock_api/macro_news.py` is dense semicolon-chained one-liners, which makes it very hard to audit. Run `ruff format`. Add `ruff`, `mypy` and `pytest` in CI (a GitHub Action or pre-commit hook).
- **One data layer.** Today there are four paths: `utils.get_stock_data`, `stock_api.data.load_prices`, `historical.runner.download`, and the per-experiment Yahoo caches. Unify them into a single point-in-time store and make every study read from it.
- **Split the docs.** `README.md` is 39 KB and mixes the API reference with research findings. Keep the README to installation plus a quickstart, and put model docs under `docs/`.
- **A results template.** Codex's reports bury the answer under defensive caveats. Every `RESULTS.md` should start with: question → setup (universe, dates, n independent periods) → headline metric with CI → verdict (supported / not supported / inconclusive). Caveats go after that.

### A3. Agent safety and operations (needed before any live capital)

- Paper-trading loop with immutable, timestamped decision logs. Each log entry holds inputs, model versions, prompt and LLM output, target weights and fills.
- Hard risk limits outside the model: maximum position size, maximum gross exposure, a daily-loss or drawdown kill switch, and a liquidity filter (ADV %).
- A "model health" monitor. Track rolling calibration (PIT, coverage) and signal IC, and fall back to a conservative portfolio (equal weight or minimum variance) when a signal degrades.
- Human approval for any change to universe, model or parameters, recorded as a new config version.

---

## (B) Studies whose settings likely produced misleading answers

### B1. The GP "drift" is a 30-day momentum score *(new diagnostic, high confidence)*

I joined the saved `forecasts/window_{A,B}_forecasts.csv` files ("GP only" rows) to the saved price files. For each origin, I computed the trailing 30-session log return ending at that origin:

| Experiment / window | Months | XS Spearman(GP μ, trailing 30d) | XS Spearman(trailing 30d, next month) | XS Spearman(GP μ, next month) |
|---|---:|---:|---:|---:|
| 22-stock, A (2019–23) | 43 | **+0.75** | −0.052 | **−0.087** |
| 22-stock, B (2025–26) | 15 | **+0.71** | +0.018 | +0.096 |
| growth-20, A (18 names) | 43 | **+0.71** | −0.010 | −0.008 |
| growth-20, B | 15 | **+0.79** | −0.039 | +0.037 |

(XS = average cross-sectional rank correlation across months.)

Why this happens: with 30 daily returns, the standard error of the mean daily return (about σ/√30) is as large as any plausible drift. So the posterior drift is a smoothed, shrunk version of the recent return. The Matérn length-scale prior is centered on 20 days, which reinforces this. The GP adds a learned smoothing kernel, but its rankings are mostly the past month's ranking.

Consequences:
- In the 6-month 22-stock study, estimated drift had **worse** MAE, RMSE and CRPS than zero drift (0.0642 vs 0.0626 MAE). Yet the drift portfolio beat the zero-drift one by 7 pp. The portfolio gain did not come from forecast skill; it came from six months in which one-month momentum happened to pay. The "GP maximum mean" variant (+13.2%) is simply the purest momentum bet.
- Window A, a regime in which one-month returns reversed (the well-documented short-term reversal effect, Jegadeesh 1990), is the only clean test, and the GP lost 47 pp against equal weight there.
- **Fix:** every GP portfolio study needs (i) a plain 1-month momentum benchmark with the same caps, costs and optimizer, (ii) a short-term-reversal benchmark, and (iii) attribution of GP returns onto those factors. If the GP adds nothing after controlling for trailing 1-month return, drop the drift and keep the GP for volatility.

### B2. Universe selection leakage invalidates every positive portfolio number

- **22-stock universe** (`reports/monthly-stock-screen-2026-10-18`): the "22 positive-skill" names were chosen because they were positive in 2017–2020, 2021–2023 **and** the Jan 2024–Sep 2026 audit. The audit period contains Window B (Jul 2025–Oct 2026) and the 6-month test (Apr–Oct 2026). The selection blocks contain Window A (2019–2023). By the screen's own numbers the choice was noise: after Holm correction 0 of 686 were significant, and direction Brier skill was −2%. On top of that, the source is the **current** Russell 1000 with continuous history since 2009, which is survivorship.
- **Growth-20**: a hand-picked list of the period's top performers, including micro-caps such as FATN (about $5), AXTI, QMCO and SAIH. "GP 475% vs EW 336%" measures how well a capped momentum tilt concentrates into ex-post winners. It says nothing about skill. Window B's shuffled-macro placebo (483%) matching the real model (483%) confirms this.
- Codex disclosed this in caveats, but the headline tables still read as performance. **Treat all of these numbers as uninformative.**

### B3. The 30-day lookback was "validated" on 12 data points

`validation_runs/return-gp-2026-09-01-benchmarks/REPORT.md`: 12 monthly origins, a 7-day horizon, one ticker (56 scored prices), 4 lookbacks × 9 models. Choosing the best of 36 configurations on 12 terminal outcomes is selection on noise. Later work cites this as "the previously identified reliable GP setting" and reuses it unchanged in a **monthly, 22-asset** portfolio problem: a different horizon, different assets and a different loss. The much larger 556-origin AMZN study (`reports/AMZN_HISTORICAL_REVIEW.md`) found **nothing beats persistence**, which is the more trustworthy result. **Fix:** choose the lookback by a pre-registered criterion on a large panel of stocks and origins (§D1), at the horizon the portfolio actually uses.

### B4. λ = 10 makes the optimizer chase the mean, causing ~95% monthly turnover

In monthly log-return units, Σ entries are about 0.003–0.01, so the risk term λΣw is small compared with μ differences of 0.01–0.03. The optimizer is dominated by the mean: 1.5 names sit at the 20% cap on average, and two-sided turnover is 88–107% per month (the news variants were the highest, because noisy μ shifts cause trades). The sensitivity grid shows net return falling steadily as λ rises, which is what you expect when μ is noise. **Fix:** set λ so the ex-ante tracking error against equal weight hits a target (for example 3–5%/yr). Better still, put signals through Black–Litterman (§C3), where signal uncertainty decides how far you tilt. Add no-trade bands.

### B5. The macro study was set up to return a null

- The ex-ante factor forecasts come from AR(1) models of monthly macro changes, and they have ≤0 skill against "no change" for 10 of 12 factor-windows (`RESULTS.md` table). The mean shift is β × (≈0 forecast), so the macro channel could only add noise.
- Each stock's mapping is a ridge regression of GP residuals on 6 factors plus SPY, using **as few as 3 completed months** (and at most 24). It is badly underdetermined. Only 2 of 132 and 1 of 132 coefficients excluded zero, which is what chance gives.
- **Better use of macro:** condition **risk** on it (regime-dependent volatility and correlation, for example VIX and credit-spread regimes), or use contemporaneous exposures for stress tests and hedging. Do not use it to forecast returns. Where it is used for means, use surprises (actual minus consensus, or market-implied moves around FOMC/CPI releases) rather than AR forecasts of levels.

### B6. The "news" study did not test news

- Input: regex flags on FOMC statements (8 per year, binary features), plus `log1p(count/coverage)/5` of CAMEO conflict events from **one 15-minute GDELT export at 12:00 UTC on the 5th and 15th**. The coverage file shows about 13–22 retained events per snapshot. That is about 30 minutes of global news per month.
- Every feature is a **market-wide scalar**, the same for all 22 stocks. It can only affect cross-sectional weights through per-stock betas estimated from 3–24 months, which amounts to noise injection. The results match: higher turnover, lower return, worse CRPS.
- The study contains no company-specific news, no headlines or text (GDELT Events has none), and no sentiment. Codex was candid about this (`llm_status.json`: NOT_RUN). The conclusion "news did not help" should be read as "this proxy did not help".

### B7. Statistical machinery too thin to support conclusions

- **Placebos:** one permutation each. A placebo needs ≥200–1000 permutations to give a p-value.
- **Block bootstrap:** Window B has 15 months, and with 3-month blocks that is only about 5 blocks. The 95% intervals are unreliable, and the "Window A news improved rank correlation, CI excludes zero" result is unadjusted across about 20 comparisons.
- **Six-month Sharpe ratios and annualized returns** (22-stock study) have standard errors of about ±1.4 Sharpe units.
- **Costs:** 10 bp all-in on micro-caps at the adjusted open understates spreads and impact by a large factor for growth-20.

### B8. Smaller modeling issues

- **`gp` / `multitask_gp` on log prices** (`stock_api/models.py:multitask_gp`) standardize log prices by the *window mean* and use a zero-mean prior. The forecast therefore reverts toward the window's average price: an unintended contrarian signal and a bias. This is why `gp` is worse than `last_price` at every lookback. Model returns instead (as the return GPs do), or use a mean function equal to the last value or a linear trend.
- **Window too short for the GP:** `train_days=30` calendar days is about 20 observations for a GP with 3+ hyperparameters. The coverage of 65–69% at 30 days shows the overconfidence.
- **Residual correlation from 30 observations of 22 assets**, with a fixed 50% shrinkage. Use Ledoit–Wolf or a factor model on a longer window (covariance does not need the 30-day drift window).
- **Optimizing log-return moments instead of arithmetic ones.** It is defensible for the reason documented (moment existence), but add the +½σ² convexity correction; otherwise high-volatility names are systematically under-rewarded relative to wealth.
- **Mixed horizons:** the forecast runs close to close while holdings run open to open. This is documented, but it matters for gap-heavy small caps.

---

## (C) Improvements to the core idea

Your plan: simple stochastic models plus LLM news sentiment, feeding a weighted portfolio. That is reasonable, but the evidence here suggests reordering the priorities. **Volatility and correlation are forecastable; expected returns barely are.** The screen found 6.4% log-variance skill against −2% direction skill. Build a strong risk engine first, then let news be a *small, uncertainty-weighted tilt* on top.

### C1. Stochastic models not yet implemented (by priority for this use case)

| Model | Why it fits here | Effort |
|---|---|---|
| **HAR-RV** (Corsi 2009) on daily/weekly/monthly realized variance | The best simple volatility forecaster in the literature. Linear, fast, hard to beat. Should be the volatility baseline that GP-vol and GJR must beat. | Low |
| **Statistical / fundamental factor risk model** (PCA or Fama–French + sector) with Ledoit–Wolf idiosyncratic shrinkage | Replaces the 30-observation correlation matrix. Scales to hundreds of names and gives hedgeable exposures. | Low–Med |
| **DCC-GARCH** (or GO-GARCH) | Time-varying correlations, which matter in crises when diversification disappears. Extends the existing univariate GJR. | Med |
| **Markov regime-switching / HMM** (2–3 states on market return and volatility, optionally VIX and credit spreads) | Gives a principled way to use macro: regime probabilities scale risk budgets. More robust than return forecasts. | Med |
| **Jump-diffusion** (Merton/Kou) and **event-variance jumps** | Earnings and FOMC dates are *known in advance*. Adding a scheduled jump variance on those days is one of the few reliable improvements to risk forecasts, and it is a natural link to the news pipeline. | Low–Med |
| **Heston / Bates stochastic volatility** (via particle filter or MCMC) | Continuous-time version of GP-vol with a mean-reverting variance. Mostly useful if you later add options data (implied volatility is strongly informative). | High |
| **Hierarchical Bayesian drift pooling** | Shrinks each stock's drift toward a sector or market mean, with partial pooling, instead of fitting 22 independent noisy drifts. Directly addresses B1. | Med |
| **Kalman filter time-varying beta** (CAPM / factor betas as random walks) | Cheap, interpretable, and gives the "exposure" piece the macro study attempted, done properly on daily data. | Low |
| **Hawkes processes** for news/jump clustering | Models how one news shock raises the intensity of subsequent ones. Good for event-driven volatility. | High |
| **Student-t / skewed-t copula** for joint tails | The current joint simulation uses Gaussian drivers, which understates crash co-movement. | Med |

Also worth trying: **implied volatility** from options (VIX for the market, single-stock IV where available) as a predictor of volatility. It usually dominates every historical-volatility model.

### C2. Doing news sentiment properly (the LLM part)

What exists now is not sentiment analysis. A credible pipeline looks like this:

1. **Company-level, timestamped text sources.** Options: SEC EDGAR 8-K/10-Q/10-K filings (free, exact acceptance timestamps, no licensing issues), earnings-call transcripts, company press releases, and a news API with first-publication timestamps (Alpaca News/Benzinga, Polygon, Tiingo, or the GDELT **GKG/DOC** article feeds rather than Events). Store the raw text, the source timestamp and your ingestion timestamp.
2. **Structured LLM extraction, not a single sentiment score.** For each (document, ticker) pair, extract a JSON schema: `relevance` (0–1), `event_type` (earnings, guidance, M&A, litigation, product, macro…), `direction` (−1…+1), `surprise/novelty` (is this already known?), `expected_magnitude`, `horizon`, `confidence`, and `affects_volatility` (bool). Cache prompt, model ID and output immutably. Use temperature 0 and fixed prompts with versions.
3. **Aggregation:** exponentially decayed, relevance-weighted sums per ticker, de-duplicated by story cluster, then **cross-sectionally standardized** (z-scores within the date). A raw level is useless; what matters is a stock's news *relative to its peers*.
4. **Two channels:**
   - *Mean channel:* sentiment → Black–Litterman view on the stock's excess return. The view uncertainty Ω comes from the historically measured hit rate of that signal, *not* from the LLM's self-reported confidence. Use the LLM confidence only as a feature after it has been calibrated.
   - *Risk channel:* `affects_volatility` and event type → a short-term variance bump in the risk model. This is probably the more reliable gain.
5. **Leakage control specific to LLMs.** The model may "know" what happened after a historical article. Mitigations:
   - (a) evaluate primarily on data after the model's training cutoff (i.e. prospectively);
   - (b) anonymize entities and dates in historical tests ("Company A announced…");
   - (c) compare against a small model or a lexicon baseline (e.g. Loughran–McDonald) to see whether the LLM's edge disappears on post-cutoff data.
6. **Measure the signal before it touches the portfolio:** IC and IC decay by horizon (1d, 5d, 21d), turnover of the signal, and IC after neutralizing momentum, reversal, size and sector. Most published news alpha decays within 1–5 days, so **monthly rebalancing will discard most of it.** You will likely need a daily or weekly rebalancing sleeve for the news tilt.

### C3. Portfolio construction

- **Black–Litterman** is the natural framework for your idea. Start from a neutral prior (market-cap or equal-weight equilibrium), add views from news, momentum or the GP with view uncertainty, and get posterior μ and Σ. This removes the λ problem (B4) and stops noisy signals from creating 100% turnover.
- Alternatives and complements: **risk parity / inverse volatility** as the base allocation (uses only the forecastable part), **fractional Kelly** on the tilt, **CVaR optimization** with the simulated paths you already produce, and **Hierarchical Risk Parity** for robustness to covariance error.
- **Turnover control:** no-trade bands (only trade when weight drift exceeds a threshold), an explicit turnover budget, and a cost model with spread + √(participation) impact.
- **Universe:** start with a liquid, survivorship-free universe. A **set of ~30–50 sector, country and factor ETFs** removes delisting and selection problems entirely, while still letting macro news matter (energy ETFs vs oil news, and so on). Move to single stocks once you have point-in-time constituent data.

### C4. Reframing the agent

```
daily:   ingest prices + news → update risk model (HAR/GARCH/factor) → LLM-extract news → signals
weekly:  BL posterior → optimizer with turnover/cost model → orders (paper) → log everything
monthly: re-estimate slow parameters, evaluation report, signal health check, human review
```

The LLM's best role may not be raw sentiment scoring. It can also (i) classify events into types whose *historical* average reaction you measure empirically, (ii) flag scheduled and unscheduled risk events for the volatility model, and (iii) write the human-readable rationale for each trade into the log.

---

## (D) Missing validation frameworks

### D1. Survivorship-free, point-in-time universe *(blocking)*

The universe-robustness study correctly stopped (`BLOCKED_DATA`). Options, from cheapest:
- **ETF universe** (§C3). No survivorship issue, free Yahoo data is adequate.
- **Norgate Data** (inexpensive, includes delisted names and historical index membership) or **Sharadar SEP/SF1** via Nasdaq Data Link.
- **CRSP via WRDS** (academic access) for the reference-quality version.

Until one of these exists, no single-stock portfolio backtest is interpretable.

### D2. Backtest-overfitting controls

- **Purged and embargoed walk-forward / combinatorial purged CV** (López de Prado 2018). This is needed because monthly horizons overlap daily features.
- **Deflated Sharpe Ratio** and **Probability of Backtest Overfitting (PBO)**. Record *every* configuration ever tried (lookbacks, λ, universes, feature sets); the repository has already tried dozens.
- **White's Reality Check / Hansen's SPA test / Model Confidence Set** when comparing many models against a benchmark.
- **A pre-registration file** per study: hypothesis, primary metric, decision rule, frozen config hash, committed *before* running. Codex's "frozen config" is a good start but is not tied to a git commit.

### D3. Forecast evaluation

- **Diebold–Mariano tests with HAC (Newey–West) errors** for paired loss differences. This replaces ad-hoc block bootstraps on 15 months.
- **Calibration of full distributions:** PIT histograms plus the Berkowitz test. For tails, use **Kupiec** (unconditional) and **Christoffersen** (independence) VaR backtests at 95%/99%. The current coverage numbers are a weak substitute.
- **Volatility forecast loss:** QLIKE and MSE on realized variance (Patton 2011 robust losses), against HAR-RV.
- **Cross-sectional signal evaluation:** IC, IC t-stat across dates, IC decay, quintile spread returns, and *factor-neutralized* IC (against momentum, reversal, size, value, sector). This alone would have caught B1.

### D4. Performance attribution

Regress every strategy's returns on market + Fama–French 5 + momentum (UMD) + short-term reversal (STR), all free from the Kenneth French library. Report alpha with HAC t-stats. A strategy whose "alpha" is all UMD/STR loading is a factor bet, not a model.

### D5. Model-level sanity tests

- **Synthetic recovery:** simulate from known data-generating processes (GBM with known drift; GARCH; a regime switch; a planted weak predictive signal with IC 0.03) and verify each model and the full pipeline recover them. In particular, check that the pipeline gives **zero alpha on pure-noise returns**. This is the most powerful bug-finder for backtests.
- **Null and placebo tests at scale:** ≥500 permutations or time-shifts of the signal, giving a real p-value. Also run a "future-shifted" signal (signal moved +1 period) as a leakage detector: if the leaked version is not dramatically better, your evaluation is not sensitive enough.
- **Invariance tests:** results unchanged when tickers are reordered, prices scaled, or future data truncated. Some of these exist (good); extend them to the full backtest engine.

### D6. Execution realism

A cost model with half-spread (from quoted spreads or a Corwin–Schultz estimate) + square-root market impact + ADV participation limits. Report capacity, i.e. the dollar size at which net alpha reaches zero. Cost sensitivity at 5/10/25/50 bp should be standard output.

### D7. Prospective (paper) forward test

The only truly out-of-sample test. Freeze universe, models, prompts and rules at a git commit, then run the agent in paper mode with immutable daily logs for ≥6–12 months. Compare it to equal weight, inverse volatility and 1-month momentum. This also serves as the clean LLM-leakage test (§C2.5).

---

## Suggested order of work

1. Commit code; move artifacts out of git; one environment plus a lock file. *(1 day)*
2. Build `evaluation/` (IC, attribution, DM, PIT/VaR tests, DSR) and the synthetic-recovery and null tests. Re-score the **existing** saved forecasts with them. *(about 1 week)*
3. Switch to an ETF universe; implement HAR-RV + factor/Ledoit–Wolf risk + risk-parity base portfolio; benchmark GP-vol and GJR against HAR. *(about 1–2 weeks)*
4. Black–Litterman layer; add momentum/reversal as explicit, measured views; turnover bands and a cost model. *(Done 2026-10-27: no momentum view survives beta adjustment and deflation; turnover control helps. See `reports/etf-black-litterman-2026-10-27/REPORT.md`.)*
5. News pipeline: EDGAR 8-Ks + one news API → LLM structured extraction → IC study on post-cutoff data → BL views + event-variance bumps.
6. Paper-trading agent with risk limits and logging; then a prospective evaluation.

## Appendix: how the B1 numbers were computed

For each experiment/window, I took rows with `strategy == "GP only"` from `forecasts/window_{A,B}_forecasts.csv`. Trailing return = `log(AdjClose[origin]) − log(AdjClose[origin − 30 sessions])` from `data/prices/{A,B}/<TICKER>.csv`. For each origin I computed Spearman correlations across tickers between `mean`, the trailing return and `actual`, then averaged over origins. This script should become a standard diagnostic in `evaluation/`.
