"""Render the stock shortlist report from frozen numerical results."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from screen_monthly_stocks import ROOT


def table(frame):
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(map(str,r))+' |' for r in frame.itertuples(index=False,name=None)]])


def main():
    selected=pd.read_csv(ROOT/'top-100.csv')
    ranked=pd.read_csv(ROOT/'all-ranked-stocks.csv')
    predictions=pd.read_csv(ROOT/'monthly-predictions.csv')
    summary=json.loads((ROOT/'summary.json').read_text())
    gjr=json.loads((ROOT/'gjr-summary.json').read_text())
    gjr_stocks=pd.read_csv(ROOT/'gjr-stock-summary.csv')
    gjr_scores=pd.read_csv(ROOT/'gjr-monthly-scores.csv')
    successful=gjr_scores[gjr_scores.status=='ok']
    matched=predictions.merge(successful[['ticker','month']],on=['ticker','month'],how='inner')
    matched_skill=100*(1-matched.brier.mean()/matched.baseline_brier.mean())
    matched_accuracy=100*matched.correct.mean()
    promising=selected[(selected.selection_score>0)&(selected.audit_brier_skill_pct>0)]
    promising.to_csv(ROOT/'positive-skill-watchlist.csv',index=False)
    selected=selected.merge(gjr_stocks[['ticker','gjr_brier_skill_pct','successful','failed']],on='ticker',validate='one_to_one')
    selected.to_csv(ROOT/'top-100-with-gjr.csv',index=False)
    all100=selected[['rank','ticker','name','sector','price_usd','selection_score','audit_brier_skill_pct','gjr_brier_skill_pct']].copy()
    all100['ticker']=all100.ticker.map(lambda t:f'[{t}](https://finance.yahoo.com/quote/{t}/history/)')
    for col in all100.columns[4:]: all100[col]=all100[col].map(lambda x:f'{x:.2f}')
    all100.columns=['Rank','Ticker / price source','Company','Sector','Close USD','Earlier score %','Recent skill %','Recent GJR skill %']
    watch=promising[['ticker','price_usd','selection_score','audit_brier_skill_pct']].copy()
    for col in watch.columns[1:]:watch[col]=watch[col].map(lambda x:f'{x:.2f}')
    watch.columns=['Ticker','Close USD','Earlier score %','Recent skill %']
    exclusions=pd.read_csv(ROOT/'excluded-stocks.csv').reason.value_counts().rename_axis('Exclusion').reset_index(name='Stocks')
    excluded_table=table(exclusions.astype(str))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(13,5))
    ax=axes[0]
    ax.scatter(ranked.selection_score,ranked.audit_brier_skill_pct,s=12,color='#bbbbbb',alpha=.5,label='Other eligible stocks')
    ax.scatter(selected.selection_score,selected.audit_brier_skill_pct,s=22,color='#2876b8',label='Selected 100')
    ax.scatter(promising.selection_score,promising.audit_brier_skill_pct,s=25,color='#16816a',label='Positive in all three periods')
    ax.axhline(0,color='#555555',lw=.7);ax.axvline(0,color='#555555',lw=.7)
    ax.set_xlabel('Earlier direction score (%)');ax.set_ylabel('2024–Sep 2026 direction skill (%)')
    ax.set_title('Earlier predictability often fails to persist')
    ax.legend(fontsize=8)
    sectors=selected.groupby('sector').size().sort_values()
    axes[1].barh(sectors.index,sectors.values,color='#2876b8')
    axes[1].set_title('Sectors represented in the 100-stock shortlist')
    axes[1].set_xlabel('Number of stocks')
    fig.tight_layout();fig.savefig(ROOT/'screen-findings.png',dpi=160);plt.close(fig)
    counts=summary['selected_pooled_audit']
    failed_types=gjr_scores.loc[gjr_scores.status!='ok','error'].value_counts().to_dict()
    report=f'''# Monthly stock predictability: 100-stock research shortlist

Prepared October 18, 2026. Latest stock closes: **October 16, 2026**. Last completed return month: **September 2026**. Scope: US-listed stocks in the issuer-published Russell 1000 ETF holdings, with prices strictly above $10.

## Finding and recommendation

I assembled and tested **100 ranked research candidates**, but the evidence does **not identify 100 stocks with demonstrated, persistent monthly directional predictability**. Out of {summary['eligible_issuers']} eligible issuers, only {summary['positive_selection_both_blocks']} of the selected 100 beat the direction baseline in both earlier selection periods; {summary['positive_selection_and_audit']} also beat it in the later 2024–September 2026 audit. None survives correction for the number of stocks tested. The full list is provided below, with negative results retained rather than replacing unsuccessful stocks after seeing the later outcomes.

For portfolio research, start with the **22 positive-skill names** listed below and retain the full 100 as the comparison universe. These are candidates for further testing, not confirmed investment signals. The evidence is stronger for anticipating the **size of fluctuations** than their direction: on the selected stocks, the simple monthly volatility model reduces squared log-variance error by {counts['volatility_skill_pct']:.1f}% relative to its historical-average baseline, whereas the direction model is worse than its baseline. Predictable volatility can help determine position sizes; it does not establish positive expected returns.

**Price requirement:** all 100 have independently downloaded latest closes above $10 and issuer-reported prices above $10. The lowest selected close is **${selected.price_usd.min():.2f}** ({selected.loc[selected.price_usd.idxmin(),'ticker']}); the smallest selected median daily dollar volume is **${selected.median_daily_dollar_volume.min()/1e6:.1f} million**. This is a liquid listed-equity screen, with funds, cash and derivatives excluded. Prices can subsequently fall through the threshold, so it must be reapplied before portfolio formation.

## Where the information came from

1. **Universe, company names, sectors and an initial price check:** [iShares Russell 1000 ETF holdings](https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/latest-holdings.csv), dated **October 15, 2026**, downloaded from the fund issuer. The [fund page](https://www.ishares.com/us/products/239707/ishares-russell-1000-etf) describes its US large- and mid-cap exposure. The saved [raw holdings](data/iwb-holdings.csv) preserve the exact vintage used; this is a practical broad universe, not every exchange-listed stock or an independently reconstructed Russell membership database.
2. **Latest closes, daily volume, adjusted historical prices and corporate-action fields:** Yahoo Finance, downloaded with the repository's existing [yfinance history API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html). Histories were requested for January 2009–October 16, 2026. Each ticker in the table links to its Yahoo history page. Raw histories and retrieval timestamps/checksums are saved in `data/`; the combined [provenance file](data-provenance.json) lists evaluated input files. Screening uses the latest `Close`; return calculations use `Adj Close`, so historical price adjustments do not turn today's $10 filter into a historical-price filter.
3. **Predictor rationale:** the [Kenneth French momentum-factor construction](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html) uses prior months 2–12. That motivates one lagged predictor here; this screen is not a replication of the French portfolio or evidence that every momentum stock is predictable.
4. **Evidence standard:** [Harvey, Liu and Zhu's multiple-testing research](https://www.nber.org/papers/w20592) motivates correcting for searching many potential signals. The specific model, ranking and thresholds below are this repository's research choices, not a ranking published by those authors or a brokerage consensus.

## Eligibility and methodology

The issuer file contains **{summary['issuer_equities']} equities**. We fetched histories for all 1,007 whose issuer price exceeded $10, plus SPY as a reference download; there were no download failures. The final screen requires:

- USD-listed equity in that universe; a current Yahoo close dated October 16 above $10 as well as an issuer price above $10.
- Median `Close × Volume` over the latest 63 sessions of at least $10 million a day.
- Enough continuous history to estimate the same models and score the same 117 completed months for every stock. Recent IPOs and some newer reorganizations therefore do not qualify.
- Finite positive adjusted closes, at least 15 observations in every evaluation month, and no adjusted daily log-return jump larger than 1 in absolute value. The last rule routes unusually large moves/corporate actions to manual review; it does not prove those observations are erroneous.
- One share class per normalized issuer name, keeping the more liquid class without inspecting its forecast score. This removes duplicate classes, but does not identify every economic relationship between separate companies.

After these filters, **{summary['eligible_issuers']} issuers** remain. Exclusions are explicit in [excluded-stocks.csv](excluded-stocks.csv):

{excluded_table}

The main forecast is a fixed **ridge linear-probability regression**, trained separately for each stock on the previous 60 completed monthly observations. Its four predictors are the previous monthly log return, the previous three months' cumulative log return, momentum over months 2–12 before the target, and the previous month's log realized variance. All predictors and training labels precede the target month. Realized variance is the sum of daily squared adjusted log returns in that month, a daily-data proxy rather than intraday realized variance.

Predictor centering and scaling are fitted on those 60 training months only. The ridge penalty is fixed at 60; the intercept is not penalized. The up-probability intercept receives Laplace smoothing, and predictions are constrained to [0.01, 0.99]. The baseline is that stock's Laplace-smoothed up-frequency over the same preceding 60 months. This controls for a tendency to rise more often than fall; beating a coin flip alone would not establish useful timing information. These model probabilities are not independently calibrated.

Each stock gets **84 selection forecasts, January 2017–December 2023**, and **33 later audit forecasts, January 2024–September 2026**. No partial October return enters either period. The score is `100 × (1 − model Brier / baseline Brier)`; higher is better, zero ties the baseline, and negative means worse. The ranking uses the **smaller** score across January 2017–December 2020 and January 2021–December 2023, rewarding consistency. The full earlier-period score breaks ties, followed by ticker. Later audit performance does not change the 100 selections or their ranks. The [design](screen-design.json) was saved before fitting.

A companion ridge return regression uses the same predictors and training setup, compared against the rolling mean return. A separate one-lag ridge regression forecasts monthly log realized variance against its rolling 60-month average. The volatility score measures squared log-variance error reduction; it is a different target and should not be interpreted as directional skill.

## What the later audit found

| Measure, January 2024–September 2026 | Selected 100 |
| --- | ---: |
| Stocks with positive earlier score in both selection blocks | {summary['positive_selection_both_blocks']} |
| Stocks with positive recent directional skill | {summary['positive_audit_skill']} |
| Stocks positive in both earlier blocks and the later audit | {summary['positive_selection_and_audit']} |
| Stocks significant after Holm correction across {summary['eligible_issuers']} eligible issuers | {summary['holm_confirmed_selected']} |
| Direction accuracy | {counts['accuracy_pct']:.2f}% |
| Historical up-frequency baseline accuracy | {counts['baseline_accuracy_pct']:.2f}% |
| Direction Brier score | {counts['brier']:.4f} |
| Baseline Brier score | {counts['baseline_brier']:.4f} |
| Direction Brier skill | {counts['brier_skill_pct']:.2f}% |
| Return-prediction R² versus rolling mean | {counts['return_r2_pct']:.2f}% |
| Log-variance forecasting skill | {counts['volatility_skill_pct']:.2f}% |
| Stocks with positive log-variance forecasting skill | {int((selected.audit_volatility_skill_pct>0).sum())} |

Aggregate scores average the 3,300 stock-month observations. Those observations are correlated, so the aggregate is descriptive and is **not** a test with 3,300 independent observations. Individual one-sided Brier-improvement tests use 4,000 circular moving-block bootstrap replications with six-month blocks. Their p-values are Holm-adjusted across all {summary['eligible_issuers']} eligible issuers. With only 33 audit months, these tests have limited power, and the ranking is not precise enough to treat adjacent ranks as meaningfully different.

![Earlier versus later skill and sector composition](screen-findings.png)

## Check using the repository's GJR-GARCH model

I additionally ran the implemented **AR(1) regression–GJR-GARCH with Student-t innovations** on the same selected 100 stocks for the 33 audit months. Each fit uses the preceding 365 calendar days and simulates 2,000 paths through all actual sessions in the next calendar month. Parameters are plug-in maximum-likelihood estimates, so these forecasts omit parameter uncertainty. This is a supplementary model check decided after the primary screen, not a second basis for changing the shortlist. The [GJR design](gjr-audit-design.json) records the settings and model/shortlist hashes.

**{gjr['successful']:,} of {gjr['records']:,} forecasts succeeded; {gjr['failures']} failed.** Failures were preserved without replacement forecasts. On exactly the successful GJR stock-months:

| Model | Direction Brier skill versus historical up-frequency | Direction accuracy |
| --- | ---: | ---: |
| GJR-GARCH | {gjr['brier_skill_pct']:.2f}% | {gjr['accuracy_pct']:.2f}% |
| Primary monthly ridge model, matching dates | {matched_skill:.2f}% | {matched_accuracy:.2f}% |
| Historical up-frequency baseline | 0.00% | {gjr['baseline_accuracy_pct']:.2f}% |

GJR has positive Brier skill for **{gjr['positive_stock_brier_skill']} of 100** stocks, with **{gjr['holm_confirmed']}** significant after Holm correction within this supplementary 100-stock family. Those tests are exploratory and do not jointly correct across every model explored in this project. Its aggregate squared log-variance error reduction is **{gjr['volatility_skill_pct']:.2f}%** on successful forecasts. Numerical success and volatility skill do not establish a profitable directional strategy. [Per-stock GJR results](gjr-stock-summary.csv) show each stock's failure count; its table column below uses only its successful months. The GP was not rerun across this universe, and these results are not a GP-versus-GJR comparison.

Failure counts by message: `{json.dumps(failed_types)}`.

## The 22 names to investigate first

These stocks show positive primary-model direction skill in both earlier blocks and the recent audit. They still have **no multiple-testing-confirmed directional edge**. They retain their original order; this is a descriptive subset, not an independently validated reranking. [Download this watchlist](positive-skill-watchlist.csv).

{table(watch)}

## Full 100-stock shortlist

The ranks are determined by the earlier consistency score, not expected return, analyst targets, or likelihood of going up next month. A positive score is **relative Brier improvement**, not a percentage-point increase in prediction accuracy. Negative recent scores explicitly identify failed replication. Prices below are the October 16 closes in USD. GJR scores can cover fewer than 33 months; see the linked CSV for successful and failed counts.

{table(all100)}

## Implications for building the portfolio

The shortlist has {int((selected.sector=='Financials').sum())} financials, {int((selected.sector=='Industrials').sum())} industrials, {int((selected.sector=='Health Care').sum())} health-care stocks and {int((selected.sector=='Utilities').sum())} utilities. It has no communication-sector stocks. The mean pairwise monthly return correlation in the recent audit is **{summary['mean_pairwise_monthly_correlation']:.3f}**, but an average hides clusters and changes during stress. Stock-by-stock predictability does not by itself determine the predictability or risk of their combined portfolio. The [correlation matrix](selected-monthly-correlations.csv) is included for subsequent allocation research.

Before assigning weights, define whether the portfolio is long-only or permits shorts, its risk target, sector limits and turnover/cost assumptions. Then test a fixed selection-and-weighting rule on later unseen months, against a broad-market and simple allocation baseline. Use monthly risk forecasts for position sizing as a separately assessed component. This report does not estimate tradable portfolio returns, transaction costs, taxes, execution delays or capacity. A monthly close-to-close forecast also assumes information available at the prior close; a live strategy needs a feasible subsequent execution convention.

The current-constituent universe and today's price/liquidity filters introduce **survivorship and selection bias**. The audit is later in time than selection and excludes its outcomes from the ranking, but it is not a fully untouched point-in-time investment backtest: membership, present-day filters and model design were chosen retrospectively. Daily adjusted histories are retrospective provider data and may include ticker/predecessor histories or corporate-action revisions. Delisted stocks and many recent IPOs are absent. The 100 are therefore the top-ranked candidates **within this stated universe and fixed model**, not the 100 most predictable stocks in the whole market.

## Files, checks and reproduction

- [100 names with primary and GJR results](top-100-with-gjr.csv), [ticker-only list](tickers.txt), and [all {summary['eligible_issuers']} ranked eligible issuers](all-ranked-stocks.csv).
- [Monthly primary forecasts](monthly-predictions.csv), [GJR monthly scores](gjr-monthly-scores.csv), and GJR fit diagnostics in `gjr-records/`.
- [Frozen design](screen-design.json), [source provenance](data-provenance.json), [environment and file hashes](reproducibility.json), and raw data in `data/`.
- Three focused checks passed for lag construction/partial-month exclusion, future-data isolation, and training-only scaling/baseline scoring. The GJR implementation is covered by the repository's existing numerical and chronology tests.

From the repository root:

```bash
.conda-env/bin/python scripts/download_monthly_screen.py
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/screen_monthly_stocks.py
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/audit_monthly_gjr.py
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit .conda-env/bin/python scripts/report_monthly_screen.py
```

The downloader reuses cached files. Preserve the frozen files and source versions when reproducing this dated screen; a fresh as-of date requires a new experiment. The GJR audit checks its saved model and shortlist hashes before resuming. No trades were placed and no existing model defaults or historical experiment results were changed by this screen.
'''
    (ROOT/'REPORT.md').write_text(report)
    artifact_paths=[ROOT/'screen-design.json',ROOT/'data/iwb-holdings.csv',ROOT/'data/universe.csv',
        ROOT/'top-100.csv',ROOT/'monthly-predictions.csv',ROOT/'gjr-monthly-scores.csv',
        *[Path('scripts')/name for name in ['download_monthly_screen.py','screen_monthly_stocks.py','audit_monthly_gjr.py','report_monthly_screen.py']]]
    provenance={'python':platform.python_version(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','yfinance','arch','matplotlib']},
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_paths}}
    (ROOT/'reproducibility.json').write_text(json.dumps(provenance,indent=2))
    print(f'Report written: {ROOT}/REPORT.md')


if __name__=='__main__':main()
