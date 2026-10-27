"""Separate, data-driven report for the supplied 20-stock replication."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from stock_api.macro_news import FACTORS,NEWS,block_difference_ci
from report_macro_news_study import plots,event_studies,headline,PRIMARY,SHORT
from portfolio_gp_report import table

BASE=ROOT/'experiments/portfolio_gp_macro_news_2026-10-18'

def fmt(frame):
    return table(frame,{col:(lambda v:f'{v:.5f}') for col in frame.select_dtypes('number').columns})

def build_report(out,c,tickers,states,records,f,exposure,diag,paired,comparison,daily,monthly,weights):
    extremes_prices=[]
    for window in ['A','B']:
        for ticker in c['window_tickers'][window]+['SPY']:
            prices=pd.read_csv(out/f'data/prices/{window}/{ticker}.csv',index_col='Date',parse_dates=True)
            changes=prices['Adj Close'].pct_change(); day=changes.abs().idxmax()
            extremes_prices.append(dict(window=window,ticker=ticker,max_abs_daily_return=changes.abs().max(),date=str(day.date()),daily_return=changes.loc[day],volume=prices.loc[day,'Volume'],median_dollar_volume=(prices.Close*prices.Volume).median()))
    pd.DataFrame(extremes_prices).to_csv(out/'data/prices/extreme_returns_audit.csv',index=False)
    news=pd.read_csv(out/'forecasts/news.csv')
    events,pe=event_studies(out,records,tickers,daily)
    cross=plots(out,tickers,states,exposure,f,diag,paired,comparison,daily,weights,news,events)
    cross['sign_agreement']=np.sign(cross.coefficient_A)==np.sign(cross.coefficient_B)
    cross['standardized_change_z']=(cross.coefficient_B-cross.coefficient_A)/np.sqrt(cross.standard_error_A**2+cross.standard_error_B**2)
    cross.to_csv(out/'exposures/cross_regime_comparison.csv',index=False)
    final=exposure[(exposure.variant=='final_frozen')&exposure.factor.isin(FACTORS)]
    extremes=[]
    for (window,factor),g in final.groupby(['window','factor']):
        for label,row in [('highest',g.loc[g.coefficient.idxmax()]),('lowest',g.loc[g.coefficient.idxmin()])]:
            extremes.append(dict(window=window,factor=factor,extreme=label,ticker=row.ticker,coefficient=row.coefficient,lower95=row.lower95,upper95=row.upper95))
    extremes=pd.DataFrame(extremes); extremes.to_csv(out/'exposures/strongest_sensitivities.csv',index=False)
    calibration=pd.read_csv(out/'forecasts/calibration.csv')
    cs=calibration.groupby(['window','variant']).apply(lambda g:pd.Series(dict(conditional_mae=g.error.abs().mean(),conditional_rmse=np.sqrt(np.mean(g.error**2)),coverage95=(g.z.abs()<=1.96).mean())),include_groups=False).reset_index()
    cs.to_csv(out/'exposures/transfer_calibration.csv',index=False)
    asset_cal=calibration[calibration.window=='B'].groupby(['ticker','variant']).error.apply(lambda x:np.mean(x**2)).unstack()
    asset_cal['transfer_minus_weak_mse']=asset_cal.transfer_prior-asset_cal.weak_prior
    asset_cal.to_csv(out/'exposures/transfer_by_asset.csv')
    attr=weights.merge(monthly[['window','strategy','period','start_equity','cost']],on=['window','strategy','period'])
    attr['gross_pnl_on_net_capital']=(attr.start_equity-attr.cost)*attr.return_contribution
    attr.to_csv(out/'portfolios/asset_contributions.csv',index=False)
    check=attr.groupby(['window','strategy','period']).gross_pnl_on_net_capital.sum()
    m=monthly.set_index(['window','strategy','period'])
    np.testing.assert_allclose(check.loc[m.index],m.end_equity-m.start_equity+m.cost,atol=1e-7)
    portfolio_pairs=[]
    for window,g in monthly.groupby('window'):
        base=g[g.strategy=='GP only'].set_index('period').net_return
        for name,h in g.groupby('strategy'):
            if name=='GP only': continue
            delta=h.set_index('period').net_return-base
            mean,lo,hi=block_difference_ci(delta,seed=c['seed'],replicates=c['bootstrap']['replicates'],block=c['bootstrap']['block_months'])
            portfolio_pairs.append(dict(window=window,strategy=name,mean_monthly_difference=mean,lower95=lo,upper95=hi))
    portfolio_pairs=pd.DataFrame(portfolio_pairs); portfolio_pairs.to_csv(out/'portfolios/paired_return_comparisons.csv',index=False)
    by_ticker=[]
    for (window,strategy,ticker),g in f.groupby(['window','strategy','ticker']):
        error=g.actual-g['mean']
        by_ticker.append(dict(window=window,strategy=strategy,ticker=ticker,n=len(g),mae=error.abs().mean(),rmse=np.sqrt(np.mean(error**2)),crps=g.crps.mean(),coverage90=((g.actual>=g.q05)&(g.actual<=g.q95)).mean(),direction_accuracy=((g.actual>0)==(g['mean']>0)).mean(),variance_ratio=g.realized_sum_squared_daily_returns.sum()/g.innovation_variance.sum()))
    pd.DataFrame(by_ticker).to_csv(out/'forecasts/ticker_diagnostics.csv',index=False)
    factor_scores=[]
    for (window,factor),g in pd.read_csv(out/'forecasts/factors.csv').groupby(['window','factor']):
        mse=np.mean((g.predicted_change-g.realized_change)**2); zero=np.mean(g.realized_change**2)
        factor_scores.append(dict(window=window,factor=factor,rmse=np.sqrt(mse),zero_change_rmse=np.sqrt(zero),mse_skill_vs_zero=1-mse/zero))
    pd.DataFrame(factor_scores).to_csv(out/'forecasts/factor_diagnostics.csv',index=False)
    risks=pd.read_csv(out/'portfolios/risk.csv'); risk_summary=[]
    for (window,strategy),g in risks.groupby(['window','strategy']):
        risk_summary.append(dict(window=window,strategy=strategy,rms_z=np.sqrt(np.mean(g.z**2)),coverage90=((g.realized_gross_log_return>=g.q05)&(g.realized_gross_log_return<=g.q95)).mean()))
    risk_summary=pd.DataFrame(risk_summary); risk_summary.to_csv(out/'portfolios/risk_calibration.csv',index=False)
    coverage=pd.read_csv(out/'data/prices/history_coverage.csv')
    below=coverage[(coverage.ticker!='SPY')&(coverage.cutoff_close<=10)]
    limit_text=f'''The requested universe is exactly: {', '.join(tickers)}. The label “top growing” is the user's selection criterion, not an independently established ranking. This is a hindsight-selected October 2026 universe, not a point-in-time screen. Selecting recent winners can strongly inflate retrospective performance. No historical profits validate the selection process.

FATN started trading April 8, 2025. SAIH's vendor history begins June 28, 2021, including its SPAC predecessor; the operating-company combination closed April 29, 2022. **Window A cannot be run on the full 20.** The earlier-window diagnostic uses the other 18 supplied names, explicitly excluding SAIH and FATN without substitutes. Window B uses all 20, but common training starts April 8, 2025, not January. Evaluation still starts July 17, 2025. The first numerical macro/news correction activates only after three completed GP forecast-error intervals. The July allocation has no augmentation. Missing prior distributions for SAIH/FATN remain exactly weak priors; no GP temporal state crosses the gap.

The supplied list supersedes the earlier screen's >$10 restriction: {', '.join(f'{r.ticker} closes at ${r.cutoff_close:.2f}' for r in below.itertuples()) or 'all supplied instruments exceed $10 at cutoff'}. Names are retained exactly as requested. A >$10 requirement throughout history is not applied. PENG includes the same issuer's pre-October-2024 SGH history. Vendor adjusted prices account for splits/dividends; raw cached actions and checksums are saved. There are zero missing required price observations in the declared usable windows, with no pre-listing filling or synthetic returns.

**No LLM was run.** News variants use the previous experiment's deterministic FOMC/CAMEO mechanisms, not sentiment, market surprises, or LLM interpretations. Numerical macro/news caches were reused in a separate copy; all GP joint distributions were recomputed for the changed asset universe. Estimated drift, 30 trading days, 10,000 paths, lambda=10, 10 bp per dollar bought or sold, $2,000 and a 20% allocation cap are unchanged. No parameter was chosen using these returns.
'''
    source='''# Data provenance

See `data/prices/history_coverage.csv` for every instrument's dates, cutoff price, corporate-action counts and raw-file SHA256. New histories use the repository's Yahoo/yfinance ingestion convention; previously cached stocks are reused only after hash verification. Raw requests/URLs/retrieval timestamps are in `data/prices/raw/*.json`. Current-vintage adjusted close is the total-return proxy; adjusted open is Open × Adj Close / Close. Retroactive vendor adjustments/corrections remain possible.

- [FatPipe SEC filing](https://www.sec.gov/Archives/edgar/data/1993400/000164117225003850/form8-k.htm) verifies April 8, 2025 commencement of FATN trading.
- [SAI.TECH SEC filing](https://www.sec.gov/Archives/edgar/data/1847075/000121390022035343/ff12022a1_saitechglobal.htm) verifies the April 2022 combination and May 2, 2022 operating-company trading; pre-combination SPAC data are not used as operating-company Window A observations.
- [Penguin Solutions SEC filing](https://www.sec.gov/Archives/edgar/data/1616533/000162828024043646/peng-20240830.htm) documents SGH → PENG on October 15, 2024. The exact supplied PENG symbol is preserved.
- [ALFRED real-time observations documentation](https://fred.stlouisfed.org/docs/api/fred/series_observations.html): copied 432 verified vintage snapshots at 72 origins/endpoints, six series. A vintage precedes its origin by at least one calendar day. Original release timestamps are unknown and remain blank. CPI uses same-vintage prior-year observations; unemployment is vintage-specific. B only uses states whose origins lie within its shortened usable window.
- [Federal Reserve calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm): 48 archived statements. Only contemporaneous statement text is classified, with a 24-hour delay after publication or conservative end-of-day bound.
- [GDELT codebook](https://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf) and [CAMEO taxonomy](https://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf): 140/142 fixed noon-UTC snapshots on the 5th/15th, 2,094 deduplicated conflict/sanction records. Two missing snapshots (2020-11-15 and 2025-06-15) are not treated as no events. Ingestion time is the availability bound; article publication time, full text and headline are not invented. Records can include domestic crime/retrospective reporting, so these are noisy proxies. Historical news coverage is incomplete.

Macro/news source copies, hashes, URLs, transformations and availability bounds are in `data/macro/`, `data/news/` and `logs/download_manifest.json`. No new LLM calls were made; `data/news/llm_status.json` records NOT_RUN. No direct consensus surprises, comprehensive tariff dataset, dollar/credit factors or company fundamentals were added. Interpretation of inflation compensation, vintage-state changes and event intensity must not turn them into causal shocks or expectation surprises.
'''
    (out/'DATA_SOURCES.md').write_text(source)
    architecture='''# Replication architecture and frozen choices

This experiment reuses `stock_api/heteroskedastic.py` (coupled drift/log-volatility Matérn-3/2 GPs, variational likelihood, two-start optimization and full future temporal posterior), `stock_api/portfolio.py` (correlated Monte Carlo paths, 50% residual-correlation shrinkage, analytic marginal variance preservation, convex constrained optimizer and self-financing fees), and `stock_api/macro_news.py` (point-in-time macro/news states, regularized exposures and incremental mappings). `scripts/run_macro_news_study.py --output ...` supplies the same walk-forward pipeline. Parameterized universe/report/audit interfaces do not overwrite the earlier experiment.

GP forecasts use 30 observed daily log returns. Monthly latent drift uncertainty and future innovations are both simulated, retaining temporal drift/log-variance dynamics and correlated cross-asset drivers. There is no multiplication of a single daily prediction by 21. Utility uses monthly log-return moments, an approximation to arithmetic wealth utility; lognormal volatility-mixture arithmetic moments need not exist. Dollar accounting uses actual simple holding returns. Macro/news models shift sample means and retain GP covariance: conditional plug-in tests of mean information, not fully integrated new risk forecasts. Mapping parameter SDs are saved but not independently added, because correlated/double-counted risks would require a joint model.

Execution is the first session on/after the 17th at adjusted open; forecasts know only the preceding session close. Close-to-close forecast targets approximate open-to-open holdings (last exit at cutoff close). No rebalance occurs after October 16. Fees are 10 bp × purchases plus sales, solved from the self-financing budget, once in realized accounting. The objective's fee penalty discourages trades but is not another cash debit. Gross wealth replays the same chosen weights without fees. Ten basis points omits market impact and may be optimistic for the smaller, less liquid names; no liquidity/volume participation filter was imposed. Daily close marks include the starting wealth anchor; volatility/Sharpe use 252 sessions and risk-free rate zero. Initial purchase counts in two-sided turnover; buys plus sells may exceed 100%.

Exposures: rolling up to 24 completed monthly intervals, ridge 6, SPY control, six fixed-scale numerical state changes. CPI YoY, two-year yield, ten-year inflation compensation, unemployment, asinh(WTI/50), log VIX. These are observed changes, not measured consensus surprises. Ex-ante zero-intercept ridge AR(1) factor forecasts use completed intervals only; coefficients capped ±0.8. Incremental GP residual models use ridge 24; macro+news is one joint regression to reduce double counting. News uses only records available at each historical origin, trailing 35 days, 24-hour delay, deterministic mechanism features. At least three completed GP-error months are required. News and numerical macro models share the same GP samples/covariance. Chronological validation observations can update the frozen rules but cannot tune them.

Window A's full-regime posterior freezes after January 2023, covariance inflated fourfold, extra precision capped at six standardized observations. Transfer applies only to the 18 observed issuers; SAIH/FATN use an availability mask and exact weak prior. No GP histories cross the gap. Placebos use one seeded permutation of completed training rows shared across stocks; they break serial dependence and are limited controls. Paired intervals use 2,000 circular three-month whole-market blocks; no independent-stock assumption or multiplicity-adjusted significance is claimed.

Window A: 18-name diagnostic, January–June 2019 warmup, July 17, 2019–January 31, 2023 evaluation. Window B: exact 20, common warmup April 8–July 16, 2025, July 17, 2025–October 16, 2026 evaluation. April 17–October 16, 2026 is a separately funded reset comparison. Calendars, exposure histories and GP states reset by window. No backfill of nonexistent securities. Missing cells for SAIH/FATN in Window A figures mean unavailable, not zero exposure.
'''
    (out/'ARCHITECTURE.md').write_text(architecture)
    reports={}
    for window in ['A','B']:
        perf=comparison[comparison.window==window]; p=perf.set_index('strategy'); d=diag[diag.window==window].set_index('strategy')
        main=perf.set_index('strategy').loc[PRIMARY].reset_index()
        month=monthly[monthly.window==window]; n=int(d.loc['GP only','months'])
        details=[]
        for name in PRIMARY[1:4]:
            ci=paired[(paired.window==window)&(paired.strategy==name)&(paired.baseline=='GP only')&(paired.metric=='crps')].iloc[0]
            details.append(f'- {name}: net return difference vs GP {(p.loc[name,"net_return"]-p.loc["GP only","net_return"])*100:+.2f} percentage points; gross difference {(p.loc[name,"gross_return"]-p.loc["GP only","gross_return"])*100:+.2f} points. CRPS difference {ci.difference:+.6f}, 95% block interval [{ci.lower95:+.6f}, {ci.upper95:+.6f}].')
        g=f[(f.window==window)&(f.strategy=='GP only')]
        variance_ratio=g.realized_sum_squared_daily_returns.sum()/g.innovation_variance.sum()
        limits=final[final.window==window]; nonzero=int(((limits.lower95>0)|(limits.upper95<0)).sum())
        extras=perf[~perf.strategy.isin(PRIMARY)]
        diff=month[month.strategy==PRIMARY[3]].set_index('execution_date').net_return-month[month.strategy=='GP only'].set_index('execution_date').net_return
        top=diff.abs().nlargest(3).index
        contributions=attr[(attr.window==window)&(attr.strategy==PRIMARY[3])].groupby('ticker').gross_pnl_on_net_capital.sum()-attr[(attr.window==window)&(attr.strategy=='GP only')].groupby('ticker').gross_pnl_on_net_capital.sum()
        asset_top=contributions.abs().nlargest(5).index
        news_last=exposure[(exposure.window==window)&(exposure.variant=='news_incremental')]; news_last=news_last[news_last.origin==news_last.origin.max()]
        news_ext=[]
        for factor,h in news_last[news_last.factor.isin(NEWS)].groupby('factor'):
            for label,r in [('highest',h.loc[h.coefficient.idxmax()]),('lowest',h.loc[h.coefficient.idxmin()])]:
                news_ext.append(dict(factor=factor,extreme=label,ticker=r.ticker,coefficient=r.coefficient,lower95=r.lower95,upper95=r.upper95))
        pd.DataFrame(news_ext).to_csv(out/f'exposures/window_{window}_news_sensitivities.csv',index=False)
        event_examples=events[(events.window==window)&~events.event.str.startswith('FOMC')].copy()
        event_examples=event_examples.loc[event_examples.groupby('event').abnormal_log_return.apply(lambda s:s.abs().nlargest(2).index).explode().tolist()] if len(event_examples) else event_examples
        title='18-stock diagnostic; SAIH and FATN unavailable' if window=='A' else 'exact 20-stock primary replication; shortened initial training'
        text=table(headline(main))+f'''

# Window {window}: {title}

{n} walk-forward holding periods. {len(c['window_tickers'][window])} assets; each strategy separately starts at $2,000. Evaluation dates: {month.execution_date.min()}–{month.holding_end.max()}. The two windows are independent and must not be compounded together. Window A ends with an 11-session partial holding. No LLM was evaluated.

{limit_text}

## Return forecasts and incremental information

{chr(10).join(details)}

{fmt(diag[(diag.window==window)&diag.strategy.isin(PRIMARY[:4])][['strategy','mae','rmse','crps','coverage90','coverage68','correlation','spearman','direction_accuracy','rms_z']])}

Errors and scores are dimensionless monthly log returns. Coverage columns target 90% and 68%; cross-sectional Spearman is averaged across months. Paired inference resamples entire market months with three-month circular blocks, 2,000 draws. Intervals are exploratory and unadjusted for multiple comparisons, selection and research reuse.

{fmt(paired[(paired.window==window)&(paired.metric=='crps')][['strategy','baseline','difference','lower95','upper95']])}

Paired cross-sectional rank-correlation differences (positive is better; same block procedure):

{fmt(paired[(paired.window==window)&(paired.metric=='spearman')][['strategy','baseline','difference','lower95','upper95']])}

Portfolio monthly differences versus GP, with matching block intervals:

{fmt(portfolio_pairs[portfolio_pairs.window==window])}

## Risk, allocations and attribution

Price-only realized daily squared-return sum / predicted innovation variance sum is {variance_ratio:.3f}; one is the aggregate target. Monthly standardized error RMS is {d.loc['GP only','rms_z']:.3f}. GP and augmented models use identical conditional covariance, so any apparent coverage change comes from the mean. Parameter uncertainty in the macro/news adjustment is not fully propagated. Portfolio predictive returns use log-sum-exp of simulated asset returns and weights; close-origin forecasts still approximate next-open execution.

{fmt(risk_summary[(risk_summary.window==window)&risk_summary.strategy.isin(PRIMARY[:4])])}

{fmt(perf[perf.strategy.isin(PRIMARY)][['strategy','mean_cap_count','mean_effective_holdings','mean_monthly_turnover','best_month','worst_month','positive_month_fraction','net_annualized_volatility']])}

Allocation caps apply at rebalance, not to weights drifting during the month. Largest monthly differences (joint macro+news minus GP): {', '.join(f'{date}: {diff.loc[date]*100:+.2f} pp' for date in top)}. Largest asset-level dollar P&L differences: {', '.join(f'{t}: ${contributions.loc[t]:+.2f}' for t in asset_top)}. Dollar attribution reflects evolving capital as well as weights, and is descriptive rather than a causal signal attribution. Exact holding-period contribution and fee identities are checked.

## Macroeconomic sensitivities

{fmt(extremes[extremes.window==window])}

These are full-window descriptive coefficients with conditional Gaussian 95% intervals. {nonzero}/{len(limits)} exclude zero before multiplicity adjustment. Rates, inflation and unemployment units are one percentage-point change; oil is one asinh(WTI/50) unit and volatility one log-VIX unit. Coefficients control for SPY and other factors; monetary-policy changes are only proxied by yields, growth by unemployment. Coefficient extremes do not establish reliable causal sensitivity. Ridge intervals omit model misspecification and are not HAC inference. Rolling coefficients, uncertainty and all asset rows are saved.

Latest available incremental news sensitivities (not LLM estimates, surprises, or causal geopolitical/trade betas):

{fmt(pd.DataFrame(news_ext))}

## Transfer, controls and events

{table(headline(extras))}

Ex-post conditional exposure calibration uses realized factors for diagnostics only; these realized factors never enter a forward allocation:

{fmt(cs[cs.window==window])}

Descriptive major-event abnormal returns, market model fitted on up to 60 prior sessions ending five sessions before the event; target event day through five following sessions:

{fmt(event_examples[['event','ticker','abnormal_log_return','lower95','upper95']] if len(event_examples) else pd.DataFrame(columns=['event','ticker']))}

The event reactions do not demonstrate advance anticipation. Scheduled FOMC statements are included by calendar rule. Named-event findings are post-event diagnostics; the initial 2025 tariff announcement precedes the full-20 common price history and is not estimable on all 20. Missing/short event windows are recorded in `exposures/event_exclusions.csv`. No hindsight narrative is used as a trading input.

## Monthly returns

{table(month[month.strategy.isin(PRIMARY)].pivot(index='execution_date',columns='strategy',values='net_return').reindex(columns=PRIMARY).reset_index(),{name:(lambda x:f'{x:.2%}') for name in PRIMARY})}

Full gross/net metrics, annualization, daily equity, weights, costs and paired controls are saved in CSVs. Sharpe uses zero risk-free rate. Annualization is descriptive, especially for the six-month reset subset. See [architecture](ARCHITECTURE.md), [sources](DATA_SOURCES.md), and [main report](RESULTS.md).
'''
        (out/f'RESULTS_WINDOW_{window}.md').write_text(text); reports[window]=text
    p=comparison[comparison.window=='B'].set_index('strategy'); dp=diag[diag.window=='B'].set_index('strategy')
    gp_months=monthly[(monthly.window=='B')&(monthly.strategy=='GP only')].copy()
    best_two=gp_months.nlargest(2,'net_return')
    log_share=np.log1p(best_two.net_return).sum()/np.log1p(gp_months.net_return).sum()
    matched=comparison[comparison.window=='B_matched_2026'].set_index('strategy').loc[PRIMARY].reset_index()
    conclusions=[]
    for window in ['A','B']:
        per=comparison[comparison.window==window].set_index('strategy'); diagnostics=diag[diag.window==window].set_index('strategy')
        improvements=[]
        for name in PRIMARY[1:4]:
            r=paired[(paired.window==window)&(paired.strategy==name)&(paired.baseline=='GP only')&(paired.metric=='crps')].iloc[0]
            evidence='lower CRPS with a pointwise interval excluding zero' if r.upper95<0 else ('higher CRPS with a pointwise interval excluding zero' if r.lower95>0 else 'a CRPS interval spanning zero')
            improvements.append(f'{SHORT[name]}: {(per.loc[name,"net_return"]-per.loc["GP only","net_return"])*100:+.2f} pp net versus GP, {evidence}')
        conclusions.append(f'- Window {window}: '+ '; '.join(improvements)+'.')
    transfer=paired[(paired.window=='B')&(paired.strategy=='GP + Macro transfer')&(paired.baseline=='GP + Macro')&(paired.metric=='crps')].iloc[0]
    a_news_return=portfolio_pairs[(portfolio_pairs.window=='A')&(portfolio_pairs.strategy=='GP + News (deterministic)')].iloc[0]
    a_news_rank=paired[(paired.window=='A')&(paired.strategy=='GP + News (deterministic)')&(paired.baseline=='GP only')&(paired.metric=='spearman')].iloc[0]
    a_joint_rank=paired[(paired.window=='A')&(paired.strategy==PRIMARY[3])&(paired.baseline=='GP only')&(paired.metric=='spearman')].iloc[0]
    sign=cross.groupby('factor').sign_agreement.mean()
    summary=table(headline(p.loc[PRIMARY].reset_index()))+f'''

# Supplied growth-stock universe: separate macro/news GP replication

**Primary table: July 17, 2025–October 16, 2026, exact 20 supplied stocks.** Net returns include 10 bp bought-plus-sold costs. News variants are deterministic, not LLM-based. The first six-month original comparison is below; this table covers 15 holding periods. Study cutoff remains October 16, 2026 for a matched replication.

{limit_text}

## Actual findings

The price-only GP returned **{p.loc['GP only','net_return']:.2%} net**, versus **{p.loc['Equal weight','net_return']:.2%} equal weight** and **{p.loc['SPY','net_return']:.2%} SPY**. Its net differences are {(p.loc['GP only','net_return']-p.loc['Equal weight','net_return'])*100:+.2f} and {(p.loc['GP only','net_return']-p.loc['SPY','net_return'])*100:+.2f} percentage points; gross differences are {(p.loc['GP only','gross_return']-p.loc['Equal weight','gross_return'])*100:+.2f} and {(p.loc['GP only','gross_return']-p.loc['SPY','gross_return'])*100:+.2f} points. Positive returns alone are not evidence that GP or economic features add value.

{chr(10).join(conclusions)}

**Macroeconomic information:** compare the GP+Macro return differences above and paired CRPS intervals in the window reports. An observed allocation improvement is not automatically improved return forecasting. Strong economic sensitivity need not imply predictable future factor changes. All six AR factor forecasts and zero-change controls are scored in `forecasts/factor_diagnostics.csv`.

**No clear incremental probabilistic-score improvement was established in either window:** every actual macro/news CRPS comparison against GP has a 95% market-month block interval containing zero. Window B's shuffled-macro placebo returned {p.loc['GP + Macro placebo','net_return']:.2%}, nearly the joint model's {p.loc[PRIMARY[3],'net_return']:.2%}. Window B paired monthly return-difference intervals for all three augmentations also include zero. This weakens any claim that the recent extra profit specifically came from useful economic information.

There is a narrower positive earlier-regime finding: Window A deterministic news improved mean monthly portfolio return by {a_news_return.mean_monthly_difference*100:.3f} percentage points (95% block interval [{a_news_return.lower95*100:.3f}, {a_news_return.upper95*100:.3f}]) and mean cross-sectional rank correlation by {a_news_rank.difference:.4f} (interval [{a_news_rank.lower95:.4f}, {a_news_rank.upper95:.4f}]). Both pointwise intervals exclude zero. This is exploratory evidence for ranking/allocation in the 18-stock earlier regime, despite no CRPS improvement. It is unadjusted for multiple comparisons and hindsight selection, and the analogous ranking/return advantage is not established in Window B. It merits a frozen prospective test rather than a claim of generalization.

Window A's joint macro+news model also improved rank correlation versus GP by {a_joint_rank.difference:.4f} (pointwise interval [{a_joint_rank.lower95:.4f}, {a_joint_rank.upper95:.4f}]); its portfolio-return interval still includes zero. The window reports show every paired ranking comparison, including placebos, rather than only favorable estimates.

**News information:** the evaluated feature layer is deterministic and sparse. Joint-vs-macro paired comparisons in `paired_comparisons.csv` isolate incremental news evidence within the joint specification. There is no evidence here about LLM value because no LLM was evaluated. Recent winners selected after the period and noisy archives preclude production claims.

**Cross-regime priors:** GP+Macro transfer returns {p.loc['GP + Macro transfer','net_return']:.2%}, versus {p.loc['GP + Macro','net_return']:.2%} without transfer. Its paired CRPS difference versus weak-prior macro is {transfer.difference:+.6f}, 95% block interval [{transfer.lower95:+.6f}, {transfer.upper95:+.6f}]. Transfer affects only the 18 issuers with Window A history; unavailable issuers keep exact weak priors. Descriptive sign agreement across those 18: {', '.join(f'{factor} {v:.0%}' for factor,v in sign.items())}. This is not causal stability. Asset-specific transfer failures are in `exposures/transfer_by_asset.csv`; changed business conditions cannot be attributed without point-in-time fundamentals.

**Risk and stability:** Window B GP has 90% interval coverage {dp.loc['GP only','coverage90']:.1%}, standardized error RMS {dp.loc['GP only','rms_z']:.3f}, average {p.loc['GP only','mean_cap_count']:.2f} names at the 20% cap, {p.loc['GP only','mean_effective_holdings']:.2f} effective holdings and {p.loc['GP only','mean_monthly_turnover']:.1%} two-sided monthly turnover. Daily volatility and drawdown, not just monthly observations, determine risk metrics. The full report separates mean accuracy from variance calibration and shows concentration, event behavior and monthly/asset attribution.

The GP had {(gp_months.net_return>0).sum()}/{len(gp_months)} positive holdings, but its two best periods—{', '.join(f'{r.execution_date} ({r.net_return:+.2%})' for r in best_two.itertuples())}—contributed {log_share:.1%} of cumulative log growth. Thus gains were materially concentrated. In particular, the April 2026 holding gained {gp_months[gp_months.execution_date=='2026-04-17'].net_return.iloc[0]:.2%}. Window B starts macro/news corrections on {f[(f.window=='B')&f.mapping_active].execution.min()}; July and August remain unadjusted because of FATN's short history.

The universe includes thinly traded stocks and very large observed daily moves. `data/prices/extreme_returns_audit.csv` records each ticker's largest adjusted daily return and median unadjusted close × volume. These observations are retained, not winsorized to improve model results. No independent second-vendor reconciliation is claimed. Ten basis points may understate spreads/impact; results assume fractional shares and fills at adjusted open without a liquidity filter.

No factor is promoted to a production trading signal on this retrospective experiment alone. Retain compact numerical macro factors and price-only/equal-weight controls for a preregistered prospective test. Treat sparse conflict counts, trade proxies and uncalibrated news mappings as speculative; discard claims of measured market surprises or LLM performance. The most informative next experiment freezes the universe and model rules now, evaluates new months with liquid-price execution/slippage, and compares a timestamped original-text LLM layer against the deterministic baseline with knowledge-contamination controls.

## April 17–October 16, 2026 reset comparison

{table(headline(matched))}

Each strategy starts anew at $2,000. These dates match the earlier 22-stock study but the universe is different; changes in returns cannot be attributed solely to improved models. The earlier 22-stock GP's 5.37% and SPY's 8.53% are context, not a control for the hindsight growth selection. All comparisons are exploratory.

## Earlier-regime diagnostic

{table(headline(comparison[(comparison.window=='A')&comparison.strategy.isin(PRIMARY)].set_index('strategy').loc[PRIMARY].reset_index()))}

**18 stocks only**, July 17, 2019–January 31, 2023, 43 holdings. This is not a full-20 result or a common-universe head-to-head comparison against Window B. Missing names are not silently replaced. Full results: [Window A](RESULTS_WINDOW_A.md), [Window B](RESULTS_WINDOW_B.md).

## Figures and reproducibility

![Independent equity curves](figures/portfolio_equity.png)
![Macro exposures; blank means unavailable](figures/exposure_heatmap.png)
![Exposure uncertainty](figures/exposure_uncertainty.png)
![Cross-regime coefficients](figures/cross_regime_exposures.png)
![Exposure evolution](figures/exposure_evolution.png)
![Dated macro and news inputs](figures/macro_news_timeline.png)
![Predictions](figures/predicted_vs_realized.png)
![Calibration](figures/forecast_calibration.png)
![Allocation weights](figures/portfolio_weights.png)
![Incremental probabilistic score](figures/incremental_information.png)
![Transfer](figures/cross_regime_transfer.png)

The exact universe is in `tickers.txt`; complete frozen settings are in `config.yaml`. Seed {c['seed']}, 10,000 antithetic paths per joint forecast. `forecasts/gp_cache` checkpoints include ordered tickers, price/code fingerprints and fitted parameters. `forecasts/parameters.csv`, mapping caches, source snapshots, software versions and execution log make the run restartable. `logs/audit.json` checks chronology/constraints/fees; `logs/independent_audit.json` independently reconstructs every daily gross and net portfolio path, checks PSD and timestamps. Prior study output checksums are preserved in `logs/previous_study_hashes.json` and verified after execution.

```bash
.conda-env/bin/python scripts/prepare_growth20_study.py
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python -m pytest tests/test_growth20_study.py tests/test_macro_news.py tests/test_portfolio.py -q
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_macro_news_study.py --output experiments/portfolio_gp_macro_news_growth20_2026-10-19 --smoke
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_macro_news_study.py --output experiments/portfolio_gp_macro_news_growth20_2026-10-19
```

Completed scope: usable-window price audit, unchanged heteroskedastic GP, all four mean-information variants, placebos, partially available cross-regime priors, matched portfolio accounting, exposures/uncertainty, event studies, forecast calibration, plots and reports. Unavailable scope: a full-20 pre-2023 history, a full six-month initial FATN history, actual LLM classification, fully integrated macro/news uncertainty, point-in-time unbiased winner selection and genuine historical consensus-surprise data. None is fabricated. [Architecture](ARCHITECTURE.md) and [data sources](DATA_SOURCES.md) state the exact conventions and deviations.
'''
    (out/'RESULTS.md').write_text(summary)
    # Protect the previous experiment's saved results, data and logs.
    import hashlib
    expected=json.loads((out/'logs/previous_study_hashes.json').read_text())
    changed=[p for p,h in expected.items() if hashlib.sha256((BASE/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    (out/'logs/previous_outputs_preserved.json').write_text(json.dumps(dict(status='PASS',files=len(expected),changed=changed),indent=2))


def main():
    """Regenerate reports from saved numerical results without refitting any model."""
    import argparse
    import hashlib
    import shutil
    from datetime import datetime,timezone
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'experiments/portfolio_gp_macro_news_growth20_2026-10-19'); args=ap.parse_args()
    out=args.output.resolve(); c=json.loads((out/'config.yaml').read_text()); tickers=(out/'tickers.txt').read_text().splitlines()
    states=pd.read_csv(out/'data/macro/states.csv'); records=[json.loads(x) for x in (out/'data/news/records.jsonl').read_text().splitlines()]
    f=pd.concat([pd.read_csv(out/f'forecasts/window_{w}_forecasts.csv') for w in ['A','B']],ignore_index=True)
    e=pd.concat([pd.read_csv(out/f'exposures/window_{w}_exposures.csv') for w in ['A','B']],ignore_index=True)
    build_report(out,c,tickers,states,records,f,e,*[pd.read_csv(out/p) for p in ['forecast_diagnostics.csv','paired_comparisons.csv','ablation_results.csv','portfolios/daily.csv','portfolios/monthly.csv','portfolios/monthly_weights.csv']])
    manifest=json.loads((out/'logs/manifest.json').read_text()); source='scripts/report_growth20_study.py'
    manifest['sources'][source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    manifest['report_regeneration']=dict(utc=datetime.now(timezone.utc).isoformat(),command=f'.conda-env/bin/python {source} --output {out.relative_to(ROOT)}')
    shutil.copyfile(ROOT/source,out/'source'/source)
    (out/'logs/manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Reports and 11 figures regenerated from saved results:',out)

if __name__=='__main__': main()
