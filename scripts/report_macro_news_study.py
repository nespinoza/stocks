"""Scientific diagnostics, event studies and reports from completed study outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from stock_api.macro_news import FACTORS,NEWS,block_difference_ci
from portfolio_gp_report import table

PRIMARY=['GP only','GP + Macro','GP + News (deterministic)','GP + Macro + News (deterministic)','Equal weight','SPY']
SHORT={'GP only':'GP','GP + Macro':'GP + Macro','GP + News (deterministic)':'GP + News*','GP + Macro + News (deterministic)':'GP + Macro + News*','Equal weight':'Equal weight','SPY':'SPY','GP + Macro transfer':'GP + Macro transfer'}


def event_studies(out,records,tickers,daily):
    events=json.loads((out/'data/news/event_study_dates.json').read_text())
    events += [dict(date=r['publication_date'],event='FOMC '+r['publication_date'],source=r['url']) for r in records if r['source'].startswith('Federal')]
    results=[]; portfolio_events=[]; excluded=[]
    for window in ['A','B']:
        window_tickers=json.loads((out/'config.yaml').read_text()).get('window_tickers',{}).get(window,tickers)
        close=pd.DataFrame({t:pd.read_csv(out/f'data/prices/{window}/{t}.csv',index_col='Date',parse_dates=True)['Adj Close'] for t in window_tickers+['SPY']})
        returns=np.log(close).diff().dropna(); idx=returns.index
        for event in events:
            date=pd.Timestamp(event['date'])
            if not idx[0]<=date<=idx[-1]: continue
            eligible=idx[idx>=date]; event_day=eligible[0]; pos=idx.get_loc(event_day)
            training=returns.iloc[max(0,pos-65):max(0,pos-5)]
            if len(training)<40 or pos+5>=len(idx):
                excluded.append(dict(window=window,event=event['event'],reason='fewer than 40 prior observations or incomplete event horizon')); continue
            target=returns.iloc[pos:pos+6]; x=np.column_stack([np.ones(len(training)),training.SPY]); xt=np.column_stack([np.ones(len(target)),target.SPY]); inv=np.linalg.inv(x.T@x)
            for ticker in window_tickers:
                coef=inv@x.T@training[ticker].to_numpy(); residual=training[ticker].to_numpy()-x@coef; variance=residual@residual/(len(training)-2)
                car=float((target[ticker].to_numpy()-xt@coef).sum()); sx=xt.sum(axis=0); sd=np.sqrt(variance*(len(target)+sx@inv@sx))
                results.append(dict(window=window,event=event['event'],event_date=str(event_day.date()),event_end=str(target.index[-1].date()),source=event['source'],ticker=ticker,training_start=str(training.index[0].date()),training_end=str(training.index[-1].date()),market_beta=coef[1],abnormal_log_return=car,conditional_standard_error=sd,lower95=car-1.96*sd,upper95=car+1.96*sd))
            for name,curve in daily[(daily.window==window)&daily.strategy.isin(PRIMARY)].groupby('strategy'):
                curve=curve.set_index(pd.to_datetime(curve.date)); before=idx[pos-1]; end=target.index[-1]
                if before in curve.index and end in curve.index:
                    portfolio_events.append(dict(window=window,event=event['event'],strategy=name,event_date=str(event_day.date()),event_end=str(end.date()),net_return=curve.loc[end,'net_equity']/curve.loc[before,'net_equity']-1))
    frame=pd.DataFrame(results); pe=pd.DataFrame(portfolio_events)
    frame.to_csv(out/'exposures/event_studies.csv',index=False); pe.to_csv(out/'portfolios/event_performance.csv',index=False)
    pd.DataFrame(excluded).to_csv(out/'exposures/event_exclusions.csv',index=False)
    return frame,pe


def plots(out,tickers,states,exposure,forecast,diag,paired,comparison,daily,weights,news,events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9})
    def save(fig,name):
        fig.tight_layout(); fig.savefig(out/'figures'/f'{name}.png',dpi=160); plt.close(fig)
    final=exposure[(exposure.variant=='final_frozen')&exposure.factor.isin(FACTORS)]
    # 1. Twenty-two by six factor panels on fixed economically interpretable shock scales.
    fig,axes=plt.subplots(1,2,figsize=(13,9),sharey=True)
    limit=np.max(np.abs(final.standardized_coefficient))*100
    for ax,window in zip(axes,['A','B']):
        a=final[final.window==window].pivot(index='ticker',columns='factor',values='standardized_coefficient').reindex(index=tickers,columns=FACTORS)*100
        im=ax.imshow(a,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit); ax.set_yticks(range(len(tickers)),tickers); ax.set_xticks(range(6),FACTORS,rotation=45,ha='right'); ax.set_title(f'Window {window}: full-window descriptive exposures')
    fig.subplots_adjust(left=.08,right=.85,bottom=.20,wspace=.12)
    color_axis=fig.add_axes([.88,.28,.02,.60])
    fig.colorbar(im,cax=color_axis,label='Monthly log-return percentage points per fixed factor shock')
    fig.savefig(out/'figures/exposure_heatmap.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    # 2. Uncertainty, shown for every ticker on rates and oil.
    fig,axes=plt.subplots(1,2,figsize=(11,9),sharey=True)
    for ax,factor in zip(axes,['rates','oil']):
        for shift,window,color in [(-.12,'A','tab:blue'),(.12,'B','tab:orange')]:
            a=final[(final.window==window)&(final.factor==factor)].set_index('ticker').reindex(tickers)
            ax.errorbar(a.coefficient*100,np.arange(len(tickers))+shift,xerr=a.standard_error*196,fmt='o',markersize=3,alpha=.7,label=window,color=color)
        ax.axvline(0,color='k',lw=.7); ax.set_yticks(range(len(tickers)),tickers); ax.set_title(factor); ax.set_xlabel('Log-return pp per factor unit, conditional 95% interval'); ax.legend()
    save(fig,'exposure_uncertainty')
    # 3. Cross-regime coefficient comparison.
    merged=final[final.window=='A'].merge(final[final.window=='B'],on=['ticker','factor'],suffixes=('_A','_B'))
    fig,ax=plt.subplots(figsize=(6,5))
    for factor,g in merged.groupby('factor'): ax.scatter(g.standardized_coefficient_A*100,g.standardized_coefficient_B*100,s=20,label=factor)
    ax.plot([-limit,limit],[-limit,limit],'k--',lw=.7); ax.set(xlabel='Window A effect (log pp)',ylabel='Window B effect (log pp)',title='Full-window descriptive exposures'); ax.legend(fontsize=7); save(fig,'cross_regime_exposures')
    # 4. Prespecified tickers, not selected because their coefficients look strongest.
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex='col')
    for j,window in enumerate(['A','B']):
        for i,factor in enumerate(['rates','oil']):
            for ticker in (['DUK','HBAN','ALK','KO'] if 'DUK' in tickers else tickers[:4]):
                g=exposure[(exposure.window==window)&(exposure.variant=='weak_prior')&(exposure.factor==factor)&(exposure.ticker==ticker)]
                axes[i,j].plot(pd.to_datetime(g.origin),g.coefficient,label=ticker)
            axes[i,j].set_title(f'{window}: {factor}'); axes[i,j].axhline(0,color='k',lw=.5); axes[i,j].legend(fontsize=7)
    fig.autofmt_xdate(); save(fig,'exposure_evolution')
    # 5. Dated numerical states and archived news signals.
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    for j,window in enumerate(['A','B']):
        g=states[states.window==window]; dates=pd.to_datetime(g.origin)
        axes[0,j].plot(dates,g.rates,label='2Y yield'); axes[0,j].plot(dates,g.inflation,label='CPI YoY'); axes[0,j].legend(); axes[0,j].set_title(f'{window}: vintage-known states (%)')
        axes[1,j].plot(dates,g.oil,label='asinh WTI/50'); axes[1,j].plot(dates,g.volatility,label='log VIX'); axes[1,j].legend()
        for name in ['policy_direction','trade_attention','conflict_attention']:
            z=news[(news.window==window)&(news.feature==name)]; axes[2,j].plot(pd.to_datetime(z.origin),z.value,label=name)
        axes[2,j].legend(fontsize=7); axes[2,j].set_title('Known-at-origin deterministic news features')
    fig.autofmt_xdate(); save(fig,'macro_news_timeline')
    # 6. Forecast means versus actual stock returns.
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for i,window in enumerate(['A','B']):
        for j,name in enumerate(PRIMARY[:4]):
            g=forecast[(forecast.window==window)&(forecast.strategy==name)]
            axes[i,j].scatter(g['mean'],g.actual,s=6,alpha=.35); axes[i,j].set_title(f'{window}: {SHORT[name]}'); axes[i,j].set_xlabel('Predicted log return'); axes[i,j].set_ylabel('Realized log return')
    save(fig,'predicted_vs_realized')
    # 7. Calibration, using common stock-month observations.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,window in zip(axes,['A','B']):
        g=diag[(diag.window==window)&diag.strategy.isin(PRIMARY[:4])].set_index('strategy').loc[PRIMARY[:4]]
        g[['coverage68','coverage90']].plot.bar(ax=ax); ax.axhline(.68,color='tab:blue',ls='--'); ax.axhline(.90,color='tab:orange',ls='--'); ax.set_xticklabels([SHORT[n] for n in g.index],rotation=30,ha='right'); ax.set_title(f'{window}: empirical coverage'); ax.set_ylim(0,1)
    save(fig,'forecast_calibration')
    # 8. Never concatenate the independent windows.
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,window in zip(axes,['A','B']):
        for name in PRIMARY:
            g=daily[(daily.window==window)&(daily.strategy==name)]; ax.plot(pd.to_datetime(g.date),g.net_equity,label=SHORT[name])
        ax.set_title(f'Window {window}: net value of $2,000'); ax.legend(fontsize=7)
    fig.autofmt_xdate(); save(fig,'portfolio_equity')
    # 9. All monthly allocations for the joint model in both windows.
    fig,axes=plt.subplots(1,2,figsize=(15,9),sharey=True)
    for ax,window in zip(axes,['A','B']):
        g=weights[(weights.window==window)&(weights.strategy==PRIMARY[3])].pivot(index='ticker',columns='execution_date',values='weight').reindex(tickers)
        im=ax.imshow(g,aspect='auto',vmin=0,vmax=.2,cmap='viridis'); ax.set_yticks(range(len(tickers)),tickers); step=max(1,len(g.columns)//8); ax.set_xticks(np.arange(0,len(g.columns),step),g.columns[::step],rotation=60,ha='right'); ax.set_title(f'Window {window}: GP + Macro + deterministic News')
    save(fig,'portfolio_weights')
    # 10. Paired error improvements with market-month block uncertainty.
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,window in zip(axes,['A','B']):
        g=paired[(paired.window==window)&(paired.metric=='crps')&(paired.baseline=='GP only')]
        y=np.arange(len(g)); ax.errorbar(g.difference,y,xerr=[g.difference-g.lower95,g.upper95-g.difference],fmt='o'); ax.set_yticks(y,[SHORT.get(n,n.replace('GP + ','')) for n in g.strategy]); ax.axvline(0,color='k',ls='--'); ax.set_title(f'{window}: CRPS difference vs GP'); ax.set_xlabel('Negative is better; 95% block interval')
    save(fig,'incremental_information')
    # 11. Transfer performance versus no transfer and price-only.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=['GP only','GP + Macro','GP + Macro transfer']; p=comparison[comparison.window=='B'].set_index('strategy').loc[names]
    axes[0].bar(range(3),p.net_return*100); axes[0].set_xticks(range(3),names,rotation=20,ha='right'); axes[0].set_ylabel('Window B net return (%)')
    g=diag[diag.window=='B'].set_index('strategy').loc[names]; axes[1].bar(range(3),g.crps); axes[1].set_xticks(range(3),names,rotation=20,ha='right'); axes[1].set_ylabel('CRPS (lower is better)'); save(fig,'cross_regime_transfer')
    return merged


def headline(perf):
    return pd.DataFrame([{'Strategy':r.strategy,'Net return':f'{r.net_return:.2%}','Ending $':f'{r.net_ending_dollars:,.2f}','Sharpe':f'{r.net_sharpe:.2f}','Max drawdown':f'{r.net_max_drawdown:.2%}','Turnover':f'{r.mean_monthly_turnover:.1%}','Costs':f'${r.costs:.2f}'} for r in perf.itertuples()])


def build_report(out,c,tickers,states,records,f,exposure,diag,paired,comparison,daily,monthly,weights):
    news=pd.read_csv(out/'forecasts/news.csv'); event,pe=event_studies(out,records,tickers,daily)
    cross=plots(out,tickers,states,exposure,f,diag,paired,comparison,daily,weights,news,event)
    cross['sign_agreement']=np.sign(cross.coefficient_A)==np.sign(cross.coefficient_B)
    cross['standardized_change_z']=(cross.coefficient_B-cross.coefficient_A)/np.sqrt(cross.standard_error_A**2+cross.standard_error_B**2)
    cross.to_csv(out/'exposures/cross_regime_comparison.csv',index=False)
    calibration=pd.read_csv(out/'forecasts/calibration.csv')
    calibration_summary=calibration.groupby(['window','variant']).apply(lambda g:pd.Series({'conditional_mae':g.error.abs().mean(),'conditional_rmse':np.sqrt(np.mean(g.error**2)),'conditional_coverage95':(g.z.abs()<=1.96).mean()}),include_groups=False).reset_index()
    calibration_summary.to_csv(out/'exposures/transfer_calibration.csv',index=False)
    extremes=[]
    final=exposure[(exposure.variant=='final_frozen')&exposure.factor.isin(FACTORS)]
    for (window,factor),g in final.groupby(['window','factor']):
        for label,row in [('highest',g.loc[g.coefficient.idxmax()]),('lowest',g.loc[g.coefficient.idxmin()])]:
            extremes.append(dict(window=window,factor=factor,extreme=label,ticker=row.ticker,coefficient=row.coefficient,lower95=row.lower95,upper95=row.upper95,n_months=row.n_months))
    extremes=pd.DataFrame(extremes); extremes.to_csv(out/'exposures/strongest_sensitivities.csv',index=False)
    factor_forecasts=pd.read_csv(out/'forecasts/factors.csv')
    factor_scores=[]
    for (window,factor),g in factor_forecasts.groupby(['window','factor']):
        mse=np.mean((g.predicted_change-g.realized_change)**2); zero=np.mean(g.realized_change**2)
        factor_scores.append(dict(window=window,factor=factor,rmse=np.sqrt(mse),zero_change_rmse=np.sqrt(zero),mse_skill_vs_zero=1-mse/zero))
    factor_scores=pd.DataFrame(factor_scores); factor_scores.to_csv(out/'forecasts/factor_forecast_diagnostics.csv',index=False)
    news_extremes=[]
    for window in ['A','B']:
        g=exposure[(exposure.window==window)&(exposure.variant=='news_incremental')]
        g=g[g.origin==g.origin.max()]
        for factor,h in g[g.factor.isin(NEWS)].groupby('factor'):
            identified=not np.allclose(h.coefficient,0)
            high=h.loc[h.coefficient.idxmax()]; low=h.loc[h.coefficient.idxmin()]
            news_extremes.append(dict(window=window,factor=factor,highest=high.ticker if identified else 'not identified',highest_coefficient=high.coefficient,
                highest_lower95=high.lower95,highest_upper95=high.upper95,lowest=low.ticker if identified else 'not identified',lowest_coefficient=low.coefficient,lowest_lower95=low.lower95,lowest_upper95=low.upper95))
    news_extremes=pd.DataFrame(news_extremes); news_extremes.to_csv(out/'exposures/news_sensitivities.csv',index=False)
    # Paired portfolio inference resamples complete market months as well.
    portfolio_pairs=[]
    for window,group in monthly.groupby('window'):
        baseline=group[group.strategy=='GP only'].set_index('period')
        for name,g in group.groupby('strategy'):
            if name=='GP only': continue
            diff=g.set_index('period').net_return-baseline.net_return
            avg,lo,hi=block_difference_ci(diff,seed=c['seed'],replicates=c['bootstrap']['replicates'],block=c['bootstrap']['block_months'])
            portfolio_pairs.append(dict(window=window,strategy=name,mean_monthly_return_difference=avg,lower95=lo,upper95=hi))
    portfolio_pairs=pd.DataFrame(portfolio_pairs); portfolio_pairs.to_csv(out/'portfolios/paired_return_comparisons.csv',index=False)
    # Dollar attribution: accounting identity for each period, including its own fees.
    attr=weights.merge(monthly[['window','strategy','period','start_equity','cost']],on=['window','strategy','period'])
    attr['gross_pnl_on_net_capital']=(attr.start_equity-attr.cost)*attr.return_contribution
    attr.to_csv(out/'portfolios/asset_contributions.csv',index=False)
    source_text='''# Data sources and availability

- **Stock prices:** the 23 previously cached Yahoo/yfinance histories, with original checksums verified and exact window slices saved under `data/prices/`. Adjusted close is the total-return proxy; adjusted open uses Open × Adj Close / Close. Zero required missing stock observations. Current-vintage adjusted histories and a hindsight-selected universe remain limitations.
- **Macroeconomics:** 432 ALFRED CSV snapshots, six series at 72 origins/endpoints. Each response header must encode the requested vintage date. See `data/macro/snapshot_index.csv`, raw CSVs and metadata, and `observation_provenance.csv`. Every vintage precedes its decision origin. [ALFRED/FRED real-time observation documentation](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) distinguishes vintage from observation dates. We verified availability by the vintage day; exact original release timestamps are unknown and intentionally blank. No current revised CPI/unemployment history is substituted for a historical vintage.
- **Factor meaning:** DGS2 is a yield level, T10YIE is market inflation compensation rather than a pure inflation forecast, CPIAUCSL is a seasonally adjusted price index, UNRATE is unemployment, DCOILWTICO is a spot-price series, and VIXCLS is market implied-volatility information. Transformations and units are recorded per observation. Monthly state differences can contain newly released information, revised information already known at that vintage, and differing observation lags. They are not structural shocks or consensus surprises. [FRED](https://fred.stlouisfed.org/) supplies the series/source definitions.
- **Official news:** [Federal Reserve FOMC calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) and [2019 historical materials](https://www.federalreserve.gov/monetarypolicy/fomchistorical2019.htm)/[2020 historical materials](https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm) supplied dated statement links. Only statement text is classified; later minutes and transcripts are excluded. Original pages and hashes are cached. Publication minute is parsed when explicit; otherwise an end-of-day availability bound is used. An additional 24-hour delay is imposed.
- **Geopolitical records:** fixed 12:00 UTC GDELT 2.0 event snapshots on days 5 and 15, rather than a return-selected event sample. [GDELT documentation](https://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf) identifies DATEADDED as ingestion time and SOURCEURL as the news reference. Original publication time/headline/full article text are not available from these records, so those fields remain null. Actor-country filters are fixed in config; conflict/sanction categories use the [CAMEO taxonomy](https://data.gdeltproject.org/documentation/CAMEO.Manual.1.1b3.pdf). URLs are deduplicated chronologically. Counts are normalized by the number of observed snapshots in the trailing 35 days; missing snapshots are not zero-event snapshots. The signal is sparse archive intensity, not complete global conflict activity, media-independent severity or probability of an economic event.
- **Missing archives:** GDELT 2020-11-15 12:00 and 2025-06-15 12:00 returned HTTP 404; 140/142 requested snapshots were available. Coverage markers are stored separately from news meaning. The DOC API probe returned 429; its [document search documentation](https://blog.gdeltproject.org/doc-geo-2-0-api-updates-full-year-searching-and-more/) does not provide the required multi-year complete archive through a simple current query. BLS archive probe returned 403; ALFRED vintages provided the usable numerical alternative. Natural gas, dollar index, credit spreads, direct payroll/industrial-production releases and genuine consensus-surprise data were not included in this compact study.
- **LLM:** no configured provider credential/interface was present. No LLM inference was run. All evaluated news variants are deterministic baselines; confidence is null, not invented. `data/news/llm_status.json` records this limitation. Adding a provider alone is insufficient: a future LLM test must use timestamped original article text, cache the model/prompt/output, control for pretrained hindsight knowledge, and run a new prespecified comparison.

`logs/download_manifest.json` records URLs, retrieval timestamps, hashes and failures. `data/news/records.jsonl` contains raw-source identifiers, text where available, timestamps, entities, event descriptions and deterministic classifications. Current retrieval establishes reproducibility but cannot certify that vendor historical files have never been corrected. Archived news is incomplete and machine event extraction can misclassify events.
'''
    source_text+='\nCAMEO assault/fight categories can include domestic violence and retrospective reports, not just economically important interstate conflicts. Country filtering does not solve that classification problem. The deterministic classifier received regression tests for word-boundary handling (for example, “forward” must not imply “war”) and for distinguishing a rate increase from securities purchases.\n'
    (out/'DATA_SOURCES.md').write_text(source_text)
    window_reports={}
    for window in ['A','B']:
        perf=comparison[comparison.window==window]; main=perf[perf.strategy.isin(PRIMARY)].set_index('strategy').loc[PRIMARY].reset_index()
        diagnostics=diag[(diag.window==window)&diag.strategy.isin(PRIMARY[:4])]
        p=perf.set_index('strategy'); d=diag[diag.window==window].set_index('strategy'); b=p.loc['GP only']
        n=int(d.loc['GP only','months']); month=monthly[monthly.window==window]
        differences=paired[(paired.window==window)&(paired.metric=='crps')]
        exp=extremes[extremes.window==window].copy()
        for col in ['coefficient','lower95','upper95']: exp[col]=exp[col].map(lambda x:f'{x:.5f}')
        details=[]
        for name in PRIMARY[1:4]:
            r=p.loc[name]; err=d.loc[name]; ci=differences[(differences.strategy==name)&(differences.baseline=='GP only')].iloc[0]
            details.append(f'- **{name}:** net return {r.net_return:.2%}, a {(r.net_return-b.net_return)*100:+.2f} percentage-point difference from GP; gross difference {(r.gross_return-b.gross_return)*100:+.2f} points. MAE {err.mae:.5f}, RMSE {err.rmse:.5f}, CRPS {err.crps:.5f}. Paired CRPS difference {ci.difference:+.6f}, 95% market-month block interval [{ci.lower95:+.6f}, {ci.upper95:+.6f}].')
        paired_table=differences[['strategy','baseline','difference','lower95','upper95']]
        conditional=calibration_summary[calibration_summary.window==window]
        returns_table=month[month.strategy.isin(PRIMARY)].pivot(index='execution_date',columns='strategy',values='net_return').reindex(columns=PRIMARY)
        extras=perf[~perf.strategy.isin(PRIMARY)]
        joint=month[month.strategy==PRIMARY[3]]; base_month=month[month.strategy=='GP only']
        mdelta=joint.set_index('period').net_return-base_month.set_index('period').net_return
        top_dates=mdelta.abs().nlargest(3)
        contributions=attr[(attr.window==window)&(attr.strategy==PRIMARY[3])].groupby('ticker').gross_pnl_on_net_capital.sum()-attr[(attr.window==window)&(attr.strategy=='GP only')].groupby('ticker').gross_pnl_on_net_capital.sum()
        strongest=contributions.abs().nlargest(5).index
        regime_start='July 17, 2019' if window=='A' else 'July 17, 2025'
        regime_end='January 31, 2023' if window=='A' else 'October 16, 2026'
        wr=table(headline(main))+f'''

# Window {window}: {regime_start}–{regime_end}

{n} walk-forward holding periods, starting after January–June warm-up. Each strategy starts independently with $2,000. All news variants here are **deterministic**, not LLM-based. The final Window A holding is a short January 2023 period; returns are not directly comparable to Window B's shorter evaluation. Mean turnover counts buys plus sells, includes the initial purchase, and can exceed 100%.

## Predictive information and portfolio results

{chr(10).join(details)}

Price-only GP reference: MAE {d.loc['GP only','mae']:.5f}, RMSE {d.loc['GP only','rmse']:.5f}, CRPS {d.loc['GP only','crps']:.5f}. All metrics use identical stock-month observations within this window, in dimensionless monthly log-return units.

{table(diagnostics[['strategy','mae','rmse','crps','coverage90','coverage68','correlation','spearman','direction_accuracy','rms_z']],{k:(lambda x:f'{x:.4f}') for k in ['mae','rmse','crps','coverage90','coverage68','correlation','spearman','direction_accuracy','rms_z']})}

Paired CRPS comparisons, with three-month circular blocks resampling entire market months, 2,000 repetitions. Negative means better. Intervals are pointwise, exploratory and not multiplicity-adjusted; they do not account for universe screening or the research choices already informed by earlier experiments.

{table(paired_table,{k:(lambda x:f'{x:+.6f}') for k in ['difference','lower95','upper95']})}

Additional controls and transfer (no strategy was selected by validation profits):

{table(headline(extras))}

Gross returns, daily volatility, annualized returns, both drawdowns, effective holdings, best/worst month and positive-month fractions are in `portfolios/window_{window}_performance.csv`. Daily curves are in `portfolios/daily.csv`; annualized return extrapolates using 252 trading sessions and Sharpe uses zero risk-free return. Transaction costs are charged once using the existing exact self-financing equation.

## Exposure sensitivities

Highest and lowest full-window ridge coefficients, with conditional Gaussian 95% intervals. These end-of-window descriptions are not retroactively used in earlier allocations. Rates/inflation/unemployment coefficients are monthly log return per one percentage-point change; oil is per unit of asinh(WTI/50), volatility per unit log VIX. All are conditional on SPY and the other factors. The complete 22 × 6 matrix and its uncertainty are in `exposures/window_{window}_exposures.csv`; the `variant` column separates rolling, full-window and news mappings.

{table(exp[['factor','extreme','ticker','coefficient','lower95','upper95','n_months']])}

This is an observational association model, not identification of monetary-policy, inflation or geopolitical causal shocks. Early exposures use few monthly observations and are prior-sensitive. Gaussian intervals are conditional on shrinkage and an approximate independent-error likelihood; they do not provide robust frequentist coverage under serial dependence. Growth is represented only by unemployment, not a complete growth factor. Trade/conflict effects are separate regularized news-feature associations, not reliable company-specific structural exposures.

Ex-post conditional exposure checks use realized macro changes only for scoring, never in a trading forecast:

{table(conditional,{k:(lambda x:f'{x:.5f}') for k in ['conditional_mae','conditional_rmse','conditional_coverage95']})}

## Month and stock contributions

Joint macro/news versus GP-only: the three largest absolute monthly return differences were {', '.join(f'period {int(i)+1}: {v*100:+.2f} pp' for i,v in top_dates.items())}. Net effects include cost differences.

The largest absolute cumulative stock-dollar contribution differences were {', '.join(f'{t}: ${contributions[t]:+.2f}' for t in strongest)}. These compare each strategy's own changing capital base; they are an accounting decomposition, not causal attribution. Fees are separate. Full data are in `portfolios/asset_contributions.csv`.

Monthly net returns:

{table(returns_table.reset_index(),{name:(lambda x:f'{x:.2%}') for name in PRIMARY})}

All results use an October 2026 hindsight-selected universe. {'The April–October 2026 subperiod was already examined in prior research and remains explicitly exploratory.' if window=='B' else 'No claim of prospective stock-selection validity follows from this earlier-window test.'}
'''
        window_reports[window]=wr; (out/f'RESULTS_WINDOW_{window}.md').write_text(wr)
    # Main report is written after every computed table, making actual findings easy to inspect.
    a=comparison[comparison.window=='A'].set_index('strategy'); b=comparison[comparison.window=='B'].set_index('strategy'); matched=comparison[comparison.window=='B_matched_2026']
    dA=diag[diag.window=='A'].set_index('strategy'); dB=diag[diag.window=='B'].set_index('strategy')
    transfer=b.loc['GP + Macro transfer']; weak=b.loc['GP + Macro']
    sign=cross.groupby('factor').sign_agreement.mean()
    transfer_delta=paired[(paired.window=='B')&(paired.strategy=='GP + Macro transfer')&(paired.baseline=='GP + Macro')&(paired.metric=='crps')].iloc[0]
    failed_stocks=calibration[calibration.window=='B'].groupby(['variant','ticker']).error.apply(lambda x:np.mean(x*x)).unstack(0)
    failed_stocks['transfer_mse_change']=failed_stocks.transfer_prior-failed_stocks.weak_prior
    failed_stocks.to_csv(out/'exposures/transfer_by_ticker.csv')
    snapshots=pd.read_csv(out/'data/news/coverage.csv'); actual_news=[r for r in records if r['source']!='Archive coverage marker']
    main_report=f'''# Macro and news information beyond the heteroskedastic GP

The complete feasible study was executed on the original 22 symbols, using the existing GP, joint simulation, optimizer and transaction-cost accounting. **No LLM was run.** News results refer solely to the deterministic, timestamped baseline. These are two independent historical experiments, with a separate reset comparison to the earlier six-month backtest.

**Window A — July 17, 2019–January 31, 2023**

{table(headline(a.loc[PRIMARY].reset_index()))}

**Window B — July 17, 2025–October 16, 2026**

{table(headline(b.loc[PRIMARY].reset_index()))}

**Hindsight selection applies to both windows.** The 22 names were selected using the October 2026 screen. The April–October 2026 interval was already studied. Neither window validates the complete stock-selection-and-allocation strategy prospectively. No configuration was selected to maximize these returns.

## I. Macro exposures

The six-factor matrix measures rate changes, market inflation compensation, vintage-known CPI inflation, oil prices, unemployment and VIX, conditional on SPY. [Window A](RESULTS_WINDOW_A.md) and [Window B](RESULTS_WINDOW_B.md) list the largest positive/negative coefficients with uncertainty for every factor. All 22 stocks, coefficient trajectories and conditional posterior intervals are saved, including negative findings.

![22-stock macro exposures](figures/exposure_heatmap.png)

![Exposure uncertainty](figures/exposure_uncertainty.png)

![Coefficient evolution](figures/exposure_evolution.png)

The matrix is descriptive. Rate changes are not pure FOMC surprises, inflation compensation includes other market components, and unemployment is only one growth proxy. Sector/economic intuition did not hard-code exposure signs. Zero-centered priors and fixed factor units prevent unsupported sector narratives from becoming signals. Large intervals and changes of sign indicate limited identification; shrinkage intervals are model-dependent rather than causal confidence bounds.

News mappings for policy, inflation/growth language, trade attention and sampled conflicts are in the exposure CSVs under `news_incremental`. They estimate incremental predictive associations, not structural sensitivities to geopolitical conflict or trade disruption. Sparse policy mentions, sampled media coverage and common market reactions make company-specific conclusions particularly uncertain.

## II. Incremental predictive information

Price-only versus macro CRPS was **{dA.loc['GP only','crps']:.5f} vs {dA.loc['GP + Macro','crps']:.5f} in A**, and **{dB.loc['GP only','crps']:.5f} vs {dB.loc['GP + Macro','crps']:.5f} in B**. Lower is better. News-only CRPS was {dA.loc[PRIMARY[2],'crps']:.5f}/{dB.loc[PRIMARY[2],'crps']:.5f}; combined CRPS was {dA.loc[PRIMARY[3],'crps']:.5f}/{dB.loc[PRIMARY[3],'crps']:.5f}. These small differences must be assessed against the paired market-month intervals in the window reports, not thousands of purported independent asset-month observations.

![Predicted and realized returns](figures/predicted_vs_realized.png)

![Forecast calibration](figures/forecast_calibration.png)

![Paired incremental probabilistic scores](figures/incremental_information.png)

The augmentation targets prior **out-of-sample GP residuals**. Model B fits those residuals on historical realized factor changes, then evaluates the coefficient mapping on ridge AR forecasts of future changes. Realized future macro observations are used only as later training labels and scoring targets. Model C maps contemporaneously available news features to completed future residuals. Model D fits macro and news jointly rather than blindly summing separately fitted correlated corrections. Intercepts and expected SPY innovation are set to zero when applying corrections, isolating these feature families from an extra unconditional-return intercept.

This is deliberately a **matched conditional mean experiment**: all models retain the same GP covariance and mean-shift the same 10,000 joint paths. It does not test a news-driven volatility GP. Mapping coefficient uncertainty is saved but not independently added to the GP's risk, because both are inferred from overlapping returns and the GP already captures macro-induced realized volatility. Consequently augmented predictive distributions omit factor-forecast and mapping uncertainty; they are plug-in conditional distributions, and calibration may be overconfident. A fully joint macro/GP latent model is a future extension, not something implemented here.

## III. Portfolio performance and event behavior

Macro net-return differences from GP were **{(a.loc['GP + Macro','net_return']-a.loc['GP only','net_return'])*100:+.2f} pp in A** and **{(b.loc['GP + Macro','net_return']-b.loc['GP only','net_return'])*100:+.2f} pp in B**. Combined macro/news differences were **{(a.loc[PRIMARY[3],'net_return']-a.loc['GP only','net_return'])*100:+.2f} pp** and **{(b.loc[PRIMARY[3],'net_return']-b.loc['GP only','net_return'])*100:+.2f} pp**. Differences in dollar gains across windows should not be compared without accounting for their different lengths. Gross and net returns, daily risk metrics, turnover and costs are all saved in `ablation_results.csv`.

![Independent-window equity curves](figures/portfolio_equity.png)

![Monthly joint-model weights](figures/portfolio_weights.png)

[Window reports](RESULTS_WINDOW_A.md) identify the largest contributing stock and month differences. Placebo variants preserve the cross-asset label vectors but shuffle completed historical predictor rows with fixed seeds. They never use unavailable future features. This retains common stock shocks but disrupts predictor serial dependence; one placebo realization is a diagnostic, not a randomization test proving information content.

Event studies cover all 48 archived FOMC statements plus the three prespecified [WHO pandemic characterization](https://www.who.int/news-room/speeches/item/who-director-general-s-opening-remarks-at-the-media-briefing-on-covid-19---11-march-2020), [Ukraine invasion](https://ukraine.un.org/en/173250-statement-secretary-general-%E2%80%93-ukraine), and [April 2025 tariff announcement](https://www.govinfo.gov/content/pkg/DCPD-202500423/html/DCPD-202500423.htm) dates. A market model is fitted on up to 60 prior daily returns ending five sessions before the event; fewer than 40 observations causes an explicit exclusion. Cumulative abnormal log returns use sessions 0 through +5. Some announcements occurred after that day's close, so this broad window includes a pre-announcement return; it is not an intraday identification design. Concurrent events confound attribution. The tariff event falls in B's warm-up, so no evaluation portfolio performance is claimed around it.

`exposures/event_studies.csv` contains {len(event)} ticker-event rows; `portfolios/event_performance.csv` saves available evaluation-portfolio event returns. These post-event observations are never backfilled into earlier news features. The augmentation cannot directly anticipate an increase in univariate predictive variance because its covariance is held fixed; it can only change expected returns and weights.

![Vintage-known macro states and dated news](figures/macro_news_timeline.png)

## IV. Cross-regime generalization

All of A's completed exposure observations, including COVID and Ukraine-era intervals, inform its final frozen posterior. B receives a weakened prior: covariance inflated by four and added precision capped at six standardized observations, so new evidence can overwhelm it. Descriptive exposures and the GP-residual macro mapping have separate frozen posteriors. GP histories and latent states reset in January 2025; no time series is concatenated across the gap.

Transfer net return in B: **{transfer.net_return:.2%}**, versus **{weak.net_return:.2%}** without transfer and **{b.loc['GP only','net_return']:.2%}** price-only. Transfer CRPS difference versus weak-prior macro: **{transfer_delta.difference:+.6f}**, 95% block interval **[{transfer_delta.lower95:+.6f}, {transfer_delta.upper95:+.6f}]**.

![Exposure regime comparison](figures/cross_regime_exposures.png)

![Transfer predictive and portfolio results](figures/cross_regime_transfer.png)

End-of-window coefficient sign agreement across 22 stocks: {', '.join(f'{factor} {value:.0%}' for factor,value in sign.items())}. Agreement is not proof of stable causal exposure; shared priors, omitted variables and different observations can matter. The largest deteriorations in ex-post conditional exposure MSE under transfer were {', '.join(f'{ticker} ({row.transfer_mse_change:+.6f})' for ticker,row in failed_stocks.sort_values('transfer_mse_change',ascending=False).head(5).iterrows())}. The study does not contain point-in-time company fundamentals sufficient to attribute those failures to acquisitions, leverage changes or business-model changes; such explanations would be speculation.

## V. Direct comparison with the earlier six-month experiment

Reset all portfolios to $2,000 on April 17, 2026, while retaining only information learned chronologically before each origin. The price-only distribution cache is reused only after matching ticker order, return window, dates, horizon and GP code hashes. The original GP net return is reproduced to numerical precision.

{table(headline(matched[matched.strategy.isin(PRIMARY)]))}

This subset is exploratory and was not used to choose coefficients. Earlier portfolio holdings are not carried into this reset comparison.

## VI. Conclusions and next experiment

1. **Incremental macro signal:** assess the displayed paired CRPS/MAE intervals jointly with portfolio differences and placebo results. Improvement in realized wealth alone does not establish forecast information. With only 43 A months and 15 B months, modest effects and regime variation remain uncertain; this is not confirmatory evidence for production use.
2. **Incremental historical news:** only deterministic features were evaluated. The combined-versus-macro paired comparison is the direct test of added news information and is saved separately. No claim of LLM improvement is possible: zero LLM classifications were run. Sparse GDELT snapshots and official statements are a narrow news baseline.
3. **Economic-prior transfer:** the numerical comparison above tests a deliberately weak transfer, with per-ticker exposure calibration, error scores, rank correlations, weights and returns saved. A favorable result on one metric does not show stability across all companies or establish a permanent economic relationship.
4. **Production inclusion:** no feature is approved for live allocation on this evidence. Vintage-safe macro state data and explicit availability controls are suitable research infrastructure. Rate, inflation, energy and volatility factors remain interpretable candidates for independent validation.
5. **Discard or treat as speculative:** current revised macro histories used as if historically known, realized future shocks used as forecasts, LLM confidence treated as market surprise, inferred company-specific conflict effects from sparse counts, and strong untempered cross-regime priors are not supported. The evaluated fixed-sample news signals remain speculative.
6. **Most informative next experiment:** freeze the universe, feature definitions and model rules, then collect a new prospective period with complete timestamped articles and a genuine held-out LLM-versus-deterministic comparison. Use an original-text-only prompt, measure leakage from pretrained event knowledge, preserve immutable inputs/outputs and compare incremental scores before optimizing for returns. Joint modeling of forecast/mapping uncertainty should precede claims about improved risk calibration.

## Reproducibility, execution and limits

The frozen [configuration](config.yaml), [architecture](ARCHITECTURE.md), [data-source audit](DATA_SOURCES.md), raw cached data and checksums are included. There are {len(states)} macro states, 432 vintage CSVs, 48 actual FOMC statements, 2,094 deduplicated sampled conflict/sanction events, and 140 archive-coverage markers (markers are not articles). Two GDELT files are unavailable. Publication timestamps are retained where known, and ingestion/availability bounds are labeled rather than presented as original publication times.

```bash
.conda-env/bin/python scripts/download_macro_news_study.py
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python -m pytest tests/test_macro_news.py tests/test_portfolio.py -q
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_macro_news_study.py --smoke
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_macro_news_study.py
```

The downloader requires network access only for absent/unverified files. GP checkpoints bind data, horizon and model hashes; all models share the same cached base paths. Logs, input/code hashes, package versions and source snapshots are included. Failures are explicit; no synthetic data or substitute forecasts are emitted. The initial smoke test exposed an evaluation-boundary bug, which was corrected before the full backtest. Core timestamp, prior/covariance, optimizer and accounting tests passed. `logs/audit.json` verifies chronological training, weight constraints, fees, monthly compounding and exact reproduction of the previous primary result.

Limitations: hindsight universe selection and survivorship; differing window lengths; already explored 2026 subset; only monthly exposure observations despite daily prices; selected compact factors and fixed shrinkage; observation/release-date differences and unknown precise first-release times; residual macro revisions within valid historical vintages; noisy/sparse archived news; no genuine market expectations or policy surprises; no LLM; possible event-code errors; plug-in augmented uncertainty; approximate independent-error exposure posterior; unadjusted multiple comparisons; the inherited monthly log-return utility approximation and prior-close-to-next-open execution clock. The final January 2023 partial horizon has fewer sessions than other exposures. No positive result should be interpreted as a causal economic sensitivity or an investable edge without new evidence.
'''
    nonzero=final.assign(nonzero=(final.lower95>0)|(final.upper95<0)).groupby('window').nonzero.sum()
    brief=[]
    for factor in FACTORS:
        row={'Factor':factor}
        for window in ['A','B']:
            g=extremes[(extremes.window==window)&(extremes.factor==factor)].set_index('extreme')
            row[f'{window}: highest / lowest']=f'{g.loc["highest","ticker"]} ({g.loc["highest","coefficient"]:+.3f}) / {g.loc["lowest","ticker"]} ({g.loc["lowest","coefficient"]:+.3f})'
        brief.append(row)
    main_report=main_report.replace('The matrix is descriptive.',
        table(pd.DataFrame(brief))+f'\n\nOnly {int(nonzero["A"])} of 132 A coefficients and {int(nonzero["B"])} of 132 B coefficients have conditional 95% intervals excluding zero, before any multiplicity correction. These counts do not establish reliable individual exposures. Highest/lowest are point-estimate rankings, not statistically established orderings; the window reports give the intervals.\n\nThe matrix is descriptive.')
    main_report=main_report.replace('Sparse policy mentions, sampled media coverage and common market reactions make company-specific conclusions particularly uncertain.',
        'Sparse policy mentions, sampled media coverage and common market reactions make company-specific conclusions particularly uncertain. Full highest/lowest news coefficients and intervals are in `exposures/news_sensitivities.csv`; zero coefficients for a feature absent from training are labeled not identified, rather than assigning an arbitrary ticker a sensitivity rank.\n\n'+table(news_extremes[['window','factor','highest','highest_coefficient','lowest','lowest_coefficient']],{'highest_coefficient':lambda x:f'{x:+.5f}','lowest_coefficient':lambda x:f'{x:+.5f}'}))
    main_report=main_report.replace('The augmentation targets prior **out-of-sample GP residuals**.',
        'Ex-ante factor-change forecasting versus predicting no change (positive MSE skill is better):\n\n'+table(factor_scores,{'rmse':lambda x:f'{x:.5f}','zero_change_rmse':lambda x:f'{x:.5f}','mse_skill_vs_zero':lambda x:f'{x:.2%}'})+'\n\nThe augmentation targets prior **out-of-sample GP residuals**.')
    named=pe[~pe.event.str.startswith('FOMC')].pivot(index=['window','event'],columns='strategy',values='net_return').reset_index()
    main_report=main_report.replace('These post-event observations are never backfilled into earlier news features.',
        'Available six-session event portfolio returns:\n\n'+table(named,{name:lambda x:f'{x:.2%}' for name in PRIMARY})+'\n\nThese post-event observations are never backfilled into earlier news features.')
    main_report=main_report.replace('1. **Incremental macro signal:** assess the displayed paired CRPS/MAE intervals jointly with portfolio differences and placebo results.',
        '1. **Incremental macro signal: no demonstrated improvement in this implementation.** Macro augmentation worsened MAE and CRPS and reduced net portfolio returns in both windows. The macro-versus-GP CRPS intervals include zero; apparent changes are not statistically decisive. This does not show that economic information is intrinsically useless—only that the tested exposures and ex-ante factor forecasts failed to add value.')
    main_report=main_report.replace('2. **Incremental historical news:** only deterministic features were evaluated.',
        '2. **Incremental historical news: no positive predictive evidence from this baseline.** News-only and joint models worsened CRPS in both windows; adding news to macro also worsened CRPS, with pointwise block intervals above zero in both windows. That is exploratory evidence of deterioration, not a multiple-testing-adjusted universal finding. News-only portfolio returns deteriorated in both windows and remained below simple benchmarks. Only deterministic features were evaluated.')
    main_report=main_report.replace('3. **Economic-prior transfer:** the numerical comparison above tests a deliberately weak transfer,',
        '3. **Economic-prior transfer: a small, inconclusive improvement over the weak-prior macro model.** Transfer slightly improves B CRPS and net returns versus macro without transfer, but its paired CRPS interval includes zero and it remains below price-only GP. Ex-post conditional exposure MAE improves while RMSE slightly worsens. The numerical comparison above tests a deliberately weak transfer,')
    main_report=main_report.replace('The complete feasible study was executed',
        '**Finding: the tested macro and deterministic-news additions did not deliver a consistent predictive or portfolio advantage.**\n\nThe complete feasible study was executed')
    main_report=main_report.replace('The study does not contain point-in-time company fundamentals sufficient to attribute those failures to acquisitions, leverage changes or business-model changes; such explanations would be speculation.',
        'Two post-hoc context checks show why stable company exposures cannot be assumed: [Alaska completed its Hawaiian acquisition on September 18, 2024](https://news.alaskaair.com/company/alaska-airlines-completes-acquisition-of-hawaiian-airlines-expanding-benefits-and-choice-for-travelers/), between the windows; [LPL completed its Commonwealth acquisition on August 1, 2025](https://www.sec.gov/Archives/edgar/data/1397911/000119312525171084/d810447d8k.htm), during Window B. Both ALK and LPLA appear among the transfer-calibration deteriorations. These facts motivate fundamental-aware priors but do not establish the cause of those errors. They were checked after scoring and were not used as predictive inputs or to choose parameters.')
    main_report=main_report.replace('![Forecast calibration](figures/forecast_calibration.png)',
        f'Price-only 90% stock-return coverage was {dA.loc["GP only","coverage90"]:.1%} in A and {dB.loc["GP only","coverage90"]:.1%} in B. Standardized residual RMS was {dA.loc["GP only","rms_z"]:.2f}/{dB.loc["GP only","rms_z"]:.2f}; realized summed daily squared returns divided by forecast innovation variance was {dA.loc["GP only","realized_to_predicted_daily_variance"]:.2f}/{dB.loc["GP only","realized_to_predicted_daily_variance"]:.2f}. Near-nominal interval coverage does not establish correct tail-risk calibration, particularly in A. All augmented models inherit the same marginal variances by design.\n\n![Forecast calibration](figures/forecast_calibration.png)')
    main_report=main_report.replace('`logs/audit.json` verifies chronological training, weight constraints, fees, monthly compounding and exact reproduction of the previous primary result.',
        '`logs/audit.json` verifies chronological training, weight constraints, fees, monthly compounding and exact reproduction of the previous primary result. `logs/independent_audit.json` independently reconstructs daily net/gross holdings and verifies every included news timestamp and macro vintage. Full mapping posterior matrices and fitted design/target data are cached in `exposures/mapping_cache/`, and complete frozen regime posteriors are saved as NPZ files.')
    main_report=main_report.replace('Core timestamp, prior/covariance, optimizer and accounting tests passed.',
        'Final classifier review also found and corrected a substring bug that could read “war” inside “forward”, and narrowed policy/inflation rules to avoid unrelated phrase matches. The corrected feature pipeline and all dependent scores/allocations were rerun; preliminary news results are superseded. No parameters were selected to improve profits. Core timestamp, prior/covariance, optimizer and accounting tests passed.')
    (out/'RESULTS.md').write_text(main_report)
