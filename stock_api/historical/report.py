"""Regenerate ticker reliability tables and a browsable gallery from checkpoints."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from .config import HistoricalConfig
from .runner import load_record,atomic_json,atomic_bytes,units,unit_id,digest,sha
from .analysis import score,add_pairs,aggregate,reliability_bins,confidence_bins,block_comparison,holm,regime_labels,disagreement
from .inference import LEVELS


def generate_report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output);meta=json.loads((root/'experiment-config.json').read_text());config=HistoricalConfig.model_validate(meta['config'])
    if sha(root/'prices.csv')!=meta['snapshot_sha256']:raise ValueError('Snapshot changed')
    rows=[];diagnostics=[]
    for unit in units(config):
        key=unit_id(*unit);path=root/'records'/f'{key}.json'
        if path.exists():
            record=load_record(path,meta['experiment_id']);rows.append(score(record))
            diagnostics.append({'unit_id':key,'origin':unit[0],'model':unit[1],'lookback':unit[2],'status':record['status'],
                                'error':record.get('error'),'diagnostics':(record.get('prediction') or {}).get('diagnostics'),
                                'rerun_command':f'python -m stock_api.historical diagnose --output {str(root.resolve())!r} --origin {unit[0]} --model {unit[1]} --lookback {unit[2]}'})
        else:
            rows.append({'unit_id':key,'origin':unit[0],'model':unit[1],'lookback':unit[2],'status':'pending','year':int(unit[0][:4]),
                         'runtime_seconds':0.,'fallback':False,'warnings':0,'training_volatility':np.nan,'training_return':np.nan})
    if not rows:raise ValueError('No origins in the configured interval')
    frame=regime_labels(add_pairs(pd.DataFrame(rows)))
    overall=aggregate(frame,min_count=config.min_bin_count)
    yearly=aggregate(frame,groups=('model','lookback','year'),min_count=config.min_bin_count)
    def csv(relative,values):atomic_bytes(root/relative,values.to_csv(index=False).encode())
    csv('summaries/forecast-scores.csv',frame);csv('summaries/overall.csv',overall);csv('yearly/metrics.csv',yearly)
    atomic_json(root/'summaries/inference-details.json',diagnostics)
    failure_rows=[{'unit_id':d['unit_id'],'origin':d['origin'],'model':d['model'],'lookback':d['lookback'],
                   'status':d['status'],'reason':(d['error'] or {}).get('message','Configured convergence checks failed')}
                  for d in diagnostics if d['status']!='ok']
    csv('summaries/failures.csv',pd.DataFrame(failure_rows,columns=['unit_id','origin','model','lookback','status','reason']))
    good=frame[frame.status=='ok'].copy()
    comparisons=[];calibrations=[];confidences=[];coverage=[];episodes=[];disagreement_bins=[]
    for (model,lookback),g in good.groupby(['model','lookback']):
        for metric in ['mae','terminal_return_mae','brier']:
            if model=='last_price':continue
            comparison=block_comparison(g.set_index('origin')['delta_'+metric],meta['origins'],
                        config.bootstrap_block_weeks,config.bootstrap_replicates,config.seed)
            comparisons.append({'model':model,'lookback':lookback,'loss':metric,**comparison})
        bins=reliability_bins(g,config.min_bin_count);bins['model']=model;bins['lookback']=lookback;calibrations.append(bins)
        bins=confidence_bins(g);bins['model']=model;bins['lookback']=lookback;confidences.append(bins)
        # All thresholds below are descriptive analysis, never prediction inputs.
        g=g.copy()
        for col in ['sigma','confidence','realized_volatility']:
            g[col+'_bin']=pd.qcut(g[col],4,labels=False,duplicates='drop').fillna(0).astype(int) if g[col].nunique()>1 else 0
        for column in ['year','sigma_bin','confidence_bin','realized_volatility_bin','trailing_volatility_regime','return_environment']:
            for label,subset in g.groupby(column):
                for level in LEVELS:
                    tag=str(int(level*100));coverage.append({'model':model,'lookback':lookback,'grouping':column,'group':str(label),
                        'nominal':level,'count':len(subset),'empirical_coverage':subset['coverage_'+tag].mean(),
                        'path_coverage':subset['path_coverage_'+tag].mean(),'mean_return_interval_width':subset['interval_width_'+tag].mean()})
        if model!='last_price':
            paired=g.dropna(subset=['delta_terminal_return_mae'])
            for sign,chosen in [('best',paired.nsmallest(3,'delta_terminal_return_mae')),('worst',paired.nlargest(3,'delta_terminal_return_mae'))]:
                for _,row in chosen.iterrows():episodes.append({'model':model,'lookback':lookback,'kind':sign,'origin':row.origin,
                    'delta_terminal_return_mae':row.delta_terminal_return_mae,'realized_return':row.realized_return,
                    'predicted_return':row.predicted_return,'record':f'records/{row.unit_id}.json'})
    paired=pd.DataFrame(comparisons)
    if not paired.empty:paired['holm_p_value']=holm(paired.p_value,(len(config.models)-1)*len(config.lookbacks)*3)
    csv('summaries/paired-comparisons.csv',paired)
    calibration=pd.concat(calibrations,ignore_index=True) if calibrations else pd.DataFrame()
    confidence=pd.concat(confidences,ignore_index=True) if confidences else pd.DataFrame()
    csv('calibration/reliability-bins.csv',calibration);csv('calibration/confidence-bins.csv',confidence)
    coverage_frame=pd.DataFrame(coverage)
    csv('calibration/coverage-strata.csv',coverage_frame);csv('summaries/episodes.csv',pd.DataFrame(episodes))
    regimes=[]
    for column in ['trailing_volatility_regime','realized_volatility_regime','return_environment']:
        if column in frame:
            result=aggregate(frame,groups=('model','lookback',column),min_count=config.min_bin_count)
            result['regime_definition']=column;result=result.rename(columns={column:'regime'});regimes.append(result)
    csv('summaries/regimes.csv',pd.concat(regimes,ignore_index=True) if regimes else pd.DataFrame())
    disagree=disagreement(frame,len(config.models));csv('summaries/disagreement.csv',disagree)
    if not disagree.empty:
        for lookback,g in disagree[disagree.complete_panel].groupby('lookback'):
            g=g.copy();g['bin']=pd.qcut(g.expected_return_dispersion,4,labels=False,duplicates='drop').fillna(0).astype(int) if g.expected_return_dispersion.nunique()>1 else 0
            for label,b in g.groupby('bin'):
                disagreement_bins.append({'lookback':lookback,'bin':int(label),'count':len(b),
                    **{c:b[c].mean() for c in ['expected_return_dispersion','probability_spread','sigma_dispersion','mean_absolute_error','mean_brier','realized_volatility']}})
    csv('summaries/disagreement-bins.csv',pd.DataFrame(disagreement_bins))
    # Reliability retains components; there is deliberately no allocation/confidence score.
    reliability=[]
    for row in overall.to_dict('records'):
        selected=confidence[(confidence.model==row['model']) & (confidence.lookback==row['lookback'])] if not confidence.empty else pd.DataFrame()
        row['confidence_relationship']=selected.to_dict('records')
        row['regime_stability']=[r for f in regimes for r in f.to_dict('records') if r['model']==row['model'] and r['lookback']==row['lookback']]
        row['disagreement']=[r for r in disagreement_bins if r['lookback']==row['lookback']]
        reliability.append(row)
    atomic_json(root/'summaries/ticker-reliability.json',{'schema_version':1,'ticker':config.target_ticker,
                'experiment_id':meta['experiment_id'],'historical_period':[str(config.start_date),str(config.evaluation_end)],
                'data_vintage':config.data_vintage,'models':reliability,'planned_comparisons':(len(config.models)-1)*len(config.lookbacks)*3,
                'interpretation':'Development evidence, no capital-allocation score; conditional metrics must be read with failures and pending counts'})
    plots=[]
    def save(fig,name):
        path=root/'plots'/name;path.parent.mkdir(exist_ok=True)
        fig.tight_layout();fig.savefig(path,dpi=110,bbox_inches='tight');plt.close(fig);plots.append('plots/'+name)
    def panels(names):
        fig,axes=plt.subplots(int(np.ceil(len(names)/3)),3,figsize=(15,4*int(np.ceil(len(names)/3))),squeeze=False)
        for ax in axes.ravel()[len(names):]:ax.axis('off')
        return fig,axes.ravel()
    cumulative=[]
    for lookback,subset in good.groupby('lookback'):
        names=[m for m in config.models if m in set(subset.model)]
        for metric in ['mae','terminal_return_mae','brier']:
            for year in [None,*sorted(subset.year.unique())]:
                fig,ax=plt.subplots(figsize=(11,5))
                for model,g in subset.groupby('model'):
                    if model=='last_price':continue
                    g=g.sort_values('origin')
                    if year is not None:g=g[g.year==year]
                    # Missing comparisons stay NaN on the full origin grid; do not
                    # draw a falsely continuous skill curve through failures.
                    series=g.set_index('origin')['delta_'+metric].reindex(meta['origins'])
                    if year is not None:series=series[[int(x[:4])==year for x in series.index]]
                    total=series.cumsum();ax.plot(pd.to_datetime(total.index),total,label=model)
                    if year is None:
                        cumulative.extend({'origin':o,'model':model,'lookback':lookback,'loss':metric,'delta_loss':v,'cumulative_delta':total.loc[o]} for o,v in series.items())
                ax.axhline(0,color='black',lw=.6);ax.set(title=f'{lookback} days: cumulative {metric} loss difference ({year or "full period"})',ylabel='Model − persistence; lower is better')
                ax.legend(fontsize=7);fig.autofmt_xdate();save(fig,f'cumulative-{lookback}-{metric}-{year or "all"}.png')
        for kind in ['calibration','confidence','coverage','pit','pit-time','residuals']:
            fig,axes=panels(names)
            for ax,model in zip(axes,names):
                g=subset[subset.model==model].sort_values('origin');ax.set_title(model,fontsize=9)
                if kind=='calibration':
                    b=calibration[(calibration.model==model)&(calibration.lookback==lookback)]
                    ax.plot([0,1],[0,1],'k--',lw=.7);ax.scatter(b.mean_probability,b.observed_positive_frequency,s=20+np.sqrt(b['count'])*4)
                    for r in b.itertuples():ax.annotate(str(r.count),(r.mean_probability,r.observed_positive_frequency),fontsize=7)
                    ax.set(xlim=(-.02,1.02),ylim=(-.03,1.05),xlabel='Posterior/model P(up)',ylabel='Observed positive frequency')
                elif kind=='confidence':
                    b=confidence[(confidence.model==model)&(confidence.lookback==lookback)]
                    ax.plot(b.mean_confidence,b.direction_accuracy,'o-',label='direction accuracy');ax.plot(b.mean_confidence,b.brier,'s-',label='Brier')
                    ax.legend(fontsize=7);ax.set(xlabel='Mean |P(up)−0.5|')
                elif kind=='coverage':
                    ax.plot(LEVELS,[g['coverage_'+str(int(x*100))].mean() for x in LEVELS],'o-');ax.plot([.5,1],[.5,1],'k--');ax.set(xlabel='Nominal',ylabel='Empirical terminal coverage',ylim=(-.03,1.03))
                elif kind=='pit':
                    ax.hist(g.pit,bins=np.linspace(0,1,11));ax.axhline(len(g)/10,color='black',ls='--');ax.set(xlabel='Predictive CDF at realized return')
                elif kind=='pit-time':
                    ax.plot(pd.to_datetime(g.origin),g.pit,'.',ms=2);ax.set(ylim=(0,1),ylabel='PIT');ax.tick_params(axis='x',rotation=30)
                else:
                    if (g.distribution=='conditional_gaussian_log_return').all():
                        z=g.z.dropna().sort_values().to_numpy();theory=norm.ppf((np.arange(len(z))+.5)/max(len(z),1));ax.plot(theory,z,'.');ax.plot(theory,theory,'k--');ax.set(xlabel='Normal reference quantile',ylabel='Standardized error')
                    else:
                        ax.hist(g.z.dropna(),bins=20);ax.set(xlabel='Standardized error (not assumed Gaussian)',ylabel='Count')
            fig.suptitle(f'{config.target_ticker}: {lookback}-day lookback — {kind}');save(fig,f'{kind}-{lookback}.png')
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ax,metric in zip(axes.ravel(),['terminal_return_mae','brier','coverage_95','skill_terminal_return_mae']):
            for model,g in yearly[yearly.lookback==lookback].groupby('model'):
                if metric in g:ax.plot(g.year,g[metric],'o-',label=model)
            ax.set(title=metric,xlabel='Year')
        axes[0,0].legend(fontsize=6);save(fig,f'yearly-{lookback}.png')
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ax,grouping in zip(axes.ravel(),['sigma_bin','confidence_bin','realized_volatility_bin','trailing_volatility_regime']):
            selected=coverage_frame[(coverage_frame.lookback==lookback)&(coverage_frame.grouping==grouping)&(coverage_frame.nominal==.95)]
            for model,g in selected.groupby('model'):
                ax.plot(g['group'],g.empirical_coverage,'o-',label=model)
            ax.axhline(.95,color='black',ls='--');ax.set(title=grouping,ylabel='95% terminal coverage',ylim=(-.03,1.03))
        axes[0,0].legend(fontsize=6);save(fig,f'coverage-strata-{lookback}.png')
        if not disagree.empty:
            d=disagree[(disagree.lookback==lookback)&disagree.complete_panel]
            fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].scatter(d.expected_return_dispersion,d.mean_absolute_error,s=8);axes[1].scatter(d.probability_spread,d.mean_brier,s=8)
            axes[0].set(xlabel='Across-model expected-return dispersion',ylabel='Mean subsequent absolute error');axes[1].set(xlabel='P(up) spread',ylabel='Mean Brier');save(fig,f'disagreement-{lookback}.png')
    csv('summaries/cumulative-loss.csv',pd.DataFrame(cumulative))
    selected_columns=[c for c in ['model','lookback','successful','failures','pending','failure_rate','mae','nmae_pct','terminal_return_mae','skill_terminal_return_mae','brier','calibration_ece','coverage_95','sigma'] if c in overall]
    table=overall[selected_columns].to_html(index=False,float_format=lambda x:f'{x:.4g}',escape=True)
    caveats='''<p>These results are historical development evidence, not a strategy or a claim of alpha. No models or lookbacks are selected or tuned by this report.</p>
<ul><li>Posterior P(up) is a model probability; empirical reliability/calibration is an observed frequency. Neither is a capital-allocation confidence score.</li>
<li>Aleatoric uncertainty is future market randomness conditional on parameters. GP forecasts also marginalize parameter/latent uncertainty. Non-GP models retain their existing plug-in parameters. Model disagreement measures differences across forecasts, not a calibrated posterior over models.</li>
<li>Only successful fits enter accuracy/calibration metrics. Failure and pending counts are shown; fallback forecasts are included but flagged. Paired skill uses the intersection with successful persistence forecasts, and paired counts are saved.</li>
<li>95% coverage in the main table is TERMINAL RETURN coverage; path_coverage_95 preserves the older path-wise coverage metric. Returns are dimensionless log returns, NMAE/NRMSE/MAPE are percentages, and price errors use the ticker's quote currency.</li>
<li>Persistence abstains on direction; its existing probabilistic random-walk envelope supplies a 0.5 event-probability null and risk baseline.</li>
<li>Weekly first-session origins can overlap around holidays. Moving-block bootstrap uncertainty resamples whole paired episodes on the full origin grid, retaining missing forecasts. Small samples receive no confidence interval/p-value. Holm corrections account for the full planned model/lookback/loss family, including unavailable tests conservatively; no significance guarantees are implied.</li>
<li>Reliability bins start at deciles and merge adjacent bins to meet minimum counts. Confidence quantiles retain ties and may have fewer than five bins. Calibration regressions are descriptive, never fed into fitting. Sparse/extreme-bin results require caution.</li>
<li>Regimes and confidence/volatility bins are analysis-only groupings; full-sample thresholds never enter forecasts. Trailing target volatility and trailing return define regimes; realized horizon volatility is explicitly retrospective. No market index was added to the models.</li>
<li>PIT and interval coverage assess non-Gaussian mixtures without assuming Normal standardized residuals. Normal QQ references and analytic log scores apply only to models with a conditional Gaussian log-return distribution. No unstable density estimate is imposed on GP mixtures.</li>
<li>GP split R-hat/ESS safeguards do not prove convergence or tail accuracy. Weak-amplitude component lengths are unidentified. Frozen settings are not automatically tuned when fits fail. No diagnostic profiles are run unless explicitly requested.</li></ul>'''
    links=['summaries/overall.csv','yearly/metrics.csv','summaries/forecast-scores.csv','summaries/paired-comparisons.csv','summaries/ticker-reliability.json','summaries/failures.csv',
           'summaries/inference-details.json','summaries/episodes.csv','summaries/regimes.csv','summaries/disagreement.csv','summaries/disagreement-bins.csv',
           'calibration/reliability-bins.csv','calibration/confidence-bins.csv','calibration/coverage-strata.csv','summaries/cumulative-loss.csv']
    body=f'<!doctype html><meta charset="utf-8"><title>{html.escape(config.target_ticker)} historical validation</title><style>body{{font:15px system-ui;margin:35px;line-height:1.5}}table{{border-collapse:collapse}}td,th{{padding:5px;border:1px solid #ddd}}img{{max-width:100%}}section{{max-width:1500px}}</style><h1>{html.escape(config.target_ticker)} historical validation</h1><p>Experiment {meta["experiment_id"]}; {config.start_date} to {config.evaluation_end} exclusive. {len(good)} successful / {meta["total_units"]} planned units. Planned paired tests: {(len(config.models)-1)*len(config.lookbacks)*3}.</p><p><strong>Data provenance:</strong> {html.escape(meta["data_limitation"])}</p>'+caveats+table
    body+='<h2>Machine-readable tables and reliability components</h2><ul>'+''.join(f'<li><a href="{x}">{x}</a></li>' for x in links)+'</ul>'
    episode_table=pd.DataFrame(episodes).to_html(index=False,escape=True)
    for e in episodes:
        escaped=html.escape(e['record']);episode_table=episode_table.replace('>'+escaped+'<','><a href="'+escaped+'">'+escaped+'</a><')
    body+='<h2>Strongest positive and negative episodes</h2>'+episode_table
    body+='<p>Individual record paths above identify folds. Use <code>python -m stock_api.historical diagnose --output OUTPUT --origin YYYY-MM-DD --model GP_MODEL --lookback DAYS</code> for detailed GP plots; this explicitly reruns that selected fold only.</p>'
    body+=''.join(f'<section><h2>{html.escape(Path(p).stem)}</h2><a href="{p}"><img loading="lazy" src="{p}" alt="{html.escape(Path(p).stem)}"></a></section>' for p in plots)
    atomic_bytes(root/'index.html',body.encode())
    atomic_json(root/'report-manifest.json',{'experiment_id':meta['experiment_id'],'successful':len(good),'planned':meta['total_units'],
                'source_records':len(diagnostics),'plots':len(plots),'tables':links,'reporting_version':'historical-report-v1',
                'report_source_sha256':{name:sha(Path(__file__).parent/name) for name in ['analysis.py','report.py']}})
    return root/'index.html'
