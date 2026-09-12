"""Scores and serially dependent paired comparisons, computed without inference."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from .inference import LEVELS


def score(record):
    row={k:record.get(k) for k in ['unit_id','target_ticker','origin','model','lookback','status','runtime_seconds','training_volatility','training_return']}
    row['year']=int(record['origin'][:4])
    pred=record.get('prediction');real=record.get('realized')
    row.update(fallback=bool(pred and pred['diagnostics'].get('fallback')),
               warnings=len(pred['diagnostics'].get('warnings',[])) if pred else 0,
               converged=pred['diagnostics'].get('converged') if pred else None)
    if pred:
        d=pred['diagnostics']
        row.update(mean_process_active=d.get('mean_process_active'),volatility_process_active=d.get('volatility_process_active'),
                   max_rhat=max(d.get('split_rhat',[np.nan])),min_ess=min(d.get('autocorrelation_ess',[np.nan])),
                   weighted_ess=d.get('weighted_ess'),extreme_hyperparameters=any(max(v.values())>.2 for v in d.get('prior_edge_mass',{}).values()))
    if record['status']!='ok' or not real:return row
    actual=np.asarray(real['prices']);median=np.asarray(pred['path_median']);error=median-actual
    actual_return=real['terminal_log_return'];location=pred['median_log_return'];sigma=pred['terminal_sigma'];p=pred['probability_up']
    absolute=abs(actual_return-location);direction=0 if abs(location)<1e-10 else int(np.sign(location))
    z=(actual_return-pred['expected_log_return'])/sigma if sigma>0 else np.nan
    if pred['distribution']=='conditional_gaussian_log_return':
        ns=pred['normal_sigma'];mu=pred['normal_location']
        # A point mass has no continuous-uniform PIT. Do not manufacture a
        # misleading midpoint statistic or randomize a new evaluation outcome.
        pit=float(norm.cdf((actual_return-mu)/ns)) if ns>0 else np.nan
        nll=float(-norm.logpdf(actual_return,loc=mu,scale=ns)) if ns>0 else np.nan
    else:
        samples=np.asarray(pred['terminal_samples'])
        pit=float(((samples<actual_return).sum()+.5*(samples==actual_return).sum())/len(samples));nll=np.nan
    row.update(distribution=pred['distribution'],observations=len(actual),mae=np.abs(error).mean(),rmse=np.sqrt(np.mean(error**2)),
               squared_price_error=np.sum(error**2),absolute_price_error=np.sum(np.abs(error)),sum_actual_price=np.sum(actual),
               nmae_pct=100*np.abs(error).mean()/actual.mean(),nrmse_pct=100*np.sqrt(np.mean(error**2))/actual.mean(),
               mape_pct=100*np.mean(np.abs(error)/actual),terminal_return_mae=absolute,
               terminal_return_squared_error=(actual_return-location)**2,expected_return=pred['expected_log_return'],
               predicted_return=location,realized_return=actual_return,probability_up=p,actual_up=int(real['positive']),
               direction_correct=float(direction==real['direction']) if direction else np.nan,
               signed_realized_return=direction*actual_return if direction else np.nan,
               brier=(p-real['positive'])**2,sigma=sigma,confidence=abs(p-.5),
               signal_to_sigma=abs(pred['expected_log_return'])/sigma if sigma>0 else np.nan,
               z=z,abs_z=abs(z),pit=pit,negative_log_score=nll,
               realized_volatility=real['horizon_realized_volatility'])
    for level in LEVELS:
        interval=pred['intervals'][str(level)];tag=str(int(level*100))
        row['coverage_'+tag]=float(interval['return_lower']<=actual_return<=interval['return_upper'])
        row['path_coverage_'+tag]=float(np.mean((actual>=interval['path_lower'])&(actual<=interval['path_upper'])))
        row['interval_width_'+tag]=interval['return_upper']-interval['return_lower']
    return row


def add_pairs(frame):
    frame=frame.copy()
    good=frame[frame.status=='ok']
    if good.empty:return frame
    baseline=good[good.model=='last_price'][['origin','lookback','mae','terminal_return_mae','brier']]
    baseline=baseline.rename(columns={x:'null_'+x for x in ['mae','terminal_return_mae','brier']})
    frame=frame.merge(baseline,on=['origin','lookback'],how='left',validate='many_to_one')
    for metric in ['mae','terminal_return_mae','brier']:
        frame['delta_'+metric]=frame[metric]-frame['null_'+metric]
    return frame


def reliability_bins(frame,min_count=20):
    frame=frame.dropna(subset=['probability_up']).copy()
    if frame.empty:return pd.DataFrame()
    frame['bin']=np.minimum((frame.probability_up*10).astype(int),9)
    groups=[];acc=[]
    for _,group in frame.groupby('bin',sort=True):
        acc.extend(group.index)
        if len(acc)>=min_count:groups.append(acc);acc=[]
    if acc:
        if groups:groups[-1]+=acc
        else:groups=[acc]
    return pd.DataFrame([{'count':len(g),'probability_lower':g.probability_up.min(),'probability_upper':g.probability_up.max(),
                          'mean_probability':g.probability_up.mean(),'observed_positive_frequency':g.actual_up.mean(),
                          'mean_realized_return':g.realized_return.mean(),'mean_absolute_error':g.terminal_return_mae.mean(),
                          'sparse':len(g)<min_count} for g in [frame.loc[idx] for idx in groups]])


def confidence_bins(frame):
    frame=frame.copy()
    if frame.confidence.nunique()<2:frame['confidence_bin']=0
    else:frame['confidence_bin']=pd.qcut(frame.confidence,5,labels=False,duplicates='drop').fillna(0).astype(int)
    rows=[]
    for label,g in frame.groupby('confidence_bin'):
        rows.append({'bin':int(label),'count':len(g),'mean_confidence':g.confidence.mean(),
                     'direction_accuracy':g.direction_correct.mean(),'directional_count':g.direction_correct.count(),
                     'mean_standardized_absolute_error':g.abs_z.mean(),'signed_realized_return':g.signed_realized_return.mean(),
                     'brier':g.brier.mean(),'calibration_gap':g.probability_up.mean()-g.actual_up.mean(),
                     'coverage_95':g.coverage_95.mean()})
    return pd.DataFrame(rows)


def calibration_fit(frame):
    # Descriptive logistic recalibration only; never fed back to forecasts.
    if len(frame)<50 or frame.probability_up.nunique()<5 or frame.actual_up.nunique()<2:return None,None
    from scipy.optimize import minimize
    x=np.log(np.clip(frame.probability_up.to_numpy(),1e-6,1-1e-6)/np.clip(1-frame.probability_up.to_numpy(),1e-6,1))
    y=frame.actual_up.to_numpy()
    def loss(beta):
        z=beta[0]+beta[1]*x
        return np.mean(np.logaddexp(0,z)-y*z)
    fit=minimize(loss,[0.,1.],method='BFGS')
    if not fit.success or not np.isfinite(fit.x).all() or np.max(np.abs(fit.x))>50:return None,None
    return float(fit.x[0]),float(fit.x[1])


def aggregate(frame,groups=('model','lookback'),min_count=20):
    rows=[]
    for key,all_group in frame.groupby(list(groups),dropna=False):
        key=key if isinstance(key,tuple) else (key,);g=all_group[all_group.status=='ok']
        attempted=all_group[all_group.status!='pending'];failed=len(attempted)-len(g)
        row=dict(zip(groups,key));row.update(planned_records=len(all_group),successful=len(g),failures=failed,pending=len(all_group)-len(attempted),
            failure_rate=failed/len(attempted) if len(attempted) else np.nan,fallback_rate=attempted.fallback.mean(),
            total_runtime_seconds=all_group.runtime_seconds.sum(),median_runtime_seconds=all_group.runtime_seconds.median(),
            warnings_count=all_group.warnings.sum())
        if not g.empty:
            for metric in ['mae','rmse','nmae_pct','nrmse_pct','mape_pct','terminal_return_mae','brier','sigma','direction_correct','coverage_50','coverage_68','coverage_90','coverage_95','path_coverage_95','interval_width_95','negative_log_score']:
                row[metric]=g[metric].mean()
            row['terminal_return_rmse']=np.sqrt(g.terminal_return_squared_error.mean())
            row['terminal_return_correlation']=g.predicted_return.corr(g.realized_return) if len(g)>2 and g.predicted_return.std()>1e-12 and g.realized_return.std()>1e-12 else np.nan
            observations=g.observations.sum()
            row.update(observations=observations,pooled_mae=g.absolute_price_error.sum()/observations,
                       pooled_rmse=np.sqrt(g.squared_price_error.sum()/observations),
                       pooled_nmae_pct=100*g.absolute_price_error.sum()/g.sum_actual_price.sum(),
                       pooled_nrmse_pct=100*np.sqrt(g.squared_price_error.sum()/observations)/(g.sum_actual_price.sum()/observations),
                       directional_origins=g.direction_correct.count(),pit_count=g.pit.count(),pit_mean=g.pit.mean(),pit_variance=g.pit.var(ddof=0),
                       pit_extreme_fraction=((g.pit.dropna()<.025)|(g.pit.dropna()>.975)).mean(),z_mean=g.z.mean(),z_variance=g.z.var(ddof=0))
            bins=reliability_bins(g,min_count)
            row['calibration_ece']=np.average(abs(bins.mean_probability-bins.observed_positive_frequency),weights=bins['count'])
            row['calibration_intercept'],row['calibration_slope']=calibration_fit(g)
            for metric in ['mae','terminal_return_mae','brier']:
                paired=g.dropna(subset=['null_'+metric]);base=paired['null_'+metric].mean()
                row['paired_count_'+metric]=len(paired)
                row['skill_'+metric]=1-paired[metric].mean()/base if base>0 else np.nan
                row['mean_delta_'+metric]=paired['delta_'+metric].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def block_comparison(series,origins,block_weeks=8,replicates=2000,seed=0):
    """Circular moving-block bootstrap on the FULL origin grid, retaining missing units."""
    values=series.reindex(origins).to_numpy(dtype=float);n=len(values);finite=values[np.isfinite(values)]
    mean=float(finite.mean()) if len(finite) else np.nan
    result={'paired_origins':len(finite),'mean_loss_difference':mean,'lower95':None,'upper95':None,'p_value':None,
            'standardized_effect':mean/finite.std(ddof=1) if len(finite)>1 and finite.std(ddof=1)>0 else None,
            'method':'circular_moving_block_bootstrap','block_weeks':block_weeks}
    dates=pd.DatetimeIndex(origins);window=pd.Timedelta(weeks=block_weeks)
    full_window=dates+window<=dates[-1] if n else np.array([],dtype=bool)
    counts=dates.searchsorted(dates+window)-np.arange(n)
    block=max(1,int(np.median(counts[full_window]))) if full_window.any() else max(1,block_weeks)
    result['block_origins']=block
    if len(finite)<max(20,2*block) or n<2*block:return result
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(replicates):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        sampled=values[idx];sampled=sampled[np.isfinite(sampled)]
        if len(sampled):boot.append(sampled.mean())
    boot=np.asarray(boot)
    if not len(boot):return result
    result.update(lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),
                  p_value=float((1+(abs(boot-mean)>=abs(mean)).sum())/(len(boot)+1)))
    return result


def holm(values,total_tests=None):
    values=np.asarray(values,float);out=np.full(len(values),np.nan);valid=np.flatnonzero(np.isfinite(values))
    order=valid[np.argsort(values[valid])]
    family=max(len(order),total_tests or len(values))
    if len(order):out[order]=np.minimum(1,np.maximum.accumulate(values[order]*(family-np.arange(len(order)))))
    return out


def regime_labels(frame):
    frame=frame.copy()
    # Thresholds are retrospective descriptive groupings, never inference inputs.
    for source,name in [('training_volatility','trailing_volatility_regime'),('realized_volatility','realized_volatility_regime')]:
        if source not in frame:continue
        unique=frame.drop_duplicates(['origin','lookback'])
        thresholds=unique[source].dropna().quantile([1/3,2/3]).to_numpy()
        if len(thresholds)==2:
            frame[name]=np.where(frame[source].isna(),'unknown',np.where(frame[source]<=thresholds[0],'low',np.where(frame[source]<=thresholds[1],'medium','high')))
    frame['return_environment']=np.where(frame.training_return.isna(),'unknown',np.where(frame.training_return>=0,'trailing_positive','trailing_negative'))
    return frame


def disagreement(frame,expected_models):
    rows=[]
    for (origin,lookback),g in frame[frame.status=='ok'].groupby(['origin','lookback']):
        rows.append({'origin':origin,'lookback':lookback,'available_models':len(g),'expected_models':expected_models,
                     'complete_panel':len(g)==expected_models,'expected_return_dispersion':g.expected_return.std(ddof=0),
                     'probability_spread':g.probability_up.max()-g.probability_up.min(),
                     'sigma_dispersion':g.sigma.std(ddof=0),'mean_absolute_error':g.terminal_return_mae.mean(),
                     'mean_brier':g.brier.mean(),'realized_volatility':g.realized_volatility.mean()})
    return pd.DataFrame(rows)
