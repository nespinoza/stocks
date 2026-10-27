#!/usr/bin/env python3
"""Walk-forward two-regime GP + macro/news experiment; restartable joint GP cache."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
import sys
import warnings
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from stock_api.portfolio import joint_gp,crps
from stock_api.macro_news import FACTORS,NEWS,macro_states,parse_news,news_vector,ridge_posterior,ar_forecast,block_difference_ci
from run_portfolio_gp_22 import extract_tickers,simulate,metrics
OUT=ROOT/'experiments/portfolio_gp_macro_news_2026-10-18'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align_priors(priors,source_tickers,target_tickers):
    """Transfer only observed instrument posteriors; absent names keep exact weak prior."""
    result={}
    for kind,post in priors.items():
        p=post['mean'].shape[0]; n=len(target_tickers)
        aligned=dict(mean=np.zeros((p,n)),covariances=np.tile(np.eye(p),(n,1,1)),available=np.zeros(n,dtype=bool),n=post['n'])
        for i,t in enumerate(target_tickers):
            if t in source_tickers:
                j=source_tickers.index(t); aligned['mean'][:,i]=post['mean'][:,j]; aligned['covariances'][i]=post['covariances'][j]; aligned['available'][i]=True
        result[kind]=aligned
    return result


def load_window(window,tickers):
    close={}; opening={}
    for ticker in tickers+['SPY']:
        f=pd.read_csv(OUT/'data/prices'/window/f'{ticker}.csv',index_col='Date',parse_dates=True,float_precision='round_trip')
        close[ticker]=f['Adj Close']; opening[ticker]=f.Open*f['Adj Close']/f.Close
    return pd.DataFrame(close),pd.DataFrame(opening)


def periods_for(index):
    executions=[]
    for month in pd.period_range(index[0],index[-1],freq='M'):
        eligible=index[index>=pd.Timestamp(f'{month}-17')]
        if len(eligible): executions.append(eligible[0])
    result=[]
    for j,ex in enumerate(executions):
        origin=index[index.get_loc(ex)-1]; following=executions[j+1] if j+1<len(executions) else None
        end=index[index.get_loc(following)-1] if following is not None else index[-1]
        result.append(dict(period=j,origin=origin,execution=ex,following=following,target_end=end,horizon=int(((index>origin)&(index<=end)).sum())))
    return result


def fit_gps(c,window,tickers,close,periods,smoke=False):
    returns=np.log(close[tickers]).diff().dropna(); models={}; params=[]
    previous=ROOT/c['previous_experiment']; old_schedule=pd.read_csv(previous/'schedule.csv')
    old_prices=pd.read_csv(previous/'data/adjusted_close.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    old_manifest=json.loads((previous/'manifest.json').read_text())
    for s in periods:
        origin=s['origin']; train=returns.loc[:origin].tail(c['gp_lookback'])
        if len(train)<c['gp_lookback']: continue
        key=f'{window}_{origin.date()}'; path=OUT/'forecasts/gp_cache'/f'{key}.npz'; info_path=path.with_suffix('.json')
        legacy=hashlib.sha256(train.to_numpy().tobytes()+str(s['horizon']).encode()+sha(ROOT/'stock_api/heteroskedastic.py').encode()+sha(ROOT/'stock_api/portfolio.py').encode()+str(c['paths']).encode()).hexdigest()
        gp_configuration=dict(tickers=tickers,lookback=c['gp_lookback'],paths=c['paths'],base_seed=c['seed'],correlation_shrinkage=c['correlation_shrinkage'],mean_gp=True,matching_original_cache_preferred=True)
        fingerprint=hashlib.sha256((legacy+json.dumps(gp_configuration,sort_keys=True)).encode()).hexdigest()
        # One-time metadata migration, only for this already verified frozen run.
        if path.exists() and info_path.exists():
            saved=json.loads(info_path.read_text()); manifest_path=OUT/'logs/manifest.json'
            if saved.get('fingerprint')==legacy and manifest_path.exists() and json.loads(manifest_path.read_text())['config_hash']==sha(OUT/'config.yaml'):
                saved.update(fingerprint=fingerprint,gp_configuration=gp_configuration)
                info_path.write_text(json.dumps(saved,indent=2))
        if path.exists() and json.loads(info_path.read_text())['fingerprint']==fingerprint:
            model=dict(np.load(path)); info=json.loads(info_path.read_text()); model['parameters']=info['parameters']
            logging.info('Reuse GP %s',key)
        else:
            matched=old_schedule[(old_schedule.origin==str(origin.date()))&(old_schedule.target_end==str(s['target_end'].date()))]
            reused=False
            if c.get('reuse_previous_gp',True) and list(old_prices.drop(columns='SPY',errors='ignore').columns)==tickers and window=='B' and len(matched)==1 and all(old_manifest['hashes'][p]==sha(ROOT/p) for p in ['stock_api/heteroskedastic.py','stock_api/portfolio.py']):
                old_train=np.log(old_prices[tickers]).diff().loc[:origin].tail(c['gp_lookback'])
                if len(old_train)==len(train) and np.allclose(old_train,train,rtol=0,atol=1e-14):
                    source=previous/'predictive'/f'estimated_drift_lb30_p{int(matched.iloc[0].period)}.npz'
                    model=dict(np.load(source)); model['parameters']=json.loads(source.with_suffix('.json').read_text())['parameters']; reused=True
                    logging.info('Reuse EXACT previous experiment distribution %s',key)
            if not reused:
                logging.info('Fit GP %s (%s sessions)',key,s['horizon'])
                with warnings.catch_warnings(record=True) as ws:
                    model=joint_gp(train.to_numpy(),s['horizon'],mean_gp=True,n_paths=c['paths'],seed=c['seed']+int(origin.strftime('%Y%m%d')),shrinkage=c['correlation_shrinkage'])
                for warning in ws: logging.warning('%s %s',key,warning.message)
            np.savez_compressed(path,**{k:v for k,v in model.items() if k!='parameters'})
            info=dict(fingerprint=fingerprint,gp_configuration=gp_configuration,parameters=model['parameters'],reused_previous=reused,origin=str(origin.date()),target_end=str(s['target_end'].date()),train_start=str(train.index[0].date()),train_end=str(train.index[-1].date()))
            info_path.write_text(json.dumps(info,indent=2))
        models[origin]=model
        for ticker,p in zip(tickers,model['parameters']): params.append(dict(window=window,origin=str(origin.date()),ticker=ticker,**p))
        if smoke and origin>=pd.Timestamp(c['windows'][window]['evaluation_start']): break
    return models,params


def design_intervals(states,close,tickers,gps,records,c):
    dates=list(pd.to_datetime(states.origin)); result=[]
    factors=states.set_index(pd.to_datetime(states.origin))[FACTORS]
    for prev,end in zip(dates[:-1],dates[1:]):
        actual=np.log(close.loc[end,tickers]/close.loc[prev,tickers]).to_numpy()
        market=float(np.log(close.loc[end,'SPY']/close.loc[prev,'SPY']))
        news,ids=news_vector(records,prev,c['news_lookback_days'])
        result.append(dict(origin=prev,end=end,y=actual,market=market,z=(factors.loc[end]-factors.loc[prev]).to_numpy(),news=news,
                           residual=actual-gps[prev]['mu'] if prev in gps else None))
    return result


def macro_design(intervals,c):
    return np.array([np.r_[1,row['market']/c['market_scale'],row['z']/c['fixed_factor_scales']] for row in intervals])


def exposure_rows(posterior,window,origin,tickers,variant,c):
    names=['alpha','market']+FACTORS; scales=np.r_[1,c['market_scale'],c['fixed_factor_scales']]; rows=[]
    for i,ticker in enumerate(tickers):
        for k,name in enumerate(names):
            mean=posterior['mean'][k,i]/scales[k]; sd=np.sqrt(posterior['covariances'][i,k,k])/scales[k]
            rows.append(dict(window=window,origin=str(origin.date()),variant=variant,ticker=ticker,factor=name,n_months=posterior['n'],coefficient=mean,standard_error=sd,lower95=mean-1.96*sd,upper95=mean+1.96*sd,
                             standardized_coefficient=posterior['mean'][k,i]))
    return rows


def shifted(model,delta):
    return {'mu':model['mu']+delta,'covariance':model['covariance'],'samples':model['samples']+delta}


def augment(c,window,tickers,close,periods,gps,states,records,priors=None,smoke=False):
    intervals=design_intervals(states,close,tickers,gps,records,c)
    names=['GP only','GP + Macro','GP + News (deterministic)','GP + Macro + News (deterministic)','GP + Macro placebo','GP + News placebo']
    if window=='B': names+=['GP + Macro transfer']
    all_models={name:{} for name in names}; exposure=[]; news_inputs=[]; forecast_rows=[]; factor_predictions=[]; transfer_calibration=[]
    for s in periods:
        origin=s['origin']
        if origin not in gps: continue
        if origin<pd.Timestamp(c['windows'][window]['evaluation_start']): continue
        history=[row for row in intervals if row['end']<=origin][-c['exposure_window_months']:]
        x=macro_design(history,c); y=np.array([row['y'] for row in history])
        descriptive=ridge_posterior(x,y,c['exposure_ridge'])
        exposure+=exposure_rows(descriptive,window,origin,tickers,'weak_prior',c)
        transfer_desc=None
        if window=='B' and priors is not None:
            transfer_desc=ridge_posterior(x,y,c['exposure_ridge'],priors['descriptive'])
            exposure+=exposure_rows(transfer_desc,window,origin,tickers,'transfer_prior',c)
        completed=[row for row in history if row['residual'] is not None]
        news,ids=news_vector(records,origin,c['news_lookback_days'])
        for name,value in zip(NEWS,news): news_inputs.append(dict(window=window,origin=str(origin.date()),feature=name,value=value,records=len(ids),identifiers=json.dumps(ids)))
        zhat=ar_forecast(np.array([row['z'] for row in history]),s['horizon'],c['ar_ridge'])
        macro_pred=np.r_[0,0,zhat/np.array(c['fixed_factor_scales'])]
        news_pred=np.r_[0,news]; combined_pred=np.r_[macro_pred,news]
        actual_interval=next(row for row in intervals if row['origin']==origin)
        for k,factor in enumerate(FACTORS): factor_predictions.append(dict(window=window,origin=str(origin.date()),target_end=str(s['target_end'].date()),factor=factor,predicted_change=zhat[k],realized_change=actual_interval['z'][k]))
        corrections={name:np.zeros(len(tickers)) for name in names}; uncertainty={name:np.zeros(len(tickers)) for name in names}
        if len(completed)>=c['minimum_completed_incremental_months']:
            mx=macro_design(completed,c); nx=np.array([np.r_[1,row['news']] for row in completed]); cx=np.column_stack([mx,nx[:,1:]])
            residuals=np.array([row['residual'] for row in completed]); rng=np.random.default_rng(c['seed']+int(origin.strftime('%Y%m%d')))
            permutation=rng.permutation(len(completed))
            designs={'GP + Macro':(mx,macro_pred,None),'GP + News (deterministic)':(nx,news_pred,None),
                'GP + Macro + News (deterministic)':(cx,combined_pred,None),
                'GP + Macro placebo':(mx[permutation],macro_pred,None),'GP + News placebo':(nx[permutation],news_pred,None)}
            if window=='B': designs['GP + Macro transfer']=(mx,macro_pred,priors['incremental'])
            for name,(design,pred,prior) in designs.items():
                fitted=ridge_posterior(design,residuals,c['incremental_ridge'],prior)
                corrections[name]=pred@fitted['mean']
                uncertainty[name]=np.array([np.sqrt(pred@v@pred) for v in fitted['covariances']])
                checkpoint=OUT/'exposures/mapping_cache'/f'{window}_{origin.date()}_{name.replace(" ","_").replace("+","plus")}.npz'
                checkpoint.parent.mkdir(exist_ok=True)
                np.savez_compressed(checkpoint,design=design,targets=residuals,prediction_features=pred,
                    posterior_mean=fitted['mean'],posterior_covariances=fitted['covariances'],
                    delta=corrections[name],training_end_dates=np.array([str(r['end'].date()) for r in completed]))
                if name=='GP + News (deterministic)':
                    for i,ticker in enumerate(tickers):
                        for k,factor in enumerate(['alpha']+NEWS):
                            mean=fitted['mean'][k,i]; sd=np.sqrt(fitted['covariances'][i,k,k])
                            exposure.append(dict(window=window,origin=str(origin.date()),variant='news_incremental',ticker=ticker,factor=factor,n_months=len(completed),coefficient=mean,standard_error=sd,lower95=mean-1.96*sd,upper95=mean+1.96*sd,standardized_coefficient=mean))
        # Ex-post exposure calibration is a diagnostic only; never fed to allocations.
        realized_design=np.r_[1,actual_interval['market']/c['market_scale'],actual_interval['z']/c['fixed_factor_scales']]
        for label,post in [('weak_prior',descriptive),('transfer_prior',transfer_desc)]:
            if post is None: continue
            pred=realized_design@post['mean']; variance=post['residual_variance']+np.array([realized_design@v@realized_design for v in post['covariances']])
            for i,ticker in enumerate(tickers): transfer_calibration.append(dict(window=window,origin=str(origin.date()),variant=label,ticker=ticker,predicted_conditional_return=pred[i],realized=actual_interval['y'][i],error=actual_interval['y'][i]-pred[i],z=(actual_interval['y'][i]-pred[i])/np.sqrt(variance[i])))
        endprices=close.loc[s['target_end'],tickers]; actual=np.log(endprices/close.loc[origin,tickers]).to_numpy()
        realized_variance=np.log(close.loc[origin:s['target_end'],tickers]).diff().iloc[1:].pow(2).sum().to_numpy()
        for name in names:
            model=shifted(gps[origin],corrections[name]); all_models[name][origin]=model
            qs=np.quantile(model['samples'],[.05,.16,.5,.84,.95],axis=0); sd=np.sqrt(np.diag(model['covariance'])); scores=crps(model['samples'],actual)
            for i,ticker in enumerate(tickers):
                row=dict(window=window,strategy=name,origin=str(origin.date()),target_end=str(s['target_end'].date()),execution=str(s['execution'].date()),horizon=s['horizon'],ticker=ticker,
                    mean=model['mu'][i],std=sd[i],actual=actual[i],delta=corrections[name][i],mapping_parameter_sd=uncertainty[name][i],crps=scores[i],z=(actual[i]-model['mu'][i])/sd[i],
                    completed_mapping_months=len(completed),mapping_active=len(completed)>=c['minimum_completed_incremental_months'],latest_training_target=str(completed[-1]['end'].date()) if completed else None,
                    probability_up=float((model['samples'][:,i]>0).mean()))
                row.update(realized_sum_squared_daily_returns=realized_variance[i],innovation_variance=sd[i]**2-gps[origin]['drift_variance'][i])
                row.update({f'q{int(q*100):02d}':qs[k,i] for k,q in enumerate([.05,.16,.5,.84,.95])}); forecast_rows.append(row)
        if smoke: break
    # Freeze the full first regime (including COVID), rather than only its last 24 months.
    final_history=[r for r in intervals if r['end']<=max(gps)] if smoke else intervals
    completed=[row for row in final_history if row['residual'] is not None]
    final=dict(descriptive=ridge_posterior(macro_design(final_history,c),np.array([r['y'] for r in final_history]),c['exposure_ridge']),
        incremental=ridge_posterior(macro_design(completed,c),np.array([r['residual'] for r in completed]),c['incremental_ridge']))
    exposure+=exposure_rows(final['descriptive'],window,close.index[-1],tickers,'final_frozen',c)
    return all_models,forecast_rows,exposure,news_inputs,factor_predictions,transfer_calibration,final


def run_portfolios(c,window,tickers,close,opening,periods,all_models,subset=False):
    old=json.loads((ROOT/c['previous_experiment']/'config.yaml').read_text())
    chosen=[s for s in periods if s['origin'] in all_models['GP only'] and (not subset or s['execution']>=pd.Timestamp('2026-04-17'))]
    chosen=[dict(s,period=j) for j,s in enumerate(chosen)]
    rets=np.log(close[tickers]).diff().dropna(); outputs=[[],[],[],[]]
    specs=[(name,'gp') for name in all_models]+[('Equal weight','equal'),('SPY','spy')]
    for name,kind in specs:
        source=all_models[name] if name in all_models else all_models['GP only']
        cache={(30,'estimated_drift',s['period']):source[s['origin']] for s in chosen}
        daily,monthly,weights,risk,_=simulate(old,tickers,close,opening,chosen,cache,rets,name,kind,30,c['lambda'],c['cost_bps'])
        for dest,items in zip(outputs,[daily,monthly,weights,risk]):
            for row in items: row['window']=window+'_matched_2026' if subset else window
            dest.extend(items)
    return tuple(pd.DataFrame(rows) for rows in outputs)


def score(f,c):
    rows=[]; paired=[]; monthly_scores=[]
    for (window,name),g in f.groupby(['window','strategy'],sort=False):
        err=g.actual-g['mean']; corr=np.corrcoef(g['mean'],g.actual)[0,1]
        ranks=[]
        for origin,h in g.groupby('origin'):
            ranks.append(spearmanr(h['mean'],h.actual).statistic)
            monthly_scores.append(dict(window=window,strategy=name,origin=origin,mae=np.abs(h.actual-h['mean']).mean(),mse=((h.actual-h['mean'])**2).mean(),crps=h.crps.mean(),spearman=ranks[-1]))
        rows.append(dict(window=window,strategy=name,n=len(g),months=g.origin.nunique(),mae=np.abs(err).mean(),rmse=np.sqrt(np.mean(err**2)),crps=g.crps.mean(),coverage90=((g.actual>=g.q05)&(g.actual<=g.q95)).mean(),coverage68=((g.actual>=g.q16)&(g.actual<=g.q84)).mean(),correlation=corr,spearman=np.mean(ranks),direction_accuracy=((g['mean']>0)==(g.actual>0)).mean(),rms_z=np.sqrt(np.mean(g.z**2)),mean_abs_adjustment=g.delta.abs().mean()))
    monthly=pd.DataFrame(monthly_scores)
    for window,group in monthly.groupby('window'):
        base=group[group.strategy=='GP only'].set_index('origin')
        for name,g in group.groupby('strategy'):
            if name=='GP only': continue
            g=g.set_index('origin').loc[base.index]
            for metric in ['mae','mse','crps','spearman']:
                avg,lo,hi=block_difference_ci(g[metric]-base[metric],seed=c['seed'],replicates=c['bootstrap']['replicates'],block=c['bootstrap']['block_months'])
                paired.append(dict(window=window,strategy=name,baseline='GP only',metric=metric,difference=avg,lower95=lo,upper95=hi,months=len(base),unit='whole market-month; unadjusted exploratory interval'))
        for name in ['GP + Macro + News (deterministic)','GP + Macro transfer']:
            if name not in set(group.strategy): continue
            base=group[group.strategy=='GP + Macro'].set_index('origin')
            g=group[group.strategy==name].set_index('origin').loc[base.index]
            for metric in ['mae','mse','crps','spearman']:
                avg,lo,hi=block_difference_ci(g[metric]-base[metric],seed=c['seed'],replicates=c['bootstrap']['replicates'],block=c['bootstrap']['block_months'])
                paired.append(dict(window=window,strategy=name,baseline='GP + Macro',metric=metric,difference=avg,lower95=lo,upper95=hi,months=len(base),unit='whole market-month; unadjusted exploratory interval'))
    scores=pd.DataFrame(rows)
    ratios=f.groupby(['window','strategy']).apply(lambda g:g.realized_sum_squared_daily_returns.sum()/g.innovation_variance.sum(),include_groups=False)
    scores['realized_to_predicted_daily_variance']=[ratios.loc[(r.window,r.strategy)] for r in scores.itertuples()]
    return scores,pd.DataFrame(paired),monthly


def audit_outputs(f,monthly,weights,daily,comparison,c):
    assert (pd.to_datetime(f.latest_training_target)<=pd.to_datetime(f.origin)).all()
    assert (pd.to_datetime(f.origin)<pd.to_datetime(f.execution)).all()
    grouped=weights.groupby(['window','strategy','period']).weight
    assert np.max(np.abs(grouped.sum()-1))<1e-8
    assert weights.weight.min()>=-1e-8
    assert weights[weights.strategy!='SPY'].weight.max()<=c['cap']+1e-8
    assert np.max(np.abs(monthly.cost-c['cost_bps']/10000*(monthly.bought_dollars+monthly.sold_dollars)))<1e-8
    for (window,name),group in monthly.groupby(['window','strategy']):
        perf=comparison[(comparison.window==window)&(comparison.strategy==name)].iloc[0]
        np.testing.assert_allclose(np.prod(1+group.net_return)-1,perf.net_return,atol=1e-10)
        curve=daily[(daily.window==window)&(daily.strategy==name)]
        np.testing.assert_allclose(curve.net_equity.iloc[-1],perf.net_ending_dollars,atol=1e-8)
    return {'status':'PASS','forecast_rows':len(f),'stock_months_per_model':f.groupby(['window','strategy']).size().to_dict().__str__(),'strategies_evaluated':len(comparison),'checks':['chronology','weight sum','positivity','cap','fees','monthly compounding','daily endpoint']}


def main():
    global OUT
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--output',type=Path,default=OUT); args=ap.parse_args()
    OUT=args.output.resolve()
    for folder in ['forecasts/gp_cache','exposures','portfolios','figures','logs','source']: (OUT/folder).mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'logs/run.log')])
    c=json.loads((OUT/'config.yaml').read_text())
    all_tickers=(OUT/'tickers.txt').read_text().splitlines() if c.get('window_tickers') else extract_tickers(ROOT/c['source_report'])
    assert len(all_tickers)==len(set(all_tickers))==c.get('ticker_count',22)
    states=macro_states(OUT); records=parse_news(OUT,c)
    logging.info('Verified %s macro states, %s dated deterministic news records; LLM NOT RUN',len(states),len(records))
    totals={k:[] for k in ['forecasts','exposure','news','factors','calibration','parameters']}; portfolios=[[],[],[],[]]; comparisons=[]; prior=None
    for window in ['A','B']:
        tickers=c.get('window_tickers',{}).get(window,all_tickers)
        if window=='B' and tickers!=c.get('window_tickers',{}).get('A',all_tickers):
            prior=align_priors(prior,c['window_tickers']['A'],tickers)
        close,opening=load_window(window,tickers); periods=periods_for(close.index)
        gps,parameters=fit_gps(c,window,tickers,close,periods,args.smoke)
        available_periods=[s for s in periods if s['origin'] in gps]
        if args.smoke:
            # Keep all original intervals for targets, but test one July allocation.
            available_periods=[s for s in available_periods if s['origin']>=pd.Timestamp(c['windows'][window]['evaluation_start'])][:1]
        window_states=states[(states.window==window)&pd.to_datetime(states.origin).isin(close.index)]
        models,forecasts,exposure,news,factors,calibration,prior_new=augment(c,window,tickers,close,periods if not args.smoke else available_periods,gps,window_states,records,prior,args.smoke)
        if window=='A': prior=prior_new
        np.savez_compressed(OUT/f'exposures/window_{window}_frozen_posterior.npz',
            **{f'{kind}_{key}':value for kind,post in prior_new.items() for key,value in post.items()})
        for key,value in [('forecasts',forecasts),('exposure',exposure),('news',news),('factors',factors),('calibration',calibration),('parameters',parameters)]: totals[key]+=value
        outputs=run_portfolios(c,window,tickers,close,opening,periods,models)
        for target,output in zip(portfolios,outputs): target.append(output)
        perf=metrics(outputs[0],outputs[1],c['initial_dollars']); perf['window']=window; comparisons.append(perf)
        if window=='B' and not args.smoke:
            outputs=run_portfolios(c,window,tickers,close,opening,periods,models,subset=True)
            for target,output in zip(portfolios,outputs): target.append(output)
            perf=metrics(outputs[0],outputs[1],c['initial_dollars']); perf['window']='B_matched_2026'; comparisons.append(perf)
        logging.info('Completed window %s',window)
        if args.smoke: break
    if args.smoke:
        (OUT/'logs/smoke.json').write_text(json.dumps({'status':'PASS','GP_origins':len(gps),'forecast_rows':len(totals['forecasts']),'portfolio_strategies':len(comparisons[0])},indent=2)); return
    f=pd.DataFrame(totals['forecasts']); exposure=pd.DataFrame(totals['exposure']); daily,monthly,weights,risk=[pd.concat(frames,ignore_index=True) for frames in portfolios]; comparison=pd.concat(comparisons,ignore_index=True)
    diag,paired,month_scores=score(f,c)
    for window in ['A','B']:
        f[f.window==window].to_csv(OUT/f'forecasts/window_{window}_forecasts.csv',index=False)
        exposure[exposure.window==window].to_csv(OUT/f'exposures/window_{window}_exposures.csv',index=False)
        comparison[comparison.window==window].to_csv(OUT/f'portfolios/window_{window}_performance.csv',index=False)
    for frame,name in [(daily,'portfolios/daily'),(monthly,'portfolios/monthly'),(weights,'portfolios/monthly_weights'),(risk,'portfolios/risk'),(comparison,'ablation_results'),(diag,'forecast_diagnostics'),(paired,'paired_comparisons'),(month_scores,'monthly_forecast_scores')]: frame.to_csv(OUT/f'{name}.csv',index=False)
    for key in ['news','factors','calibration','parameters']: pd.DataFrame(totals[key]).to_csv(OUT/f'forecasts/{key}.csv',index=False)
    audit=audit_outputs(f,monthly,weights,daily,comparison,c)
    # Exact matched baseline is an explicit regression check against the earlier experiment.
    if c.get('reuse_previous_gp',True):
        previous=pd.read_csv(ROOT/c['previous_experiment']/'benchmark_comparison.csv').set_index('strategy')
        matched=comparison[comparison.window=='B_matched_2026'].set_index('strategy')
        np.testing.assert_allclose(matched.loc['GP only','net_return'],previous.loc['Heteroskedastic GP optimizer','net_return'],atol=1e-10)
        audit['original_GP_net_return_reproduced']=float(matched.loc['GP only','net_return'])
    (OUT/'logs/audit.json').write_text(json.dumps(audit,indent=2))
    if c.get('window_tickers'):
        from report_growth20_study import build_report
    else:
        from report_macro_news_study import build_report
    build_report(OUT,c,all_tickers,states,records,f,exposure,diag,paired,comparison,daily,monthly,weights)
    from audit_macro_news_study import audit as independent_audit
    independent_audit(OUT)
    files=['stock_api/macro_news.py','stock_api/portfolio.py','stock_api/heteroskedastic.py','scripts/run_macro_news_study.py','scripts/download_macro_news_study.py','scripts/report_macro_news_study.py','scripts/run_portfolio_gp_22.py','scripts/audit_macro_news_study.py','tests/test_macro_news.py']
    if c.get('window_tickers'): files+=['scripts/prepare_growth20_study.py','scripts/report_growth20_study.py','tests/test_growth20_study.py']
    manifest=dict(config_hash=sha(OUT/'config.yaml'),command='MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_macro_news_study.py --output '+str(OUT.relative_to(ROOT)),python=sys.version,versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib','requests','beautifulsoup4','exchange-calendars']},sources={p:sha(ROOT/p) for p in files})
    for p in files:
        target=OUT/'source'/p; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/p,target)
    (OUT/'logs/manifest.json').write_text(json.dumps(manifest,indent=2)); logging.info('COMPLETE: %s',OUT)

if __name__=='__main__':
    try: main()
    except Exception:
        logging.exception('Study stopped. Correct external/input error and resume same command.'); raise
