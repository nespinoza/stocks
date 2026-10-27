#!/usr/bin/env python3
"""Reproducible conditional portfolio experiment. Run from repository root."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr
from stock_api.data import calendar
from stock_api.portfolio import joint_gp, optimize_weights, trade, shrunk_covariance, crps, check_chronology


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_tickers(path):
    section = Path(path).read_text().split('## The 22 names to investigate first',1)[1].split('\n## ',1)[0]
    tickers = re.findall(r'^\| ([A-Z][A-Z0-9.\-]*) \|', section, re.M)
    if len(tickers) != 22 or len(set(tickers)) != 22:
        raise ValueError(f'Expected exactly 22 unique report tickers: {tickers}')
    return tickers


def load_data(c, out, tickers):
    sessions = calendar().sessions_in_range(c['price_anchor'],c['validation_end']).tz_localize(None)
    closes, opens, audits = {}, {}, []
    (out/'data').mkdir(exist_ok=True)
    for ticker in tickers+['SPY']:
        source = ROOT/c['source_data']/f'{ticker}.csv'
        meta = json.loads(source.with_suffix('.json').read_text())
        if digest(source) != meta['sha256']:
            raise ValueError(f'Checksum mismatch: {ticker}')
        frame = pd.read_csv(source,index_col='Date',parse_dates=True,float_precision='round_trip')
        if frame.index.has_duplicates:
            raise ValueError(f'Duplicate dates: {ticker}')
        frame = frame.reindex(sessions)
        if frame[['Open','Close','Adj Close']].isna().any().any() or not np.isfinite(frame[['Open','Close','Adj Close']]).all().all() or (frame[['Open','Close','Adj Close']]<=0).any().any():
            raise ValueError(f'Missing/invalid prices: {ticker}')
        closes[ticker] = frame['Adj Close']
        opens[ticker] = frame['Open']*frame['Adj Close']/frame['Close']
        audits.append(dict(ticker=ticker,source=meta['source'],retrieved_utc=meta['retrieved_utc'],sha256=meta['sha256'],
            rows=len(frame),missing=0,dividend_events=int((frame['Dividends']!=0).sum()),split_events=int((frame['Stock Splits']!=0).sum()),
            maximum_abs_log_return=float(np.log(frame['Adj Close']).diff().abs().max()),last_unadjusted_close=float(frame['Close'].iloc[-1])))
        frame.to_csv(out/'data'/f'{ticker}.csv',index_label='Date')
        (out/'data'/f'{ticker}.provenance.json').write_text(json.dumps(meta,indent=2))
        if ticker != 'SPY' and frame['Close'].iloc[-1] <= 10:
            raise ValueError(f'Latest stock price fails the $10 screen: {ticker}')
    close, opening = pd.DataFrame(closes), pd.DataFrame(opens)
    close.to_csv(out/'data'/'adjusted_close.csv',index_label='Date')
    opening.to_csv(out/'data'/'adjusted_open.csv',index_label='Date')
    pd.DataFrame(audits).to_csv(out/'data_integrity.csv',index=False)
    return close, opening


def schedule(c, index):
    dates=[]
    for month in pd.period_range(c['validation_start'][:7],c['validation_end'][:7],freq='M'):
        candidate=pd.Timestamp(f'{month}-{c["rebalance_day_of_month"]:02d}')
        valid=index[(index>=candidate)&(index<=pd.Timestamp(c['validation_end']))]
        if len(valid): dates.append(valid[0])
    result=[]
    for j, execution in enumerate(dates):
        origin=index[index.get_loc(execution)-1]
        following=dates[j+1] if j+1<len(dates) else None
        target_end=index[index.get_loc(following)-1] if following is not None else index[-1]
        targets=index[(index>origin)&(index<=target_end)]
        result.append(dict(period=j,origin=origin,execution=execution,following=following,target_end=target_end,horizon=len(targets)))
    if len(result)!=6: raise ValueError('Expected six validation periods')
    return result


def forecasts(c,out,tickers,close,opening,periods):
    returns=np.log(close[tickers]).diff().dropna()
    forecast_rows=[]; parameter_rows=[]; latent_rows=[]; cache={}
    (out/'predictive').mkdir(exist_ok=True)
    for lookback in c['lookback_grid']:
        for mean_gp in (True,False):
            variant='estimated_drift' if mean_gp else 'native_zero_drift'
            for s in periods:
                j=s['period']; origin=s['origin']; h=s['horizon']
                train=returns.loc[:origin].tail(lookback)
                check_chronology(train.index,origin,s['execution'],returns.loc[s['execution']:s['target_end']].index)
                assert len(train)==lookback
                key=f'{variant}_lb{lookback}_p{j}'
                path=out/'predictive'/f'{key}.npz'; param_path=path.with_suffix('.json')
                fingerprint=hashlib.sha256(train.to_numpy().tobytes()+json.dumps(c,sort_keys=True).encode()+digest(ROOT/'stock_api/portfolio.py').encode()+digest(ROOT/'stock_api/heteroskedastic.py').encode()).hexdigest()
                if path.exists() and param_path.exists() and json.loads(param_path.read_text()).get('fingerprint')==fingerprint:
                    model=dict(np.load(path)); model['parameters']=json.loads(param_path.read_text())['parameters']
                    logging.info('Reuse %s',key)
                else:
                    logging.info('Fit %s: 22 tickers, %s through %s, horizon %s',key,train.index[0].date(),origin.date(),h)
                    with warnings.catch_warnings(record=True) as recorded:
                        warnings.simplefilter('always')
                        model=joint_gp(train.to_numpy(),h,mean_gp=mean_gp,n_paths=c['predictive_paths'],
                                       seed=c['seed']+lookback*100+j*2+int(mean_gp),shrinkage=c['correlation_identity_shrinkage'])
                    for warning in recorded: logging.warning('%s: %s',key,warning.message)
                    np.savez_compressed(path,**{k:v for k,v in model.items() if k!='parameters'})
                    param_path.write_text(json.dumps({'fingerprint':fingerprint,'parameters':model['parameters']},indent=2))
                cache[lookback,variant,j]=model
                actual=np.log(close.loc[s['target_end'],tickers]/close.loc[origin,tickers]).to_numpy()
                end_prices=opening.loc[s['following'],tickers] if s['following'] is not None else close.loc[s['target_end'],tickers]
                held=np.log(end_prices/opening.loc[s['execution'],tickers]).to_numpy()
                samples=model['samples']; qs=np.quantile(samples,[.05,.16,.5,.84,.95],axis=0)
                sigma=np.sqrt(np.diag(model['covariance'])); scores=crps(samples,actual)
                for i,ticker in enumerate(tickers):
                    row=dict(ticker=ticker,lookback=lookback,variant=variant,period=j,forecast_origin=str(origin.date()),
                        execution_date=str(s['execution'].date()),target_end=str(s['target_end'].date()),horizon=h,
                        training_start=str(train.index[0].date()),training_end=str(train.index[-1].date()),
                        predicted_mean=model['mu'][i],predicted_std=sigma[i],realized_log_return=actual[i],holding_log_return=held[i],
                        realized_simple_return=np.expm1(actual[i]),standardized_residual=(actual[i]-model['mu'][i])/sigma[i],
                        latent_drift_variance=model['drift_variance'][i],innovation_variance=sigma[i]**2-model['drift_variance'][i],
                        realized_sum_squared_daily_returns=float((returns.loc[s['execution']:s['target_end'],ticker]**2).sum()),
                        mc_variance_correction=model['mc_variance_correction'][i],crps=scores[i],probability_up=(samples[:,i]>0).mean())
                    row.update({f'q{int(q*100):02d}':qs[k,i] for k,q in enumerate([.05,.16,.5,.84,.95])})
                    forecast_rows.append(row)
                    parameter_rows.append(dict(ticker=ticker,lookback=lookback,variant=variant,period=j,forecast_origin=str(origin.date()),**model['parameters'][i]))
                    for day,vals in enumerate(model['latent'][i]):
                        latent_rows.append(dict(ticker=ticker,lookback=lookback,variant=variant,period=j,step=day+1,
                            daily_drift=vals[0],latent_drift_variance=vals[1],log_variance_mean=vals[2],log_variance_uncertainty=vals[3],expected_innovation_variance=vals[4]))
    f=pd.DataFrame(forecast_rows); f.to_csv(out/'monthly_forecasts.csv',index=False)
    pd.DataFrame(parameter_rows).to_csv(out/'gp_parameter_history.csv',index=False)
    pd.DataFrame(latent_rows).to_csv(out/'gp_forward_latents.csv',index=False)
    returns.loc[c['training_start']:c['training_end']].describe().to_csv(out/'training_return_summary.csv')
    return cache, f, returns


def simulate(c,tickers,close,opening,periods,cache,returns,name,kind='gp',lookback=30,lam=10.,bps=10):
    symbols=['SPY'] if kind=='spy' else tickers
    qty=np.zeros(len(symbols)); gross_qty=qty.copy(); cash=c['initial_dollars']; gross_cash=cash
    daily=[dict(strategy=name,date=str(periods[0]['origin'].date()),net_equity=cash,gross_equity=cash)]
    monthly=[]; weights=[]; risk=[]; perturb=[]
    for s in periods:
        j=s['period']; op=opening.loc[s['execution'],symbols].to_numpy()
        old=qty*op; wealth=old.sum()+cash; gross_wealth=(gross_qty*op).sum()+gross_cash
        pre=old/wealth
        model=cache[lookback,'native_zero_drift' if kind=='native_zero' else 'estimated_drift',j]
        mu=model['mu'].copy(); covariance=model['covariance'].copy()
        if kind in ('zero','native_zero','minvar'): mu[:]=0
        if kind=='minvar':
            hist=returns.loc[:s['origin']].tail(c['historical_covariance_returns'])
            covariance=shrunk_covariance(hist.to_numpy(),c['correlation_identity_shrinkage'])[0]*s['horizon']
        info={'optimizer_success':True,'optimizer_message':'fixed allocation','optimizer_iterations':0,'optimizer_objective':np.nan}
        if kind=='spy': w=np.ones(1)
        elif kind=='equal': w=np.full(len(symbols),1/len(symbols))
        else: w,info=optimize_weights(mu,covariance,pre,0. if kind=='maxmean' else lam,bps/10000,c['weight_cap'])
        if kind=='spy' and j>0:
            dollars=old; fee=buys=sells=0.; gross_dollars=gross_qty*op
        else:
            dollars,fee,buys,sells=trade(wealth,old,w,bps/10000)
            gross_dollars=gross_wealth*w
        qty=dollars/op; gross_qty=gross_dollars/op; cash=gross_cash=0.
        end_prices=opening.loc[s['following'],symbols].to_numpy() if s['following'] is not None else close.loc[s['target_end'],symbols].to_numpy()
        terminal=qty@end_prices; gross_terminal=gross_qty@end_prices
        holding_log=np.log(end_prices/op)
        row=dict(strategy=name,period=j,execution_date=str(s['execution'].date()),
            holding_end=str((s['following'] if s['following'] is not None else s['target_end']).date()),
            start_equity=wealth,end_equity=terminal,net_return=terminal/wealth-1,gross_return=gross_terminal/gross_wealth-1,
            cost=fee,bought_dollars=buys,sold_dollars=sells,two_sided_turnover=(buys+sells)/wealth,one_way_turnover=.5*(buys+sells)/wealth,
            max_weight=w.max(),cap_count=int((w>=c['weight_cap']-1e-6).sum()) if kind!='spy' else 0,
            effective_holdings=1/(w@w),lookback=lookback,risk_aversion=lam,cost_bps=bps,**info)
        monthly.append(row)
        for i,ticker in enumerate(symbols):
            weights.append(dict(strategy=name,period=j,execution_date=str(s['execution'].date()),ticker=ticker,weight=w[i],pretrade_weight=pre[i],return_contribution=w[i]*np.expm1(holding_log[i])))
        dates=close.index[(close.index>=s['execution'])&(close.index<=s['target_end'])]
        for d in dates:
            p=close.loc[d,symbols].to_numpy()
            daily.append(dict(strategy=name,date=str(d.date()),net_equity=float(qty@p),gross_equity=float(gross_qty@p)))
        if kind!='spy':
            sample=model['samples'].copy()
            if kind=='zero': sample-=model['mu']
            # For equal/historical minvar, GP risk is diagnostic, not their allocation model.
            active=w>0
            port_samples=logsumexp(sample[:,active]+np.log(w[active]),axis=1)
            realized=logsumexp(holding_log[active]+np.log(w[active]))
            q=np.quantile(port_samples,[.05,.16,.5,.84,.95])
            risk.append(dict(strategy=name,period=j,predicted_log_mean=port_samples.mean(),predicted_log_std=port_samples.std(ddof=1),
                realized_gross_log_return=realized,z=(realized-port_samples.mean())/port_samples.std(ddof=1),q05=q[0],q16=q[1],q50=q[2],q84=q[3],q95=q[4],crps=float(crps(port_samples,realized))))
        if name=='Heteroskedastic GP optimizer':
            rng=np.random.default_rng(c['seed']+j)
            for k in range(c['drift_perturbation_replicates']):
                perturbed=mu+rng.normal(0,c['drift_perturbation_sd_log_return'],len(mu))
                wp,_=optimize_weights(perturbed,covariance,pre,lam,bps/10000,c['weight_cap'])
                perturb.append(dict(period=j,replicate=k,l1_weight_change=np.abs(wp-w).sum(),max_weight_change=np.abs(wp-w).max()))
    return daily,monthly,weights,risk,perturb


def metrics(daily,monthly,initial):
    rows=[]
    for name,d in daily.groupby('strategy',sort=False):
        m=monthly[monthly.strategy==name]
        row=dict(strategy=name,costs=m.cost.sum(),mean_monthly_turnover=m.two_sided_turnover.mean(),best_month=m.net_return.max(),worst_month=m.net_return.min(),positive_month_fraction=(m.net_return>0).mean(),mean_cap_count=m.cap_count.mean(),mean_effective_holdings=m.effective_holdings.mean())
        for prefix in ('net','gross'):
            equity=d[f'{prefix}_equity'].to_numpy(); r=equity[1:]/equity[:-1]-1
            row.update({f'{prefix}_return':equity[-1]/initial-1,f'{prefix}_ending_dollars':equity[-1],
                f'{prefix}_annualized_return':(equity[-1]/initial)**(252/len(r))-1,
                f'{prefix}_annualized_volatility':r.std(ddof=1)*np.sqrt(252),
                f'{prefix}_sharpe':r.mean()/r.std(ddof=1)*np.sqrt(252),
                f'{prefix}_max_drawdown':(equity/np.maximum.accumulate(equity)-1).min()})
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics(f,cache,tickers):
    rows=[]
    for (lookback,variant),group in f.groupby(['lookback','variant']):
        for control in ([False,True] if variant=='estimated_drift' else [False]):
            g=group.copy(); name='same_covariance_zero_drift' if control else variant
            if control:
                for q in ['q05','q16','q50','q84','q95']: g[q]-=g.predicted_mean
                g.predicted_mean=0.
                for period,h in g.groupby('period'):
                    model=cache[lookback,variant,period]
                    g.loc[h.index,'crps']=crps(model['samples']-model['mu'],h.realized_log_return.to_numpy())
            for ticker,h in [('ALL',g)]+list(g.groupby('ticker',sort=False)):
                error=h.realized_log_return-h.predicted_mean; std=h.predicted_std
                ranks=[]
                if ticker=='ALL' and not control and variant=='estimated_drift':
                    ranks=[spearmanr(z.predicted_mean,z.realized_log_return).statistic for _,z in h.groupby('period')]
                corr=np.corrcoef(h.predicted_mean,h.realized_log_return)[0,1] if h.predicted_mean.std()>1e-12 else np.nan
                predvar=h.innovation_variance; realizedvar=h.realized_sum_squared_daily_returns
                rows.append(dict(lookback=lookback,variant=name,ticker=ticker,n=len(h),mae=np.abs(error).mean(),rmse=np.sqrt(np.mean(error**2)),
                    coverage90=((h.realized_log_return>=h.q05)&(h.realized_log_return<=h.q95)).mean(),
                    coverage68=((h.realized_log_return>=h.q16)&(h.realized_log_return<=h.q84)).mean(),
                    mean_z=(error/std).mean(),rms_z=np.sqrt(np.mean((error/std)**2)),correlation=corr,mean_cross_section_spearman=np.mean(ranks) if ranks else np.nan,
                    gaussian_log_density=np.mean(-.5*np.log(2*np.pi*std**2)-.5*(error/std)**2),crps=h.crps.mean(),
                    realized_to_predicted_variance=realizedvar.sum()/predvar.sum(),volatility_qlike=(np.log(predvar)+realizedvar/predvar).mean(),
                    mean_absolute_predicted_drift=h.predicted_mean.abs().mean(),mean_drift_variance_share=(h.latent_drift_variance/h.predicted_std**2).mean()))
    result=pd.DataFrame(rows)
    ranking=result[(result.lookback==30)&(result.variant=='estimated_drift')&(result.ticker!='ALL')].sort_values('mae').copy()
    ranking.insert(0,'rank',np.arange(1,len(ranking)+1))
    return result,ranking


def figures(out,primary,daily,monthly,weights,forecast,comparison,sensitivity):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out/'figures').mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,5))
    for name in primary:
        d=daily[daily.strategy==name]
        ax.plot(pd.to_datetime(d.date),d.net_equity,label=name)
    ax.set(ylabel='Net value of initial $2,000',title='Conditional retrospective validation: daily adjusted-close wealth')
    ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out/'figures/equity_curves.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,8))
    w=weights[weights.strategy==primary[0]].pivot(index='ticker',columns='execution_date',values='weight')
    im=ax.imshow(w.values,vmin=0,vmax=.2,aspect='auto',cmap='viridis'); ax.set_yticks(range(len(w)),w.index); ax.set_xticks(range(len(w.columns)),w.columns,rotation=45,ha='right')
    fig.colorbar(im,ax=ax,label='Portfolio weight'); ax.set_title('Primary GP monthly allocations'); fig.tight_layout(); fig.savefig(out/'figures/monthly_weights.png',dpi=160); plt.close(fig)
    f=forecast[(forecast.lookback==30)&(forecast.variant=='estimated_drift')]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].scatter(f.predicted_mean,f.realized_log_return,alpha=.6,s=15); axes[0].set(xlabel='Predicted monthly log return',ylabel='Realized monthly log return')
    from scipy.stats import norm
    zs=np.sort(f.standardized_residual); normal=norm.ppf((np.arange(len(zs))+.5)/len(zs)); axes[1].scatter(normal,zs,s=12); axes[1].plot([-3,3],[-3,3],'k--'); axes[1].set(xlabel='Standard normal quantile',ylabel='Standardized prediction residual')
    axes[2].scatter(f.innovation_variance,f.realized_sum_squared_daily_returns,s=15); limit=max(f.innovation_variance.max(),f.realized_sum_squared_daily_returns.max()); axes[2].plot([0,limit],[0,limit],'k--'); axes[2].set(xlabel='Forecast summed innovation variance',ylabel='Realized sum of daily squared returns')
    fig.tight_layout(); fig.savefig(out/'figures/forecast_calibration.png',dpi=160); plt.close(fig)
    p=comparison.set_index('strategy').loc[primary]
    fig,axes=plt.subplots(1,2,figsize=(12,5)); p[['gross_return','net_return']].mul(100).plot.barh(ax=axes[0]); axes[0].set_xlabel('Six-month return (%)'); p[['net_annualized_volatility','net_max_drawdown']].mul(100).plot.barh(ax=axes[1]); axes[1].set_xlabel('Volatility (annualized) / drawdown (%)'); fig.tight_layout(); fig.savefig(out/'figures/benchmark_comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,lb in zip(axes,[30,60]):
        z=sensitivity[sensitivity.lookback==lb].pivot(index='risk_aversion',columns='cost_bps',values='net_return')*100
        im=ax.imshow(z,aspect='auto',cmap='RdYlGn'); ax.set_xticks(range(len(z.columns)),z.columns); ax.set_yticks(range(len(z)),z.index); ax.set(xlabel='Trading cost (bp)',ylabel='Risk aversion',title=f'{lb} trading-day lookback')
        for i in range(len(z)):
            for j in range(len(z.columns)): ax.text(j,i,f'{z.iloc[i,j]:.2f}%',ha='center',va='center')
    fig.tight_layout(); fig.savefig(out/'figures/sensitivity.png',dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='experiments/portfolio_gp_22_2026-10-18'); args=parser.parse_args()
    out=ROOT/args.output; out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    c=json.loads((out/'config.yaml').read_text()); tickers=extract_tickers(ROOT/c['source_report']); (out/'tickers.txt').write_text('\n'.join(tickers)+'\n')
    manifest=dict(command='MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_portfolio_gp_22.py',python=sys.version,
        packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib','exchange-calendars','yfinance']},
        hashes={p:digest(ROOT/p) for p in [c['source_report'],'stock_api/heteroskedastic.py','stock_api/sde.py','stock_api/data.py','stock_api/portfolio.py','scripts/run_portfolio_gp_22.py','scripts/portfolio_gp_report.py','scripts/audit_portfolio_gp_22.py','tests/test_portfolio.py','environment.yml','pyproject.toml']},config_sha256=digest(out/'config.yaml'))
    for source in manifest['hashes']:
        target=out/'source'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/source,target)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    close,opening=load_data(c,out,tickers); periods=schedule(c,close.index)
    pd.DataFrame(periods).to_csv(out/'schedule.csv',index=False)
    logging.info('Verified %s unique stocks plus SPY, %s price sessions; six validation periods',len(tickers),len(close))
    cache,forecast,returns=forecasts(c,out,tickers,close,opening,periods)
    specs=[('Heteroskedastic GP optimizer','gp'),('Equal weight','equal'),('Minimum variance','minvar'),('Zero-drift GP','zero'),('SPY','spy'),('Native zero-drift GP','native_zero'),('GP maximum mean','maxmean')]
    results=[[],[],[],[],[]]; settings=[]
    for name,kind in specs:
        logging.info('Account strategy %s',name)
        run=simulate(c,tickers,close,opening,periods,cache,returns,name,kind,
                     c['primary_lookback_returns'],c['primary_lambda'],c['primary_cost_bps'])
        for all_rows,rows in zip(results,run): all_rows.extend(rows)
    for lb in c['lookback_grid']:
        for lam in c['lambda_grid']:
            for bps in c['cost_grid_bps']:
                name='Heteroskedastic GP optimizer' if (lb,lam,bps)==(30,10,10) else f'GP sensitivity L{lb} lambda{lam:g} cost{bps}'
                settings.append(dict(strategy=name,lookback=lb,risk_aversion=lam,cost_bps=bps))
                if name==specs[0][0]: continue
                run=simulate(c,tickers,close,opening,periods,cache,returns,name,'gp',lb,lam,bps)
                for all_rows,rows in zip(results,run): all_rows.extend(rows)
    daily,monthly,weights,risk,perturb=map(pd.DataFrame,results)
    comparison=metrics(daily,monthly,c['initial_dollars']); sensitivity=pd.DataFrame(settings).merge(comparison,on='strategy')
    diag,ranking=diagnostics(forecast,cache,tickers)
    for frame,file in [(daily,'portfolio_daily'),(monthly,'monthly_performance'),(weights,'monthly_weights'),(risk,'portfolio_risk_diagnostics'),(perturb,'drift_perturbation'),(comparison,'benchmark_comparison'),(sensitivity,'sensitivity'),(diag,'forecast_diagnostics'),(ranking,'ticker_error_ranking')]: frame.to_csv(out/f'{file}.csv',index=False)
    primary=[s[0] for s in specs[:5]]
    figures(out,primary,daily,monthly,weights,forecast,comparison,sensitivity)
    from portfolio_gp_report import write_report
    write_report(out,c,primary,comparison,monthly,weights,risk,perturb,diag,ranking,sensitivity,forecast)
    from audit_portfolio_gp_22 import audit
    audit(out)
    logging.info('COMPLETE: %s',out)

if __name__=='__main__':
    try:
        main()
    except Exception:
        logging.exception('Experiment stopped; rerun the same command to resume verified checkpoints')
        raise
