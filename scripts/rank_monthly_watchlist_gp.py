"""Monthly heteroskedastic GP evaluation for the frozen 22-stock watchlist.

Uses the repository's standard variational GP, calendar-month targets, and only
the preceding 30 calendar days for training. No downloads or mid-month updates.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stock_api.data import calendar
from stock_api.heteroskedastic import heteroskedastic_gp_returns
from stock_api.historical.runner import (atomic_json, atomic_bytes, digest,
                                         experiment_lock, load_record, save_record, sha)

SOURCE = Path('reports/monthly-stock-screen-2026-10-18')
OUTPUT = SOURCE/'gp-monthly-2025-08-to-2026-09'
SETTINGS = {
    'model':'heteroskedastic_gp_returns', 'inference':'collapsed_gaussian_variational',
    'first_month':'2025-08', 'last_month':'2026-09', 'train_calendar_days':30,
    'forecast_horizon':'all exchange sessions of each calendar month',
    'forecast_origin':'before the first trading session; cutoff is calendar month start',
    'n_paths':10000, 'seed':20261018, 'calendar':'XNYS',
    'point_forecast':'conditional predictive price median',
    'primary_ranking':'equal-month mean of 100 * path MAE / mean actual monthly price',
    'failure_policy':'no fallback; common successful months for the primary ranking',
    'parameter_uncertainty':'latent functions approximated; fitted hyperparameter uncertainty omitted',
}


def periods():
    return pd.period_range(SETTINGS['first_month'],SETTINGS['last_month'],freq='M')


def fold_dates(month):
    month=pd.Period(month,freq='M')
    origin=month.start_time
    # Explicit timedelta unit avoids NumPy's deprecated generic timedelta path.
    lower=origin-pd.Timedelta(30,unit='D')
    end=(month+1).start_time
    sessions=calendar().sessions_in_range(lower,end-pd.Timedelta(1,unit='D'))
    if sessions.tz is not None:sessions=sessions.tz_localize(None)
    return sessions[sessions<origin],sessions[sessions>=origin]


def prepare(root):
    watch_path=SOURCE/'positive-skill-watchlist.csv'
    watch=pd.read_csv(watch_path)
    if len(watch)!=22 or watch.ticker.nunique()!=22:raise ValueError('Expected exactly 22 watchlist tickers')
    inputs={};full={}
    for ticker in watch.ticker:
        path=SOURCE/'data'/f'{ticker}.csv'
        meta=json.loads((SOURCE/'data'/f'{ticker}.json').read_text())
        if sha(path)!=meta['sha256']:raise ValueError(f'Original data checksum mismatch: {ticker}')
        frame=pd.read_csv(path,index_col=0,parse_dates=True,float_precision='round_trip')
        if frame.index.has_duplicates:raise ValueError(f'Duplicate price dates: {ticker}')
        full[ticker]=frame['Adj Close']
        inputs[ticker]={'source':str(path),'sha256':sha(path),'provider_url':meta['source'],
                        'retrieved_utc':meta['retrieved_utc']}
    needed=pd.DatetimeIndex([])
    for month in periods():
        train,test=fold_dates(month);needed=needed.union(train).union(test)
    prices=pd.DataFrame(full).reindex(needed)
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all() or (prices<=0).any().any():
        raise ValueError('Missing/nonpositive required price; no filling allowed')
    payload=prices.to_csv().encode()
    implementation={'python':platform.python_version(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','exchange-calendars','threadpoolctl']},
        'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path('stock_api/heteroskedastic.py'),
                            Path('stock_api/sde.py'),Path('stock_api/historical/runner.py')]}}
    identity={'settings':SETTINGS,'tickers':watch.ticker.tolist(),'watchlist_sha256':sha(watch_path),
              'prices_sha256':hashlib.sha256(payload).hexdigest(),'inputs':inputs,'implementation':implementation}
    manifest_path=root/'experiment-config.json'
    experiment_id=digest(identity)
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text())
        if existing['experiment_id']!=experiment_id:raise ValueError('Inputs, settings or implementation changed; use a new output directory')
        if sha(root/'prices.csv')!=identity['prices_sha256']:raise ValueError('Experiment price snapshot changed')
    else:
        atomic_bytes(root/'prices.csv',payload)
        atomic_json(manifest_path,dict(identity,experiment_id=experiment_id,
                    created_utc=datetime.now(timezone.utc).isoformat(),planned_fits=len(watch)*len(periods())))
    return watch,prices,experiment_id


def score_paths(actual, predicted, initial):
    actual=np.asarray(actual);predicted=np.asarray(predicted)
    if actual.shape!=predicted.shape or not len(actual):raise ValueError('Forecast and target shapes differ')
    errors=predicted-actual
    baseline=np.abs(initial-actual)
    return {'forecast_sessions':len(actual),'mae_usd':float(np.abs(errors).mean()),
        'rmse_usd':float(np.sqrt(np.mean(errors**2))),
        'nmae_pct':float(100*np.abs(errors).mean()/actual.mean()),
        'mape_pct':float(100*np.mean(np.abs(errors)/actual)),
        'month_end_ape_pct':float(100*abs(errors[-1])/actual[-1]),
        'terminal_return_mae_log_pp':float(100*abs(np.log(predicted[-1]/actual[-1]))),
        'baseline_nmae_pct':float(100*baseline.mean()/actual.mean()),
        'baseline_month_end_ape_pct':float(100*baseline[-1]/actual[-1]),
        'actual_terminal_log_return':float(np.log(actual[-1]/initial)),
        'predicted_terminal_log_return':float(np.log(predicted[-1]/initial))}


def execute(ticker,month,series,experiment_id):
    started=time.perf_counter()
    train_dates,test_dates=fold_dates(month)
    record={'experiment_id':experiment_id,'ticker':ticker,'month':str(month),
        'origin':str(pd.Period(month,freq='M').start_time.date()),
        'train_dates':[str(d.date()) for d in train_dates],
        'forecast_dates':[str(d.date()) for d in test_dates],
        'status':'error','prediction':None,'scores':None}
    seed=int(digest([SETTINGS['seed'],ticker,str(month),SETTINGS['model']])[:8],16)
    record['seed']=seed
    try:
        train=series.reindex(train_dates).to_numpy(dtype=float)
        actual=series.reindex(test_dates).to_numpy(dtype=float)
        if not np.isfinite(train).all() or (train<=0).any() or len(train)<9:raise ValueError('Invalid training prices')
        record['training_sha256']=hashlib.sha256(train.tobytes()).hexdigest()
        record['training_observations']=len(train)
        record['starting_price']=float(train[-1])
        # The model receives only the copied training array and a forecast length.
        _,_,info=heteroskedastic_gp_returns(train[:,None].copy(),len(test_dates),
                                           seed=seed,n_paths=SETTINGS['n_paths'])
        stats=info.pop('_statistics');ensemble=info.pop('_ensemble')
        predicted=np.asarray(stats['median_price'])
        record['prediction']={'median':predicted.tolist(),
            'lower95':np.asarray(stats['lower_95']).tolist(),
            'upper95':np.asarray(stats['upper_95']).tolist(),
            'probability_up':np.asarray(stats['probability_up']).tolist(),
            'terminal_log_returns':ensemble.terminal_log_returns.tolist(),
            'price_mean':None,'price_std':None,'diagnostics':info}
        if not info['converged']:raise ValueError('GP inference did not converge')
        if not np.isfinite(actual).all() or (actual<=0).any():raise ValueError('Invalid held-out prices')
        if not np.isfinite(predicted).all() or (predicted<=0).any():raise ValueError('Invalid price predictions')
        scores=score_paths(actual,predicted,train[-1])
        scores['path_coverage_95']=float(np.mean((actual>=stats['lower_95'])&(actual<=stats['upper_95'])))
        scores['terminal_coverage_95']=float(stats['lower_95'][-1]<=actual[-1]<=stats['upper_95'][-1])
        scores['brier']=float((stats['probability_up'][-1]-float(actual[-1]>train[-1]))**2)
        scores['direction_correct']=float((predicted[-1]>train[-1])==(actual[-1]>train[-1]))
        record.update(status='ok',actual=actual.tolist(),scores=scores)
    except Exception as exc:
        record['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    record['runtime_seconds']=time.perf_counter()-started
    return record


def worker_init():
    from threadpoolctl import threadpool_limits
    global THREAD_LIMIT
    THREAD_LIMIT=threadpool_limits(limits=1)


def summarize(root,watch,records):
    by_stock={ticker:{r['month'] for r in records if r['ticker']==ticker and r['status']=='ok'} for ticker in watch.ticker}
    common=set.intersection(*by_stock.values())
    rows=[];daily=[]
    for r in records:
        row={k:r[k] for k in ['ticker','month','status','runtime_seconds']}
        row['training_observations']=r.get('training_observations')
        row.update(r.get('scores') or {})
        row['error_message']=r.get('error',{}).get('message')
        rows.append(row)
        if r['status']=='ok':
            for i,date in enumerate(r['forecast_dates']):
                daily.append({'ticker':r['ticker'],'month':r['month'],'date':date,
                    'actual':r['actual'][i],'predicted_median':r['prediction']['median'][i],
                    'lower95':r['prediction']['lower95'][i],'upper95':r['prediction']['upper95'][i],
                    'baseline':r['starting_price']})
    scores=pd.DataFrame(rows)
    atomic_bytes(root/'monthly-scores.csv',scores.to_csv(index=False).encode())
    atomic_bytes(root/'daily-predictions.csv',pd.DataFrame(daily).to_csv(index=False).encode())
    metrics=['nmae_pct','mape_pct','mae_usd','rmse_usd','month_end_ape_pct','terminal_return_mae_log_pp',
             'baseline_nmae_pct','baseline_month_end_ape_pct','brier','path_coverage_95','terminal_coverage_95','direction_correct']
    ranking=[]
    for ticker in watch.ticker:
        all_stock=scores[scores.ticker==ticker]
        good=all_stock[all_stock.status=='ok']
        shared=good[good.month.isin(common)]
        item={'ticker':ticker,'successful_months':len(good),'failed_months':len(all_stock)-len(good),
              'ranking_months':len(shared),'all_success_nmae_pct':good.nmae_pct.mean() if len(good) else np.nan,
              'min_training_observations':all_stock.training_observations.min(),
              'max_training_observations':all_stock.training_observations.max()}
        item.update({m:shared[m].mean() if len(shared) else np.nan for m in metrics})
        item['improvement_vs_last_price_pct']=100*(1-item['nmae_pct']/item['baseline_nmae_pct']) if len(shared) else np.nan
        ranking.append(item)
    ranking=pd.DataFrame(ranking).merge(watch[['ticker','name','sector']],on='ticker',validate='one_to_one')
    ranking=ranking.sort_values(['nmae_pct','ticker'],na_position='last')
    ranking.insert(0,'rank',np.arange(1,len(ranking)+1) if common else [None]*len(ranking))
    atomic_bytes(root/'ranking.csv',ranking.to_csv(index=False).encode())
    endpoint=ranking.sort_values(['month_end_ape_pct','ticker']).drop(columns='rank')
    endpoint.insert(0,'rank',np.arange(1,len(endpoint)+1) if common else [None]*len(endpoint))
    atomic_bytes(root/'month-end-ranking.csv',endpoint.to_csv(index=False).encode())
    display=ranking[['rank','ticker','nmae_pct','month_end_ape_pct','baseline_nmae_pct','improvement_vs_last_price_pct','successful_months','failed_months']].copy()
    for c in display.columns[2:6]:display[c]=display[c].map(lambda v:f'{v:.3f}')
    display.columns=['Rank','Ticker','GP path NMAE %','Month-end APE %','Last-price NMAE %','Improvement vs baseline %','Successful','Failed']
    table='\n'.join(['| '+' | '.join(display.columns)+' |', '| '+' | '.join(['---']*len(display.columns))+' |',
        *['| '+' | '.join(map(str,r))+' |' for r in display.itertuples(index=False,name=None)]])
    ok=[r for r in records if r['status']=='ok']
    metadata={'completed_fits':len(records),'successful_fits':len(ok),'failed_fits':len(records)-len(ok),
              'common_ranking_months':sorted(common),'total_forecast_sessions':len(daily),
              'stocks_beating_baseline':int((ranking.improvement_vs_last_price_pct>0).sum()),
              'mean_gp_nmae_pct':ranking.nmae_pct.mean(),'mean_baseline_nmae_pct':ranking.baseline_nmae_pct.mean(),
              'median_fit_seconds':scores.runtime_seconds.median()}
    atomic_json(root/'summary.json',metadata)
    report=f'''# Monthly heteroskedastic GP: 22-stock error ranking

The requested interval is interpreted as **August 2025 through September 2026, inclusive: 14 calendar months**. Each forecast is fitted once using the **30 calendar days before that month's first calendar day**, then predicts every exchange session in the entire target month. No target-month prices enter training and there is no mid-month refit.

The model is the repository's `heteroskedastic_gp_returns`, using its **standard collapsed Gaussian variational inference** and 10,000 predictive paths. This is the public API's default fit, **not the Bayesian MCMC used in the earlier archived Amazon study**. Hyperparameters are fitted by the existing regularized ELBO optimizer; latent uncertainty is approximated, while hyperparameter uncertainty is omitted. Model settings were fixed before fitting and were not tuned against these outcomes.

## Ranking: lowest to highest predictive error

The primary error is normalized mean absolute error: for each month, `100 × mean(|predicted median − actual price|) / mean(actual price)`, then averaged equally over months. This makes errors comparable across stocks with different share prices. The ranking reflects accuracy over all daily predictions within each month. **Month-end APE** instead scores only the final session's absolute percentage error; its separate ordering is in [month-end-ranking.csv](month-end-ranking.csv).

{table}

There were **{len(records)} attempted fits: {len(ok)} successful and {len(records)-len(ok)} failed**. All tickers are ranked on the same **{len(common)} common successful months**, avoiding an advantage from omitting difficult months for a particular stock. Common months: {', '.join(sorted(common)) or 'none; a comparable ranking is unavailable'}. [Monthly scores](monthly-scores.csv) include all attempts and failure reasons; [ranking.csv](ranking.csv) additionally gives results over each ticker's own successful months.

Positive improvement means lower error than holding the prior month's final close constant throughout the target month. **{metadata['stocks_beating_baseline']} of 22 stocks** improve on that baseline by the primary metric. Equal-stock average GP NMAE is **{metadata['mean_gp_nmae_pct']:.3f}%**, versus **{metadata['mean_baseline_nmae_pct']:.3f}%** for last price. Low raw forecast error can simply reflect calmer price movements; improvement against the same stock's baseline is the more relevant indication of incremental model value. The ordering is descriptive, with only 14 attempted months per stock, and is not a significance claim or a portfolio-return ranking.

## Dates, data and reproducibility

- Tickers are exactly the 22 in the original [positive-skill watchlist](../positive-skill-watchlist.csv), checked before running. No stock is removed or substituted after looking at its GP error.
- Prices reuse the saved Yahoo **adjusted-close** histories downloaded on October 18, 2026 for the [original report](../REPORT.md). There was no new download. Per-ticker original-file checksums, provider URLs and retrieval timestamps are retained in [experiment-config.json](experiment-config.json). The exact common price snapshot is [prices.csv](prices.csv).
- A 30-calendar-day history includes only its trading sessions (see the per-ticker training counts in the CSV), and returns are formed within that price window. The date cutoff is the first calendar day, including months that begin on a weekend or holiday. Training and forecast dates are saved for every fit.
- All months use actual XNYS sessions; forecasts end at the target month's last session and never spill into the next month. The predictive horizon is usually longer than the number of training returns; this is the requested design, not a recommendation that one month of training is sufficient.
- Seeds are deterministic by ticker and month. Source hashes, dependency versions, the watchlist hash and input hashes freeze the experiment. Each atomic checkpoint has a content checksum. Completed fits are reused on resume; failed fits are preserved without fallback or post-hoc retuning.
- [Daily predictions and 95% bands](daily-predictions.csv), [monthly scores](monthly-scores.csv), and per-fit distributions/diagnostics in `records/` make the ranking auditable. Population price means/standard deviations are omitted because the latent volatility mixture may have nonfinite price moments; point scoring uses medians.

The 22 names were selected using prior analysis that overlaps this interval. This is therefore a **retrospective evaluation of a preselected watchlist**, not an independent validation of its selection procedure. Current membership and retrospective price adjustments also retain the original screen's survivorship/data-vintage limitations.

Reproduce/resume from the repository root:

```bash
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 \\
  .conda-env/bin/python scripts/rank_monthly_watchlist_gp.py --workers 4
```

Use a new `--output` directory if model code, settings or inputs change. The original screening report and its rankings are preserved.
'''
    atomic_bytes(root/'REPORT.md',report.encode())
    if common:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,8))
        pos=np.arange(len(ranking))
        ax.barh(pos,ranking.nmae_pct,label='Heteroskedastic GP',color='#2677ad')
        ax.scatter(ranking.baseline_nmae_pct,pos,color='#c66c22',marker='D',s=25,label='Last-price baseline',zorder=3)
        ax.set_yticks(pos,ranking.ticker);ax.invert_yaxis()
        ax.set_xlabel('Mean monthly normalized absolute price error (%) — lower is better')
        ax.set_title('22-stock monthly GP forecasts: Aug 2025–Sep 2026')
        ax.legend();fig.tight_layout();fig.savefig(root/'ranking.png',dpi=160);plt.close(fig)
    print(display.to_string(index=False),flush=True)
    print(json.dumps(metadata,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-units',type=int)
    args=parser.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    root=args.output
    with experiment_lock(root):
        watch,prices,experiment_id=prepare(root)
        records=[];pending=[]
        for ticker in watch.ticker:
            for month in periods():
                path=root/'records'/f'{ticker}-{month}.json'
                if path.exists():records.append(load_record(path,experiment_id))
                else:pending.append((ticker,str(month)))
        if args.max_units is not None:pending=pending[:args.max_units]
        print(f'{len(records)}/308 saved; {len(pending)} fits queued',flush=True)
        import multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init) as pool:
            futures=[pool.submit(execute,ticker,month,prices[ticker],experiment_id) for ticker,month in pending]
            for future in as_completed(futures):
                record=future.result()
                save_record(root/'records'/f'{record["ticker"]}-{record["month"]}.json',record)
                records.append(record)
                atomic_json(root/'progress.json',{'completed':len(records),'total':308,
                    'failures':sum(r['status']!='ok' for r in records),
                    'updated_utc':datetime.now(timezone.utc).isoformat()})
                if len(records)%22==0 or record['status']!='ok':
                    print(f'{len(records)}/308 completed; failures {sum(r["status"]!="ok" for r in records)}',flush=True)
        if len(records)==308:summarize(root,watch,records)
        else:print('Partial checkpoint run; ranking requires all 308 attempts',flush=True)


if __name__=='__main__':main()
