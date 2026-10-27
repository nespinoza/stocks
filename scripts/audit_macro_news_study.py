#!/usr/bin/env python3
"""Reconstruct all daily holdings independently and verify information cutoffs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/portfolio_gp_macro_news_2026-10-18'


def audit(out=OUT):
    c=json.loads((out/'config.yaml').read_text()); tickers=(out/'tickers.txt').read_text().splitlines(); assert len(tickers)==len(set(tickers))==c.get('ticker_count',22)
    monthly=pd.read_csv(out/'portfolios/monthly.csv'); weights=pd.read_csv(out/'portfolios/monthly_weights.csv'); daily=pd.read_csv(out/'portfolios/daily.csv')
    records={r['identifier']:r for r in (json.loads(x) for x in (out/'data/news/records.jsonl').read_text().splitlines())}
    inputs=pd.read_csv(out/'forecasts/news.csv')
    for row in inputs.itertuples():
        cutoff=pd.Timestamp(row.origin).tz_localize('America/New_York')+pd.Timedelta(16,unit='h')
        for id in json.loads(row.identifiers): assert pd.Timestamp(records[id]['availability_timestamp'])<=cutoff
    macro=pd.read_csv(out/'data/macro/observation_provenance.csv')
    assert (pd.to_datetime(macro.vintage)<pd.to_datetime(macro.origin)).all()
    assert (pd.to_datetime(macro.observation_date)<=pd.to_datetime(macro.vintage)).all()
    max_error=0.; min_eigenvalue=float('inf'); paths=0
    for window,group in monthly.groupby('window'):
        label='B' if window.startswith('B') else 'A'
        window_tickers=c.get('window_tickers',{}).get(label,tickers)
        raw={t:pd.read_csv(out/f'data/prices/{label}/{t}.csv',index_col='Date') for t in window_tickers+['SPY']}
        close=pd.DataFrame({t:f['Adj Close'] for t,f in raw.items()}); opening=pd.DataFrame({t:f.Open*f['Adj Close']/f.Close for t,f in raw.items()})
        for name,months in group.groupby('strategy'):
            symbols=['SPY'] if name=='SPY' else window_tickers; qty=np.zeros(len(symbols)); cash=2000.; gross_qty=qty.copy(); gross_cash=2000.
            curve=daily[(daily.window==window)&(daily.strategy==name)].set_index('date')
            for _,m in months.iterrows():
                w=weights[(weights.window==window)&(weights.strategy==name)&(weights.period==m.period)].set_index('ticker').loc[symbols]
                op=opening.loc[m.execution_date,symbols].to_numpy(); old=qty*op; wealth=old.sum()+cash
                np.testing.assert_allclose(wealth,m.start_equity,atol=1e-8)
                np.testing.assert_allclose(w.pretrade_weight,old/wealth,atol=1e-8)
                new=(wealth-m.cost)*w.weight.to_numpy(); buy=np.maximum(new-old,0).sum(); sell=np.maximum(old-new,0).sum()
                np.testing.assert_allclose([buy,sell,.001*(buy+sell)],[m.bought_dollars,m.sold_dollars,m.cost],atol=1e-8)
                qty=new/op; cash=0.; gross_qty=((gross_qty*op).sum()+gross_cash)*w.weight.to_numpy()/op; gross_cash=0.
                dates=curve.index[(curve.index>=m.execution_date)&((curve.index<m.holding_end) if m.period<months.period.max() else (curve.index<=m.holding_end))]
                for date in dates:
                    error=abs(qty@close.loc[date,symbols].to_numpy()-curve.loc[date,'net_equity']); max_error=max(max_error,error); assert error<1e-7
                    np.testing.assert_allclose(gross_qty@close.loc[date,symbols].to_numpy(),curve.loc[date,'gross_equity'],atol=1e-7)
            paths+=1
    for path in (out/'forecasts/gp_cache').glob('*.npz'):
        z=np.load(path); min_eigenvalue=min(min_eigenvalue,np.linalg.eigvalsh(z['covariance']).min()); assert min_eigenvalue>-1e-9
        np.testing.assert_allclose(z['samples'].mean(axis=0),z['mu'],atol=1e-12)
        info=json.loads(path.with_suffix('.json').read_text()); window=path.stem[0]
        if 'train_start' in info: assert pd.Timestamp(info['train_start'])>=pd.Timestamp(c['windows'][window]['start'])
    for path in (out/'exposures/mapping_cache').glob('*.npz'):
        z=np.load(path); origin=path.stem[2:12]
        assert all(date<=origin for date in z['training_end_dates'])
        np.testing.assert_allclose(z['prediction_features']@z['posterior_mean'],z['delta'],atol=1e-12)
        assert all(np.linalg.eigvalsh(v).min()>0 for v in z['posterior_covariances'])
    a=np.load(out/'exposures/window_A_frozen_posterior.npz'); assert int(a['descriptive_n'])==49
    old=ROOT/c['previous_experiment']; manifest=json.loads((old/'manifest.json').read_text())
    # Original outputs are never destinations in the new pipeline.
    report_hash=hashlib.sha256((old/'RESULTS.md').read_bytes()).hexdigest()
    result=dict(status='PASS',portfolio_paths=paths,max_reconstructed_daily_equity_error=float(max_error),minimum_GP_covariance_eigenvalue=float(min_eigenvalue),macro_vintages=len(macro),news_timestamp_rows=len(inputs),frozen_A_exposure_months=int(a['descriptive_n']),original_report_sha256=report_hash,
        checks=['vintage dates','news availability','historical training cutoffs','frozen full-regime prior','parameter covariance PSD','prediction reconstruction','drifted pretrade weights','fees and budget','daily net and gross wealth'])
    (out/'logs/independent_audit.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); return result

if __name__=='__main__': audit()
