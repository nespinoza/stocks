#!/usr/bin/env python3
"""Acquire the exact user universe; copy verified macro/news caches without mutations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from stock_api.data import calendar

OUT=ROOT/'experiments/portfolio_gp_macro_news_growth20_2026-10-19'
BASE=ROOT/'experiments/portfolio_gp_macro_news_2026-10-18'
TICKERS='MXL DELL AXTI BAND SAIH SILC PENG MU AEHR RNG QMCO MRVL AMD FATN INTC HPE SIMO OKTA STX FTNT'.split()

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert len(TICKERS)==len(set(TICKERS))==20
    for folder in ['data/prices/raw','logs','forecasts/gp_cache','exposures','portfolios','figures','source']:
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    (OUT/'tickers.txt').write_text('\n'.join(TICKERS)+'\n')
    import yfinance as yf
    yf.set_tz_cache_location(str(OUT/'data/prices/yahoo-cache'))
    frames={}; coverage=[]
    for ticker in TICKERS+['SPY']:
        target=OUT/'data/prices/raw'/f'{ticker}.csv'
        source=ROOT/'reports/monthly-stock-screen-2026-10-18/data'/f'{ticker}.csv'
        if not target.exists():
            if source.exists():
                meta=json.loads(source.with_suffix('.json').read_text())
                assert sha(source)==meta['sha256']
                shutil.copyfile(source,target); shutil.copyfile(source.with_suffix('.json'),target.with_suffix('.json'))
            else:
                print('Downloading',ticker,flush=True)
                f=yf.Ticker(ticker).history(start='2009-01-01',end='2026-10-18',interval='1d',auto_adjust=False,actions=True,raise_errors=True,timeout=25)
                if f.empty: raise ValueError(f'Empty history {ticker}')
                f.index=f.index.tz_localize(None); f.to_csv(target,index_label='Date')
                meta=dict(ticker=ticker,retrieved_utc=datetime.now(timezone.utc).isoformat(),source=f'https://finance.yahoo.com/quote/{ticker}/history/',request=dict(start='2009-01-01',end='2026-10-18',interval='1d',auto_adjust=False),sha256=sha(target))
                target.with_suffix('.json').write_text(json.dumps(meta,indent=2))
        assert sha(target)==json.loads(target.with_suffix('.json').read_text())['sha256']
        f=pd.read_csv(target,index_col='Date',parse_dates=True,float_precision='round_trip')
        assert f.index.is_unique and f.index.is_monotonic_increasing
        assert np.isfinite(f[['Open','Close','Adj Close']]).all().all() and (f[['Open','Close','Adj Close']]>0).all().all()
        frames[ticker]=f
        coverage.append(dict(ticker=ticker,first=str(f.index[0].date()),last=str(f.index[-1].date()),rows=len(f),cutoff_close=float(f.Close.iloc[-1]),splits=int(f['Stock Splits'].ne(0).sum()),dividend_days=int(f.Dividends.ne(0).sum()),sha256=sha(target)))
        print(ticker,coverage[-1]['first'],coverage[-1]['last'],flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'data/prices/history_coverage.csv',index=False)
    c=json.loads((BASE/'config.yaml').read_text())
    c.update(study=OUT.name,source_report=None,universe_source='Exact user-supplied symbols on 2026-10-19; growth ranking not independently validated',ticker_count=20,reuse_previous_gp=False)
    c['window_tickers']={'A':[t for t in TICKERS if t not in ['SAIH','FATN']],'B':TICKERS}
    c['windows']['B']['requested_start']='2025-01-01'
    c['windows']['B']['start']='2025-04-08'
    c['missing_history_policy']='A: explicitly labelled 18-stock diagnostic excludes SAIH/FATN, no substitutes. B: all 20, common history starts FATN IPO 2025-04-08; retain July evaluation; short warmup disclosed; transfer only 18 eligible stocks.'
    c['bootstrap']['unit']='whole market month, all eligible assets together'
    c['selection_limitation']='User selected growing stocks in October 2026: hindsight winner selection in both windows. The 2026 interval and methodology were already examined; exploratory replication.'
    c['transfer_unavailable']=['SAIH','FATN']
    (OUT/'config.yaml').write_text(json.dumps(c,indent=2))
    audit=[]
    for window,w in c['windows'].items():
        idx=calendar().sessions_in_range(w['start'],w['end']).tz_localize(None)
        folder=OUT/'data/prices'/window; folder.mkdir(exist_ok=True)
        for ticker in c['window_tickers'][window]+['SPY']:
            f=frames[ticker].reindex(idx)
            missing=int(f[['Open','Close','Adj Close']].isna().any(axis=1).sum())
            audit.append(dict(window=window,ticker=ticker,first=str(idx[0].date()),last=str(idx[-1].date()),sessions=len(idx),missing=missing))
            if missing: raise ValueError(f'{ticker} {window}: {missing} missing sessions; no filling allowed')
            f.to_csv(folder/f'{ticker}.csv',index_label='Date')
    pd.DataFrame(audit).to_csv(OUT/'data/prices/audit.csv',index=False)
    for name in ['macro','news']:
        if not (OUT/'data'/name).exists(): shutil.copytree(BASE/'data'/name,OUT/'data'/name)
    shutil.copyfile(BASE/'logs/download_manifest.json',OUT/'logs/download_manifest.json')
    manifest={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file() and 'source' not in p.relative_to(BASE).parts}
    (OUT/'logs/previous_study_hashes.json').write_text(json.dumps(manifest,indent=2))
    print('Prepared',OUT,flush=True)

if __name__=='__main__': main()
