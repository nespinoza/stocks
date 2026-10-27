#!/usr/bin/env python3
"""Cache dated ALFRED snapshots, official FOMC statements and fixed GDELT samples."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sys
import time
from urllib.parse import urljoin

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
import requests
from bs4 import BeautifulSoup
from stock_api.data import calendar
from scripts.run_portfolio_gp_22 import extract_tickers
OUT=ROOT/'experiments/portfolio_gp_macro_news_2026-10-18'
C=json.loads((OUT/'config.yaml').read_text())


def fetch(url,path):
    meta=path.with_suffix(path.suffix+'.json')
    if path.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=saved['sha256']:
            raise ValueError(f'Cache checksum mismatch: {path}')
        if saved['url']==url: return saved
    error=None
    for attempt in range(2):
        try:
            r=requests.get(url,timeout=20)
            r.raise_for_status()
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(r.content)
            saved=dict(url=url,retrieved_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content),path=str(path.relative_to(OUT)))
            meta.write_text(json.dumps(saved,indent=2)); return saved
        except requests.RequestException as e:
            error=e; print(f'Request attempt {attempt+1}: {url}: {type(e).__name__}',flush=True); time.sleep(1+attempt)
    return dict(url=url,error=str(error),path=str(path.relative_to(OUT)))


def origins(window):
    idx=calendar().sessions_in_range(window['start'],window['end']).tz_localize(None)
    values=[]
    for month in pd.period_range(window['start'][:7],window['end'][:7],freq='M'):
        eligible=idx[idx>=pd.Timestamp(f'{month}-17')]
        if len(eligible):
            execution=eligible[0]; values.append(idx[idx.get_loc(execution)-1])
    return values+[idx[-1]]


def main():
    (OUT/'logs').mkdir(exist_ok=True)
    tickers=extract_tickers(ROOT/C['source_report']); (OUT/'tickers.txt').write_text('\n'.join(tickers)+'\n')
    audit=[]
    for name,window in C['windows'].items():
        idx=calendar().sessions_in_range(window['start'],window['end']).tz_localize(None)
        for ticker in tickers+['SPY']:
            source=ROOT/C['source_prices']/f'{ticker}.csv'; meta=json.loads(source.with_suffix('.json').read_text())
            assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['sha256']
            f=pd.read_csv(source,index_col='Date',parse_dates=True,float_precision='round_trip').reindex(idx)
            if f[['Open','Close','Adj Close']].isna().any().any(): raise ValueError(f'Missing {ticker} {name}')
            folder=OUT/'data/prices'/name; folder.mkdir(parents=True,exist_ok=True)
            f.to_csv(folder/f'{ticker}.csv',index_label='Date'); (folder/f'{ticker}.json').write_text(json.dumps(meta,indent=2))
            audit.append(dict(window=name,ticker=ticker,rows=len(f),missing=0,first=str(idx[0].date()),last=str(idx[-1].date()),source_sha256=meta['sha256']))
    pd.DataFrame(audit).to_csv(OUT/'data/prices/audit.csv',index=False)
    jobs=[]; states=[]
    for name,window in C['windows'].items():
        for origin in origins(window):
            vintage=(origin-pd.Timedelta(1,unit='D')).date().isoformat()
            for series in C['macro_series']:
                # Prior CPI observations establish a level, not pre-window exposure training.
                start=(pd.Timestamp(window['start'])-pd.DateOffset(years=2 if series=='CPIAUCSL' else 1)).date().isoformat()
                url=f'https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={series}&cosd={start}&coed={vintage}&vintage_date={vintage}'
                path=OUT/'data/macro'/f'{name}_{series}_{vintage}.csv'
                jobs.append((url,path)); states.append(dict(window=name,origin=str(origin.date()),vintage=vintage,series=series,path=str(path.relative_to(OUT))))
    # Objective calendar sampling, not event/return-dependent retrieval.
    for name,window in C['windows'].items():
        for month in pd.period_range(window['start'][:7],window['end'][:7],freq='M'):
            for day in C['news_sample_days']:
                date=pd.Timestamp(f'{month}-{day:02d}')
                if date>pd.Timestamp(window['end']): continue
                stamp=date.strftime('%Y%m%d')+C['news_sample_utc']
                jobs.append((f'https://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip',OUT/'data/news/gdelt'/f'{stamp}.zip'))
    archives=[]
    for year in [2019,2020]:
        url=f'https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm'
        path=OUT/'data/news/fed'/f'archive_{year}.html'
        archives.append(fetch(url,path))
    # Current calendars also index recent years that have not become historical pages.
    archives.append(fetch('https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',OUT/'data/news/fed/calendar.html'))
    links=set()
    for item in archives:
        if 'error' in item: continue
        html=(OUT/item['path']).read_text(encoding='utf-8-sig',errors='replace')
        soup=BeautifulSoup(html,'html.parser')
        for a in soup.find_all('a',href=True):
            href=urljoin(item['url'],a['href'])
            match=re.search(r'/newsevents/pressreleases/monetary(\d{8})[a-z]\.htm$',href)
            # Calendar labels Statement/HTML may omit full title; validate fetched title downstream.
            if match:
                date=pd.to_datetime(match[1],format='%Y%m%d')
                if any(pd.Timestamp(w['start'])<=date<=pd.Timestamp(w['end']) for w in C['windows'].values()): links.add(href)
    for link in sorted(links): jobs.append((link,OUT/'data/news/fed'/link.rsplit('/',1)[-1]))
    pd.DataFrame(states).to_csv(OUT/'data/macro/snapshot_index.csv',index=False)
    results=archives[:]
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures={pool.submit(fetch,*job):job for job in jobs}
        for i,future in enumerate(as_completed(futures),1):
            results.append(future.result())
            if i%10==0 or i==len(jobs): print(f'{i}/{len(jobs)} files; failures {sum("error" in r for r in results)}',flush=True)
    (OUT/'logs/download_manifest.json').write_text(json.dumps(results,indent=2))
    failures=[r for r in results if 'error' in r]
    print(json.dumps({'files':len(results),'failures':failures,'fed_links':len(links)},indent=2),flush=True)

if __name__=='__main__': main()
