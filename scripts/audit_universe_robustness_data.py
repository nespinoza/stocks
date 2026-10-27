#!/usr/bin/env python3
"""Audit historical universe sources before allocating or inspecting future returns."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import requests

OUT=ROOT/'experiments/portfolio_gp_universe_robustness_2026-10-19'

def fetch(name,url):
    folder=OUT/'data/source_audit'; folder.mkdir(parents=True,exist_ok=True)
    target=folder/(name+'.body'); meta=folder/(name+'.json')
    if meta.exists():
        cached=json.loads(meta.read_text())
        if 'error' not in cached: return cached
    record=dict(name=name,url=url,retrieved_utc=datetime.now(timezone.utc).isoformat())
    try:
        response=requests.get(url,timeout=30)
        target.write_bytes(response.content)
        record.update(status=response.status_code,final_url=response.url,bytes=len(response.content),sha256=hashlib.sha256(response.content).hexdigest(),content_type=response.headers.get('Content-Type'),body=str(target.relative_to(OUT)))
        print(name,response.status_code,len(response.content),response.text[:130].replace('\n',' '),flush=True)
    except requests.RequestException as exc:
        record['error']=type(exc).__name__+': '+str(exc); print(name,record['error'],flush=True)
    meta.write_text(json.dumps(record,indent=2)); return record

def main():
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    names=['WRDS_USERNAME','WRDS_PASSWORD','NASDAQ_DATA_LINK_API_KEY','QUANDL_API_KEY','EODHD_API_KEY','TIINGO_API_KEY','POLYGON_API_KEY','FMP_API_KEY']
    (OUT/'logs/access_audit.json').write_text(json.dumps(dict(environment_presence_only={n:bool(os.environ.get(n)) for n in names},credential_file_presence={n:Path(n).expanduser().exists() for n in ['~/.pgpass','~/.netrc','~/.quandl/apikey','.env']},platform=sys.platform),indent=2))
    jobs=[]
    for window,date in [('A','20190716'),('B','20250716'),('A_monthend','20190628'),('B_monthend','20250630')]:
        jobs.append(('iwb_'+window,f'https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund&asOfDate={date}'))
    jobs += [
        ('historical_membership_csv','https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv'),
        ('iwb_dated_latest_probe','https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/latest-holdings.csv?asOfDate=20190716'),
        ('historical_membership_readme','https://raw.githubusercontent.com/fja05680/sp500/master/README.md'),
        ('historical_membership_files','https://api.github.com/repos/fja05680/sp500/contents'),
        ('github_membership_2019','https://api.github.com/repos/datasets/s-and-p-500-companies/commits?path=data/constituents.csv&until=2019-07-16T00:00:00Z&per_page=1'),
        ('github_membership_2025','https://api.github.com/repos/datasets/s-and-p-500-companies/commits?path=data/constituents.csv&until=2025-07-16T00:00:00Z&per_page=1'),
        ('sharadar_sample_access','https://data.nasdaq.com/api/v3/datatables/SHARADAR/SEP.csv?ticker=CELG&date.gte=2019-07-01&date.lte=2019-07-16'),
        ('norgate_documentation','https://norgatedata.com/data-content-tables.php'),
        ('crsp_documentation','https://www.crsp.org/research/crsp-us-stock-databases/'),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda args:fetch(*args),jobs))
    for label in ['2019','2025']:
        p=OUT/f'data/source_audit/github_membership_{label}.body'
        if p.exists():
            commits=json.loads(p.read_text())
            if isinstance(commits,list) and commits:
                commit=commits[0]['sha']
                results.append(fetch('github_snapshot_'+label,f'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/{commit}/data/constituents.csv'))
    # These are data availability probes, not selected portfolios or performance filters.
    import yfinance as yf
    yf.set_tz_cache_location(str(OUT/'data/yahoo-cache'))
    prices=[]
    for ticker in ['CELG','ABMD','TWTR','SIVB']:
        old_probe=OUT/'logs/delisted_price_probes.json'
        if old_probe.exists():
            previous=next((r for r in json.loads(old_probe.read_text()) if r['ticker']==ticker),None)
            if previous is not None:
                prices.append(previous); continue
        record=dict(ticker=ticker,request_start='2018-06-01',request_end='2023-02-01')
        try:
            f=yf.Ticker(ticker).history(start=record['request_start'],end=record['request_end'],auto_adjust=False,actions=True,raise_errors=True,timeout=20)
            record.update(rows=len(f),first=str(f.index[0]) if len(f) else None,last=str(f.index[-1]) if len(f) else None)
            if len(f): f.to_csv(OUT/f'data/source_audit/yahoo_{ticker}.csv')
        except Exception as exc: record.update(rows=0,error=type(exc).__name__+': '+str(exc))
        print('Delisted history probe:',record,flush=True); prices.append(record)
    (OUT/'logs/delisted_price_probes.json').write_text(json.dumps(prices,indent=2))
    (OUT/'logs/source_audit.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__': main()
