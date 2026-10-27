"""Freeze issuer holdings and Yahoo price histories for a monthly stock screen."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import re
import hashlib

import pandas as pd
import requests
import yfinance as yf

ROOT = Path('reports/monthly-stock-screen-2026-10-18')
URL = 'https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/latest-holdings.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    data = ROOT/'data'; data.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(data/'yahoo-cache'))
    raw = data/'iwb-holdings.csv'
    if not raw.exists():
        response = requests.get(URL, timeout=40)
        response.raise_for_status()
        if 'Ticker,Name,Sector,Asset Class' not in response.text:
            raise ValueError('Issuer response is not the expected holdings CSV')
        raw.write_bytes(response.content)
        (data/'holdings-source.json').write_text(json.dumps({
            'url':URL,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
            'sha256':hashlib.sha256(response.content).hexdigest()}, indent=2))
    content = raw.read_text(encoding='utf-8-sig')
    lines = content.splitlines()
    start = next(i for i,line in enumerate(lines) if line.startswith('Ticker,Name,'))
    frame = pd.read_csv(StringIO('\n'.join(lines[start:])), on_bad_lines='error')
    frame = frame[(frame['Asset Class']=='Equity') & (frame['Currency']=='USD')].copy()
    frame['Price'] = pd.to_numeric(frame.Price.astype(str).str.replace(',',''), errors='coerce')
    frame['ticker'] = frame.Ticker.str.replace(' ', '-', regex=False).str.replace('.', '-', regex=False)
    frame = frame[frame.ticker.map(lambda x:bool(re.fullmatch(r'[A-Z][A-Z0-9-]{0,9}', x)))].copy()
    frame.to_csv(data/'universe.csv', index=False)
    print('Issuer universe:', len(frame), 'Header:', lines[:3], flush=True)
    candidates = frame[frame.Price>10].ticker.tolist()
    # Benchmarks are downloaded for comparison and not included as stock candidates.
    symbols = candidates + ['SPY']
    if args.limit: symbols = symbols[:args.limit]
    def download(symbol):
        path = data/f'{symbol}.csv'
        if path.exists(): return symbol, 'cached', len(pd.read_csv(path))
        try:
            obj = yf.Ticker(symbol)
            history = obj.history(start='2009-01-01', end='2026-10-18', interval='1d',
                                  auto_adjust=False, actions=True, raise_errors=True, timeout=25)
            if history.empty: raise ValueError('Empty history')
            history.index = history.index.tz_localize(None)
            history.to_csv(path)
            metadata = {'ticker':symbol,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                        'source':f'https://finance.yahoo.com/quote/{symbol}/history/',
                        'request':{'start':'2009-01-01','end':'2026-10-18','interval':'1d','auto_adjust':False},
                        'rows':len(history),'first_date':str(history.index[0].date()),
                        'last_date':str(history.index[-1].date()),
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            (data/f'{symbol}.json').write_text(json.dumps(metadata, indent=2))
            return symbol, 'ok', len(history)
        except Exception as exc:
            return symbol, str(exc), 0
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download,s) for s in symbols]
        for i,f in enumerate(as_completed(futures),1):
            records.append(f.result())
            if i%25==0 or i==len(symbols):
                print(f'{i}/{len(symbols)} completed; failures {sum(r[2]==0 for r in records)}; last {records[-1]}',flush=True)
    pd.DataFrame(records,columns=['ticker','status','rows']).to_csv(data/'download-status.csv',index=False)
    print(pd.DataFrame(records).head().to_string(index=False),flush=True)


if __name__ == '__main__': main()
