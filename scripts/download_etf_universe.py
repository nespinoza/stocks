"""Download daily OHLCV + adjusted closes for configs/etf_universe.json into data_store/etf/.

Writes one CSV per ticker (raw provider fields, auto_adjust=False) and a
manifest with checksums. Existing files are kept unless --refresh is given.

    .conda-env/bin/python scripts/download_etf_universe.py --end 2026-10-27
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end', required=True, help='Exclusive end date, YYYY-MM-DD')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/etf_universe.json').read_text())
    tickers = [t for group in config['groups'].values() for t in group]
    out = ROOT/'data_store/etf'
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'provider': 'Yahoo Finance via yfinance',
                'yfinance': yf.__version__, 'start': '2007-01-01', 'end_exclusive': args.end,
                'auto_adjust': False, 'files': {}}
    for ticker in tickers:
        path = out/f'{ticker}.csv'
        if args.refresh or not path.exists():
            frame = yf.Ticker(ticker).history(start='2007-01-01', end=args.end, interval='1d',
                                              auto_adjust=False, actions=True, timeout=30)
            if frame.empty:
                raise RuntimeError(f'No data for {ticker}')
            frame.index = frame.index.tz_localize(None).normalize()
            frame.index.name = 'Date'
            frame.to_csv(path)
        text = path.read_bytes()
        lines = text.decode().splitlines()
        manifest['files'][ticker] = {'sha256': hashlib.sha256(text).hexdigest(), 'rows': len(lines)-1,
                                     'first': lines[1][:10], 'last': lines[-1][:10]}
        print(ticker, manifest['files'][ticker]['first'], manifest['files'][ticker]['last'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
