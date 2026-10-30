"""Download daily OHLCV + adjusted closes for a universe config into data_store/.

Default: configs/etf_universe.json into data_store/etf/. For the news stocks:
    --config configs/news_universe.json --out data_store/stocks --start 2024-01-01
For the broad small/mid-cap universe:
    --config configs/broad_universe.json --out data_store/broad --start 2010-01-01 --skip-missing

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
    parser.add_argument('--config', default='configs/etf_universe.json')
    parser.add_argument('--out', default='data_store/etf')
    parser.add_argument('--start', default='2007-01-01')
    parser.add_argument('--skip-missing', action='store_true', help='Skip tickers with no data instead of failing')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    tickers = (list(config['stocks']) if 'stocks' in config
               else [t for group in config['groups'].values() for t in group])
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'provider': 'Yahoo Finance via yfinance',
                'yfinance': yf.__version__, 'start': args.start, 'end_exclusive': args.end,
                'auto_adjust': False, 'files': {}}
    if (out/'manifest.json').exists():  # several configs can share one folder
        manifest['files'] = json.loads((out/'manifest.json').read_text()).get('files', {})
    for ticker in tickers:
        path = out/f'{ticker}.csv'
        if args.refresh or not path.exists():
            frame = yf.Ticker(ticker).history(start=args.start, end=args.end, interval='1d',
                                              auto_adjust=False, actions=True, timeout=30)
            if frame.empty:
                if not args.skip_missing:
                    raise RuntimeError(f'No data for {ticker}')
                print(ticker, 'no data, skipped')
                continue
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
