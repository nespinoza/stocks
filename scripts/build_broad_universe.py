"""Build a broad small/mid-cap US stock universe from free sources.

1. SEC company_tickers_exchange.json: every listed ticker with its CIK and exchange
   (NYSE and Nasdaq kept; first ticker per CIK, which is the primary class).
2. SEC XBRL frames (dei:EntityCommonStockSharesOutstanding): the latest cover-page
   share count per company over the last four quarters. Only 10-K/10-Q filers
   report it, which drops funds, ETFs and foreign private issuers (who also do
   not file Form 4).
3. Yahoo prices over the last ~3 months: last close and median dollar volume.
4. SEC submissions: SIC code, for sector controls.

Kept: market cap in [--min-cap, --max-cap], price >= --min-price, median daily
dollar volume >= --min-dollar-volume. The universe is fixed at the build date,
so a historical backtest on it has survivorship bias (companies that died or
were acquired before today are missing); a forward test does not.

    SEC_USER_AGENT="<name> <email>" .conda-env/bin/python scripts/build_broad_universe.py
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from stock_api.news import sources

ROOT = Path(__file__).resolve().parents[1]
TICKERS_URL = 'https://www.sec.gov/files/company_tickers_exchange.json'
FRAME_URL = 'https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/{period}.json'


def recent_quarters(today, count=4):
    quarter = (today.month-1)//3+1
    year = today.year
    out = []
    for _ in range(count):
        out.append(f'CY{year}Q{quarter}I')
        year, quarter = (year-1, 4) if quarter == 1 else (year, quarter-1)
    return out


def share_counts(session, throttle, today):
    rows = []
    for period in recent_quarters(today):
        throttle.wait()
        response = session.get(FRAME_URL.format(period=period), headers=sources.sec_headers(), timeout=120)
        if response.status_code == 404:  # the current quarter's frame may not exist yet
            continue
        response.raise_for_status()
        rows += response.json()['data']
    frame = pd.DataFrame(rows).sort_values('end')
    return frame.groupby('cik').last()[['val', 'end']].rename(columns={'val': 'shares', 'end': 'shares_as_of'})


def market_data(tickers, chunk=200):
    """Last close and median daily dollar volume over ~3 months, via bulk Yahoo downloads."""
    rows = {}
    for start in range(0, len(tickers), chunk):
        batch = tickers[start:start+chunk]
        data = yf.download(batch, period='3mo', auto_adjust=False, progress=False, group_by='ticker', threads=True)
        for ticker in batch:
            if ticker not in data.columns.get_level_values(0):
                continue
            close, volume = data[ticker]['Close'].dropna(), data[ticker]['Volume']
            if len(close) < 40:
                continue
            rows[ticker] = dict(price=float(close.iloc[-1]),
                                dollar_volume=float((data[ticker]['Close']*volume).dropna().median()))
        print(f'yahoo {min(start+chunk, len(tickers))}/{len(tickers)}', flush=True)
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-cap', type=float, default=3e8)
    parser.add_argument('--max-cap', type=float, default=1e10)
    parser.add_argument('--min-price', type=float, default=3.)
    parser.add_argument('--min-dollar-volume', type=float, default=2e6)
    parser.add_argument('--out', default=str(ROOT/'configs/broad_universe.json'))
    args = parser.parse_args()
    session, throttle = requests.Session(), sources.Throttle(sources.SEC_SPACING)
    today = date.today()

    throttle.wait()
    listing = session.get(TICKERS_URL, headers=sources.sec_headers(), timeout=60).json()
    listed = pd.DataFrame(listing['data'], columns=listing['fields'])
    listed = listed[listed.exchange.isin(['NYSE', 'Nasdaq'])].drop_duplicates('cik')
    listed = listed.join(share_counts(session, throttle, today), on='cik', how='inner')
    listed['yahoo'] = listed.ticker.str.replace('.', '-', regex=False)
    print(f'{len(listed)} NYSE/Nasdaq companies with a reported share count')

    market = market_data(listed.yahoo.tolist())
    listed = listed.join(market, on='yahoo', how='inner')
    listed['market_cap'] = listed.shares*listed.price
    keep = listed[listed.market_cap.between(args.min_cap, args.max_cap) & (listed.price >= args.min_price)
                  & (listed.dollar_volume >= args.min_dollar_volume)].copy()
    print(f'{len(keep)} pass the cap, price and liquidity filters')

    sic = {}
    for n, cik in enumerate(keep.cik, 1):
        try:
            throttle.wait()
            info = session.get(f'https://data.sec.gov/submissions/CIK{cik:010d}.json', headers=sources.sec_headers(),
                               timeout=60).json()
            sic[cik] = (info.get('sic') or '', info.get('sicDescription') or '')
        except (requests.RequestException, ValueError):
            sic[cik] = ('', '')
        if n % 250 == 0:
            print(f'sic {n}/{len(keep)}', flush=True)
    stocks = {r.yahoo: dict(name=r.name, cik=int(r.cik), exchange=r.exchange, sic=sic[r.cik][0],
                            industry=sic[r.cik][1], market_cap=round(float(r.market_cap)),
                            dollar_volume=round(float(r.dollar_volume)), shares_as_of=r.shares_as_of)
              for r in keep.sort_values('market_cap', ascending=False).itertuples()}
    config = dict(description='Small/mid-cap NYSE+Nasdaq operating companies from free SEC + Yahoo data; fixed at the '
                              'build date, so historical backtests on it carry survivorship bias.',
                  built=str(today), filters=dict(min_cap=args.min_cap, max_cap=args.max_cap, min_price=args.min_price,
                                                 min_dollar_volume=args.min_dollar_volume),
                  stocks=stocks)
    Path(args.out).write_text(json.dumps(config, indent=1))
    print(f'wrote {len(stocks)} stocks to {args.out}')


if __name__ == '__main__':
    main()
