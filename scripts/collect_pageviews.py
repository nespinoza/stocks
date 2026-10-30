"""Daily English-Wikipedia pageviews for the broad universe (free attention proxy).

1. Map each stock to its English Wikipedia article through Wikidata: first by
   SEC CIK (property P5531), else by a current NYSE/Nasdaq ticker (P414 with a
   P249 qualifier and no end date) whose article title shares a word with the
   SEC company name. Unmatched stocks (typically the least-known ones) are skipped.
2. Fetch daily user (non-bot) pageviews from the Wikimedia REST API since
   July 2015, the API's first day.

Writes data_store/attention/wiki_titles.json and pageviews.csv.gz (date ×
ticker). Pageviews are counted per UTC day, so day d is complete only after
00:00 UTC of d+1 (20:00 New York); signals use views through d−1.

    .conda-env/bin/python scripts/collect_pageviews.py --end 2026-10-29
"""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {'User-Agent': 'stocks-research/0.1 (https://github.com/nespinoza/stocks)'}
SPARQL = 'https://query.wikidata.org/sparql'
PAGEVIEWS = ('https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/'
             '{title}/daily/20150701/{end}')
BY_CIK = ('SELECT ?cik ?article WHERE { ?item wdt:P5531 ?cik . ?article schema:about ?item ; '
          'schema:isPartOf <https://en.wikipedia.org/> . }')
BY_TICKER = ('SELECT ?ticker ?article WHERE { VALUES ?exchange { wd:Q13677 wd:Q82059 } ?item p:P414 ?st . '
             '?st ps:P414 ?exchange ; pq:P249 ?ticker . FILTER NOT EXISTS { ?st pq:P582 ?end } '
             '?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }')
STOP = {'inc', 'corp', 'corporation', 'co', 'company', 'holdings', 'group', 'the', 'ltd', 'plc', 'de', 'llc', 'lp',
        'trust', 'and', 'of', 'international', 'technologies', 'bancorp', 'financial'}


def sparql(query):
    response = requests.get(SPARQL, params={'query': query}, timeout=300,
                            headers={**HEADERS, 'Accept': 'application/sparql-results+json'})
    response.raise_for_status()
    return response.json()['results']['bindings']


def words(text):
    return {w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOP and len(w) > 1}


def map_titles(stocks):
    by_cik = {int(r['cik']['value']): r['article']['value'] for r in sparql(BY_CIK) if r['cik']['value'].isdigit()}
    by_ticker = {}
    for r in sparql(BY_TICKER):
        by_ticker.setdefault(r['ticker']['value'].upper(), set()).add(r['article']['value'])
    titles = {}
    for ticker, stock in stocks.items():
        if stock['cik'] in by_cik:
            url, how = by_cik[stock['cik']], 'cik'
        else:
            candidates = by_ticker.get(ticker.replace('-', '.'), set()) | by_ticker.get(ticker, set())
            if len(candidates) != 1:
                continue
            url, how = next(iter(candidates)), 'ticker'
            if not words(stock['name']) & words(unquote(url.rsplit('/', 1)[1]).replace('_', ' ')):
                continue
        titles[ticker] = dict(title=unquote(url.rsplit('/', 1)[1]), matched_by=how)
    return titles


def fetch_views(title, end, session):
    for attempt in range(4):
        response = session.get(PAGEVIEWS.format(title=quote(title, safe=''), end=end), headers=HEADERS, timeout=60)
        if response.status_code == 404:
            return pd.Series(dtype=float)
        if response.status_code == 429:
            time.sleep(10*(attempt+1))
            continue
        response.raise_for_status()
        items = response.json()['items']
        return pd.Series({pd.Timestamp(i['timestamp'][:8]): float(i['views']) for i in items})
    raise RuntimeError(f'pageviews rate limit for {title}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end', required=True, help='Last UTC day to fetch, YYYY-MM-DD')
    parser.add_argument('--spacing', type=float, default=.2, help='Seconds between pageview requests')
    args = parser.parse_args()
    out = ROOT/'data_store/attention'
    out.mkdir(parents=True, exist_ok=True)
    stocks = json.loads((ROOT/'configs/broad_universe.json').read_text())['stocks']
    titles = map_titles(stocks)
    (out/'wiki_titles.json').write_text(json.dumps(titles, indent=1))
    print(f"{len(titles)} of {len(stocks)} stocks mapped "
          f"({sum(v['matched_by'] == 'cik' for v in titles.values())} by CIK)", flush=True)
    session, series = requests.Session(), {}
    for n, (ticker, info) in enumerate(titles.items(), 1):
        time.sleep(args.spacing)
        series[ticker] = fetch_views(info['title'], args.end.replace('-', ''), session)
        if n % 100 == 0:
            print(f'pageviews {n}/{len(titles)}', flush=True)
    views = pd.DataFrame(series).sort_index()
    views.index.name = 'date'
    views.to_csv(out/'pageviews.csv.gz', compression='gzip')
    print(f'wrote {views.shape[1]} series, {views.index[0].date()} to {views.index[-1].date()}')


if __name__ == '__main__':
    main()
