"""Collect timestamped news for configs/news_universe.json into data_store/news/documents.jsonl.

GDELT DOC headlines (rolling ~3-month window, one request per ~6 s, so the
full universe takes a few hours) and SEC EDGAR 8-K filings (needs
SEC_USER_AGENT="<name> <email>"). Re-running only adds new documents, so this
can run daily from cron going forward.

    .conda-env/bin/python scripts/collect_news.py --start 2026-07-28 --end 2026-10-27 --sources gdelt
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from stock_api.news import sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', required=True, help='UTC start date, YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='UTC end date (exclusive), YYYY-MM-DD')
    parser.add_argument('--sources', default='gdelt,edgar')
    parser.add_argument('--tickers', default='', help='Comma-separated subset (default: all)')
    parser.add_argument('--max-records', type=int, default=75, help='GDELT articles per ticker-week query')
    parser.add_argument('--store', default=str(ROOT/'data_store/news/documents.jsonl'))
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/news_universe.json').read_text())
    tickers = args.tickers.split(',') if args.tickers else list(config['stocks'])
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    store = sources.DocumentStore(args.store)
    session = requests.Session()
    log = (ROOT/'data_store/news/collect_log.jsonl').open('a')
    wanted = args.sources.split(',')
    if 'gdelt' in wanted:
        throttle = sources.Throttle(sources.GDELT_SPACING)
        for n, ticker in enumerate(tickers, 1):
            query = sources.gdelt_query(config['stocks'][ticker]['gdelt_phrases'])
            for a, b in sources.weekly_windows(start, end):
                try:
                    articles = sources.fetch_gdelt(query, a, b, args.max_records, throttle, session)
                    added = store.add(sources.gdelt_documents(ticker, articles))
                    status = 'ok'
                except Exception as error:  # Logged and skipped; a re-run fills the gap.
                    articles, added, status = [], 0, f'error: {error}'[:300]
                log.write(json.dumps(dict(source='gdelt', ticker=ticker, query=query, start=a.isoformat(),
                                          end=b.isoformat(), returned=len(articles), added=added, status=status,
                                          at=sources.now_utc()))+'\n')
                log.flush()
            print(f'gdelt {n}/{len(tickers)} {ticker}: {len(store.ids)} documents', flush=True)
    if 'edgar' in wanted:
        throttle = sources.Throttle(sources.SEC_SPACING)
        for ticker in tickers:
            cik = config['stocks'][ticker]['cik']
            docs = []
            for filing in sources.fetch_8k_index(cik, session, throttle):
                accepted = datetime.fromisoformat(filing['accepted'].replace('Z', '+00:00'))
                if not start <= accepted < end:
                    continue
                doc = sources.eight_k_document(ticker, cik, filing)
                if doc['doc_id'] in store.ids:
                    continue
                try:
                    doc['text'] = sources.fetch_8k_body(doc['url'], session, throttle)
                except requests.HTTPError:
                    pass
                docs.append(doc)
            added = store.add(docs)
            log.write(json.dumps(dict(source='edgar', ticker=ticker, added=added, at=sources.now_utc()))+'\n')
            print(f'edgar {ticker}: +{added}', flush=True)


if __name__ == '__main__':
    main()
