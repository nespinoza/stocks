"""Collect timestamped news for configs/news_universe.json into data_store/news/documents.jsonl.

GDELT DOC headlines (rolling ~3-month window, one request per ~6 s, so the
full universe takes a few hours) and SEC EDGAR 8-K filings (needs
SEC_USER_AGENT="<name> <email>"). Re-running only adds new documents, so this
can run daily from cron going forward. GDELT throttles bulk back-fills
heavily, so the practical route is a small daily pull of the last few days:

    .conda-env/bin/python scripts/collect_news.py --days-back 3
    .conda-env/bin/python scripts/collect_news.py --start 2026-07-28 --end 2026-10-27 --sources edgar

8-Ks for the broad small/mid-cap universe go to a separate store:

    .conda-env/bin/python scripts/collect_news.py --start 2026-07-01 --end 2026-10-31 --sources edgar \
        --config configs/broad_universe.json --store data_store/news_broad/documents.jsonl
"""
import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from stock_api.news import sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', help='UTC start date, YYYY-MM-DD')
    parser.add_argument('--end', help='UTC end date (exclusive), YYYY-MM-DD')
    parser.add_argument('--days-back', type=int, default=0, help='Instead of --start/--end: the last N full UTC days')
    parser.add_argument('--sources', default='gdelt,edgar')
    parser.add_argument('--tickers', default='', help='Comma-separated subset (default: all)')
    parser.add_argument('--max-records', type=int, default=75, help='GDELT articles per ticker-window query (max 250)')
    parser.add_argument('--window-days', type=int, default=7, help='GDELT query window length')
    parser.add_argument('--spacing', type=float, default=sources.GDELT_SPACING, help='Seconds between GDELT requests')
    parser.add_argument('--cooldown', type=float, default=900., help='Pause after a window exhausts its rate-limit retries')
    parser.add_argument('--store', default=str(ROOT/'data_store/news/documents.jsonl'))
    parser.add_argument('--config', default=str(ROOT/'configs/news_universe.json'),
                        help='Universe file; GDELT needs gdelt_phrases, EDGAR only a cik per stock')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    tickers = args.tickers.split(',') if args.tickers else list(config['stocks'])
    if args.days_back:
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end-timedelta(days=args.days_back)
    elif args.start and args.end:
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    else:
        parser.error('give --start and --end, or --days-back')
    store = sources.DocumentStore(args.store)
    session = requests.Session()
    log_path = Path(args.store).parent/'collect_log.jsonl'
    # Windows already fetched successfully are skipped, so a re-run only fills gaps.
    done = {(r['query'], r['start'], r['end']) for r in map(json.loads, log_path.open()) if r.get('status') == 'ok'} \
        if log_path.exists() else set()
    log = log_path.open('a')
    wanted = args.sources.split(',')
    if 'gdelt' in wanted:
        throttle = sources.Throttle(args.spacing)
        for n, ticker in enumerate(tickers, 1):
            query = sources.gdelt_query(config['stocks'][ticker]['gdelt_phrases'])
            for a, b in sources.windows(start, end, args.window_days):
                if (query, a.isoformat(), b.isoformat()) in done:
                    continue
                try:
                    articles = sources.fetch_gdelt(query, a, b, args.max_records, throttle, session)
                    added = store.add(sources.gdelt_documents(ticker, articles))
                    status = 'ok'
                except Exception as error:  # Logged and skipped; a re-run fills the gap.
                    articles, added, status = [], 0, f'error: {error}'[:300]
                    if 'rate limit' in status:
                        time.sleep(args.cooldown)
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
            try:
                filings = sources.fetch_8k_index(cik, session, throttle)
            except requests.RequestException as error:  # Logged and skipped; a re-run fills the gap.
                log.write(json.dumps(dict(source='edgar', ticker=ticker, status=f'error: {error}'[:300],
                                          at=sources.now_utc()))+'\n')
                print(f'edgar {ticker}: error {error}', flush=True)
                continue
            for filing in filings:
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
