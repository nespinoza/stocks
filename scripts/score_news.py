"""Score collected news: Loughran–McDonald tone for every document, LLM extraction for a capped subset.

The LLM step defaults to headless Claude Code on the user's plan
(`--backend claude-code --model haiku`); switch to `--backend anthropic
--model claude-haiku-5-5` once API billing is set up. Cached results are reused.
Only documents published on or after --since are scored by the LLM (default:
after Haiku 5.5's June 2026 training cutoff, so the evaluation is leak-free).

    .conda-env/bin/python scripts/score_news.py --limit 500

Broad small/mid-cap universe (documents from collect_news.py --config ...):

    NEWS_DIR=data_store/news_broad .conda-env/bin/python scripts/score_news.py \
        --config configs/broad_universe.json --limit 100000

Results are cached after every batch. If the plan's usage limit is reached the
run stops with exit code 75; re-run the same command after the limit resets and
it continues from the first unscored batch.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from stock_api.news import (AnthropicBackend, ClaudeCodeBackend, ExtractionCache, MockBackend, UsageLimitError, dedupe,
                            extract, item_key, load_lm, tone)
from stock_api.news.aggregate import assign_sessions

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['claude-code', 'anthropic', 'mock'], default='claude-code')
    parser.add_argument('--model', default='haiku', help="'haiku' for claude-code, 'claude-haiku-5-5' for anthropic")
    parser.add_argument('--since', default='2026-07-01')
    parser.add_argument('--max-per-ticker-day', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0, help='Max new LLM items this run (0 = no LLM step)')
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--config', default=str(ROOT/'configs/news_universe.json'))
    args = parser.parse_args()
    news = Path(os.environ.get('NEWS_DIR', ROOT/'data_store/news'))
    config = json.loads(Path(args.config).read_text())
    docs = [json.loads(line) for line in (news/'documents.jsonl').open() if line.strip()]
    frame = dedupe(docs)
    sessions = pd.bdate_range('2024-01-01', '2027-12-31')  # calendar days; exchange holidays matter little here
    frame['session'] = assign_sessions(frame.published_utc, sessions)
    print(f'{len(docs)} documents, {len(frame)} after de-duplication')

    positive, negative, uncertainty = load_lm(ROOT/'data_store/lexicon/LM_MasterDictionary.csv')
    scored = pd.DataFrame([tone(f"{t} {x or ''}", positive, negative, uncertainty) for t, x in zip(frame.title, frame.text)],
                          index=frame.index)
    lexicon = pd.concat([frame[['doc_id', 'query_ticker', 'published_utc', 'source']].rename(columns={'query_ticker': 'ticker'}),
                         scored], axis=1)
    lexicon.to_csv(news/'lexicon_scores.csv', index=False)
    print(f'lexicon: scored {len(lexicon)} documents')

    if args.limit <= 0:
        return
    backend = {'claude-code': lambda: ClaudeCodeBackend(args.model), 'anthropic': lambda: AnthropicBackend(args.model),
               'mock': MockBackend}[args.backend]()
    recent = frame[frame.published_utc >= args.since]
    # EDGAR filings are always kept; headlines are capped per ticker-session, earliest first.
    capped = pd.concat([recent[recent.source != 'gdelt'],
                        recent[recent.source == 'gdelt'].sort_values('published_utc')
                        .groupby(['query_ticker', 'session']).head(args.max_per_ticker_day)])
    cache = ExtractionCache(news/'extractions.jsonl')
    items = [dict(id=item_key(r.doc_id, r.query_ticker), doc_id=r.doc_id, ticker=r.query_ticker,
                  company=config['stocks'][r.query_ticker]['name'], source=r.source, published_utc=r.published_utc,
                  title=r.title, text=r.text or '')
             for r in capped.sort_values('published_utc').itertuples()]
    todo = [i for i in items if not cache.has(args.model, i['id'])][:args.limit]
    print(f'LLM: {len(items)} eligible since {args.since}, {len(items)-len(todo)} cached or deferred, scoring {len(todo)}')
    try:
        added = extract(todo, backend, cache, args.model, args.batch_size)
    except UsageLimitError as error:
        left = sum(not cache.has(args.model, i['id']) for i in todo)
        print(f'LLM: stopped at the usage limit with {left} items left ({error}).\n'
              'Progress is saved; re-run the same command after the limit resets.', flush=True)
        sys.exit(75)
    print(f'LLM: added {added} rows')


if __name__ == '__main__':
    main()
