"""Year-over-year text similarity of 10-K and 10-Q filings ("Lazy Prices", Cohen, Malloy & Nguyen 2020).

For each broad-universe company, stream its 10-K and 10-Q primary documents from
EDGAR, oldest first, and compare each one with the same form for the same fiscal
period a year earlier (report date 335–395 days before). This avoids seasonal
differences between quarters. Only the scores are kept. Each filing's word
counts stay in memory until the company is done, and the text is never written
to disk, so the whole run needs a few MB.

Fixed before looking at results:
- Text = primary document with HTML tags, scripts and hidden inline-XBRL facts
  removed, lower-cased, words of 2+ letters (numbers dropped, since they change
  every period anyway).
- cosine = cosine similarity of word counts; jaccard = overlap of the word sets.
- The same two measures for the Risk Factors section (Item 1A) when it can be
  found (the longest "Item 1A Risk Factors" span up to the next item heading).

Writes data_store/filings/similarity.csv (one row per filing) and resumes by
company (data_store/filings/done.txt). EDGAR asks for at most 10 requests a
second; requests are spaced across all threads.

    SEC_USER_AGENT="<name> <email>" .conda-env/bin/python scripts/collect_filing_similarity.py --since 2018-01-01
"""
import argparse
import csv
import json
import math
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import requests

from stock_api.news.sources import Throttle, sec_headers, strip_html

ROOT = Path(__file__).resolve().parents[1]
FORMS = ('10-K', '10-Q')
COLUMNS = ['ticker', 'cik', 'form', 'accession', 'filing_date', 'accepted', 'report_date', 'prior_accession',
           'prior_report_date', 'words', 'prior_words', 'cosine', 'jaccard', 'risk_words', 'prior_risk_words',
           'risk_cosine', 'risk_jaccard']
RISK = re.compile(r'item\s*1a\s*[\.:\-–—]?\s*risk\s+factors')
NEXT_ITEM = re.compile(r'item\s*(1b|1c|2|3|4|5|6)\s*[\.:\-–—]?\s*[a-z]')


class SharedThrottle(Throttle):
    def __init__(self, spacing):
        super().__init__(spacing)
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            super().wait()


def get(session, throttle, url):
    for attempt in range(5):
        throttle.wait()
        try:
            response = session.get(url, headers=sec_headers(), timeout=120)
        except requests.RequestException:
            time.sleep(5*(attempt+1))
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(15*(attempt+1))
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f'gave up on {url}')


def filing_list(session, throttle, cik, since):
    """10-K and 10-Q filings (no amendments) whose report date is on or after `since`, oldest first."""
    base = 'https://data.sec.gov/submissions/'
    data = get(session, throttle, f'{base}CIK{cik:010d}.json').json()
    blocks = [data['filings']['recent']]
    for extra in data['filings'].get('files', []):
        if extra['filingTo'] >= since:
            blocks.append(get(session, throttle, base+extra['name']).json())
    rows = []
    for block in blocks:
        for i, form in enumerate(block['form']):
            report = block['reportDate'][i]
            if form in FORMS and report and report >= since and block['primaryDocument'][i]:
                rows.append(dict(form=form, accession=block['accessionNumber'][i], filing_date=block['filingDate'][i],
                                 accepted=block['acceptanceDateTime'][i], report_date=report,
                                 primary=block['primaryDocument'][i]))
    unique = {r['accession']: r for r in rows}
    return sorted(unique.values(), key=lambda r: (r['filing_date'], r['accession']))


def risk_section(text):
    best = ''
    for match in RISK.finditer(text):
        end = NEXT_ITEM.search(text, match.end()+200)
        span = text[match.end():end.start() if end else match.end()+400000]
        if len(span) > len(best):
            best = span
    return best if len(best) > 2000 else ''


def counts(text):
    return Counter(re.findall(r'[a-z]{2,}', text))


def cosine(a, b):
    if not a or not b:
        return math.nan
    dot = sum(v*b.get(k, 0) for k, v in a.items())
    return dot/math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))


def jaccard(a, b):
    if not a or not b:
        return math.nan
    return len(a.keys() & b.keys())/len(a.keys() | b.keys())


def find_prior(filing, history):
    report = date.fromisoformat(filing['report_date'])
    matches = [h for h in history if h['form'] == filing['form'] and 'all' in h
               and 335 <= (report-date.fromisoformat(h['report_date'])).days <= 395]
    return min(matches, key=lambda h: abs((report-date.fromisoformat(h['report_date'])).days-365)) if matches else None


def process(ticker, cik, since, session, throttle):
    history, rows, failures = [], [], 0
    for filing in filing_list(session, throttle, cik, since):
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing['accession'].replace('-', '')}/{filing['primary']}"
        try:
            text = strip_html(get(session, throttle, url).text, 10**9).lower()
        except Exception as error:  # keep going; one bad document should not drop the company
            print(f'{ticker} {filing["accession"]}: {error}', flush=True)
            failures += 1
            continue
        filing.update(all=counts(text), risk=counts(risk_section(text)))
        prior = find_prior(filing, history)
        history.append(filing)
        row = dict(ticker=ticker, cik=cik, **{k: filing[k] for k in ('form', 'accession', 'filing_date', 'accepted', 'report_date')},
                   words=sum(filing['all'].values()), risk_words=sum(filing['risk'].values()))
        if prior:
            row.update(prior_accession=prior['accession'], prior_report_date=prior['report_date'],
                       prior_words=sum(prior['all'].values()), prior_risk_words=sum(prior['risk'].values()),
                       cosine=cosine(filing['all'], prior['all']), jaccard=jaccard(filing['all'], prior['all']),
                       risk_cosine=cosine(filing['risk'], prior['risk']), risk_jaccard=jaccard(filing['risk'], prior['risk']))
        rows.append(row)
        for old in history[:-8]:  # a year-ago match is normally ~4 filings back; extra filings can push it further
            old.pop('all', None), old.pop('risk', None)
    return rows, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--since', default='2018-01-01', help='Earliest report date to download')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Only the first N pending companies (for a timing test)')
    args = parser.parse_args()
    out = ROOT/'data_store/filings'
    out.mkdir(parents=True, exist_ok=True)
    table, done_path = out/'similarity.csv', out/'done.txt'
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    stocks = json.loads((ROOT/'configs/broad_universe.json').read_text())['stocks']
    pending = [(t, int(s['cik'])) for t, s in stocks.items() if t not in done][:args.limit]
    print(f'{len(pending)} companies pending, {len(done)} done', flush=True)
    throttle, lock, local = SharedThrottle(.11), threading.Lock(), threading.local()
    started = time.monotonic()
    if not table.exists():
        table.write_text(','.join(COLUMNS)+'\n')

    def work(item):
        ticker, cik = item
        local.session = getattr(local, 'session', None) or requests.Session()
        try:
            rows, failures = process(ticker, cik, args.since, local.session, throttle)
        except Exception as error:
            print(f'{ticker}: skipped ({error})', flush=True)
            return
        with lock:
            with table.open('a', newline='') as handle:
                csv.DictWriter(handle, COLUMNS).writerows(rows)
            with done_path.open('a') as handle:
                handle.write(ticker+'\n')
            done.add(ticker)
            n = len(done)
        if n % 25 == 0:
            print(f'{n}/{len(stocks)} companies, {(time.monotonic()-started)/60:.0f} min'
                  + (f', {failures} failed documents' if failures else ''), flush=True)

    with ThreadPoolExecutor(args.threads) as pool:
        list(pool.map(work, pending))


if __name__ == '__main__':
    main()
