"""Timestamped text sources: GDELT DOC 2.0 headlines and SEC EDGAR 8-K filings.

Every document is a dict with a stable doc_id, the source's own timestamp
(`published_utc`, the moment the text became observable) and our
`ingested_utc`. Downstream code must only use `published_utc` for timing.
"""
import hashlib
import html
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

GDELT_URL = 'https://api.gdeltproject.org/api/v2/doc/doc'
GDELT_SPACING = 6.  # GDELT asks for at most one request every 5 seconds
SEC_SPACING = .15   # SEC allows 10 requests per second

# 8-K item numbers (Form 8-K General Instructions B).
EIGHT_K_ITEMS = {
    '1.01': 'Entry into a material definitive agreement', '1.02': 'Termination of a material definitive agreement',
    '1.03': 'Bankruptcy or receivership', '1.05': 'Material cybersecurity incident',
    '2.01': 'Completion of acquisition or disposition of assets', '2.02': 'Results of operations and financial condition',
    '2.03': 'Creation of a direct financial obligation', '2.04': 'Triggering events that accelerate an obligation',
    '2.05': 'Costs associated with exit or disposal activities', '2.06': 'Material impairments',
    '3.01': 'Notice of delisting or failure to satisfy listing rule', '3.02': 'Unregistered sales of equity securities',
    '3.03': 'Material modification to rights of security holders', '4.01': "Changes in registrant's certifying accountant",
    '4.02': 'Non-reliance on previously issued financial statements', '5.01': 'Changes in control of registrant',
    '5.02': 'Departure or appointment of directors or officers', '5.03': 'Amendments to articles or bylaws',
    '5.07': 'Submission of matters to a vote of security holders', '7.01': 'Regulation FD disclosure',
    '8.01': 'Other events', '9.01': 'Financial statements and exhibits'}


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def doc_id(source, key):
    return hashlib.sha256(f'{source}|{key}'.encode()).hexdigest()[:20]


class Throttle:
    def __init__(self, spacing):
        self.spacing, self.last = spacing, 0.

    def wait(self):
        delay = self.last+self.spacing-time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.last = time.monotonic()


def gdelt_query(phrases):
    """OR of quoted phrases, English sources only."""
    terms = [f'"{p}"' if ' ' in p or not p.isalnum() else p for p in phrases]
    body = terms[0] if len(terms) == 1 else '('+' OR '.join(terms)+')'
    return f'{body} sourcelang:english'


def fetch_gdelt(query, start, end, max_records=75, throttle=None, session=None, retries=4):
    """Article list (title, url, domain, seendate) for one query and UTC window.

    The DOC API only searches a rolling ~3-month window. `seendate` is when
    GDELT first saw the article; it is the observable time we use.
    """
    throttle = throttle or Throttle(GDELT_SPACING)
    session = session or requests.Session()
    params = dict(query=query, mode='artlist', format='json', maxrecords=max_records, sort='HybridRel',
                  startdatetime=start.strftime('%Y%m%d%H%M%S'), enddatetime=end.strftime('%Y%m%d%H%M%S'))
    for attempt in range(retries):
        throttle.wait()
        response = session.get(GDELT_URL, params=params, timeout=60)
        text = response.text.strip()
        if response.status_code == 200 and text.startswith('{'):
            return json.loads(text).get('articles', []) if text != '{}' else []
        if 'limit requests' in text or response.status_code == 429:
            time.sleep(30*(attempt+1))
            continue
        raise RuntimeError(f'GDELT error {response.status_code}: {text[:200]}')
    raise RuntimeError('GDELT rate limit persisted')


def gdelt_documents(ticker, articles, ingested=None):
    ingested = ingested or now_utc()
    docs = []
    for a in articles:
        seen = datetime.strptime(a['seendate'], '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
        docs.append(dict(doc_id=doc_id('gdelt', f"{a['url']}|{ticker}"), source='gdelt', query_ticker=ticker,
                         published_utc=seen.isoformat(timespec='seconds'), ingested_utc=ingested,
                         title=html.unescape(a.get('title', '')).strip(), url=a['url'], domain=a.get('domain', ''),
                         language=a.get('language', ''), text=''))
    return docs


def sec_headers():
    agent = os.environ.get('SEC_USER_AGENT')
    if not agent:
        raise RuntimeError('Set SEC_USER_AGENT to "<name> <contact email>" as SEC requires')
    return {'User-Agent': agent, 'Accept-Encoding': 'gzip, deflate'}


def fetch_8k_index(cik, session=None, throttle=None):
    """Recent 8-K filings of one company from the EDGAR submissions API."""
    throttle = throttle or Throttle(SEC_SPACING)
    session = session or requests.Session()
    throttle.wait()
    response = session.get(f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json', headers=sec_headers(), timeout=60)
    response.raise_for_status()
    recent = response.json()['filings']['recent']
    rows = []
    for i, form in enumerate(recent['form']):
        if form != '8-K':
            continue
        rows.append(dict(accession=recent['accessionNumber'][i], accepted=recent['acceptanceDateTime'][i],
                         items=recent['items'][i], primary=recent['primaryDocument'][i]))
    return rows


def strip_html(raw, limit=4000):
    text = re.sub(r'(?is)<(script|style|ix:header).*?</\1>', ' ', raw)  # ix:header = hidden inline-XBRL facts
    text = re.sub(r'(?s)<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', html.unescape(text)).strip()[:limit]


def eight_k_document(ticker, cik, filing, body='', ingested=None):
    """One 8-K as a document. acceptanceDateTime is EDGAR's public dissemination time (UTC)."""
    accepted = datetime.fromisoformat(filing['accepted'].replace('Z', '+00:00'))
    items = [i.strip() for i in filing['items'].split(',') if i.strip()]
    described = '; '.join(f'Item {i} {EIGHT_K_ITEMS.get(i, "")}'.strip() for i in items)
    accession = filing['accession'].replace('-', '')
    return dict(doc_id=doc_id('edgar', filing['accession']), source='edgar-8k', query_ticker=ticker,
                published_utc=accepted.astimezone(timezone.utc).isoformat(timespec='seconds'),
                ingested_utc=ingested or now_utc(), title=f'8-K: {described}', items=items,
                url=f'https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{filing["primary"]}',
                domain='sec.gov', language='english', text=body)


def fetch_8k_body(url, session=None, throttle=None, limit=4000):
    """Text of the filing's press release (Exhibit 99.x) when present, else of the 8-K itself.

    The 8-K cover is mostly boilerplate; the news is usually in Exhibit 99.1.
    """
    throttle = throttle or Throttle(SEC_SPACING)
    session = session or requests.Session()
    folder = url.rsplit('/', 1)[0]
    throttle.wait()
    listing = session.get(f'{folder}/index.json', headers=sec_headers(), timeout=60)
    listing.raise_for_status()
    names = [item['name'] for item in listing.json()['directory']['item']]
    exhibits = [n for n in names if re.search(r'ex(hibit)?[-_]?99', n, re.I) and n.lower().endswith(('.htm', '.html', '.txt'))]
    exhibits.sort(key=lambda n: (not re.search(r'99[-_.]?0?1(?!\d)', n), n))  # Exhibit 99.1 first
    target = f'{folder}/{exhibits[0]}' if exhibits else url
    throttle.wait()
    response = session.get(target, headers=sec_headers(), timeout=60)
    response.raise_for_status()
    text = strip_html(response.text, 50000)
    if not exhibits:  # skip the cover-page boilerplate up to the first reported item
        match = re.search(r'Item\s+\d\.\d\d\b', text[500:])
        text = text[500+match.start():] if match else text
    return text[:limit]


def weekly_windows(start, end):
    cursor = start
    while cursor < end:
        yield cursor, min(cursor+timedelta(days=7), end)
        cursor += timedelta(days=7)


class DocumentStore:
    """Append-only JSONL store, de-duplicated by doc_id."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ids = set()
        if self.path.exists():
            with self.path.open() as handle:
                self.ids = {json.loads(line)['doc_id'] for line in handle if line.strip()}

    def add(self, docs):
        new = [d for d in docs if d['doc_id'] not in self.ids]
        with self.path.open('a') as handle:
            for d in new:
                handle.write(json.dumps(d, sort_keys=True)+'\n')
                self.ids.add(d['doc_id'])
        return len(new)

    def load(self):
        if not self.path.exists():
            return []
        with self.path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
