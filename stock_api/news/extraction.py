"""Structured LLM extraction of (document, ticker) news events, with an immutable cache.

Backends share one prompt and schema, so cached results stay comparable:
- ClaudeCodeBackend: headless `claude -p` on the user's Claude plan (testing).
- AnthropicBackend: the billed API via the `anthropic` SDK (production/paper trading).
- MockBackend: deterministic, for tests.

Each cached row records prompt version, model, backend and timestamp, so
scores from different prompts or models are never silently mixed.
"""
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROMPT_VERSION = 'news-extract-v1'
EVENT_TYPES = ['earnings', 'guidance', 'analyst', 'mna', 'product', 'legal_regulatory', 'management', 'capital',
               'operations', 'macro_sector', 'market_move', 'other']

ITEM_SCHEMA = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'relevance': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'event_type': {'type': 'string', 'enum': EVENT_TYPES},
        'direction': {'type': 'number', 'minimum': -1, 'maximum': 1},
        'surprise': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'magnitude': {'type': 'string', 'enum': ['none', 'small', 'medium', 'large']},
        'affects_volatility': {'type': 'boolean'},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    },
    'required': ['id', 'relevance', 'event_type', 'direction', 'surprise', 'magnitude', 'affects_volatility', 'confidence'],
    'additionalProperties': False,
}
SCHEMA = {'type': 'object', 'properties': {'items': {'type': 'array', 'items': ITEM_SCHEMA}},
          'required': ['items'], 'additionalProperties': False}

SYSTEM = f"""You read financial news headlines and filings and extract structured signals for one named company per item.
Judge only from the text given; do not use knowledge of what happened afterwards.
For each item return:
- relevance (0-1): is the text materially about this company (not a namesake, not a passing mention)?
- event_type: one of {', '.join(EVENT_TYPES)}. Use market_move when the text only reports a price move.
- direction (-1 to 1): expected effect on the company's stock relative to the market, from the text alone.
- surprise (0-1): how new and unexpected the information looks (routine or already-known items are low).
- magnitude: none, small, medium or large expected price impact.
- affects_volatility: true if the event plausibly raises near-term volatility (earnings, litigation, M&A, guidance changes, crises).
- confidence (0-1): your confidence in direction.
Return every item id exactly once. Prompt version {PROMPT_VERSION}."""


def item_key(doc_id, ticker):
    return f'{doc_id}:{ticker}'


def format_batch(items):
    """items: list of dicts with id, ticker, company, source, published_utc, title, text."""
    lines = []
    for item in items:
        body = f"\n  text: {item['text'][:1500]}" if item.get('text') else ''
        lines.append(f"- id: {item['id']}\n  company: {item['company']} ({item['ticker']})\n  source: {item['source']}"
                     f"\n  date: {item['published_utc'][:10]}\n  headline: {item['title']}{body}")
    return 'Extract signals for these items:\n'+'\n'.join(lines)


def validate(result, ids):
    out = {}
    for row in result.get('items', []):
        if row.get('id') in ids and all(k in row for k in ITEM_SCHEMA['required']):
            out[row['id']] = row
    return out


# How `claude -p` reports an exhausted plan allowance (wording varies by version).
LIMIT_PATTERN = re.compile(r'usage limit|limit reached|hit your limit|rate.?limit|resets? (at|in)', re.I)


class UsageLimitError(RuntimeError):
    """The plan's usage limit is exhausted; re-run after it resets (cached batches are kept)."""


class MockBackend:
    """Deterministic keyword scorer used in tests (no network)."""
    name, model = 'mock', 'mock'

    def __call__(self, items):
        rows = []
        for item in items:
            title = item['title'].lower()
            direction = (('beat' in title or 'jump' in title or 'raise' in title)
                         - ('miss' in title or 'fall' in title or 'cut' in title or 'probe' in title))
            rows.append(dict(id=item['id'], relevance=1. if item['ticker'].lower() in title or item['company'].split()[0].lower() in title else .2,
                             event_type='other', direction=float(direction), surprise=.5, magnitude='small',
                             affects_volatility='probe' in title, confidence=.5))
        return {'items': rows}, {'model': self.model}


class ClaudeCodeBackend:
    """Headless Claude Code (`claude -p`), billed to the user's Claude plan. For testing volumes only."""
    name = 'claude-code'

    def __init__(self, model='haiku', timeout=600, executable='claude'):
        self.model, self.timeout, self.executable = model, timeout, executable

    def __call__(self, items):
        command = [self.executable, '-p', '--model', self.model, '--output-format', 'json', '--tools', '',
                   '--system-prompt', SYSTEM, '--json-schema', json.dumps(SCHEMA), '--no-session-persistence',
                   '--setting-sources', '', '--strict-mcp-config', '--disable-slash-commands']
        done = subprocess.run(command, input=format_batch(items), capture_output=True, text=True, timeout=self.timeout)
        if done.returncode != 0:
            message = f'claude -p failed ({done.returncode}): {done.stderr[:500] or done.stdout[:500]}'
            raise (UsageLimitError if LIMIT_PATTERN.search(done.stderr+done.stdout) else RuntimeError)(message)
        envelope = json.loads(done.stdout)
        if envelope.get('is_error') or 'structured_output' not in envelope:
            message = f"claude -p returned no structured output: {str(envelope.get('result'))[:300]}"
            raise (UsageLimitError if LIMIT_PATTERN.search(str(envelope.get('result'))) else RuntimeError)(message)
        models = list(envelope.get('modelUsage', {}))
        return envelope['structured_output'], {'model': models[0] if models else self.model,
                                               'usage': envelope.get('usage', {}),
                                               'cost_usd_equivalent': envelope.get('total_cost_usd')}


class AnthropicBackend:
    """Billed Claude API with structured outputs. Use for paper trading and large backfills."""
    name = 'anthropic-api'

    def __init__(self, model='claude-haiku-5-5', effort='low', max_tokens=16000):
        import anthropic
        self.client, self.model, self.effort, self.max_tokens = anthropic.Anthropic(), model, effort, max_tokens

    def __call__(self, items):
        response = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
            messages=[{'role': 'user', 'content': format_batch(items)}],
            output_config={'effort': self.effort, 'format': {'type': 'json_schema', 'schema': SCHEMA}})
        if response.stop_reason in ('refusal', 'max_tokens'):
            raise RuntimeError(f'API stop_reason {response.stop_reason} (request {response._request_id})')
        text = next(block.text for block in response.content if block.type == 'text')
        return json.loads(text), {'model': response.model, 'usage': response.usage.to_dict(),
                                  'request_id': response._request_id}


class ExtractionCache:
    """Append-only JSONL of extraction results keyed by (prompt version, model, doc, ticker)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        if self.path.exists():
            with self.path.open() as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.rows[(row['prompt_version'], row['model_alias'], row['key'])] = row

    def has(self, model_alias, key):
        return (PROMPT_VERSION, model_alias, key) in self.rows

    def add(self, rows):
        with self.path.open('a') as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True)+'\n')
                self.rows[(row['prompt_version'], row['model_alias'], row['key'])] = row

    def frame(self, model_alias=None, prompt_version=PROMPT_VERSION):
        import pandas as pd
        rows = [r for (p, m, _), r in self.rows.items()
                if p == prompt_version and (model_alias is None or m == model_alias)]
        return pd.DataFrame(rows)


def call_with_retry(backend, batch, retries=2, wait=30., log=print):
    """Retry transient backend failures; a usage-limit error is raised at once."""
    for attempt in range(retries+1):
        try:
            return backend(batch)
        except UsageLimitError:
            raise
        except Exception as error:
            if attempt == retries:
                raise
            if log:
                log(f'batch failed ({str(error)[:200]}); retrying in {wait*(attempt+1):.0f} s')
            time.sleep(wait*(attempt+1))


def extract(items, backend, cache, model_alias, batch_size=25, log=print):
    """Score uncached items in batches. Items need id = item_key(doc_id, ticker). Returns new row count.

    Each batch is cached as soon as it returns, so an interrupted run (e.g. a
    UsageLimitError) resumes where it stopped when called again.
    """
    todo = [i for i in items if not cache.has(model_alias, i['id'])]
    added = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start+batch_size]
        result, meta = call_with_retry(backend, batch, log=log)
        valid = validate(result, {i['id'] for i in batch})
        stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
        prompt_hash = hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]
        rows = [dict(key=i['id'], doc_id=i['doc_id'], ticker=i['ticker'], published_utc=i['published_utc'],
                     source=i['source'], prompt_version=PROMPT_VERSION, prompt_sha=prompt_hash, model_alias=model_alias,
                     model=meta.get('model'), backend=backend.name, extracted_utc=stamp,
                     **{k: v for k, v in valid[i['id']].items() if k != 'id'})
                for i in batch if i['id'] in valid]
        cache.add(rows)
        added += len(rows)
        if log:
            cost = meta.get('cost_usd_equivalent')
            log(f'batch {start//batch_size+1}/{-(-len(todo)//batch_size)}: {len(rows)}/{len(batch)} valid'+(f', ≈${cost:.4f} API-equivalent' if cost else ''))
    return added
