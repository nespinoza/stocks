import json

import numpy as np
import pandas as pd
import pytest

from stock_api.news import aggregate, extraction, lexicon, sources


def test_sessions_respect_the_close_and_weekends():
    sessions = pd.bdate_range('2026-10-19', '2026-10-30')
    stamps = ['2026-10-20T19:59:00+00:00',  # 15:59 New York: same session
              '2026-10-20T20:01:00+00:00',  # 16:01: next session
              '2026-10-24T15:00:00+00:00',  # Saturday: Monday
              '2026-10-23T21:00:00+00:00']  # Friday after close: Monday
    got = pd.DatetimeIndex(aggregate.assign_sessions(stamps, sessions)).strftime('%m-%d').tolist()
    assert got == ['10-20', '10-21', '10-26', '10-26']


def test_dedupe_collapses_syndicated_titles_per_ticker():
    docs = [dict(query_ticker='AAPL', title='Apple beats estimates - Reuters', published_utc='2026-10-20T12:00:00+00:00'),
            dict(query_ticker='AAPL', title='Apple Beats Estimates | Yahoo', published_utc='2026-10-20T11:00:00+00:00'),
            dict(query_ticker='MSFT', title='Apple beats estimates - Reuters', published_utc='2026-10-20T12:00:00+00:00')]
    kept = aggregate.dedupe(docs)
    assert len(kept) == 2 and kept[kept.query_ticker == 'AAPL'].published_utc.iloc[0].startswith('2026-10-20T11')


def test_decay_is_causal_and_sums_with_halflife_weights():
    panel = pd.DataFrame({'A': [0., 1., 0., 0., 0.]}, index=pd.bdate_range('2026-01-05', periods=5))
    out = aggregate.decay(panel, halflife=2)
    assert out.A.iloc[0] == 0
    assert out.A.iloc[1] == pytest.approx(1)
    assert out.A.iloc[3] == pytest.approx(.5)


def test_item_scores_panel_z_and_etf_mapping():
    x = pd.DataFrame(dict(ticker=['A', 'A', 'B', 'C'], relevance=[1., .2, 1., 1.], magnitude=['large', 'large', 'small', 'none'],
                          direction=[1., -1., -1., 1.], affects_volatility=[True, True, False, True],
                          published_utc=['2026-10-20T14:00:00+00:00']*4))
    scored = aggregate.item_scores(x)
    assert scored.score.tolist() == [2., 0., -.5, 0.]  # low relevance and 'none' magnitude drop out
    panel = aggregate.daily_panel(scored, pd.bdate_range('2026-10-19', periods=5), tickers=['A', 'B', 'C', 'D'])
    assert panel.loc['2026-10-20', 'A'] == 2 and panel.loc['2026-10-21'].abs().sum() == 0
    z = aggregate.cross_sectional_z(panel)
    assert z.loc['2026-10-20'].sum() == pytest.approx(0) and z.loc['2026-10-21'].abs().sum() == 0
    etf = aggregate.etf_signal(panel, {'A': 'XLK', 'B': 'XLK', 'C': 'XLF', 'D': None})
    assert etf.loc['2026-10-20', 'XLK'] == pytest.approx(.75)
    assert aggregate.variance_bump(np.array([1.]), np.array([2.]), .25)[0] == pytest.approx(1.5)


def test_extraction_cache_batches_and_skips_cached(tmp_path):
    items = [dict(id=extraction.item_key(f'd{i}', 'AAPL'), doc_id=f'd{i}', ticker='AAPL', company='Apple Inc',
                  source='gdelt', published_utc='2026-10-20T14:00:00+00:00', title=t, text='')
             for i, t in enumerate(['Apple beats estimates', 'Apple shares fall on probe', 'Banana prices'])]
    cache = extraction.ExtractionCache(tmp_path/'x.jsonl')
    assert extraction.extract(items, extraction.MockBackend(), cache, 'mock', batch_size=2, log=None) == 3
    assert extraction.extract(items, extraction.MockBackend(), cache, 'mock', log=None) == 0
    frame = extraction.ExtractionCache(tmp_path/'x.jsonl').frame('mock')
    assert frame.set_index('doc_id').direction.to_dict() == {'d0': 1., 'd1': -1., 'd2': 0.}
    assert set(frame.prompt_version) == {extraction.PROMPT_VERSION}


def test_validate_drops_unknown_ids_and_incomplete_rows():
    good = dict(id='a', relevance=1, event_type='other', direction=0, surprise=0, magnitude='none',
                affects_volatility=False, confidence=.5)
    out = extraction.validate({'items': [good, dict(good, id='zzz'), {'id': 'b'}]}, {'a', 'b'})
    assert list(out) == ['a']


def test_claude_code_backend_parses_envelope(monkeypatch):
    class Done:
        returncode, stderr = 0, ''
        stdout = json.dumps(dict(is_error=False, structured_output={'items': []}, total_cost_usd=.001,
                                 modelUsage={'claude-haiku-5-5': {}}, usage={}))
    captured = {}

    def fake_run(command, **kwargs):
        captured['command'], captured['input'] = command, kwargs['input']
        return Done()
    monkeypatch.setattr(extraction.subprocess, 'run', fake_run)
    item = dict(id='x', ticker='AAPL', company='Apple Inc', source='gdelt', published_utc='2026-10-20T00:00:00+00:00',
                title='Apple beats', text='')
    result, meta = extraction.ClaudeCodeBackend()([item])
    assert result == {'items': []} and meta['model'] == 'claude-haiku-5-5'
    assert '--json-schema' in captured['command'] and captured['command'][captured['command'].index('--tools')+1] == ''
    assert 'Apple beats' in captured['input']


def test_lexicon_tone_with_negation():
    positive, negative = {'STRONG', 'BEAT'}, {'LOSS', 'WEAK'}
    assert lexicon.tone('Strong quarter, beat estimates', positive, negative)['lm_tone'] == 1
    assert lexicon.tone('Not strong; weak loss', positive, negative)['lm_tone'] == -1
    assert lexicon.tone('No signal words', positive, negative)['lm_tone'] == 0


def test_source_documents_and_query():
    assert sources.gdelt_query(['Microsoft']) == 'Microsoft sourcelang:english'
    assert sources.gdelt_query(['Apple Inc', 'iPhone']) == '("Apple Inc" OR iPhone) sourcelang:english'
    docs = sources.gdelt_documents('MSFT', [dict(url='u', title='A &amp; B', seendate='20261020T143000Z')])
    assert docs[0]['published_utc'] == '2026-10-20T14:30:00+00:00' and docs[0]['title'] == 'A & B'
    filing = dict(accession='0000789019-26-000001', accepted='2026-10-20T16:05:12.000Z', items='2.02,9.01', primary='x.htm')
    doc = sources.eight_k_document('MSFT', 789019, filing)
    assert doc['published_utc'] == '2026-10-20T16:05:12+00:00' and 'Results of operations' in doc['title']
    assert sources.strip_html('<p>Hi&nbsp;<b>there</b></p><script>x</script>') == 'Hi there'
