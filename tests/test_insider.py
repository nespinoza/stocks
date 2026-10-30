import pandas as pd

from stock_api import insider


def test_public_session_is_strictly_after_filing():
    sessions = pd.bdate_range('2026-10-19', '2026-10-30')
    got = insider.public_session(pd.to_datetime(['2026-10-20', '2026-10-23', '2026-10-24']), sessions)
    assert pd.DatetimeIndex(got).strftime('%m-%d').tolist() == ['10-21', '10-26', '10-26']


def test_clean_trades_keeps_insiders_and_merges_fills():
    raw = pd.DataFrame(dict(accession_number=['a', 'a', 'b', 'c'], issuer_cik=[1, 1, 1, 2], owner_cik=[9, 9, 8, 7],
                            code=['P', 'P', 'P', 'P'], shares=['100', '50', '10', '10'], price=['10', '12', '5', '5'],
                            relationship=['Director', 'Director', 'TenPercentOwner', 'Officer'],
                            filing_date=['2026-10-20']*4, trans_date=['2026-10-19', '2026-10-16', '2026-10-19', '2026-10-19']))
    out = insider.clean_trades(raw, {1})
    assert len(out) == 1 and out.value.iloc[0] == 1600 and str(out.trans_date.iloc[0].date()) == '2026-10-16'


def test_routine_flags():
    rows = [(1, f'{y}-03-10') for y in (2020, 2021, 2022, 2023)] + [(2, '2020-01-05'), (2, '2023-06-01'), (3, '2023-03-01')]
    t = pd.DataFrame(rows, columns=['owner_cik', 'trans_date'])
    t['trans_date'] = pd.to_datetime(t.trans_date)
    t['filing_date'] = t.trans_date
    flags = insider.routine_flags(t).tolist()
    assert flags[3] == 'routine' and flags[:3] == ['unclassified']*3
    assert flags[5] == 'opportunistic' and flags[6] == 'unclassified'
