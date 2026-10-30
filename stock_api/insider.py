"""Point-in-time insider-trading signals from SEC Form 4 open-market trades.

Timing: the Form 345 data sets give only the filing date, so a filing is
treated as public at the close of the first session *after* its filing date
(`public_session`). Signals at a decision close use filings public by then.
"""
import numpy as np
import pandas as pd

INSIDER_ROLES = ('Director', 'Officer')


def clean_trades(trades, ciks):
    """Officer/director open-market trades in the given issuers, with dollar value."""
    t = trades.copy()
    t = t[t.issuer_cik.isin(ciks) & t.relationship.fillna('').str.contains('|'.join(INSIDER_ROLES))]
    t['shares'], t['price'] = pd.to_numeric(t.shares, errors='coerce'), pd.to_numeric(t.price, errors='coerce')
    t = t[(t.shares > 0) & (t.price > 0)]
    t['value'] = t.shares*t.price
    t['filing_date'] = pd.to_datetime(t.filing_date)
    t['trans_date'] = pd.to_datetime(t.trans_date, errors='coerce')
    # One row per (filing, code): a filing often reports several fills of one decision.
    keys = ['accession_number', 'issuer_cik', 'owner_cik', 'code', 'filing_date']
    return t.groupby(keys, as_index=False).agg(value=('value', 'sum'), shares=('shares', 'sum'),
                                               trans_date=('trans_date', 'min'))


def public_session(filing_dates, sessions):
    """First session strictly after each filing date."""
    sessions = pd.DatetimeIndex(sessions)
    position = sessions.searchsorted(pd.DatetimeIndex(filing_dates), side='right')
    out = pd.Series(pd.NaT, index=range(len(position)), dtype='datetime64[ns]')
    valid = position < len(sessions)
    out[valid] = sessions[position[valid]]
    return out.to_numpy()


def routine_flags(trades):
    """Cohen–Malloy–Pomorski routine insiders: traded in the same calendar month in each of the
    three preceding years. Returns 'routine', 'opportunistic', or 'unclassified' (under three
    years of history for that insider in the data) per row."""
    month = trades.trans_date.fillna(trades.filing_date).dt.to_period('M')
    seen = set(zip(trades.owner_cik, month))
    first = trades.assign(month=month).groupby('owner_cik').month.min()
    flags = []
    for owner, m in zip(trades.owner_cik, month):
        if first[owner] > m-36:
            flags.append('unclassified')
        elif all((owner, m-12*k) in seen for k in (1, 2, 3)):
            flags.append('routine')
        else:
            flags.append('opportunistic')
    return np.array(flags)


def event_panel(events, sessions, tickers, value='one', how='sum'):
    """(session × ticker) panel of event values on their public session; zero elsewhere."""
    e = events.assign(one=1.)
    panel = e.pivot_table(index='session', columns='ticker', values=value, aggfunc=how)
    return panel.reindex(index=pd.DatetimeIndex(sessions), columns=tickers).fillna(0.)


def trailing(panel, window):
    """Sum over the last `window` sessions, including the current one."""
    return panel.rolling(window, min_periods=1).sum()
