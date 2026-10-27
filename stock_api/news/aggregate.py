"""Point-in-time aggregation of extracted news into per-stock and per-ETF signals.

Timing rule: an item published at or before the 16:00 New York close of
session d belongs to session d; later items belong to the next session. A
signal "at d" therefore uses only text observable by the close of d, and the
engine trades it at the close of d+1 (lag 1).
"""
import re

import numpy as np
import pandas as pd

MAGNITUDE_WEIGHT = {'none': 0., 'small': .5, 'medium': 1., 'large': 2.}


def assign_sessions(published_utc, sessions):
    """Map UTC timestamps to the trading session whose close first follows them."""
    sessions = pd.DatetimeIndex(sessions).normalize()
    local = pd.to_datetime(pd.Series(published_utc), utc=True).dt.tz_convert('America/New_York')
    day = local.dt.tz_localize(None).dt.normalize()
    after_close = local.dt.hour*60+local.dt.minute > 16*60
    day = day+pd.to_timedelta(after_close.astype(int), unit='D')
    position = sessions.searchsorted(day.to_numpy())
    out = pd.Series(pd.NaT, index=day.index, dtype='datetime64[ns]')
    valid = position < len(sessions)
    out[valid] = sessions[position[valid]]
    return out.to_numpy()


def normalize_title(title):
    """Syndicated copies differ by outlet suffixes and punctuation; this collapses them."""
    title = re.sub(r'\s+[-|–]\s+[^-|–]+$', '', title.lower())
    return re.sub(r'[^a-z0-9 ]', '', title).strip()


def dedupe(docs):
    """Keep the earliest copy of each (ticker, normalized title)."""
    frame = pd.DataFrame(docs)
    frame['norm'] = frame.title.map(normalize_title)
    frame = frame.sort_values('published_utc')
    return frame.drop_duplicates(['query_ticker', 'norm']).drop(columns='norm')


def item_scores(extractions, min_relevance=.5):
    """Signed score and volatility intensity per extracted (doc, ticker) item."""
    x = extractions.copy()
    weight = x.relevance.where(x.relevance >= min_relevance, 0.)*x.magnitude.map(MAGNITUDE_WEIGHT).fillna(0.)
    x['score'] = weight*x.direction
    x['vol_intensity'] = weight*x.affects_volatility.astype(float)
    return x


def daily_panel(items, sessions, value='score', tickers=None):
    """Sum of item values per (session, ticker); missing = 0 (no news)."""
    items = items.assign(session=assign_sessions(items.published_utc, sessions)).dropna(subset=['session'])
    panel = items.pivot_table(index='session', columns='ticker', values=value, aggfunc='sum')
    panel = panel.reindex(pd.DatetimeIndex(sessions).normalize())
    if tickers is not None:
        panel = panel.reindex(columns=tickers)
    return panel.fillna(0.)


def decay(panel, halflife):
    """Exponentially decayed sum over sessions (halflife in sessions), causal."""
    return panel.ewm(halflife=halflife, adjust=False).mean()*(1/(1-.5**(1/halflife)))


def cross_sectional_z(panel, clip=3.):
    z = panel.sub(panel.mean(axis=1), axis=0).div(panel.std(axis=1).replace(0, np.nan), axis=0)
    return z.clip(-clip, clip).fillna(0.)


def etf_signal(stock_signal, stock_to_etf):
    """Equal-weighted mean of constituent signals per sector ETF."""
    mapping = pd.Series(stock_to_etf).dropna()
    columns = [t for t in stock_signal.columns if t in mapping.index]
    return stock_signal[columns].T.groupby(mapping[columns]).mean().T


def variance_bump(variance, intensity, beta):
    """Scale a variance forecast by (1 + beta · event intensity), floored at the base forecast."""
    return variance*(1+np.maximum(beta*intensity, 0.))
