"""Is the LLM's read of an out-of-hours 8-K tradable at the next open? (broad universe, post-cutoff)

94% of 8-Ks arrive outside market hours, so their price reaction is mostly the
overnight gap. A scorer that runs within minutes could act at the next open.
Fixed before looking at results:
- Items: Haiku extractions with relevance >= 0.5, published outside 09:30–16:00
  New York time. A (stock, session) with any in-hours item is dropped.
- Signal: the summed item score (relevance × magnitude × direction) for the
  session whose open first follows the item.
- Outcomes from raw Yahoo prices: the gap (close d−1 → open d, which a 16:00
  trader cannot capture), open d → close d (tradable at the open), and the
  intraday move after controlling for the gap.
- IC per session with >= 5 stocks, Newey–West t; and the mean open→close
  return of stocks with a positive vs negative score.

    NEWS_DIR=data_store/news_broad .conda-env/bin/python scripts/news_open_check.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stock_api.evaluation import hac_mean_test, ic_summary, information_coefficient, neutralize
from stock_api.news import ExtractionCache, item_scores
from stock_api.news.aggregate import assign_sessions

ROOT = Path(__file__).resolve().parents[1]
SINCE = '2026-07-01'


def main():
    news = Path(os.environ.get('NEWS_DIR', ROOT/'data_store/news_broad'))
    out = ROOT/'reports/news-broad-2026-10-31'
    items = item_scores(ExtractionCache(news/'extractions.jsonl').frame('haiku'))
    items = items[(items.published_utc >= SINCE) & (items.relevance >= .5)].copy()
    local = pd.to_datetime(items.published_utc, utc=True).dt.tz_convert('America/New_York')
    minutes = local.dt.hour*60+local.dt.minute
    items['in_hours'] = (local.dt.weekday < 5) & (minutes >= 9*60+30) & (minutes <= 16*60)

    tickers = sorted(items.ticker.unique())
    frames = {}
    for t in tickers:
        path = ROOT/'data_store/broad'/f'{t}.csv'
        if path.exists():
            f = pd.read_csv(path, parse_dates=['Date'], index_col='Date', usecols=['Date', 'Open', 'Close'])
            frames[t] = f[~f.index.duplicated()].loc['2026-06-01':]
    opens = pd.DataFrame({t: f.Open for t, f in frames.items()}).sort_index()
    closes = pd.DataFrame({t: f.Close for t, f in frames.items()}).reindex(opens.index)
    sessions = opens.index
    items['session'] = assign_sessions(items.published_utc, sessions)
    items = items.dropna(subset=['session'])
    blocked = set(map(tuple, items.loc[items.in_hours, ['session', 'ticker']].to_numpy()))
    outside = items[~items.in_hours]
    signal = outside.groupby(['session', 'ticker']).score.sum()
    signal = signal[[key not in blocked for key in signal.index]]

    gap = np.log(opens/closes.shift(1))
    intraday = np.log(closes/opens)
    frame = signal.rename('score').rename_axis(['date', 'asset']).reset_index()
    for name, panel in (('gap', gap), ('intraday', intraday)):
        frame[name] = panel.stack().reindex(pd.MultiIndex.from_frame(frame[['date', 'asset']])).to_numpy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[(frame.gap.abs() < .5) & (frame.intraday.abs() < .5)]  # split days and bad prints
    frame['intraday_ex_gap'] = neutralize(frame, 'intraday', ['gap'])

    rows = []
    for outcome in ('gap', 'intraday', 'intraday_ex_gap'):
        ic = information_coefficient(frame, 'score', outcome, min_assets=5)
        test = ic_summary(ic, 0)
        rows.append(dict(outcome=outcome, mean_ic=float(ic.mean()), t=test['t'], p=test['p'], dates=len(ic)))
    side = frame.assign(side=np.sign(frame.score)).query('side != 0')
    daily = side.groupby(['date', 'side']).intraday.mean().unstack()
    spread = (daily[1.]-daily[-1.]).dropna()
    test = hac_mean_test(spread.to_numpy())
    info = dict(stock_sessions=len(frame), sessions=int(frame.date.nunique()),
                outside_hours_share=float(1-items.in_hours.mean()),
                positive=int((side.side > 0).sum()), negative=int((side.side < 0).sum()),
                open_to_close_long_minus_short_bp=1e4*test['mean'], t=test['t'], spread_days=len(spread),
                mean_open_to_close_positive_bp=1e4*float(side[side.side > 0].intraday.mean()),
                mean_open_to_close_negative_bp=1e4*float(side[side.side < 0].intraday.mean()))
    pd.DataFrame(rows).to_csv(out/'open_check.csv', index=False)
    (out/'open_check.json').write_text(json.dumps(info, indent=2))
    print(pd.DataFrame(rows).round(4).to_string())
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
