"""Daily abnormal returns around Form 4 filing dates (diagnostic for run_insider_study.py).

Prints the mean abnormal return (vs the equal-weighted eligible universe) on each
session relative to the filing date, and the post-filing drift split by liquidity.

    OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/insider_timing_check.py
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from run_insider_study import load_prices, ROOT, START
from stock_api import insider
from stock_api.evaluation import hac_mean_test
config = json.loads((ROOT/'configs/broad_universe.json').read_text())
adj, close, volume = load_prices(config); adj = adj.loc[START:]; close, volume = close.reindex_like(adj), volume.reindex_like(adj)
sessions, tickers = adj.index, list(adj.columns)
returns = adj.pct_change(fill_method=None).clip(-.5, 1.)
dv = (close*volume).rolling(63, min_periods=40).median()
eligible = (close >= 3) & (dv >= 1e6) & adj.notna()
cik = {s['cik']: t for t, s in config['stocks'].items() if t in tickers}
raw = pd.read_csv(ROOT/'data_store/insider/trades.csv.gz', dtype={'relationship': str, 'title': str})
t = insider.clean_trades(raw, set(cik)); t['ticker'] = t.issuer_cik.map(cik)
u = returns.where(eligible.shift(1, fill_value=False)).mean(axis=1)
ab = returns.sub(u, axis=0)
for code in ('P', 'S'):
    e = t[t.code == code].drop_duplicates(['ticker', 'filing_date'])
    pos = sessions.searchsorted(pd.DatetimeIndex(e.filing_date))  # filing-date session (or next if non-trading)
    tr = sessions.searchsorted(pd.DatetimeIndex(e.trans_date.fillna(e.filing_date)))
    col = ab.columns.get_indexer(e.ticker); A = ab.to_numpy()
    month = pd.DatetimeIndex(e.filing_date).to_period('M')
    def stat(offsets, base):
        idx = np.stack([base+k for k in offsets]); ok = (idx.min(0) >= 0) & (idx.max(0) < len(sessions))
        v = np.nansum(A[idx[:, ok], col[ok]], axis=0)
        m = pd.Series(v).groupby(np.asarray(month)[ok]).mean()
        r = hac_mean_test(m.to_numpy()); return f'{100*np.mean(v):+.2f}% (monthly {100*r["mean"]:+.2f}, t {r["t"]:.1f})'
    print(code, 'trade day..filing day', stat(range(0, 1), tr))
    print(code, 'filing day (day 0)', stat([0], pos), '| day +1', stat([1], pos), '| day +2..+6', stat(range(2, 7), pos), '| +2..+22', stat(range(2, 23), pos))
print('--- purchases by day after filing, and by liquidity (63d median $ volume at filing)')
e = t[t.code == 'P'].drop_duplicates(['ticker', 'filing_date'])
pos = sessions.searchsorted(pd.DatetimeIndex(e.filing_date)); col = ab.columns.get_indexer(e.ticker); A = ab.to_numpy()
ok = pos+25 < len(sessions); pos, col = pos[ok], col[ok]
liq = dv.to_numpy()[np.maximum(pos-1, 0), col]
for k in range(-2, 6):
    print(f'day {k:+d}: {100*np.nanmean(A[pos+k, col]):+.3f}%')
hi = liq >= np.nanmedian(liq)
for name, m in (('liquid half', hi), ('illiquid half', ~hi)):
    car = np.nansum(np.stack([A[pos+k, col] for k in range(2, 23)])[:, m], axis=0)
    car5 = np.nansum(np.stack([A[pos+k, col] for k in range(2, 7)])[:, m], axis=0)
    print(name, f'median $vol {np.nanmedian(liq[m])/1e6:.1f}M', f'day+1 {100*np.nanmean(A[pos[m]+1, col[m]]):+.2f}%',
          f'+2..+6 {100*car5.mean():+.2f}%', f'+2..+22 {100*car.mean():+.2f}%')
