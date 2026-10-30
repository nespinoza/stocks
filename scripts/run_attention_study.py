"""Does Wikipedia attention predict small/mid-cap returns or volatility? 2015–2026.

Inputs: configs/broad_universe.json, Yahoo prices in data_store/broad/,
pageviews from collect_pageviews.py. Fixed before looking at results:

- Views are per UTC day, so a signal at session s uses views through day s−1.
- shock = log(1 + mean views over the last 5 days) − log(1 + median views over
  days −61..−6): abnormal attention (in the spirit of Da, Engelberg & Gao 2011).
- level = log(1 + median views over the last 60 days): how well-known the stock is.
- Eligible: price >= $3, 63-day median dollar volume >= $1M, a mapped article and
  a 60-day median of at least 5 views.
- IC of shock against returns from the close of s (the views are known before
  it) over 1 and 5 sessions, and from the close of s+1 over 5 and 21
  sessions; raw and after neutralizing the trailing 5-day return, its absolute
  value, the trailing 21-day return, log dollar volume and abnormal dollar
  volume (5-day vs 63-day), the obvious confounds of attention.
- IC of level against 21-day returns after neutralizing log dollar volume (is
  being little-known rewarded beyond being small?).
- Volatility: log(next-5-day realized variance / trailing 63-day variance) on the
  shock with the same controls, standard errors clustered by date.
- Portfolios (equal weight, lag 1, costs 0 and 20 bp one-way): weekly top and bottom
  shock quintiles; monthly bottom and top level quintiles; benchmark = all
  eligible mapped stocks.

Same survivorship caveat as the insider study (today's listed companies).

    OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_attention_study.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_insider_study import load_prices
from run_news_study import clustered_ols
from stock_api.engine import month_end_dates, performance, run_backtest
from stock_api.evaluation import (deflated_sharpe, factor_regression, hac_mean_test, ic_summary,
                                  information_coefficient, load_french_factors, neutralize)

ROOT = Path(__file__).resolve().parents[1]
START = '2015-10-01'  # two months after the first pageview day
# Sub-periods. The universe is selected on today's size, which depends most on recent returns, so its
# look-ahead bias is, if anything, strongest in the latest window; these windows test stability, not bias.
PERIODS = [('2015-10-01', '2018-12-31'), ('2019-01-01', '2022-12-31'), ('2023-01-01', '2026-12-31')]
CONTROLS = ['trail_5d', 'abs_trail_5d', 'trail_21d', 'log_dollar_volume', 'abnormal_volume']


def attention_features(views, sessions):
    """shock and level at each session from views through the previous UTC day."""
    daily = views.asfreq('D')
    recent = np.log1p(daily.rolling(5, min_periods=4).mean())
    base = np.log1p(daily.shift(5).rolling(56, min_periods=40).median())
    level = daily.rolling(60, min_periods=40).median()
    lagged = lambda frame: frame.shift(1).reindex(sessions)  # value at day s uses days <= s−1
    return lagged(recent-base), lagged(np.log1p(level)), lagged(level)


def week_end_dates(index):
    index = pd.DatetimeIndex(index)
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(index.to_period('W-FRI')).max().to_numpy())


def quintile_decisions(signal, eligible, dates, top):
    out, sizes = {}, []
    for date in dates:
        values = signal.loc[date].where(eligible.loc[date]).dropna()
        cut = values.quantile(.8 if top else .2)
        pick = values >= cut if top else values <= cut
        sizes.append(int(pick.sum()))
        out[date] = (pick/pick.sum()).reindex(signal.columns).fillna(0.)
    return out, float(np.mean(sizes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'reports/wikipedia-attention-2026-10-30'))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'configs/broad_universe.json').read_text())
    views = pd.read_csv(ROOT/'data_store/attention/pageviews.csv.gz', index_col=0, parse_dates=True)
    adj, close, volume = load_prices({'stocks': {t: None for t in views.columns}})
    adj = adj.loc['2015-01-01':]
    close, volume = close.reindex_like(adj), volume.reindex_like(adj)
    sessions, tickers = adj.index, list(adj.columns)
    views = views[tickers]
    returns = adj.pct_change(fill_method=None).clip(-.5, 1.)
    dollar = close*volume
    dollar_volume = dollar.rolling(63, min_periods=40).median()
    shock, level, raw_level = attention_features(views, sessions)
    eligible = (close >= 3) & (dollar_volume >= 1e6) & adj.notna() & (raw_level >= 5) & shock.notna()

    logs = np.log(adj)
    panels = {'shock': shock.where(eligible), 'level': level.where(eligible),
              'ret_1d_lag0': logs.shift(-1)-logs, 'ret_5d_lag0': logs.shift(-5)-logs,
              'ret_5d_lag1': logs.shift(-6)-logs.shift(-1), 'ret_21d_lag1': logs.shift(-22)-logs.shift(-1),
              'trail_5d': logs-logs.shift(5), 'trail_21d': logs-logs.shift(21),
              'log_dollar_volume': np.log(dollar_volume),
              'abnormal_volume': np.log(dollar.rolling(5).mean()/dollar_volume)}
    panels['abs_trail_5d'] = panels['trail_5d'].abs()
    sample = sessions[(sessions >= START) & (sessions < sessions[-25])]
    frame = pd.DataFrame({k: v.loc[sample].stack() for k, v in panels.items()}).rename_axis(['date', 'asset']).reset_index()
    frame = frame.dropna(subset=['shock', *CONTROLS])
    frame['shock_neutral'] = neutralize(frame, 'shock', CONTROLS)
    frame['level_neutral'] = neutralize(frame, 'level', ['log_dollar_volume'])
    counts = dict(stocks=len(tickers), eligible_per_date=float(frame.groupby('date').size().mean()),
                  sessions=int(frame.date.nunique()), first=str(frame.date.min().date()), last=str(frame.date.max().date()))

    ic_rows = []
    for signal, outcomes in (('shock', ['ret_1d_lag0', 'ret_5d_lag0', 'ret_5d_lag1', 'ret_21d_lag1']),
                             ('shock_neutral', ['ret_1d_lag0', 'ret_5d_lag0', 'ret_5d_lag1', 'ret_21d_lag1']),
                             ('level', ['ret_21d_lag1']), ('level_neutral', ['ret_21d_lag1'])):
        for outcome in outcomes:
            h = int(outcome.split('_')[1][:-1])
            ic = information_coefficient(frame, signal, outcome, min_assets=50)
            test = ic_summary(ic.iloc[::h])  # non-overlapping dates for inference
            ic_rows.append(dict(signal=signal, outcome=outcome, mean_ic=float(ic.mean()), t=test['t'], p=test['p'],
                                independent_dates=test['dates']))
    ics = pd.DataFrame(ic_rows)
    ics.to_csv(out/'ic.csv', index=False)

    # Volatility channel.
    squared = np.log(adj).diff()**2
    future = squared.rolling(5).sum().shift(-6)
    baseline = squared.rolling(63).mean().shift(1)*5
    frame['vol_ratio'] = pd.DataFrame({'v': np.log(future/baseline).loc[sample].stack()}).v.reindex(
        pd.MultiIndex.from_frame(frame[['date', 'asset']])).to_numpy()
    vol_data = frame.dropna(subset=['vol_ratio'])
    vol_data = vol_data[np.isfinite(vol_data.vol_ratio)]
    vol_rows = []
    for spec in (['shock'], CONTROLS, ['shock', *CONTROLS]):
        beta, t, r2 = clustered_ols(vol_data.vol_ratio.to_numpy(), vol_data[spec].to_numpy(), vol_data.date.to_numpy())
        vol_rows.append(dict(spec='+'.join(spec), n=len(vol_data), r2=r2, b_shock=beta[1] if spec[0] == 'shock' else np.nan,
                             t_shock=t[1] if spec[0] == 'shock' else np.nan))
    pd.DataFrame(vol_rows).to_csv(out/'volatility_channel.csv', index=False)

    # Portfolios.
    weekly = week_end_dates(sessions)
    weekly = weekly[(weekly >= START) & (weekly < sessions[-2])]
    monthly = month_end_dates(sessions)
    monthly = monthly[(monthly >= START) & (monthly < sessions[-2])]
    designs = {'shock_top_weekly': (shock, weekly, True), 'shock_bottom_weekly': (shock, weekly, False),
               'level_bottom_monthly': (level, monthly, False), 'level_top_monthly': (level, monthly, True)}
    factors = load_french_factors('daily', cache_dir=str(ROOT/'data_store/french'))
    rows, nets = [], {}
    for cost in (0., 20.):
        bench_w = {d: (eligible.loc[d]/eligible.loc[d].sum()).astype(float) for d in monthly}
        bench = run_backtest(returns, bench_w, cost)[0].net
        nets[('benchmark', cost)] = bench
        rows.append(dict(portfolio='eligible_ew', cost_bps=cost,
                         **{k: v for k, v in performance(bench).items() if k in ('cagr', 'volatility', 'sharpe', 'max_drawdown')}))
        for name, (signal, dates, top) in designs.items():
            decisions, size = quintile_decisions(signal, eligible, dates, top)
            result, _ = run_backtest(returns, decisions, cost)
            net = result.net
            nets[(name, cost)] = net
            active = (net-bench.reindex(net.index)).dropna()
            monthly_active = (1+active).resample('ME').prod()-1
            test = hac_mean_test(monthly_active.to_numpy())
            hedged = factor_regression(active, factors.assign(RF=0.))
            rows.append(dict(portfolio=name, cost_bps=cost, avg_holdings=size,
                             turnover_per_rebalance=float(result.turnover[result.turnover > 0].mean()),
                             **{k: v for k, v in performance(net).items() if k in ('cagr', 'volatility', 'sharpe', 'max_drawdown')},
                             active_pct_yr=100*12*test['mean'], active_t=test['t'],
                             active_ff_alpha_pct_yr=100*hedged['alpha_annualized'], active_ff_alpha_t=hedged['t']['alpha'],
                             beta_smb=hedged['coefficients']['SMB'],
                             **{f'active_{a[:4]}_{b[2:4]}': 100*12*float(((1+active.loc[a:b]).resample('ME').prod()-1).mean())
                                for a, b in PERIODS}))
    table = pd.DataFrame(rows)
    table.to_csv(out/'portfolios.csv', index=False)
    trials = pd.DataFrame({n: nets[(n, 0.)]-nets[('benchmark', 0.)] for n in designs}).dropna()
    sharpes = trials.mean()/trials.std()
    best = sharpes.idxmax()
    info = dict(counts=counts, best_gross_variant=best,
                deflated_sharpe=deflated_sharpe(trials[best].to_numpy(), len(designs), float(sharpes.var())),
                factors_end=str(factors.index[-1].date()), universe_built=config['built'])
    (out/'run_info.json').write_text(json.dumps(info, indent=2, default=str))
    pd.set_option('display.width', 250)
    print(json.dumps(info, indent=1, default=str))
    print(ics.round(4).to_string())
    print(pd.DataFrame(vol_rows).round(4).to_string())
    print(table.round(3).to_string())


if __name__ == '__main__':
    main()
