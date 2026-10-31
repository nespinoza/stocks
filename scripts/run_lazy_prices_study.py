"""Do year-over-year changes in 10-K/10-Q text predict small/mid-cap returns? ("Lazy Prices") 2019–2026.

Inputs: configs/broad_universe.json, Yahoo prices in data_store/broad/, and
data_store/filings/similarity.csv from collect_filing_similarity.py. Fixed before
looking at results:

- A filing is public at the close of the first session after its filing date;
  portfolios trade one session after the decision close.
- At each month end, a stock's score is the similarity of its latest filing made
  public in the last 63 sessions (about a quarter). The measures are whole-document
  cosine (the main one), whole-document Jaccard, and Risk Factors cosine and
  Jaccard. High similarity means a "non-changer".
- Eligible: price >= $3, 63-day median dollar volume >= $1M, and a score.
  Daily returns are clipped to [-50%, +100%], as in the other studies.
- Portfolios (equal weight, monthly): the top (non-changers) and bottom
  (changers) quintiles, and their difference. Active returns are measured against
  all eligible stocks. One-way costs are 0, 20 and 40 bp. FF5 + momentum +
  reversal alpha. The deflated Sharpe ratio covers the four long–short variants.
- Monthly IC of each measure against the 21- and 63-session returns from the
  next close: raw, and after neutralizing log dollar volume, the trailing 21- and
  252-day returns, log document length and |log length change| (a large rewrite
  is also a long or short one).

Same survivorship caveat as the other broad-universe studies (today's listed
companies).

    OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_lazy_prices_study.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_insider_study import load_prices  # noqa: E402
from stock_api.engine import month_end_dates, performance, run_backtest  # noqa: E402
from stock_api.evaluation import (deflated_sharpe, factor_regression, hac_mean_test, ic_summary,  # noqa: E402
                                  information_coefficient, load_french_factors, neutralize)
from stock_api.insider import public_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
START, COSTS = '2019-03-31', (0., 20., 40.)
PERIODS = [('2019-01-01', '2021-12-31'), ('2022-01-01', '2026-12-31')]
MEASURES = ['cosine', 'jaccard', 'risk_cosine', 'risk_jaccard']
CONTROLS = ['log_dollar_volume', 'trail_21d', 'trail_252d', 'log_words', 'abs_length_change']


def latest(filings, column, sessions, tickers, window=63):
    """Value of each stock's latest public filing, carried forward for `window` sessions."""
    rows = filings.dropna(subset=[column]).sort_values(['public', 'filing_date'])
    panel = rows.pivot_table(index='public', columns='ticker', values=column, aggfunc='last')
    return panel.reindex(sessions).ffill(limit=window).reindex(columns=tickers)


def quintile(signal, eligible, dates, top):
    out, sizes = {}, []
    for date in dates:
        values = signal.loc[date].where(eligible.loc[date]).dropna()
        pick = values >= values.quantile(.8) if top else values <= values.quantile(.2)
        sizes.append(int(pick.sum()))
        out[date] = (pick/pick.sum()).reindex(signal.columns).fillna(0.)
    return out, float(np.mean(sizes))


def summarize(name, cost, net, bench, factors, extra):
    active = (net-bench.reindex(net.index)).dropna()
    monthly = (1+active).resample('ME').prod()-1
    test = hac_mean_test(monthly.to_numpy())
    hedged = factor_regression(active, factors.assign(RF=0.))
    return dict(portfolio=name, cost_bps=cost, **extra,
                **{k: v for k, v in performance(net).items() if k in ('cagr', 'volatility', 'sharpe', 'max_drawdown')},
                active_pct_yr=100*12*test['mean'], active_t=test['t'],
                ff_alpha_pct_yr=100*hedged['alpha_annualized'], ff_alpha_t=hedged['t']['alpha'],
                **{f'active_{a[:4]}_{b[2:4]}': 100*12*float(((1+active.loc[a:b]).resample('ME').prod()-1).mean())
                   for a, b in PERIODS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'reports/lazy-prices-2026-10-30'))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'configs/broad_universe.json').read_text())
    filings = pd.read_csv(ROOT/'data_store/filings/similarity.csv', parse_dates=['filing_date'])
    adj, close, volume = load_prices(config)
    adj = adj.loc['2017-01-01':]
    close, volume = close.reindex_like(adj), volume.reindex_like(adj)
    sessions, tickers = adj.index, list(adj.columns)
    filings = filings[filings.ticker.isin(tickers)].copy()
    filings['public'] = public_session(filings.filing_date, sessions)
    filings = filings.dropna(subset=['public'])
    filings['log_words'] = np.log(filings.words)
    filings['abs_length_change'] = np.log(filings.words/filings.prior_words).abs()

    returns = adj.pct_change(fill_method=None).clip(-.5, 1.)
    dollar_volume = (close*volume).rolling(63, min_periods=40).median()
    liquid = (close >= 3) & (dollar_volume >= 1e6) & adj.notna()
    signals = {m: latest(filings, m, sessions, tickers) for m in MEASURES+['log_words', 'abs_length_change']}
    monthly = month_end_dates(sessions)
    monthly = monthly[(monthly >= START) & (monthly < sessions[-2])]

    # IC at month ends.
    logs = np.log(adj)
    panels = {**{m: signals[m].where(liquid) for m in MEASURES},
              'log_words': signals['log_words'], 'abs_length_change': signals['abs_length_change'],
              'log_dollar_volume': np.log(dollar_volume), 'trail_21d': logs-logs.shift(21),
              'trail_252d': logs-logs.shift(252),
              'ret_21d': logs.shift(-22)-logs.shift(-1), 'ret_63d': logs.shift(-64)-logs.shift(-1)}
    frame = pd.DataFrame({k: v.loc[monthly].stack() for k, v in panels.items()}).rename_axis(['date', 'asset']).reset_index()
    ic_rows = []
    for measure in MEASURES:
        data = frame.dropna(subset=[measure, *CONTROLS]).copy()
        data['neutral'] = neutralize(data, measure, CONTROLS)
        for column, label in ((measure, 'raw'), ('neutral', 'neutral')):
            for outcome, step in (('ret_21d', 1), ('ret_63d', 3)):
                ic = information_coefficient(data.dropna(subset=[outcome]), column, outcome, min_assets=50)
                test = ic_summary(ic.iloc[::step])
                ic_rows.append(dict(measure=measure, version=label, outcome=outcome, mean_ic=float(ic.mean()),
                                    t=test['t'], p=test['p'], dates=test['dates'],
                                    stocks_per_date=float(data.groupby('date').size().mean())))
    ics = pd.DataFrame(ic_rows)
    ics.to_csv(out/'ic.csv', index=False)

    # Portfolios.
    factors = load_french_factors('daily', cache_dir=str(ROOT/'data_store/french'))
    rows, spreads = [], {}
    for cost in COSTS:
        for measure in MEASURES:
            eligible = liquid & signals[measure].notna()
            bench_w = {d: (eligible.loc[d]/eligible.loc[d].sum()).astype(float) for d in monthly}
            bench = run_backtest(returns, bench_w, cost)[0].net
            legs = {}
            for side, top in (('non_changers', True), ('changers', False)):
                decisions, size = quintile(signals[measure], eligible, monthly, top)
                result, _ = run_backtest(returns, decisions, cost)
                legs[side] = result.net
                rows.append(summarize(f'{measure}_{side}', cost, result.net, bench, factors,
                                      dict(avg_holdings=size, eligible=float(eligible.loc[monthly].sum(1).mean()),
                                           turnover_per_rebalance=float(result.turnover[result.turnover > 0].mean()))))
            gross_short = spreads.setdefault(('changers_gross', measure), legs['changers'])  # cost 0 runs first
            # Shorting pays costs too: subtract the short leg's gross return and its cost.
            spread = (legs['non_changers']-gross_short-(gross_short-legs['changers'])).dropna()
            spreads[(measure, cost)] = spread
            rows.append(summarize(f'{measure}_long_short', cost, spread+bench.reindex(spread.index), bench, factors, {}))
    table = pd.DataFrame(rows)
    table.to_csv(out/'portfolios.csv', index=False)

    trials = pd.DataFrame({m: spreads[(m, 0.)] for m in MEASURES}).dropna()
    sharpes = trials.mean()/trials.std()
    best = sharpes.idxmax()
    info = dict(filings=len(filings), with_prior=int(filings.cosine.notna().sum()),
                with_risk=int(filings.risk_cosine.notna().sum()), companies=int(filings.ticker.nunique()),
                months=len(monthly), first=str(monthly[0].date()), last=str(monthly[-1].date()),
                best_gross_long_short=best,
                deflated_sharpe=deflated_sharpe(trials[best].to_numpy(), len(MEASURES), float(sharpes.var())),
                factors_end=str(factors.index[-1].date()), universe_built=config['built'])
    (out/'run_info.json').write_text(json.dumps(info, indent=2, default=str))
    pd.set_option('display.width', 250)
    print(json.dumps(info, indent=1, default=str))
    print(ics.round(4).to_string())
    print(table.round(3).to_string())


if __name__ == '__main__':
    main()
