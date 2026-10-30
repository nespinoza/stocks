"""Do officer/director open-market purchases predict small/mid-cap returns? 2010–2026.

Inputs (all free): configs/broad_universe.json (build_broad_universe.py),
Yahoo prices in data_store/broad/ (download_etf_universe.py), SEC Form 4
trades in data_store/insider/trades.csv.gz (collect_insider_trades.py), Kenneth
French factors.

Everything below was fixed before looking at results:
- Events: Form 4 code-P purchases (and code-S sales) by officers or directors,
  public at the close of the session after the filing date.
- Eligible at a decision close: price >= $3 and 63-day median dollar volume
  >= $1M, from data through that close. Daily returns are clipped to
  [-50%, +100%] to blunt Yahoo data errors in small caps.
- Portfolios (equal weight, decided at month-end closes, traded at the next
  close, held a month): stocks with >= 1 buyer in the last 21 / 63 sessions,
  >= 2 distinct buyers in the last 21 sessions ("cluster"), >= 1
  opportunistic buyer (Cohen–Malloy–Pomorski) in the last 21 sessions, and,
  as a contrast, >= 2 sellers and no buyers. Benchmark: all eligible stocks,
  equal weight. One-way costs of 0 (signal), 20 and 40 bp (small-cap spreads).
- Cross-sectional IC of the 63-session net purchase ratio, raw and after
  neutralizing the trailing 21/252-day return and log dollar volume.
- Event study: cumulative simple abnormal returns (vs the equal-weighted
  eligible universe) from the close after the public session, for 5, 21 and
  63 sessions, with random eligible stock-days as a placebo.

The universe is today's listed companies, so the history omits firms that died
or were acquired (survivorship bias). The benchmark carries the same bias,
which offsets part of it; 2023–26 is reported separately as the least-affected
window.

    OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_insider_study.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_api import insider
from stock_api.engine import month_end_dates, performance, run_backtest
from stock_api.evaluation import (deflated_sharpe, factor_regression, hac_mean_test, ic_summary,
                                  information_coefficient, load_french_factors, neutralize)

ROOT = Path(__file__).resolve().parents[1]
START, COSTS = '2010-01-01', (0., 20., 40.)
PERIODS = [('2010-01-01', '2017-12-31'), ('2018-01-01', '2026-12-31'), ('2023-01-01', '2026-12-31')]
HORIZONS = (5, 21, 63)


def load_prices(config):
    adj, close, volume = {}, {}, {}
    for ticker in config['stocks']:
        path = ROOT/'data_store/broad'/f'{ticker}.csv'
        if path.exists():
            frame = pd.read_csv(path, parse_dates=['Date'], index_col='Date', usecols=['Date', 'Close', 'Adj Close', 'Volume'])
            frame = frame[~frame.index.duplicated()]
            adj[ticker], close[ticker], volume[ticker] = frame['Adj Close'], frame.Close, frame.Volume
    return pd.DataFrame(adj).sort_index(), pd.DataFrame(close).sort_index(), pd.DataFrame(volume).sort_index()


def portfolio_decisions(members, eligible, dates):
    """Equal weight over members ∩ eligible at each date; the eligible universe when empty."""
    out, counts = {}, {}
    for date in dates:
        pick = members.loc[date] & eligible.loc[date]
        if pick.sum() == 0:
            pick = eligible.loc[date]
        counts[date] = int(pick.sum())
        out[date] = pick.astype(float)/pick.sum()
    return out, pd.Series(counts)


def event_study(events, abnormal_cum, sessions):
    """Mean cumulative abnormal log return after events, t from monthly event-averages (HAC)."""
    position = sessions.get_indexer(pd.DatetimeIndex(events.session))
    rows = []
    for h in HORIZONS:
        start, end = position+1, position+1+h  # from the close of session+1 (lag-1 execution)
        valid = (position >= 0) & (end < len(sessions))
        col = abnormal_cum.columns.get_indexer(events.ticker)
        car = abnormal_cum.to_numpy()[end[valid], col[valid]]-abnormal_cum.to_numpy()[start[valid], col[valid]]
        frame = pd.DataFrame({'car': car, 'month': pd.DatetimeIndex(events.session[valid]).to_period('M')}).dropna()
        monthly = frame.groupby('month').car.mean()
        test = hac_mean_test(monthly.to_numpy(), max(h//21, 1)+2)
        rows.append(dict(horizon=h, events=len(frame), months=len(monthly), mean_car_pct=100*frame.car.mean(),
                         monthly_mean_pct=100*test['mean'], t=test['t'], hit_rate=float((frame.car > 0).mean())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'reports/insider-purchases-2026-10-30'))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/'configs/broad_universe.json').read_text())
    adj, close, volume = load_prices(config)
    adj = adj.loc[START:]
    close, volume = close.reindex_like(adj), volume.reindex_like(adj)
    sessions, tickers = adj.index, list(adj.columns)
    returns = adj.pct_change(fill_method=None).clip(-.5, 1.)
    dollar_volume = (close*volume).rolling(63, min_periods=40).median()
    eligible = (close >= 3) & (dollar_volume >= 1e6) & adj.notna()

    cik_to_ticker = {s['cik']: t for t, s in config['stocks'].items() if t in tickers}
    raw = pd.read_csv(ROOT/'data_store/insider/trades.csv.gz', dtype={'relationship': str, 'title': str})
    trades = insider.clean_trades(raw, set(cik_to_ticker))
    trades = trades.sort_values(['owner_cik', 'filing_date']).reset_index(drop=True)
    trades['flag'] = insider.routine_flags(trades)
    trades['ticker'] = trades.issuer_cik.map(cik_to_ticker)
    trades['session'] = insider.public_session(trades.filing_date, sessions)
    trades = trades.dropna(subset=['session'])
    buys, sells = trades[trades.code == 'P'], trades[trades.code == 'S']
    counts = dict(trades=len(trades), purchases=len(buys), sales=len(sells), stocks_with_purchases=buys.ticker.nunique(),
                  opportunistic_purchases=int((buys.flag == 'opportunistic').sum()),
                  routine_purchases=int((buys.flag == 'routine').sum()), universe=len(tickers))

    def distinct_owners(events, window):
        """Distinct insiders with an event in the trailing window, per session and ticker."""
        first = events.drop_duplicates(['ticker', 'owner_cik', 'session'])
        panel = insider.event_panel(first, sessions, tickers)
        # Approximation: counts insider-sessions; repeat filings by one insider inside a window are rare.
        return insider.trailing(panel, window)

    buyers_21, buyers_63 = distinct_owners(buys, 21), distinct_owners(buys, 63)
    sellers_21 = distinct_owners(sells, 21)
    opportunistic_21 = distinct_owners(buys[buys.flag == 'opportunistic'], 21)
    members = {'buy_1m': buyers_21 >= 1, 'buy_3m': buyers_63 >= 1, 'buy_cluster_1m': buyers_21 >= 2,
               'buy_opportunistic_1m': opportunistic_21 >= 1, 'sell_cluster_no_buy_1m': (sellers_21 >= 2) & (buyers_21 == 0)}
    dates = month_end_dates(sessions)
    dates = dates[(dates >= '2010-04-30') & (dates < sessions[-1])]
    universe_decisions, _ = portfolio_decisions(eligible, eligible, dates)

    daily, rows, holdings = {}, [], {}
    for cost in COSTS:
        base = run_backtest(returns, universe_decisions, cost)[0].net
        daily[('universe_ew', cost)] = base
        for name, member in members.items():
            decisions, n = portfolio_decisions(member, eligible, dates)
            holdings[name] = n
            net, _ = run_backtest(returns, decisions, cost)
            daily[(name, cost)] = net.net
            daily[(name, cost, 'turnover')] = net.turnover
    factors = load_french_factors('daily', cache_dir=str(ROOT/'data_store/french'))
    cash = factors['RF']
    for (name, cost, *extra), net in list(daily.items()):
        if extra:
            continue
        base = daily[('universe_ew', cost)].reindex(net.index)
        active = net-base
        row = dict(portfolio=name, cost_bps=cost,
                   **{k: v for k, v in performance(net).items() if k in ('cagr', 'volatility', 'sharpe', 'max_drawdown')})
        if name != 'universe_ew':
            monthly_active = (1+active).resample('ME').prod()-1
            test = hac_mean_test(monthly_active.to_numpy())
            row.update(avg_holdings=float(holdings[name].mean()), min_holdings=int(holdings[name].min()),
                       turnover_per_month=float(daily[(name, cost, 'turnover')].sum()/len(dates)),
                       active_pct_yr=100*12*test['mean'], active_t=test['t'])
            for a, b in PERIODS:
                part = (1+active.loc[a:b]).resample('ME').prod()-1
                row[f'active_{a[:4]}_{b[2:4]}'] = 100*12*part.mean()
                row[f't_{a[:4]}_{b[2:4]}'] = hac_mean_test(part.to_numpy())['t']
        fit = factor_regression(net, factors)
        row.update(ff_alpha_pct_yr=100*fit['alpha_annualized'], ff_alpha_t=fit['t']['alpha'],
                   beta_mkt=fit['coefficients']['MKT'], beta_smb=fit['coefficients']['SMB'],
                   beta_str=fit['coefficients']['STR'])
        if name != 'universe_ew':
            hedged = factor_regression(active, factors.assign(RF=0.))
            row.update(active_ff_alpha_pct_yr=100*hedged['alpha_annualized'], active_ff_alpha_t=hedged['t']['alpha'])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out/'portfolios.csv', index=False)

    # Deflated Sharpe on the gross (zero-cost) active returns: is there a signal at all?
    trials = pd.DataFrame({name: daily[(name, 0.)]-daily[('universe_ew', 0.)] for name in members}).dropna()
    sharpes = trials.mean()/trials.std()
    best = sharpes.idxmax()
    dsr = deflated_sharpe(trials[best].to_numpy(), len(members), float(sharpes.var()))

    # Cross-sectional IC of the 63-session net purchase ratio among stocks with insider trades.
    value_panel = lambda events: insider.trailing(insider.event_panel(events, sessions, tickers, value='value'), 63)
    bought, sold = value_panel(buys), value_panel(sells)
    traded = (bought+sold) > 0
    npr = ((bought-sold)/(bought+sold)).where(traded & eligible)
    logs = np.log(adj)
    panels = {'npr': npr, 'fwd_21d': logs.shift(-22)-logs.shift(-1), 'fwd_63d': logs.shift(-64)-logs.shift(-1),
              'trail_21d': logs-logs.shift(21), 'trail_252d': logs-logs.shift(252), 'log_dollar_volume': np.log(dollar_volume)}
    frame = pd.DataFrame({k: v.loc[dates].stack() for k, v in panels.items()}).rename_axis(['date', 'asset']).reset_index()
    frame = frame.dropna(subset=['npr'])
    frame['npr_neutral'] = neutralize(frame, 'npr', ['trail_21d', 'trail_252d', 'log_dollar_volume'])
    ic_rows = []
    for signal in ('npr', 'npr_neutral'):
        for outcome, h in (('fwd_21d', 1), ('fwd_63d', 3)):
            ic = information_coefficient(frame, signal, outcome, min_assets=20)
            independent = ic.iloc[::h]
            ic_rows.append(dict(signal=signal, outcome=outcome, stocks_per_date=float(frame.groupby('date').size().mean()),
                                **{k: v for k, v in ic_summary(independent, None).items() if k in ('mean_ic', 't', 'p', 'dates')},
                                mean_ic_all_dates=float(ic.mean())))
    pd.DataFrame(ic_rows).to_csv(out/'ic.csv', index=False)

    # Event study against the equal-weighted eligible universe.
    # Simple-return abnormal returns, summed (CAR). Log returns would carry a volatility-drag bias of
    # about (σ_stock² − σ_index²)/2 against a rebalanced index, i.e. ~-2% a quarter for small caps.
    universe_return = returns.where(eligible.shift(1, fill_value=False)).mean(axis=1)
    abnormal = returns.fillna(0.).sub(universe_return, axis=0)
    abnormal_cum = abnormal.cumsum()
    rng = np.random.default_rng(0)
    stacked = eligible.loc['2010-04-01':].stack()
    stacked = stacked[stacked].index.to_frame(index=False, name=['session', 'ticker'])
    placebo = stacked.iloc[rng.choice(len(stacked), 25000, replace=False)]
    event_rows = []
    for label, events in (('purchase', buys), ('opportunistic purchase', buys[buys.flag == 'opportunistic']),
                          ('routine purchase', buys[buys.flag == 'routine']), ('sale', sells),
                          ('random eligible stock-day (placebo)', placebo)):
        events = events.drop_duplicates(['ticker', 'session'])
        events = events[[bool(eligible.at[s, t]) for s, t in zip(events.session, events.ticker)]]
        for row in event_study(events, abnormal_cum, sessions):
            event_rows.append(dict(event=label, **row))
    pd.DataFrame(event_rows).to_csv(out/'event_study.csv', index=False)

    info = dict(counts=counts, start=str(sessions[0].date()), end=str(sessions[-1].date()), costs_bps=COSTS,
                best_variant=best, deflated_sharpe=dsr, factors_end=str(factors.index[-1].date()),
                universe_built=config['built'])
    (out/'run_info.json').write_text(json.dumps(info, indent=2, default=str))
    pd.set_option('display.width', 250)
    print(json.dumps(info, indent=1, default=str))
    print(table.round(3).to_string())
    print(pd.DataFrame(ic_rows).round(3).to_string())
    print(pd.DataFrame(event_rows).round(3).to_string())


if __name__ == '__main__':
    main()
