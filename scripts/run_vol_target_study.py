"""Volatility targeting of SPY and 60/40 with HAR forecasts, without leverage.

At each decision close the risky weight is min(target / forecast vol, 1); the
rest sits in BIL (T-bills). The forecast is the log-HAR 21-day variance
(Garman–Klass proxy for SPY, squared returns for the 60/40 return series) or
EWMA (λ = 0.94) for comparison. Targets are fixed in advance, not fitted.

The fair benchmark is not the unmanaged portfolio but a *static* mix of it with
BIL at the same average risky weight: only then does the test isolate timing
from simply holding less equity. Alpha is the intercept of the managed excess
returns on the static mix's excess returns (monthly, Newey–West).

    OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_vol_target_study.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_api.engine import month_end_dates, performance, run_backtest
from stock_api.evaluation import deflated_sharpe, factor_regression
from stock_api.risk import ewma_forecast, garman_klass, har_forecast

ROOT = Path(__file__).resolve().parents[1]
START, HORIZON, COST_BPS = '2010-01-01', 21, 5.
TARGETS = {'SPY': [.10, .15], '60/40': [.06, .08]}
PERIODS = [('2010-01-01', '2017-12-31'), ('2018-01-01', '2026-12-31')]


def load(ticker):
    return pd.read_csv(ROOT/'data_store/etf'/f'{ticker}.csv', parse_dates=['Date'], index_col='Date')


def week_end_dates(index):
    index = pd.DatetimeIndex(index)
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(index.to_period('W-FRI')).max().to_numpy())


def forecasts(log_returns, proxy, dates):
    """Annualized volatility forecasts at each decision date, from data through that close."""
    rows = {}
    for date in dates:
        history = log_returns.loc[:date]
        rows[date] = dict(har=np.sqrt(har_forecast(proxy.loc[:date], history, HORIZON)[0]*252/HORIZON),
                          ewma=np.sqrt(ewma_forecast(history.iloc[-500:], HORIZON)*252/HORIZON))
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'reports/vol-targeting-2026-10-30'))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frames = {t: load(t) for t in ('SPY', 'IEF', 'BIL')}
    adj = pd.DataFrame({t: f['Adj Close'] for t, f in frames.items()}).dropna()
    simple = adj.pct_change().dropna()
    cash = simple['BIL']
    # Constant-mix 60/40 return series, used only as the forecasting input for that base.
    mix = .6*simple['SPY']+.4*simple['IEF']
    spy = frames['SPY'].loc[adj.index]
    bases = {'SPY': (np.log(adj['SPY']).diff().dropna(),
                     garman_klass(spy.Open, spy.High, spy.Low, spy.Close).reindex(simple.index)),
             '60/40': (np.log1p(mix), np.log1p(mix)**2)}
    weights_of = {'SPY': pd.Series({'SPY': 1.}), '60/40': pd.Series({'SPY': .6, 'IEF': .4})}
    returns = simple[['SPY', 'IEF', 'BIL']]
    schedules = {'monthly': month_end_dates(returns.index), 'weekly': week_end_dates(returns.index)}

    strategies, risky_weights = {}, {}
    for base, (log_ret, proxy) in bases.items():
        for freq, dates in schedules.items():
            dates = dates[(dates >= START) & (dates < returns.index[-1])]
            vol = forecasts(log_ret, proxy, dates)
            vol.to_csv(out/f"forecast_{base.replace('/', '_')}_{freq}.csv")
            if freq == 'monthly':
                plain = {d: weights_of[base] for d in dates}
                strategies[(base, 'unmanaged', '-', freq, 0.)] = run_backtest(returns, plain, COST_BPS)[0].net
            for model in ('har', 'ewma'):
                for target in TARGETS[base]:
                    for band in (0., .05):
                        risky = (target/vol[model]).clip(upper=1.)
                        decisions = {d: (weights_of[base]*risky[d]).reindex(returns.columns).fillna(0.)
                                     + pd.Series({'BIL': 1-risky[d]}).reindex(returns.columns).fillna(0.) for d in dates}
                        key = (base, model, f'{target:.0%}', freq, band)
                        strategies[key] = run_backtest(returns, decisions, COST_BPS, band=band)[0].net
                        risky_weights[key] = risky

    # Static benchmark: the base mixed with BIL at each managed strategy's average risky weight.
    monthly = month_end_dates(returns.index)
    monthly = monthly[(monthly >= START) & (monthly < returns.index[-1])]
    common = max(s.index[0] for s in strategies.values())
    rows, active = [], {}
    for key, net in strategies.items():
        net = net.loc[common:]
        base = key[0]
        row = dict(base=base, forecast=key[1], target=key[2], rebalance=key[3], band=key[4],
                   **{k: v for k, v in performance(net).items() if k in ('cagr', 'volatility', 'sharpe', 'max_drawdown')})
        excess = net-cash.loc[net.index]
        row['sharpe_excess'] = float(excess.mean()/excess.std()*np.sqrt(252))
        for a, b in PERIODS:
            part = excess.loc[a:b]
            row[f'sharpe_{a[:4]}_{b[2:4]}'] = float(part.mean()/part.std()*np.sqrt(252))
        if key in risky_weights:
            average = float(risky_weights[key].mean())
            static_w = (weights_of[base]*average).reindex(returns.columns).fillna(0.)+pd.Series(
                {'BIL': 1-average}).reindex(returns.columns).fillna(0.)
            static = run_backtest(returns, {d: static_w for d in monthly}, COST_BPS)[0].net.loc[common:]
            static_excess = static-cash.loc[static.index]
            row['avg_risky_weight'] = average
            row['static_sharpe_excess'] = float(static_excess.mean()/static_excess.std()*np.sqrt(252))
            row['static_volatility'] = performance(static)['volatility']
            row['static_max_drawdown'] = performance(static)['max_drawdown']
            m = lambda s: (1+s).resample('ME').prod()-1
            fit = factor_regression(m(excess), pd.DataFrame({'BASE': m(static_excess)}), columns=('BASE',),
                                    periods_per_year=12)
            row['alpha_vs_static_pct'] = fit['coefficients']['alpha']*1200
            row['alpha_t'] = fit['t']['alpha']
            row['beta_vs_static'] = fit['coefficients']['BASE']
            active[key] = excess-static_excess*fit['coefficients']['BASE']
        rows.append(row)
    table = pd.DataFrame(rows)
    trials = pd.DataFrame(active)
    sharpes = trials.mean()/trials.std()
    best = sharpes.idxmax()
    dsr = deflated_sharpe(trials[best].to_numpy(), len(trials.columns), float(sharpes.var()))
    table.to_csv(out/'strategies.csv', index=False)
    info = dict(start=str(common.date()), end=str(returns.index[-1].date()), cost_bps=COST_BPS, horizon=HORIZON,
                targets=TARGETS, best_by_hedged_active_sharpe=list(map(str, best)), deflated_sharpe=dsr,
                data='Yahoo adjusted closes SPY/IEF/BIL from data_store/etf (manifest checksums)')
    (out/'run_info.json').write_text(json.dumps(info, indent=2, default=str))
    pd.set_option('display.width', 250)
    print(table.round(3).to_string())
    print(json.dumps(info, indent=1, default=str))


if __name__ == '__main__':
    main()
