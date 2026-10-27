"""Black–Litterman tilts on the ETF risk-parity base: are momentum views worth it?

1. Measures each signal's cross-sectional IC on the ETF panel (no portfolio).
2. Tilts the HAR + Ledoit–Wolf risk-parity portfolio with Black–Litterman
   views from each signal, at pre-specified (not fitted) ICs, with and without a
   turnover penalty, and with no-trade bands on the base.
3. Scores active returns against the base: HAC t-stats, deflated Sharpe over
   the tilt grid, PBO, and costs under a spread + square-root-impact model.

Signals are risk-adjusted trailing returns: 12-1 month momentum and 1-month
momentum (the stand-in for the GP drift, which tracked 30-day momentum with
Spearman 0.71–0.79 in the saved studies). Run after run_etf_risk_study.py:

    OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_etf_bl_study.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_etf_risk_study import FIRST_ORIGIN, HORIZON, ROOT, load_panel
from stock_api.allocation import equal_risk_contribution, mean_variance
from stock_api.costs import spread_impact_cost
from stock_api.engine import month_end_dates, performance, run_backtest
from stock_api.evaluation import (deflated_sharpe, factor_regression, hac_mean_test, ic_summary, information_coefficient,
                                  load_french_factors, pbo_cscv)
from stock_api.risk import covariance_from_vol_corr, shrunk_correlation
from stock_api.views import black_litterman, grinold_alpha, implied_returns, risk_aversion_for_sharpe, view_uncertainty

COST_BPS = 5.
PRIOR_SHARPE = .5/np.sqrt(12)  # monthly Sharpe assumed for the risk-parity equilibrium
TAU = .05
CONFIDENCE = .5
SIGNALS = ['mom_12_1', 'mom_1m', 'combined']
ICS = [.05, .10]
AUMS = [1e7, 1e9]
PERIODS = [('2010-01-01', '2017-12-31'), ('2018-01-01', '2026-12-31')]


def signal_panel(adj):
    """Risk-adjusted trailing returns at every date (data through that close only)."""
    logs = np.log(adj)
    vol = logs.diff().rolling(252).std()
    raw = {'mom_12_1': (logs.shift(21)-logs.shift(252))/(vol*np.sqrt(231)),
           'mom_1m': (logs-logs.shift(21))/(vol*np.sqrt(21))}
    zscore = lambda f: f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1), axis=0)
    raw['combined'] = (zscore(raw['mom_12_1'])+zscore(raw['mom_1m']))/2
    return raw


def ic_study(adj, signals, origins, groups):
    logs = np.log(adj)
    rows = []
    for horizon in (5, 21, 63):
        # Outcome: return from the execution close (t+1) to t+1+horizon.
        outcome = (logs.shift(-(1+horizon))-logs.shift(-1)).loc[origins]
        for name in SIGNALS:
            frame = pd.DataFrame({'signal': signals[name].loc[origins].stack(), 'outcome': outcome.stack()})
            frame = frame.rename_axis(['date', 'asset']).reset_index().dropna()
            for universe, members in [('all 41', None), ('equity 30', groups['equity'])]:
                sub = frame if members is None else frame[frame.asset.isin(members)]
                ic = information_coefficient(sub, 'signal', 'outcome')
                for period, (a, b) in [('2010–2026', ('2000', '2100')), *[(f'{a[:4]}–{b[:4]}', (a, b)) for a, b in PERIODS]]:
                    part = ic.loc[a:b]
                    lags = max(int(4*(len(part)/100)**(2/9)), horizon//21)
                    rows.append(dict(signal=name, horizon=horizon, universe=universe, period=period,
                                     **ic_summary(part, lags)))
    return pd.DataFrame(rows)


def risk_inputs(adj, har, origins):
    logs = np.log(adj).diff().iloc[1:]
    tickers = list(adj.columns)
    inputs = {}
    for origin in origins:
        vol = np.sqrt(har.loc[origin, tickers].to_numpy())
        corr, _ = shrunk_correlation(logs.loc[:origin].iloc[-252:].to_numpy())
        cov = covariance_from_vol_corr(vol, corr)
        prior = equal_risk_contribution(cov)
        delta = risk_aversion_for_sharpe(cov, prior, PRIOR_SHARPE)
        inputs[origin] = dict(vol=vol, cov=cov, prior=prior, delta=delta, pi=implied_returns(cov, prior, delta),
                              omega=view_uncertainty(np.eye(len(tickers)), cov, TAU, CONFIDENCE))
    return inputs


def bl_policy(inputs, scores, ic, penalty):
    """Posterior mean from absolute views Q = π + α on every asset, then a cost-aware optimizer."""
    def policy(date, held):
        x = inputs[date]
        alpha = grinold_alpha(scores.loc[date, held.index].to_numpy(), x['vol'], ic)
        mean, _ = black_litterman(x['pi'], x['cov'], np.eye(len(alpha)), x['pi']+alpha, x['omega'], TAU)
        invested = held.sum() > 0
        # Σ (not the posterior Σ + M) keeps "no views" exactly at the risk-parity weights.
        return mean_variance(mean, x['cov'], x['delta'], current=held.to_numpy() if invested else None,
                             cost=penalty if invested else 0.)
    return policy


def run(simple, policy, origins, adv, daily_vol, band=0.):
    trades = []

    def recorder(date, trade):
        trades.append((date, trade.copy()))
        return COST_BPS/1e4*np.abs(trade).sum()
    daily, weights = run_backtest(simple, policy, decision_dates=origins, band=band, cost_model=recorder)
    alt = {}
    for aum in AUMS:
        cost = pd.Series(0., index=daily.index)
        for date, trade in trades:
            cost[date] = spread_impact_cost(trade, aum, adv.loc[date].to_numpy(), daily_vol.loc[date].to_numpy())
        alt[aum] = (1+daily.gross)*(1-cost)-1
    return daily, weights, alt


def summarize(name, daily, weights, alt, base=None, base_alt=None):
    perf = performance(daily.net)
    months = daily.turnover.groupby(daily.index.to_period('M')).sum()
    row = dict(strategy=name, cagr=perf['cagr'], vol=perf['volatility'], sharpe=perf['sharpe'],
               max_dd=perf['max_drawdown'], turnover_month=float(months.iloc[1:].mean()),
               cost_5bp_year=float(daily.cost.sum()/(len(daily)/252)),
               effective_holdings=float((1/(weights**2).sum(axis=1)).mean()))
    for aum, net in alt.items():
        row[f'sharpe_aum_{aum:.0e}'] = performance(net)['sharpe']
        row[f'cost_aum_{aum:.0e}_year'] = float(((1+daily.gross)/(1+net)-1).sum()/(len(daily)/252))
    if base is not None:
        active = (daily.net-base.net).dropna()
        monthly = active.groupby(active.index.to_period('M')).sum()
        test = hac_mean_test(monthly.to_numpy())
        row.update(active_return_year=12*test['mean'], tracking_error=float(monthly.std(ddof=1)*np.sqrt(12)),
                   information_ratio=float(test['mean']/monthly.std(ddof=1)*np.sqrt(12)), active_t=test['t'],
                   active_p=test['p'])
        for a, b in PERIODS:
            part = monthly.loc[a:b]
            row[f'ir_{a[:4]}_{b[2:4]}'] = float(part.mean()/part.std(ddof=1)*np.sqrt(12))
        for aum, net in alt.items():
            act = net-base_alt[aum]
            row[f'active_aum_{aum:.0e}_year'] = float(act.mean()*252)
    return row


def attribution(daily_net, simple):
    """Monthly alphas: vs the risk-parity base (beta-adjusted) and vs asset-class ETF betas.

    A tilt that only adds equity or duration risk shows up as beta here, not alpha.
    """
    monthly = lambda r: (1+r).resample('ME').prod()-1
    rf = load_french_factors('monthly', cache_dir=str(ROOT/'data_store/french'))['RF']
    assets = {t: monthly(simple[t]) for t in ('SPY', 'IEF', 'GLD', 'DBC')}
    base = monthly(daily_net['Risk parity (base)'])
    rows = []
    for name, net in daily_net.items():
        r = monthly(net)
        specs = {'vs base': (pd.DataFrame({'BASE': base-rf, 'RF': rf}), ('BASE',)),
                 'vs SPY/IEF/GLD/DBC': (pd.DataFrame({**{t: a-rf for t, a in assets.items()}, 'RF': rf}),
                                        ('SPY', 'IEF', 'GLD', 'DBC'))}
        for spec, (factors, columns) in specs.items():
            if spec == 'vs base' and name == 'Risk parity (base)':
                continue
            fit = factor_regression(r, factors, columns=columns, periods_per_year=12)
            rows.append(dict(strategy=name, model=spec, alpha_year=fit['alpha_annualized'], alpha_t=fit['t']['alpha'],
                             r2=fit['r2'], months=fit['n'],
                             **{f'beta_{c}': fit['coefficients'][c] for c in columns},
                             **{f't_{c}': fit['t'][c] for c in columns}))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'reports/etf-black-litterman-2026-10-27'))
    out = Path(parser.parse_args().output)
    out.mkdir(parents=True, exist_ok=True)
    panel, config = load_panel()
    adj = panel['Adj Close']
    simple = adj.pct_change().iloc[1:]
    forecasts = pd.read_csv(ROOT/'data_store/etf_study/vol_forecasts.csv')
    har = forecasts.pivot(index='origin', columns='ticker', values='har_gk')
    har.index = pd.to_datetime(har.index)
    origins = [d for d in month_end_dates(simple.index) if pd.Timestamp(FIRST_ORIGIN) <= d < simple.index[-1]]
    groups = {'equity': config['groups']['us_equity']+config['groups']['us_sectors']+config['groups']['international_equity']}
    signals = signal_panel(adj)

    ics = ic_study(adj, signals, origins, groups)
    ics.to_csv(out/'signal_ic.csv', index=False)

    # Costs: dollar volume and daily volatility known before the execution day.
    adv = (panel['Close']*panel['Volume']).rolling(63).mean().shift(1).reindex(simple.index)
    daily_vol = simple.rolling(63).std().shift(1)
    inputs = risk_inputs(adj, har, origins)
    tickers = list(adj.columns)
    risk_parity = lambda date, held: pd.Series(inputs[date]['prior'], index=tickers)
    sixty_forty = lambda date, held: pd.Series({'SPY': .6, 'IEF': .4}).reindex(tickers).fillna(0.)

    results, daily_net, active_daily = [], {}, {}
    base, base_w, base_alt = run(simple, risk_parity, origins, adv, daily_vol)
    results.append(summarize('Risk parity (base)', base, base_w, base_alt))
    daily_net['Risk parity (base)'] = base.net
    for name, policy, band in [('60/40 SPY/IEF', sixty_forty, 0.), ('Risk parity, 1% band', risk_parity, .01),
                               ('Risk parity, 2% band', risk_parity, .02)]:
        daily, weights, alt = run(simple, policy, origins, adv, daily_vol, band)
        results.append(summarize(name, daily, weights, alt, base, base_alt))
        daily_net[name] = daily.net
    grid = [(s, ic, p) for s in SIGNALS for ic in ICS for p in (0., COST_BPS/1e4)]
    for signal, ic, penalty in grid:
        name = f'BL {signal}, IC {ic:.2f}, ' + ('cost-penalized' if penalty else 'no penalty')
        daily, weights, alt = run(simple, bl_policy(inputs, signals[signal], ic, penalty), origins, adv, daily_vol)
        results.append(summarize(name, daily, weights, alt, base, base_alt))
        daily_net[name] = daily.net
        active_daily[name] = daily.net-base.net
        print(name, f"IR {results[-1]['information_ratio']:.2f} t {results[-1]['active_t']:.2f}", flush=True)

    table = pd.DataFrame(results)
    active = pd.DataFrame(active_daily).dropna()
    trial_sharpes = active.mean()/active.std(ddof=1)
    best = trial_sharpes.idxmax()
    dsr = deflated_sharpe(active[best].to_numpy(), len(grid), float(trial_sharpes.var(ddof=1)))
    pbo = pbo_cscv(active.to_numpy(), blocks=16)
    table.to_csv(out/'strategies.csv', index=False)
    attr = attribution(daily_net, simple)
    attr.to_csv(out/'attribution.csv', index=False)
    pd.DataFrame(daily_net).to_csv(out/'daily_net.csv')
    (out/'run_info.json').write_text(json.dumps(dict(
        cost_bps=COST_BPS, prior_sharpe_monthly=PRIOR_SHARPE, tau=TAU, view_confidence=CONFIDENCE, ics=ICS,
        aums=AUMS, origins=[str(origins[0].date()), str(origins[-1].date()), len(origins)],
        tilt_trials=len(grid), best_tilt=best, deflated_sharpe=dsr, pbo=pbo), indent=2, default=float))
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(ics[(ics.period == '2010–2026')].round(3).to_string())
    print(table.round(3).to_string())
    print(attr.round(3).to_string())
    print('best', best, dsr, 'PBO', pbo)


if __name__ == '__main__':
    main()
