"""ETF risk-engine study: volatility-forecast benchmark and risk-based portfolios.

Part 1 forecasts each ETF's next-21-session realized variance at every month end
with rolling, EWMA, HAR (Garman–Klass and squared-return proxies), the
repository's regression GJR-GARCH and the volatility GP, then scores them.
Part 2 backtests risk-only allocations (no return forecasts) with the shared
engine. Run from the repository root after scripts/download_etf_universe.py:

    OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_etf_risk_study.py
"""
import argparse
import hashlib
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from stock_api.allocation import equal_risk_contribution, inverse_volatility, minimum_variance
from stock_api.engine import month_end_dates, performance, run_backtest
from stock_api.evaluation import diebold_mariano, kupiec, probabilistic_sharpe, qlike
from stock_api.risk import (covariance_from_vol_corr, ewma_forecast, factor_covariance, garman_klass,
                            har_forecast, ledoit_wolf, rolling_forecast, shrunk_correlation)

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 21
FIRST_ORIGIN = '2009-12-31'
COST_BPS = 5.
VOL_MODELS = ['rolling_63d', 'ewma_0.94', 'har_gk', 'har_sq', 'gjr_garch_252d', 'volatility_gp_63d']


def load_panel():
    config = json.loads((ROOT/'configs/etf_universe.json').read_text())
    tickers = [t for group in config['groups'].values() for t in group]
    fields = {}
    for ticker in tickers:
        frame = pd.read_csv(ROOT/'data_store/etf'/f'{ticker}.csv', parse_dates=['Date'], index_col='Date')
        for field in ('Open', 'High', 'Low', 'Close', 'Adj Close'):
            fields.setdefault(field, {})[ticker] = frame[field]
    panel = {k: pd.DataFrame(v).loc[config['start']:] for k, v in fields.items()}
    missing = int(panel['Adj Close'].isna().sum().sum())
    if missing:
        raise ValueError(f'{missing} missing adjusted closes in the common window')
    return panel, config


def vol_forecasts(args):
    """All volatility forecasts for one ticker at every month-end origin."""
    ticker, adj, open_, high, low, close = args
    from stock_api.models import MODELS
    warnings.filterwarnings('ignore')
    logs = np.log(adj)
    returns = logs.diff().dropna()
    gk = garman_klass(open_, high, low, close).reindex(returns.index)
    rows = []
    for origin in month_end_dates(returns.index):
        if origin < pd.Timestamp(FIRST_ORIGIN):
            continue
        history = returns.loc[:origin]
        future = returns.loc[origin:].iloc[1:HORIZON+1]
        realized = float((future**2).sum()) if len(future) == HORIZON else np.nan
        row = dict(ticker=ticker, origin=origin.date().isoformat(), realized=realized,
                   realized_return=float(future.sum()) if len(future) == HORIZON else np.nan)
        row['rolling_63d'] = rolling_forecast(history, HORIZON, 63)
        row['ewma_0.94'] = ewma_forecast(history.iloc[-500:], HORIZON)
        row['har_gk'] = har_forecast(gk.loc[:origin], history, HORIZON)[0]
        row['har_sq'] = har_forecast(history**2, history, HORIZON)[0]
        prices = adj.loc[:origin].to_numpy()
        seed = int(hashlib.sha256(f"{ticker}{row['origin']}".encode()).hexdigest()[:8], 16)
        for name, model, lookback in [('gjr_garch_252d', 'regression_gjr_garch', 253),
                                      ('volatility_gp_63d', 'volatility_gp_returns', 64)]:
            try:
                _, variance, _ = MODELS[model](prices[-lookback:, None], HORIZON, seed=seed, n_paths=2000)
                row[name] = float(variance[-1])
            except Exception:  # Recorded as a failed fit, never replaced.
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def score_vol(forecasts):
    scored = forecasts.dropna(subset=['realized']).copy()
    common = scored.dropna(subset=VOL_MODELS)
    rows = []
    monthly = {}
    for model in VOL_MODELS:
        loss = pd.Series(qlike(common[model], common.realized), index=common.index)
        monthly[model] = loss.groupby(common.origin).mean()
        z = common.realized_return/np.sqrt(common[model])
        rows.append(dict(model=model, failed_fits=int(scored[model].isna().sum()), mean_qlike=float(loss.mean()),
                         mse_log=float(np.mean((np.log(common[model])-np.log(common.realized))**2)),
                         mean_log_bias=float(np.mean(np.log(common.realized/common[model]))),
                         coverage_95=float(np.mean(np.abs(z) <= 1.96)),
                         kupiec_95_p=kupiec((np.abs(z) > 1.96).to_numpy(), .95)['p']))
    table = pd.DataFrame(rows)
    for model in VOL_MODELS:
        dm = diebold_mariano(monthly[model], monthly['har_gk'])
        wins = sum(qlike(g[model], g.realized).mean() < qlike(g['har_gk'], g.realized).mean()
                   for _, g in common.groupby('ticker'))
        table.loc[table.model == model, ['dm_vs_har_gk', 'dm_p', 'tickers_beating_har_gk']] = [
            dm['statistic'], dm['p'], wins]
    return table, len(common), common.origin.nunique()


def portfolios(panel, forecasts):
    adj = panel['Adj Close']
    simple = adj.pct_change().iloc[1:]
    logs = np.log(adj).diff().iloc[1:]
    har = forecasts.pivot(index='origin', columns='ticker', values='har_gk')
    har.index = pd.to_datetime(har.index)
    tickers = list(adj.columns)
    decisions = {name: {} for name in ['SPY', '60/40 SPY/IEF', 'Equal weight', 'Inverse vol (HAR)',
                                       'Risk parity (HAR + LW corr)', 'Risk parity (sample cov)',
                                       'Min var (sample cov)', 'Min var (Ledoit–Wolf)', 'Min var (PCA-5)',
                                       'Min var (HAR + LW corr)']}
    predicted = []
    cap = .25
    for origin in month_end_dates(simple.index):
        if origin < pd.Timestamp(FIRST_ORIGIN) or origin >= simple.index[-1]:
            continue
        window = logs.loc[:origin].iloc[-252:].to_numpy()
        vol = np.sqrt(har.loc[origin, tickers].to_numpy())
        corr, _ = shrunk_correlation(window)
        models = {'HAR + LW corr': covariance_from_vol_corr(vol, corr),
                  'sample cov': HORIZON*np.cov(window, rowvar=False),
                  'Ledoit–Wolf': HORIZON*ledoit_wolf(window)[0],
                  'PCA-5': HORIZON*factor_covariance(window, 5)}
        one_hot = lambda weights: pd.Series(weights, index=tickers).fillna(0.)
        chosen = {
            'SPY': (one_hot({'SPY': 1.}), 'HAR + LW corr'),
            '60/40 SPY/IEF': (one_hot({'SPY': .6, 'IEF': .4}), 'HAR + LW corr'),
            'Equal weight': (pd.Series(1/len(tickers), index=tickers), 'HAR + LW corr'),
            'Inverse vol (HAR)': (pd.Series(inverse_volatility(vol), index=tickers), 'HAR + LW corr'),
            'Risk parity (HAR + LW corr)': (pd.Series(equal_risk_contribution(models['HAR + LW corr']), index=tickers), 'HAR + LW corr'),
            'Risk parity (sample cov)': (pd.Series(equal_risk_contribution(models['sample cov']), index=tickers), 'sample cov'),
        }
        for model in ('sample cov', 'Ledoit–Wolf', 'PCA-5', 'HAR + LW corr'):
            chosen[f'Min var ({model})'] = (pd.Series(minimum_variance(models[model], cap), index=tickers), model)
        for name, (weights, model) in chosen.items():
            decisions[name][origin] = weights
            w = weights.to_numpy()
            predicted.append(dict(strategy=name, origin=origin, risk_model=model,
                                  predicted_var=float(w@models[model]@w)))
    results, daily = [], {}
    predicted = pd.DataFrame(predicted)
    for name, schedule in decisions.items():
        returns, weights = run_backtest(simple, schedule, cost_bps=COST_BPS)
        daily[name] = returns.net
        perf = performance(returns.net)
        months = returns.turnover.groupby(returns.index.to_period('M')).sum()
        # Risk-model check: predicted monthly variance vs next-month realized portfolio variance.
        realized = np.log1p(returns.gross).pow(2)
        realized_month = []
        for origin in predicted[predicted.strategy == name].origin:
            window = realized.loc[origin:].iloc[2:HORIZON+2]  # holding starts after the lagged execution
            realized_month.append(window.sum() if len(window) == HORIZON else np.nan)
        p = predicted[predicted.strategy == name].assign(realized_var=realized_month).dropna()
        sub = {f'sharpe_{a[:4]}_{b[:4]}': performance(returns.net.loc[a:b])['sharpe']
               for a, b in [('2010-01-01', '2017-12-31'), ('2018-01-01', '2026-12-31')]}
        results.append(dict(strategy=name, **perf, psr_vs_0=probabilistic_sharpe(returns.net),
                            mean_monthly_turnover=float(months.iloc[1:].mean()), total_cost=float(returns.cost.sum()),
                            mean_effective_holdings=float((1/(weights**2).sum(axis=1)).mean()),
                            risk_model=p.risk_model.iloc[0] if len(p) else '',
                            realized_to_predicted_vol=float(np.sqrt(p.realized_var.mean()/p.predicted_var.mean())),
                            portfolio_qlike=float(qlike(p.predicted_var, p.realized_var).mean()), **sub))
    return pd.DataFrame(results), pd.DataFrame(daily)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'reports/etf-risk-study-2026-10-27'))
    parser.add_argument('--workers', type=int, default=8)
    out = Path(parser.parse_args().output)
    out.mkdir(parents=True, exist_ok=True)
    panel, config = load_panel()
    cache = ROOT/'data_store/etf_study/vol_forecasts.csv'
    if cache.exists():
        forecasts = pd.read_csv(cache)
    else:
        jobs = [(t, panel['Adj Close'][t], panel['Open'][t], panel['High'][t], panel['Low'][t], panel['Close'][t])
                for t in panel['Adj Close'].columns]
        with ProcessPoolExecutor(parser.parse_args().workers) as pool:
            forecasts = pd.concat(pool.map(vol_forecasts, jobs), ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(cache, index=False)
    vol_table, n_rows, n_months = score_vol(forecasts)
    vol_table.to_csv(out/'vol_benchmark.csv', index=False)
    portfolio_table, daily = portfolios(panel, forecasts)
    portfolio_table.to_csv(out/'portfolios.csv', index=False)
    daily.to_csv(out/'portfolio_daily_net.csv')
    manifest = json.loads((ROOT/'data_store/etf/manifest.json').read_text())
    (out/'run_info.json').write_text(json.dumps(dict(
        tickers=list(panel['Adj Close'].columns), horizon=HORIZON, first_origin=FIRST_ORIGIN, cost_bps=COST_BPS,
        vol_rows_scored_common=n_rows, vol_months=n_months, data_retrieved_utc=manifest['retrieved_utc'],
        data_sha256={k: v['sha256'] for k, v in manifest['files'].items()}), indent=2))
    pd.set_option('display.width', 250)
    print(vol_table.round(4).to_string())
    print(portfolio_table.drop(columns=['start', 'end']).round(3).to_string())


if __name__ == '__main__':
    main()
