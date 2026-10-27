"""Re-score saved GP portfolio experiments with stock_api.evaluation (no refitting).

Reads the local (git-ignored) price caches and saved forecasts of the three GP
portfolio studies and writes CSV tables plus tables.md to the output folder.
Run from the repository root:

    .conda-env/bin/python scripts/rescore_saved_forecasts.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from stock_api.evaluation import (berkowitz, christoffersen, deflated_sharpe, diebold_mariano,
                                  factor_regression, ic_summary, information_coefficient, kupiec,
                                  load_french_factors, neutralize, pbo_cscv, quantile_spread, sharpe)

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {'22-stock': ROOT/'experiments/portfolio_gp_macro_news_2026-10-18',
           'growth-20': ROOT/'experiments/portfolio_gp_macro_news_growth20_2026-10-19'}
SIX_MONTH = ROOT/'experiments/portfolio_gp_22_2026-10-18'
TRAILING = 30


def trailing_returns(study, window, forecasts):
    """Log return over the TRAILING sessions ending at each forecast origin."""
    values = []
    cache = {}
    for ticker, origin in zip(forecasts.ticker, forecasts.origin):
        if ticker not in cache:
            path = study/'data/prices'/window/f'{ticker}.csv'
            prices = pd.read_csv(path, parse_dates=['Date'], index_col='Date')['Adj Close']
            cache[ticker] = np.log(prices)
        history = cache[ticker][cache[ticker].index <= origin]
        values.append(history.iloc[-1]-history.iloc[-1-TRAILING] if len(history) > TRAILING else np.nan)
    return np.array(values)


def signal_tables(name, study, window):
    forecasts = pd.read_csv(study/'forecasts'/f'window_{window}_forecasts.csv', parse_dates=['origin'])
    base = forecasts[forecasts.strategy == 'GP only'].copy()
    base['trailing'] = trailing_returns(study, window, base)
    base['gp_resid'] = neutralize(base, 'mean', ['trailing'], date='origin')
    rows = []
    for label, column, frame in [('GP drift', 'mean', base), (f'Trailing {TRAILING}d return', 'trailing', base),
                                 ('GP drift | trailing (residual)', 'gp_resid', base)]:
        ic = information_coefficient(frame, column, 'actual', date='origin')
        spread = quantile_spread(frame, column, 'actual', date='origin', quantiles=4)
        rows.append(dict(study=name, window=window, signal=label, **ic_summary(ic),
                         mean_top_minus_bottom_quartile=float(spread.mean())))
    rows.append(dict(study=name, window=window, signal='corr(GP drift, trailing) [mean XS Spearman]',
                     mean_ic=float(information_coefficient(base, 'mean', 'trailing', date='origin').mean())))
    for strategy, frame in forecasts.groupby('strategy'):
        if strategy == 'GP only':
            continue
        ic = information_coefficient(frame, 'mean', 'actual', date='origin')
        rows.append(dict(study=name, window=window, signal=f'{strategy} mean', **ic_summary(ic)))
    # Monthly CRPS (average over tickers) versus the price-only GP.
    crps = forecasts.groupby(['strategy', 'origin']).crps.mean().unstack(0)
    comparisons = [dict(study=name, window=window, strategy=s,
                        **diebold_mariano(crps[s], crps['GP only']))
                   for s in crps.columns if s != 'GP only']
    calibration = calibration_rows(name, window, base)
    return rows, comparisons, calibration


def calibration_rows(name, window, base):
    rows = []
    for level, low, high in [(.9, 'q05', 'q95'), (.68, 'q16', 'q84')]:
        hits = (base.actual < base[low]) | (base.actual > base[high])
        pooled = kupiec(hits.to_numpy(), level)
        per_ticker = [christoffersen(group.sort_values('origin').pipe(
            lambda g: ((g.actual < g[low]) | (g.actual > g[high])).to_numpy()), level)['p_conditional']
            for _, group in base.groupby('ticker')]
        rows.append(dict(study=name, window=window, test=f'{int(level*100)}% interval',
                         failure_rate=pooled['failure_rate'], expected=pooled['expected_rate'],
                         pooled_kupiec_p=pooled['p'],
                         tickers_rejecting_conditional_coverage=f'{np.sum(np.array(per_ticker) < .05)}/{len(per_ticker)}'))
    berk = [berkowitz(stats.norm.cdf(group.sort_values('origin').z))['p'] for _, group in base.groupby('ticker')]
    pit = stats.norm.cdf(base.z)
    deciles = np.histogram(pit, bins=10, range=(0, 1))[0]
    rows.append(dict(study=name, window=window, test='Gaussian-PIT Berkowitz',
                     tickers_rejecting=f'{np.sum(np.array(berk) < .05)}/{len(berk)}',
                     pit_decile_counts=' '.join(map(str, deciles))))
    return rows


def attribution_rows(name, daily, factors, gp='GP only'):
    """Factor regressions per strategy, plus the GP's active return over equal weight."""
    rows = []
    series = {key: group.set_index('date').net_equity.pct_change().dropna()
              for key, group in daily.groupby(['strategy', 'window'])}
    for window in daily.window.unique():
        if (gp, window) in series and ('Equal weight', window) in series:
            active = series[(gp, window)]-series[('Equal weight', window)]
            series[(f'{gp} minus Equal weight', window)] = active+factors.RF.reindex(active.index).fillna(0)
    for (strategy, window), returns in series.items():
        try:
            result = factor_regression(returns, factors)
        except ValueError:
            continue
        rows.append(dict(study=name, window=window, strategy=strategy, start=result['start'], end=result['end'],
                         alpha_annualized=result['alpha_annualized'], alpha_t=result['t']['alpha'],
                         **{f'beta_{k}': result['coefficients'][k] for k in ('MKT', 'UMD', 'STR')},
                         **{f't_{k}': result['t'][k] for k in ('MKT', 'UMD', 'STR')}, r2=result['r2']))
    return rows


def overfitting_rows():
    """DSR and PBO for the six-month 22-stock study's predefined sensitivity grid."""
    daily = pd.read_csv(SIX_MONTH/'portfolio_daily.csv', parse_dates=['date'])
    returns = daily.pivot(index='date', columns='strategy', values='net_equity').pct_change().dropna()
    grid = [c for c in returns.columns if c.startswith('GP sensitivity')]+['Heteroskedastic GP optimizer']
    trial_sharpes = np.array([sharpe(returns[c]) for c in grid])
    primary = returns['Heteroskedastic GP optimizer']
    rows = [dict(test=f'Deflated Sharpe, primary GP vs its {len(grid)}-config grid', **deflated_sharpe(
        primary, len(grid), trial_sharpes.var(ddof=1)), annualized_sharpe=sharpe(primary)*np.sqrt(252))]
    for trials in (50, 100):
        rows.append(dict(test=f'Deflated Sharpe, assuming {trials} configurations tried repository-wide',
                         **deflated_sharpe(primary, trials, trial_sharpes.var(ddof=1))))
    rows.append(dict(test='PBO (CSCV, 8 blocks) across the grid', **pbo_cscv(returns[grid].to_numpy(), blocks=8)))
    for study_name, study in STUDIES.items():
        study_daily = pd.read_csv(study/'portfolios/daily.csv', parse_dates=['date'])
        for window in ('A', 'B'):
            frame = study_daily[(study_daily.window == window) & study_daily.strategy.str.startswith('GP')]
            matrix = frame.pivot(index='date', columns='strategy', values='net_equity').pct_change().dropna()
            rows.append(dict(test=f'PBO across GP variants, {study_name} window {window}',
                             **pbo_cscv(matrix.to_numpy(), blocks=8), configurations=matrix.shape[1]))
    return rows


def markdown(frame, digits=3):
    frame = frame.copy()
    for column in frame.select_dtypes('number'):
        frame[column] = frame[column].map(lambda v: '' if pd.isna(v) else f'{v:.{digits}g}' if abs(v) < 1e-3 and v else f'{v:.{digits}f}')
    lines = ['| '+' | '.join(frame.columns)+' |', '|'+'---|'*len(frame.columns)]
    lines += ['| '+' | '.join(map(str, row))+' |' for row in frame.fillna('').to_numpy()]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'reports/evaluation-rescore-2026-10-27'))
    out = Path(parser.parse_args().output)
    out.mkdir(parents=True, exist_ok=True)
    factors = load_french_factors('daily', cache_dir=ROOT/'data_store/french')
    signals, comparisons, calibration, attribution = [], [], [], []
    for name, study in STUDIES.items():
        for window in ('A', 'B'):
            s, c, k = signal_tables(name, study, window)
            signals += s; comparisons += c; calibration += k
        attribution += attribution_rows(name, pd.read_csv(study/'portfolios/daily.csv', parse_dates=['date']), factors)
    six = pd.read_csv(SIX_MONTH/'portfolio_daily.csv', parse_dates=['date']).assign(window='Apr-Oct 2026')
    six = six[~six.strategy.str.startswith('GP sensitivity')]
    attribution += attribution_rows('22-stock 6-month', six, factors, gp='Heteroskedastic GP optimizer')
    tables = {'signals': pd.DataFrame(signals), 'crps_dm': pd.DataFrame(comparisons),
              'calibration': pd.DataFrame(calibration), 'attribution': pd.DataFrame(attribution),
              'overfitting': pd.DataFrame(overfitting_rows())}
    with open(out/'tables.md', 'w') as handle:
        for key, frame in tables.items():
            frame.to_csv(out/f'{key}.csv', index=False)
            handle.write(f'## {key}\n\n{markdown(frame)}\n\n')
    print(f'Wrote {len(tables)} tables to {out}')


if __name__ == '__main__':
    main()
