"""Cross-sectional signal evaluation on long panels (date, asset, signal, outcome)."""
import numpy as np
import pandas as pd

from stock_api.evaluation.stats import hac_mean_test


def information_coefficient(frame, signal, outcome, date='date', method='spearman', min_assets=5):
    """Per-date cross-sectional correlation between a signal and the realized outcome."""
    def one(group):
        group = group[[signal, outcome]].dropna()
        if len(group) < min_assets or group[signal].nunique() < 2:
            return np.nan
        return group[signal].corr(group[outcome], method=method)
    return frame.groupby(date).apply(one, include_groups=False).dropna().rename('ic')


def ic_summary(ic, lags=None):
    """Mean IC with a Newey–West t-statistic; dates are the units of evidence."""
    test = hac_mean_test(ic.to_numpy(), lags)
    return dict(mean_ic=test['mean'], ic_std=float(ic.std(ddof=1)), t=test['t'], p=test['p'],
                hit_rate=float((ic > 0).mean()), dates=test['n'],
                ic_ir=float(test['mean']/ic.std(ddof=1)) if len(ic) > 1 else np.nan)


def neutralize(frame, signal, controls, date='date'):
    """Residual of the signal after per-date cross-sectional OLS on the controls."""
    def one(group):
        clean = group[[signal, *controls]].dropna()
        out = pd.Series(np.nan, index=group.index)
        if len(clean) <= len(controls)+1:
            return out
        x = np.column_stack([np.ones(len(clean)), clean[controls].to_numpy()])
        beta, *_ = np.linalg.lstsq(x, clean[signal].to_numpy(), rcond=None)
        out.loc[clean.index] = clean[signal].to_numpy()-x@beta
        return out
    return frame.groupby(date, group_keys=False)[[signal, *controls]].apply(one).reindex(frame.index)


def quantile_spread(frame, signal, outcome, date='date', quantiles=5):
    """Per-date mean outcome of the top minus the bottom signal quantile."""
    def one(group):
        group = group[[signal, outcome]].dropna()
        if len(group) < quantiles:
            return np.nan
        ranks = pd.qcut(group[signal].rank(method='first'), quantiles, labels=False)
        return group[outcome][ranks == quantiles-1].mean()-group[outcome][ranks == 0].mean()
    return frame.groupby(date).apply(one, include_groups=False).dropna().rename('spread')
