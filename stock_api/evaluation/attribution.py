"""Factor attribution against the Kenneth French data library."""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from stock_api.evaluation.stats import long_run_variance

FRENCH_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip'
FRENCH_FILES = {
    'daily': ['F-F_Research_Data_5_Factors_2x3_daily', 'F-F_Momentum_Factor_daily',
              'F-F_ST_Reversal_Factor_daily'],
    'monthly': ['F-F_Research_Data_5_Factors_2x3', 'F-F_Momentum_Factor', 'F-F_ST_Reversal_Factor'],
}


def parse_french_csv(text, frequency):
    """Parse the first data block of a French library CSV into decimal returns."""
    digits = 8 if frequency == 'daily' else 6
    header, rows = None, []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(',')]
        if header is None:
            if len(cells) > 1 and cells[0] == '' and any(cells[1:]):
                header = cells[1:]
            continue
        if re.fullmatch(r'\d{%d}' % digits, cells[0]):
            rows.append(cells[:len(header)+1])
        elif rows:
            break  # The annual block follows the monthly block.
    if header is None or not rows:
        raise ValueError('Unrecognized French data file')
    fmt = '%Y%m%d' if frequency == 'daily' else '%Y%m'
    index = pd.to_datetime([r[0] for r in rows], format=fmt)
    if frequency == 'monthly':
        index = index+pd.offsets.MonthEnd(0)
    values = np.array([[float(v) for v in r[1:]] for r in rows])/100
    columns = [c.replace('Mom', 'UMD').replace('ST_Rev', 'STR').replace('Mkt-RF', 'MKT') for c in header]
    return pd.DataFrame(values, index=index, columns=columns)


def load_french_factors(frequency='daily', cache_dir='data_store/french', refresh=False):
    """FF5 + momentum (UMD) + short-term reversal (STR), cached locally."""
    import requests
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in FRENCH_FILES[frequency]:
        path = cache/f'{name}.zip'
        if refresh or not path.exists():
            response = requests.get(FRENCH_URL.format(name), timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        with zipfile.ZipFile(path) as archive:
            text = archive.read(archive.namelist()[0]).decode('latin-1')
        frames.append(parse_french_csv(text, frequency))
    factors = pd.concat(frames, axis=1, join='inner')
    return factors.loc[:, ~factors.columns.duplicated()]


def factor_regression(returns, factors, columns=('MKT', 'SMB', 'HML', 'RMW', 'CMA', 'UMD', 'STR'),
                      periods_per_year=252, lags=None):
    """OLS of excess strategy returns on factors with Newey–West standard errors.

    returns: simple period returns indexed by date. RF is subtracted when present.
    """
    data = pd.concat([returns.rename('r'), factors], axis=1, join='inner').dropna()
    if len(data) < len(columns)+10:
        raise ValueError('Too few overlapping return/factor observations')
    y = data['r']-(data['RF'] if 'RF' in data else 0)
    x = np.column_stack([np.ones(len(data)), data[list(columns)].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, y.to_numpy(), rcond=None)
    residual = y.to_numpy()-x@beta
    n, k = x.shape
    lags = int(np.floor(4*(n/100)**(2/9))) if lags is None else lags
    # Newey–West sandwich covariance.
    scores = x*residual[:, None]
    meat = scores.T@scores
    for lag in range(1, lags+1):
        gamma = scores[lag:].T@scores[:-lag]
        meat += (1-lag/(lags+1))*(gamma+gamma.T)
    bread = np.linalg.inv(x.T@x)
    se = np.sqrt(np.diag(bread@meat@bread))
    names = ['alpha', *columns]
    r2 = 1-residual@residual/np.sum((y-y.mean())**2)
    return dict(coefficients=dict(zip(names, beta.tolist())), t=dict(zip(names, (beta/se).tolist())),
                alpha_annualized=float(beta[0]*periods_per_year), r2=float(r2), n=n,
                start=str(data.index[0].date()), end=str(data.index[-1].date()),
                residual_vol_annualized=float(np.sqrt(long_run_variance(residual, 0)*periods_per_year)))
