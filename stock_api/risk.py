"""Volatility and covariance forecasts: realized-variance proxies, HAR-RV, EWMA,
Ledoit–Wolf shrinkage and a statistical (PCA) factor model.

All functions use only the data they are given; callers slice history at the
decision date.
"""
import numpy as np
import pandas as pd


def garman_klass(open_, high, low, close):
    """Garman–Klass daily variance from same-day OHLC (adjustment-invariant ratios)."""
    hl = np.log(high/low)
    co = np.log(close/open_)
    return np.maximum(.5*hl**2-(2*np.log(2)-1)*co**2, 0)


def _har_design(proxy):
    """log of 1-, 5- and 22-day mean proxy variance, aligned to the last day of each window."""
    floor = max(float(proxy[proxy > 0].min()) if (proxy > 0).any() else 1e-12, 1e-12)
    clean = proxy.clip(lower=floor)
    return pd.DataFrame({'d': np.log(clean), 'w': np.log(clean.rolling(5).mean()),
                         'm': np.log(clean.rolling(22).mean())})


def har_forecast(proxy, log_returns, horizon=21, window=1500, min_obs=500):
    """Log-HAR forecast of the next `horizon` days' summed squared log returns.

    proxy: daily variance proxy (Garman–Klass or squared returns) on the same
    index as log_returns, ending at the decision date. The target is close-to-
    close realized variance, so an intraday proxy's missing overnight variance
    is absorbed by the regression. Returns (forecast, fit diagnostics).
    """
    proxy, log_returns = proxy.iloc[-window-horizon:], log_returns.iloc[-window-horizon:]
    design = _har_design(proxy)
    squared = log_returns**2
    # Target at row t: sum of squared returns over t+1..t+horizon (observed only for t <= T-horizon).
    target = np.log(squared[::-1].rolling(horizon).sum()[::-1].shift(-1).clip(lower=1e-12))
    data = pd.concat([design, target.rename('y')], axis=1).dropna()
    if len(data) < min_obs:
        raise ValueError('Not enough history for HAR')
    x = np.column_stack([np.ones(len(data)), data[['d', 'w', 'm']].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, data.y.to_numpy(), rcond=None)
    residual = data.y.to_numpy()-x@beta
    # Overlapping targets make residuals autocorrelated; the variance is still consistent.
    s2 = residual@residual/(len(data)-4)
    latest = np.r_[1., design.iloc[-1].to_numpy()]
    return float(np.exp(latest@beta+s2/2)), {'beta': beta.tolist(), 'residual_var': float(s2), 'n': len(data)}


def ewma_forecast(log_returns, horizon=21, decay=.94):
    """RiskMetrics EWMA daily variance, scaled to the horizon (flat term structure)."""
    weights = (1-decay)*decay**np.arange(len(log_returns))[::-1]
    return float(horizon*np.sum(weights*log_returns.to_numpy()**2)/weights.sum())


def rolling_forecast(log_returns, horizon=21, window=63):
    """Sample variance of the last `window` daily log returns, scaled to the horizon."""
    return float(horizon*np.var(log_returns.to_numpy()[-window:], ddof=1))


def ledoit_wolf(returns):
    """Ledoit–Wolf (2004) shrinkage of the sample covariance toward a scaled identity.

    returns: (n, p) array. Returns (covariance, shrinkage intensity).
    """
    x = np.asarray(returns, float)
    x = x-x.mean(axis=0)
    n, p = x.shape
    sample = x.T@x/n
    mu = np.trace(sample)/p
    target = mu*np.eye(p)
    d2 = np.sum((sample-target)**2)/p
    b2 = sum(np.sum((np.outer(row, row)-sample)**2) for row in x)/(n**2*p)
    shrinkage = min(b2, d2)/d2 if d2 > 0 else 1.
    return shrinkage*target+(1-shrinkage)*sample, float(shrinkage)


def shrunk_correlation(returns):
    """Ledoit–Wolf shrinkage applied to standardized returns, rescaled to unit diagonal."""
    x = np.asarray(returns, float)
    z = (x-x.mean(axis=0))/x.std(axis=0, ddof=1)
    cov, shrinkage = ledoit_wolf(z)
    scale = np.sqrt(np.diag(cov))
    return cov/np.outer(scale, scale), shrinkage


def factor_covariance(returns, factors=5):
    """Statistical factor model: top principal components plus diagonal residuals."""
    x = np.asarray(returns, float)
    sample = np.cov(x, rowvar=False)
    values, vectors = np.linalg.eigh(sample)
    order = np.argsort(values)[::-1][:factors]
    loadings = vectors[:, order]*np.sqrt(values[order])
    common = loadings@loadings.T
    residual = np.clip(np.diag(sample-common), 1e-12, None)
    return common+np.diag(residual)


def covariance_from_vol_corr(volatility, correlation):
    volatility = np.asarray(volatility, float)
    return np.asarray(correlation)*np.outer(volatility, volatility)
