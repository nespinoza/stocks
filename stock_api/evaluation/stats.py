"""HAC standard errors and Diebold–Mariano forecast comparison."""
import numpy as np
from scipy import stats


def newey_west_lags(n):
    """Andrews/Newey–West rule-of-thumb bandwidth."""
    return int(np.floor(4*(n/100)**(2/9)))


def long_run_variance(x, lags=None):
    """Bartlett-kernel (Newey–West) long-run variance of a 1-D series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        raise ValueError('Need at least two finite observations')
    lags = newey_west_lags(n) if lags is None else int(lags)
    e = x-x.mean()
    variance = e@e/n
    for lag in range(1, min(lags, n-1)+1):
        variance += 2*(1-lag/(lags+1))*(e[lag:]@e[:-lag])/n
    return max(variance, 1e-300)


def hac_mean_test(x, lags=None):
    """Mean, HAC standard error, t-statistic and two-sided p-value (t, n-1 df)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    se = np.sqrt(long_run_variance(x, lags)/n)
    t = x.mean()/se
    return dict(mean=float(x.mean()), se=float(se), t=float(t),
                p=float(2*stats.t.sf(abs(t), n-1)), n=n)


def diebold_mariano(loss_a, loss_b, horizon=1):
    """Test equal expected loss of paired forecasts A and B.

    Negative statistic: A has lower loss. Uses Newey–West variance with
    horizon-1 lags at minimum and the Harvey–Leybourne–Newbold small-sample
    correction, compared with a t distribution.
    """
    d = np.asarray(loss_a, float)-np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        raise ValueError('Diebold–Mariano needs at least three paired losses')
    lags = max(horizon-1, newey_west_lags(n))
    dm = d.mean()/np.sqrt(long_run_variance(d, lags)/n)
    h = horizon
    correction = np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    statistic = dm*correction
    return dict(mean_difference=float(d.mean()), statistic=float(statistic),
                p=float(2*stats.t.sf(abs(statistic), n-1)), n=n, lags=lags)
