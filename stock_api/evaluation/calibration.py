"""Calibration of predictive distributions: PIT, Berkowitz, VaR backtests, QLIKE."""
import numpy as np
from scipy import stats
from scipy.special import xlogy


def pit_from_samples(samples, actual):
    """Probability integral transform; samples have shape (draws, ...)."""
    samples = np.asarray(samples, float)
    return (samples <= np.asarray(actual, float)).mean(axis=0)


def pit_from_quantiles(levels, quantiles, actual):
    """Linearly interpolated PIT from predictive quantiles (rows = forecasts).

    Values beyond the outermost quantiles are clipped to half the tail mass,
    so with few quantiles the PIT is coarse in the tails.
    """
    levels = np.asarray(levels, float)
    quantiles = np.atleast_2d(np.asarray(quantiles, float))
    actual = np.atleast_1d(np.asarray(actual, float))
    pit = np.array([np.interp(a, q, levels, left=levels[0]/2, right=(1+levels[-1])/2)
                    for q, a in zip(quantiles, actual)])
    return pit


def berkowitz(pit, eps=1e-6):
    """Berkowitz (2001) LR test of iid N(0,1) for inverse-normal PIT values.

    Fits z_t = c + rho z_{t-1} + sigma e_t and tests c=0, rho=0, sigma=1 (chi2, 3 df).
    """
    z = stats.norm.ppf(np.clip(np.asarray(pit, float), eps, 1-eps))
    if len(z) < 10:
        raise ValueError('Berkowitz test needs at least ten PIT values')
    y, x = z[1:], np.column_stack([np.ones(len(z)-1), z[:-1]])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y-x@beta
    sigma2 = residual@residual/len(y)
    fitted = stats.norm.logpdf(residual, scale=np.sqrt(sigma2)).sum()
    null = stats.norm.logpdf(y).sum()
    lr = 2*(fitted-null)
    return dict(lr=float(lr), p=float(stats.chi2.sf(lr, 3)), intercept=float(beta[0]),
                rho=float(beta[1]), sigma=float(np.sqrt(sigma2)), n=len(z))


def kupiec(hits, coverage):
    """Kupiec proportion-of-failures test; hits are True when outside the interval.

    coverage is the nominal interval probability (e.g. 0.9), so the expected
    failure rate is 1-coverage.
    """
    hits = np.asarray(hits, bool)
    n, x, p = len(hits), int(hits.sum()), 1-coverage
    rate = x/n
    lr = -2*(xlogy(n-x, 1-p)+xlogy(x, p)-xlogy(n-x, 1-rate)-xlogy(x, rate))
    return dict(failures=x, n=n, failure_rate=rate, expected_rate=p,
                lr=float(lr), p=float(stats.chi2.sf(lr, 1)))


def christoffersen(hits, coverage):
    """Christoffersen independence and conditional-coverage tests for a time-ordered hit series."""
    hits = np.asarray(hits, int)
    prev, curr = hits[:-1], hits[1:]
    n00, n01 = np.sum((prev == 0) & (curr == 0)), np.sum((prev == 0) & (curr == 1))
    n10, n11 = np.sum((prev == 1) & (curr == 0)), np.sum((prev == 1) & (curr == 1))
    pi01 = n01/max(n00+n01, 1)
    pi11 = n11/max(n10+n11, 1)
    pi = (n01+n11)/max(n00+n01+n10+n11, 1)
    restricted = xlogy(n00+n10, 1-pi)+xlogy(n01+n11, pi)
    unrestricted = xlogy(n00, 1-pi01)+xlogy(n01, pi01)+xlogy(n10, 1-pi11)+xlogy(n11, pi11)
    lr_ind = float(-2*(restricted-unrestricted))
    lr_cc = kupiec(hits.astype(bool), coverage)['lr']+lr_ind
    return dict(lr_independence=lr_ind, p_independence=float(stats.chi2.sf(lr_ind, 1)),
                lr_conditional=float(lr_cc), p_conditional=float(stats.chi2.sf(lr_cc, 2)))


def qlike(predicted_variance, realized_variance):
    """Patton (2011) QLIKE loss, robust to noise in the realized-variance proxy."""
    ratio = np.asarray(realized_variance, float)/np.asarray(predicted_variance, float)
    return ratio-np.log(ratio)-1
