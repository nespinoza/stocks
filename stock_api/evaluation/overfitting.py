"""Backtest-overfitting statistics: PSR, deflated Sharpe ratio and CSCV PBO.

References: Bailey & López de Prado (2012, 2014); Bailey et al. (2017) for PBO.
Sharpe ratios here are per period (not annualized) unless stated otherwise.
"""
from itertools import combinations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe(returns):
    returns = np.asarray(returns, float)
    return float(returns.mean()/returns.std(ddof=1))


def probabilistic_sharpe(returns, benchmark_sharpe=0.0):
    """P(true per-period Sharpe > benchmark), allowing for skew and fat tails."""
    returns = np.asarray(returns, float)
    n = len(returns)
    sr = sharpe(returns)
    skew = stats.skew(returns)
    kurtosis = stats.kurtosis(returns, fisher=False)
    denominator = np.sqrt(max(1-skew*sr+(kurtosis-1)/4*sr**2, 1e-12))
    return float(stats.norm.cdf((sr-benchmark_sharpe)*np.sqrt(n-1)/denominator))


def expected_max_sharpe(n_trials, trial_sharpe_variance):
    """Expected maximum per-period Sharpe among n_trials skill-less strategies."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1-1/n_trials)
    z2 = stats.norm.ppf(1-1/(n_trials*np.e))
    return float(np.sqrt(trial_sharpe_variance)*((1-EULER_GAMMA)*z1+EULER_GAMMA*z2))


def deflated_sharpe(returns, n_trials, trial_sharpe_variance):
    """PSR against the expected best Sharpe of n_trials independent null trials."""
    threshold = expected_max_sharpe(n_trials, trial_sharpe_variance)
    return dict(dsr=probabilistic_sharpe(returns, threshold), sharpe=sharpe(returns),
                threshold_sharpe=threshold, n_trials=int(n_trials), n=len(returns))


def pbo_cscv(returns, blocks=10, metric=sharpe):
    """Probability of backtest overfitting by combinatorially symmetric CV.

    returns: (T, N) per-period returns of N configurations over the same dates.
    For each half/half split of the T rows into blocks, the in-sample best
    configuration's out-of-sample relative rank is recorded; PBO is the share
    of splits in which it falls at or below the out-of-sample median.
    """
    returns = np.asarray(returns, float)
    t, n = returns.shape
    if blocks % 2 or blocks < 2 or t < blocks or n < 2:
        raise ValueError('Need an even block count, at least that many rows, and two configurations')
    parts = np.array_split(np.arange(t), blocks)
    logits = []
    for chosen in combinations(range(blocks), blocks//2):
        train = np.concatenate([parts[i] for i in chosen])
        test = np.concatenate([parts[i] for i in range(blocks) if i not in chosen])
        is_scores = np.array([metric(returns[train, j]) for j in range(n)])
        os_scores = np.array([metric(returns[test, j]) for j in range(n)])
        best = int(np.argmax(is_scores))
        rank = (stats.rankdata(os_scores)[best])/(n+1)
        logits.append(np.log(rank/(1-rank)))
    logits = np.array(logits)
    return dict(pbo=float((logits <= 0).mean()), splits=len(logits), median_logit=float(np.median(logits)))
