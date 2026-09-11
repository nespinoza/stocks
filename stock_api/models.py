"""Forecast log prices; return target marginal means and variances.

All transforms and parameter estimates use only the supplied training window.
The target is always column zero. No future auxiliary prices are required.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def random_walk(prices, steps, *, include_fit=False):
    logs = np.log(prices[:, 0])
    variance = max(float(np.var(np.diff(logs), ddof=1)), 1e-12)
    info = {}
    if include_fit:
        info["_fit"] = (1, logs[:-1], np.full(len(logs)-1, variance))
    return np.repeat(logs[-1], steps), variance * np.arange(1, steps + 1), info


def multitask_gp(prices, steps, *, include_fit=False, train_times=None, future_times=None):
    """Independent ticker GPs, fitted jointly with one shared Matern-3/2 timescale.

Each ticker has its own amplitude and observation noise. The joint likelihood
is the sum of ticker likelihoods; other tickers constrain only the timescale.
    """
    logs = np.log(prices)
    center = logs.mean(axis=0)
    scale = np.maximum(logs.std(axis=0), 1e-6)
    y = (logs-center)/scale
    n, tasks = y.shape
    x = np.arange(n, dtype=float)
    future = np.arange(n, n+steps, dtype=float)
    if train_times is not None or future_times is not None:
        x = np.asarray(train_times, dtype=float)
        future = np.asarray(future_times, dtype=float)
        if (x.shape != (n,) or future.shape != (steps,) or
                not np.isfinite(x).all() or not np.isfinite(future).all() or
                (np.diff(x) <= 0).any() or (np.diff(future) <= 0).any() or future[0] <= x[-1]):
            raise ValueError('Provide increasing training and future times matching the observations')
        future = future-x[0]
        x = x-x[0]
    identity = np.eye(n)
    distance = np.sqrt(3)*np.abs(x[:, None]-x[None, :])

    def objective(params):
        length = np.exp(params[0])
        amplitudes = np.exp(params[1:1+tasks])
        noises = np.exp(params[1+tasks:])
        r = distance/length
        base = (1+r)*np.exp(-r)
        length_derivative = r*r*np.exp(-r)
        value = 0.0
        gradient = np.zeros_like(params)
        for j in range(tasks):
            k = amplitudes[j]**2*base
            factor = cho_factor(k+(noises[j]**2+1e-8)*identity, lower=True)
            alpha = cho_solve(factor, y[:, j])
            value += .5*y[:, j]@alpha+np.log(np.diag(factor[0])).sum()
            q = cho_solve(factor, identity)-np.outer(alpha, alpha)
            gradient[0] += .5*np.sum(q*(amplitudes[j]**2*length_derivative))
            gradient[1+j] = np.sum(q*k)
            gradient[1+tasks+j] = noises[j]**2*np.trace(q)
        return float(value), gradient

    fit = minimize(objective, np.log([20]+[1]*tasks+[.1]*tasks), jac=True, method='L-BFGS-B',
                   bounds=np.log([[1,500]]+[[.01,10]]*tasks+[[.001,2]]*tasks),
                   options={'maxiter':150})
    if not np.isfinite(fit.fun):
        raise ValueError('GP optimization did not produce a finite fit')
    length = np.exp(fit.x[0])
    amplitudes = np.exp(fit.x[1:1+tasks])
    noises = np.exp(fit.x[1+tasks:])
    r = distance/length
    k = amplitudes[0]**2*(1+r)*np.exp(-r)
    factor = cho_factor(k+(noises[0]**2+1e-8)*identity, lower=True)
    evaluation = np.r_[x, future] if include_fit else future
    r_cross = np.sqrt(3)*np.abs(x[:, None]-evaluation[None, :])/length
    cross = amplitudes[0]**2*(1+r_cross)*np.exp(-r_cross)
    mean = cross.T@cho_solve(factor, y[:, 0])
    variance = np.maximum(amplitudes[0]**2+noises[0]**2
                          -np.sum(cross*cho_solve(factor, cross), axis=0), 1e-12)
    mean = mean*scale[0]+center[0]
    variance = variance*scale[0]**2
    info = {'length_scale':float(length),
            'time_unit':'calendar_days' if train_times is not None else 'trading_sessions',
            'amplitudes':(amplitudes*scale).tolist(),
            'observation_noises':(noises*scale).tolist(),
            'standardized_amplitudes':amplitudes.tolist(),
            'standardized_noises':noises.tolist(), 'converged':bool(fit.success),
            'kernel':'matern_3_2', 'shared_parameter':'length_scale'}
    if include_fit:
        info['_fit'] = (0, mean[:n], variance[:n])
        mean, variance = mean[n:], variance[n:]
    return mean, variance, info


def var(prices, steps, lags=5, ridge=10.0, *, include_fit=False):
    """Ridge VAR on standardized returns, with propagated innovation variance."""
    logs = np.log(prices)
    returns = np.diff(logs, axis=0)
    scale = np.maximum(returns.std(axis=0), 1e-6)
    z = returns / scale
    tasks = z.shape[1]
    x = np.array([z[i-lags:i][::-1].reshape(-1) for i in range(lags, len(z))])
    x = np.column_stack([np.ones(len(x)), x])
    y = z[lags:]
    penalty = np.diag([0.0] + [ridge] * (tasks * lags))
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    residuals = y - x @ beta
    innovation = residuals.T @ residuals / max(len(y) - 1, 1)
    # State = latest lags of returns followed by cumulative target log return.
    size = tasks * lags
    transition = np.zeros((size + 1, size + 1))
    transition[:tasks, :size] = beta[1:].T
    transition[tasks:size, :size-tasks] = np.eye(size-tasks)
    transition[-1, :size] = scale[0] * beta[1:, 0]
    transition[-1, -1] = 1
    injection = np.zeros((size + 1, tasks))
    injection[:tasks] = np.eye(tasks)
    injection[-1, 0] = scale[0]
    q = injection @ innovation @ injection.T
    state = np.r_[z[-lags:][::-1].reshape(-1), 0.0]
    intercept = np.r_[beta[0], np.zeros(size-tasks), scale[0] * beta[0, 0]]
    covariance = np.zeros_like(transition)
    means, variances = [], []
    for _ in range(steps):
        state = transition @ state + intercept
        covariance = transition @ covariance @ transition.T + q
        means.append(logs[-1, 0] + state[-1])
        variances.append(max(covariance[-1, -1], 1e-12))
    info = {"lags":lags, "ridge":ridge}
    if include_fit:
        fitted = logs[lags:-1, 0] + (x @ beta)[:, 0] * scale[0]
        info["_fit"] = (lags + 1, fitted,
                        np.full(len(fitted), max(innovation[0, 0] * scale[0]**2, 1e-12)))
    return np.array(means), np.array(variances), info


# Explicit persistence baseline; random_walk remains a compatible name.
last_price = random_walk


def gp(prices, steps, *, include_fit=False, train_times=None, future_times=None):
    """Single-ticker Matern-3/2 GP; auxiliary columns never enter the fit."""
    return multitask_gp(prices[:, :1], steps, include_fit=include_fit,
                        train_times=train_times, future_times=future_times)


MODELS = {"last_price":last_price, "random_walk":random_walk, "gp":gp,
          "multitask_gp":multitask_gp, "var":var}

# SDE benchmarks use the same log-marginal / diagnostics interface.
from stock_api.sde import SDE_MODELS
MODELS.update(SDE_MODELS)
