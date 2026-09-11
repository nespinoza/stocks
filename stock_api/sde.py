"""Scalar Itô simulation and training-only SDE benchmarks.

One time unit is one trading session: daily close-to-close returns use dt=1.
GBM uses exact transitions. Euler–Maruyama is an independent extension point.
All variances condition on fitted parameters (no parameter uncertainty).
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def _rng(seed=None, rng=None):
    if seed is not None and rng is not None:
        raise ValueError('Pass either seed or rng, not both')
    return np.random.default_rng(seed) if rng is None else rng


def _simulation_inputs(x0, steps, dt, n_paths):
    if not np.isfinite(x0) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('x0 must be finite and dt must be finite and positive')
    for name, value in [('steps', steps), ('n_paths', n_paths)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')


def euler_maruyama(x0, drift, diffusion, *, steps, dt=1.0, n_paths=10_000,
                   seed=None, rng=None, normal_draws=None):
    """Simulate dX=a(X,t)dt+b(X,t)dW; return (paths, steps+1), including x0.

    drift/diffusion accept a vector of states and a scalar time in trading days.
    Scalar outputs are broadcast. No positivity clipping is applied to a generic
    SDE. Subdivide dt for convergence; nonfinite states raise an error.
    normal_draws optionally supplies standard normals for coupled solver tests.
    """
    _simulation_inputs(x0, steps, dt, n_paths)
    generator = _rng(seed, rng)
    if normal_draws is not None:
        normal_draws = np.asarray(normal_draws, dtype=float)
        if normal_draws.shape != (n_paths, steps) or not np.isfinite(normal_draws).all():
            raise ValueError('normal_draws must have shape (n_paths, steps) and be finite')
    paths = np.empty((n_paths, steps+1))
    paths[:, 0] = x0
    with np.errstate(over='raise', invalid='raise'):
        for i in range(steps):
            x = paths[:, i]
            a = np.broadcast_to(np.asarray(drift(x.copy(), i*dt), dtype=float), x.shape)
            b = np.broadcast_to(np.asarray(diffusion(x.copy(), i*dt), dtype=float), x.shape)
            z = generator.standard_normal(n_paths) if normal_draws is None else normal_draws[:, i]
            paths[:, i+1] = x+a*dt+b*np.sqrt(dt)*z
            if not np.isfinite(paths[:, i+1]).all():
                raise ValueError('SDE produced nonfinite states')
    return paths


class ItoProcess:
    """Override drift and diffusion, or use euler_maruyama with callables."""
    def drift(self, x, t):
        raise NotImplementedError

    def diffusion(self, x, t):
        raise NotImplementedError

    def simulate(self, x0, **kwargs):
        return euler_maruyama(x0, self.drift, self.diffusion, **kwargs)


def simulate_gbm(s0, mu, sigma, *, steps, dt=1.0, n_paths=10_000, seed=None, rng=None):
    """Exact GBM paths, including initial price; mu is arithmetic-price drift/day."""
    _simulation_inputs(s0, steps, dt, n_paths)
    if s0 <= 0 or not np.isfinite(mu) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('GBM requires positive s0, finite mu and nonnegative sigma')
    generator = _rng(seed, rng)
    increments = (mu-.5*sigma**2)*dt+sigma*np.sqrt(dt)*generator.standard_normal((n_paths, steps))
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        paths = s0*np.exp(np.column_stack([np.zeros(n_paths), np.cumsum(increments, axis=1)]))
    if not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError('GBM paths overflowed or underflowed')
    return paths


def lognormal_statistics(means, variances, initial):
    """Exact price moments, quantiles and strict P(S>S0) of Gaussian log prices."""
    means, variances = np.asarray(means), np.asarray(variances)
    if not np.isfinite(means).all() or not np.isfinite(variances).all() or (variances < 0).any():
        raise ValueError('Invalid Gaussian log-price distribution')
    sd = np.sqrt(variances)
    with np.errstate(over='raise', invalid='raise'):
        median = np.exp(means)
        price_mean = np.exp(means+.5*variances)
        price_std = price_mean*np.sqrt(np.expm1(variances))
        probability = norm.cdf((means-np.log(initial))/np.maximum(sd, 1e-300))
        probability = np.where(sd == 0, (means > np.log(initial)).astype(float), probability)
        return {'mean_price':price_mean, 'median_price':median, 'price_std':price_std,
                'lower_95':np.exp(means+norm.ppf(.025)*sd),
                'upper_95':np.exp(means+norm.ppf(.975)*sd), 'probability_up':probability}


@dataclass
class PredictiveEnsemble:
    """Price trajectories include day zero. Return arrays are dimensionless."""
    paths: np.ndarray

    @property
    def terminal_log_returns(self):
        return np.log(self.paths[:, -1]/self.paths[:, 0])

    @property
    def terminal_returns(self):
        return self.paths[:, -1]/self.paths[:, 0]-1


def _training(prices):
    values = np.asarray(prices, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or len(values) < 4 or not np.isfinite(values[:, 0]).all() or (values[:, 0] <= 0).any():
        raise ValueError('At least four finite positive target prices are required')
    logs = np.log(values[:, 0])
    return logs, np.diff(logs)


def _gbm(prices, steps, *, estimate_drift, include_fit=False, rng=None, seed=None, n_paths=10_000):
    logs, returns = _training(prices)
    variance = float(np.var(returns, ddof=1))
    mu = float(returns.mean()+.5*variance) if estimate_drift else 0.0
    log_drift = mu-.5*variance
    generator = _rng(seed if rng is not None or seed is not None else 0, rng)
    paths = simulate_gbm(float(prices[-1, 0]), mu, np.sqrt(variance), steps=steps,
                         n_paths=n_paths, rng=generator)
    time = np.arange(1, steps+1, dtype=float)
    means, variances = logs[-1]+log_drift*time, variance*time
    info = {'mu':mu, 'sigma':float(np.sqrt(variance)), 'mean_log_return':log_drift,
            'time_unit':'trading_days', 'dt':1.0, 'n_paths':n_paths, 'transition':'exact_gbm',
            'point_forecast':'median', '_ensemble':PredictiveEnsemble(paths)}
    if include_fit:
        info['_fit'] = (1, logs[:-1]+log_drift, np.full(len(returns), variance))
    return means, variances, info


def gbm_zero_drift(prices, steps, **kwargs):
    """mu=0: expected simple return is zero; log drift is -sigma²/2."""
    return _gbm(prices, steps, estimate_drift=False, **kwargs)


def gbm_estimated_drift(prices, steps, **kwargs):
    """Training mean log return + sigma²/2 estimates arithmetic-price drift mu."""
    return _gbm(prices, steps, estimate_drift=True, **kwargs)


def ou_returns(prices, steps, *, include_fit=False, rng=None, seed=None, n_paths=10_000):
    """Exact AR(1) transition of an OU model for daily log returns.

    AR coefficient must lie in (0,.98) for stable continuous-time OU. With fewer
    than 20 returns, degenerate predictors, unstable coefficient or implausible
    equilibrium, explicitly fall back to IID Gaussian daily log returns.
    """
    logs, returns = _training(prices)
    _simulation_inputs(float(prices[-1, 0]), steps, 1, n_paths)
    generator = _rng(seed if rng is not None or seed is not None else 0, rng)
    x, y = returns[:-1], returns[1:]
    fallback = None
    phi = 0.0
    equilibrium = float(returns.mean())
    q = float(np.var(returns, ddof=1))
    if len(returns) < 20:
        fallback = 'fewer_than_20_returns'
    elif np.var(x) < 1e-16:
        fallback = 'degenerate_return_history'
    else:
        fitted_phi = float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2))
        if not 0 < fitted_phi < .98:
            fallback = 'ar_coefficient_outside_stable_ou_range'
        else:
            candidate_mu = float((y.mean()-fitted_phi*x.mean())/(1-fitted_phi))
            if abs(candidate_mu-returns.mean()) > 3*np.std(returns):
                fallback = 'unstable_equilibrium_estimate'
            else:
                phi, equilibrium = fitted_phi, candidate_mu
                residual = y-(equilibrium*(1-phi)+phi*x)
                q = float(np.sum(residual**2)/max(len(y)-2, 1))
    intercept = (1-phi)*equilibrium
    paths = np.empty((n_paths, steps+1))
    paths[:, 0] = float(prices[-1, 0])
    r = np.full(n_paths, returns[-1])
    state = np.array([returns[-1], logs[-1]])
    covariance = np.zeros((2, 2))
    transition = np.array([[phi, 0], [phi, 1]])
    means, variances = [], []
    with np.errstate(over='raise', invalid='raise'):
        for i in range(steps):
            r = intercept+phi*r+np.sqrt(q)*generator.standard_normal(n_paths)
            paths[:, i+1] = paths[:, i]*np.exp(r)
            state = transition@state+intercept
            covariance = transition@covariance@transition.T+q*np.ones((2, 2))
            means.append(state[1])
            variances.append(max(covariance[1, 1], 0.0))
    if not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError('OU price paths overflowed or underflowed')
    theta = float(-np.log(phi)) if phi else None
    diffusion = float(np.sqrt(q*2*theta/(1-phi**2))) if theta is not None else None
    info = {'theta':theta, 'mu_r':equilibrium, 'sigma':diffusion, 'ar_phi':phi,
            'innovation_std':float(np.sqrt(q)), 'fallback':fallback,
            'time_unit':'trading_days', 'dt':1.0, 'n_paths':n_paths,
            'transition':'iid_gaussian_returns' if fallback else 'exact_ou_returns',
            'point_forecast':'median', '_ensemble':PredictiveEnsemble(paths)}
    if include_fit:
        info['_fit'] = (2, logs[1:-1]+intercept+phi*returns[:-1], np.full(len(y), q))
    return np.array(means), np.array(variances), info


SDE_MODELS = {'gbm_zero_drift':gbm_zero_drift, 'gbm_estimated_drift':gbm_estimated_drift,
              'ou_returns':ou_returns}
