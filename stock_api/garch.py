"""ARX return regression with Student-t GJR-GARCH(1,1) innovations.

The ``arch`` dependency estimates mean and variance jointly. Simulated
paths update both the lagged return and conditional variance at every session.
Parameters are plug-in MLEs; simulation does not add parameter uncertainty.
"""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from stock_api.sde import PredictiveEnsemble, _rng, _simulation_inputs


def next_variance(residual, variance, omega, alpha, gamma, beta):
    """GJR variance transition, in the same squared units as the residual."""
    return omega + (alpha + gamma * (residual < 0)) * residual**2 + beta * variance


@dataclass
class RegressionGJRFit:
    """Fitted percent-return parameters and training-only regressor scaling."""
    parameters: np.ndarray
    regressor_center: np.ndarray
    regressor_scale: np.ndarray
    last_return: float
    last_residual: float
    last_variance: float
    fitted_mean: np.ndarray
    fitted_variance: np.ndarray
    diagnostics: dict

    def simulate_returns(self, steps, *, n_paths=10_000, seed=None, rng=None,
                         future_regressors=None):
        """Return daily log returns of shape (n_paths, steps).

        Training regressor row i explains return i and must be available before
        that return. Future regressors must be supplied explicitly: a known or
        forecast (steps, k) path, or joint scenarios (n_paths, steps, k). Their
        availability and scenario-generating model are the caller's responsibility.
        Deterministic future inputs condition on that path and omit its uncertainty.
        """
        _simulation_inputs(1., steps, 1., n_paths)
        generator = _rng(seed, rng)
        k = len(self.regressor_center)
        if k:
            if future_regressors is None:
                raise ValueError('Future regressors are required; no future values are inferred or filled')
            x = np.asarray(future_regressors, dtype=float)
            if x.shape not in ((steps, k), (n_paths, steps, k)) or not np.isfinite(x).all():
                raise ValueError('Future regressors must be finite (steps, k) or (n_paths, steps, k)')
            x = (x - self.regressor_center) / self.regressor_scale
        elif future_regressors is not None:
            raise ValueError('This fit has no external regressors')
        c, phi = self.parameters[:2]
        coefficients = self.parameters[2:2+k]
        omega, alpha, gamma, beta, nu = self.parameters[2+k:]
        previous = np.full(n_paths, self.last_return)
        residual = np.full(n_paths, self.last_residual)
        variance = np.full(n_paths, self.last_variance)
        draws = np.empty((n_paths, steps))
        with np.errstate(over='raise', invalid='raise'):
            for j in range(steps):
                variance = next_variance(residual, variance, omega, alpha, gamma, beta)
                if not np.isfinite(variance).all() or (variance <= 0).any():
                    raise ValueError('Invalid GJR conditional variance')
                mean = c + phi * previous
                if k:
                    mean = mean + (x[j] if x.ndim == 2 else x[:, j]) @ coefficients
                innovation = generator.standard_t(nu, size=n_paths) * np.sqrt((nu-2)/nu)
                residual = np.sqrt(variance) * innovation
                previous = mean + residual
                draws[:, j] = previous / 100.
        if not np.isfinite(draws).all():
            raise ValueError('Nonfinite simulated return')
        return draws


def fit_regression_gjr(returns, *, regressors=None):
    """Fit intercept + AR(1) + optional exogenous regression, jointly with GJR.

    ``returns`` are dimensionless daily log returns. ``regressors`` must have
    one row per return, already aligned to information available before that
    return. No future rows are accepted, and all scaling is training-only.
    At least 31 returns (30 regression observations) are required; such short
    windows are supported for sensitivity checks, not recommended for estimation.
    """
    try:
        from arch.univariate import ARX, GARCH, StudentsT
    except ImportError as exc:
        raise ValueError("Install the project dependencies with pip install . (requires arch>=8,<9)") from exc
    y = np.asarray(returns, dtype=float)
    if y.ndim != 1 or len(y) < 31 or not np.isfinite(y).all() or y.std() < 1e-10:
        raise ValueError('GJR requires at least 31 finite, nondegenerate daily log returns')
    x = np.empty((len(y), 0)) if regressors is None else np.asarray(regressors, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(y) or not np.isfinite(x).all():
        raise ValueError('Training regressors must be a finite matrix with one row per return')
    center, scale = x[1:].mean(axis=0), x[1:].std(axis=0)
    if (scale < 1e-12).any():
        raise ValueError('Constant regressors duplicate the intercept')
    standardized = (x-center)/scale
    design = np.column_stack([np.ones(len(y)-1), y[:-1], standardized[1:]])
    if len(design) <= design.shape[1]+5 or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('Insufficient or collinear regression observations')
    series = pd.Series(100*y, name='return')
    inputs = pd.DataFrame(standardized, columns=[f'x{i}' for i in range(x.shape[1])]) if x.shape[1] else None
    model = ARX(series, x=inputs, lags=1, constant=True, rescale=False)
    model.volatility = GARCH(p=1, o=1, q=1)
    model.distribution = StudentsT()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted = model.fit(disp='off', update_freq=0, show_warning=False,
                           options={'maxiter':2000, 'ftol':1e-9})
    if fitted.convergence_flag != 0 or not np.isfinite(fitted.loglikelihood):
        raise ValueError(f'GJR optimizer failed: {fitted.optimization_result.message}; no fallback used')
    params = fitted.params.to_numpy()
    c, phi = params[:2]
    omega, alpha, gamma, beta, nu = params[-5:]
    if (not np.isfinite(params).all() or abs(phi) >= 1 or omega <= 0
            or alpha < 0 or beta < 0 or alpha+gamma < -1e-10
            or alpha+gamma/2+beta > 1+1e-8 or nu <= 2):
        raise ValueError('GJR fit violates variance, finite-return-variance or AR stability constraints')
    residuals = np.asarray(fitted.resid)
    variances = np.asarray(fitted.conditional_volatility)**2
    if not np.isfinite(residuals[1:]).all() or not np.isfinite(variances[1:]).all():
        raise ValueError('Nonfinite fitted GJR residuals or variances')
    coefficients = params[2:2+x.shape[1]] / scale / 100
    notes = [str(w.message) for w in caught]
    if alpha+gamma/2+beta >= .995:
        notes.append('Variance persistence is near its unit boundary; inspect stability')
    if nu <= 2.1:
        notes.append('Student-t degrees of freedom are near 2; simulated variance is unstable')
    diagnostics = {
        'inference':'joint_maximum_likelihood', 'converged':True,
        'mean_model':'intercept + AR(1) + explicit exogenous regressors',
        'volatility_model':'GJR-GARCH(1,1)', 'innovation_distribution':'standardized_student_t',
        'parameters_percent_units':dict(zip(fitted.params.index, map(float, params))),
        'intercept_log_return':float(c/100 - center @ coefficients),
        'ar_coefficient':float(phi), 'regressor_coefficients_log_return':coefficients.tolist(),
        'regressor_center':center.tolist(), 'regressor_scale':scale.tolist(),
        'variance_persistence':float(alpha+gamma/2+beta),
        'training_returns':len(y), 'regression_observations':len(y)-1,
        'log_likelihood_percent_units':float(fitted.loglikelihood),
        'optimizer_message':str(fitted.optimization_result.message),
        'iterations':int(fitted.optimization_result.nit), 'warnings':notes,
        'parameter_uncertainty':'Plug-in MLE; innovation uncertainty only',
        'price_moment_note':'Undefined: Student-t log returns have no finite exponential moments',
        'time_unit':'trading_sessions',
    }
    return RegressionGJRFit(params, center, scale, float(series.iloc[-1]),
                            float(residuals[-1]), float(variances[-1]),
                            (series.to_numpy()[1:]-residuals[1:])/100,
                            variances[1:]/10000, diagnostics)


def regression_gjr_garch(prices, steps, *, include_fit=False, seed=None, rng=None,
                         n_paths=10_000, regressors=None, future_regressors=None):
    """Built-in target-only AR(1)-GJR adapter; optional regressors for direct use."""
    values = np.asarray(prices, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values[:, 0]).all() or (values[:, 0] <= 0).any():
        raise ValueError('Positive finite target prices are required')
    logs = np.log(values[:, 0])
    fit = fit_regression_gjr(np.diff(logs), regressors=regressors)
    increments = fit.simulate_returns(steps, n_paths=n_paths, seed=seed, rng=rng,
                                     future_regressors=future_regressors)
    cumulative = increments.cumsum(axis=1)
    with np.errstate(over='raise', invalid='raise'):
        paths = np.exp(np.column_stack([np.full(n_paths, logs[-1]), logs[-1]+cumulative]))
    if not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError('GJR sample price overflow/underflow; no tail clipping applied')
    medians = np.median(logs[-1]+cumulative, axis=0)
    variance = cumulative.var(axis=0)
    statistics = {'median_price':np.exp(medians), 'mean_price':[None]*steps,
                  'price_std':[None]*steps, 'probability_up':(cumulative > 0).mean(axis=0)}
    statistics.update(lower_95=np.exp(logs[-1]+np.quantile(cumulative,.025,axis=0)),
                      upper_95=np.exp(logs[-1]+np.quantile(cumulative,.975,axis=0)))
    for n in (1, 2, 3):
        statistics[f'sigma_{n}_lower'] = np.exp(logs[-1]+np.quantile(cumulative,norm.cdf(-n),axis=0))
        statistics[f'sigma_{n}_upper'] = np.exp(logs[-1]+np.quantile(cumulative,norm.cdf(n),axis=0))
    info = dict(fit.diagnostics, n_paths=n_paths, _statistics=statistics,
                _ensemble=PredictiveEnsemble(paths))
    if include_fit:
        mean = logs[1:-1]+fit.fitted_mean
        sd = np.sqrt(fit.fitted_variance)
        nu = fit.parameters[-1]
        info['_fit'] = (2, mean, fit.fitted_variance)
        info['_fit_statistics'] = {
            f'sigma_{n}_{side}':np.exp(mean + sd*t.ppf(norm.cdf(sign*n), nu)*np.sqrt((nu-2)/nu))
            for n in (1, 2, 3) for side, sign in [('lower', -1), ('upper', 1)]}
    return medians, variance, info


def historical_prediction(train, steps, *, seed, n_paths):
    """Historical record distribution; never approximate Student-t paths as Gaussian."""
    _, _, info = regression_gjr_garch(np.asarray(train), steps, seed=seed, n_paths=n_paths)
    ensemble = info.pop('_ensemble')
    statistics = info.pop('_statistics')
    initial = float(np.asarray(train)[-1, 0])
    cumulative = np.log(ensemble.paths[:, 1:]/initial)
    terminal = cumulative[:, -1]
    return {
        'path_mean':None, 'path_median':statistics['median_price'].tolist(),
        'expected_log_return':float(terminal.mean()), 'median_log_return':float(np.median(terminal)),
        'terminal_sigma':float(terminal.std()), 'probability_up':float((terminal > 0).mean()),
        'distribution':'conditional_student_t_gjr_simulation', 'terminal_samples':terminal.tolist(),
        'normal_location':None, 'normal_sigma':None,
        'intervals':{str(level):{
            'path_lower':(initial*np.exp(np.quantile(cumulative,(1-level)/2,axis=0))).tolist(),
            'path_upper':(initial*np.exp(np.quantile(cumulative,(1+level)/2,axis=0))).tolist(),
            'return_lower':float(np.quantile(terminal,(1-level)/2)),
            'return_upper':float(np.quantile(terminal,(1+level)/2))} for level in (.5,.68,.9,.95)},
        'parameter_uncertainty':info['parameter_uncertainty'],
        'price_moments_note':info['price_moment_note'], 'diagnostics':info,
    }


GARCH_MODELS = {'regression_gjr_garch':regression_gjr_garch}
