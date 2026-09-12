"""Public response adapter for the existing posterior GP inference machinery."""
import warnings

import numpy as np
from scipy.stats import norm

from stock_api.diagnostics.problems import GaussianProblem, ReturnProblem, component_activity
from stock_api.diagnostics.sampling import sample_gaussian, sample_latent, weighted_quantile
from stock_api.diagnostics.predictive import draw_predictive
from stock_api.inference_options import inference_settings
from stock_api.schemas import FitPoint, ForecastPoint, ForecastResponse, SigmaInterval, TrainingPoint


def posterior_fitted(problem, samples, weights, predictive, settings, *, gaussian):
    """Pointwise posterior predictive fits, conditional on the entire training window.

    These are descriptive in-sample fits, not walk-forward predictions. Separate
    random draws leave the existing forecast ensemble unchanged. Independent
    draws across dates suffice because only marginal bands are stored/plotted.
    """
    rng = np.random.default_rng(settings.seed + 2)
    if gaussian:
        selected = rng.choice(len(samples), size=settings.predictive_draws, p=weights)
        logs = np.empty((settings.predictive_draws, problem.n))
        # Factor each selected hyperparameter state once, including when nested
        # sampling weights select it repeatedly. No optimizer or new sampler.
        for index in np.unique(selected):
            rows = np.flatnonzero(selected == index)
            mean, covariance = problem.predict(samples[index], problem.x)
            variance = np.maximum(np.diag(covariance), 0.)
            logs[rows] = mean + np.sqrt(variance) * rng.normal(size=(len(rows), problem.n))
        dates = problem.prices.index
    else:
        # Existing predictive draws already contain posterior training f and
        # sigma draws. Anchor each replicated return to the observed prior close,
        # matching the legacy return-GP in-sample fit convention.
        f = predictive['f'][:, :problem.n]
        sigma = predictive['sigma'][:, :problem.n]
        previous = np.log(problem.prices.iloc[:-1, 0].to_numpy())
        logs = previous + f + sigma * rng.normal(size=f.shape)
        dates = problem.prices.index[1:]
    with np.errstate(over='raise', invalid='raise'):
        prices = np.exp(logs)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Invalid posterior fitted price draws')
    medians = np.median(prices, axis=0)
    if not gaussian and not problem.mean_gp:
        medians = problem.prices.iloc[:-1, 0].to_numpy()
    std = logs.std(axis=0)
    bands = {n: np.quantile(prices, [norm.cdf(-n), norm.cdf(n)], axis=0) for n in (1, 2, 3)}
    return [FitPoint(date=date.date().isoformat(), predicted_close=float(medians[i]),
                     log_std=float(std[i]),
                     **{f'sigma_{n}': SigmaInterval(lower=float(bands[n][0, i]),
                                                  upper=float(bands[n][1, i]))
                        for n in (1, 2, 3)}) for i, date in enumerate(dates)]


def posterior_forecast(request, prices, as_of, dates):
    settings = inference_settings(request.model, request.inference, request.inference_setup,
                                  seed=request.seed, n_paths=request.n_paths)
    gaussian = request.inference == 'dynesty'
    kind = GaussianProblem if gaussian else ReturnProblem
    # A copied training snapshot is the only data supplied to inference.
    problem = kind(prices.copy(deep=True), request.model, settings.priors, fit_old=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            samples, weights, latents, _, info = (
                sample_gaussian(problem, settings) if gaussian else
                sample_latent(problem, settings, progress=lambda _: None))
            predictive = draw_predictive(problem, samples, weights, latents, len(dates), settings,
                                         include_comparison=False)
            paths = predictive['paths'][:, 1:]
            fitted = posterior_fitted(problem, samples, weights, predictive, settings, gaussian=gaussian)
        except RuntimeError as exc:
            raise ValueError(f'{request.inference} inference failed: {exc}') from exc
    if not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError('Invalid posterior predictive price paths')
    initial = float(prices.iloc[-1, 0])
    logs = np.log(paths)
    terminal = logs[:, -1] - np.log(initial)
    zero_mean = request.model == 'volatility_gp_returns'
    medians = np.full(len(dates), initial) if zero_mean else np.median(paths, axis=0)
    probabilities = np.full(len(dates), .5) if zero_mean else (paths > initial).mean(axis=0)
    log_std = logs.std(axis=0)
    lower, upper = np.quantile(paths, [.025, .975], axis=0)
    bands = {n: np.quantile(paths, [norm.cdf(-n), norm.cdf(n)], axis=0) for n in (1, 2, 3)}
    # The latent log-variance mixture need not have finite population price
    # moments. Do not present finite Monte Carlo moments as those moments.
    means = paths.mean(axis=0) if gaussian else None
    price_std = paths.std(axis=0) if gaussian else None
    if gaussian and (not np.isfinite(means).all() or not np.isfinite(price_std).all()):
        raise ValueError('Nonfinite posterior predictive price moments')
    changes = 100 * (medians / initial - 1)
    with np.errstate(over='raise', invalid='raise'):
        terminal_returns = np.expm1(terminal)
    points = [ForecastPoint(
        date=date.date().isoformat(), predicted_close=float(medians[i]),
        median_price=float(medians[i]), mean_price=float(means[i]) if gaussian else None,
        price_std=float(price_std[i]) if gaussian else None, log_std=float(log_std[i]),
        lower_95=float(lower[i]), upper_95=float(upper[i]),
        predicted_change_pct=float(changes[i]), probability_up=float(probabilities[i]),
        **{f'sigma_{n}': SigmaInterval(lower=float(bands[n][0, i]), upper=float(bands[n][1, i]))
           for n in (1, 2, 3)}) for i, date in enumerate(dates)]
    physical = problem.physical(samples)
    info['hyperparameters'] = {
        name: dict(zip(['lower95', 'lower68', 'median', 'upper68', 'upper95'],
                       map(float, weighted_quantile(physical[:, j], weights))))
        for j, name in enumerate(problem.names)}
    info.update(component_activity(problem, physical, weights))
    info['prior_edge_mass'] = {
        name: {'lower': float(weights[samples[:, j] < lo + .05 * (hi - lo)].sum()),
               'upper': float(weights[samples[:, j] > hi - .05 * (hi - lo)].sum())}
        for j, (name, (lo, hi)) in enumerate(zip(problem.names, problem.bounds))}
    info.update(inference=request.inference, inference_setup=request.inference_setup or {},
                sampler_config=settings.model_dump(mode='json'),
                distribution='posterior_mixture',
                parameter_uncertainty='Hyperparameters and applicable latent uncertainty marginalized',
                fitted_note=('Pointwise posterior predictive fits conditional on the entire training window; '
                             'includes observation/return noise, not out-of-sample forecasts. '
                             + ('Log-price fits at each training date.' if gaussian else
                                'Return fits anchored to each observed previous close.')),
                terminal_expected_log_return=0. if zero_mean else float(terminal.mean()),
                terminal_median_log_return=0. if zero_mean else float(np.median(terminal)),
                terminal_log_return_sigma=float(terminal.std()))
    messages = ['Experimental forecasts; accuracy and interval calibration have not been established.',
                'Adjusted closes, completed US market sessions, and XNYS calendar only.',
                'Posterior mixture bands use empirical predictive quantiles; sigma labels denote nominal Gaussian coverage levels.']
    if not gaussian:
        messages.append('Price mean and standard deviation are omitted because latent variance mixtures may have nonfinite population price moments.')
    if info.get('converged') is False:
        messages.append('Posterior sampling did not pass convergence checks; inspect diagnostics before relying on this forecast.')
    info['warnings'] = list(dict.fromkeys(str(w.message) for w in caught))
    messages.extend(info['warnings'])
    change = float(changes[-1])
    return ForecastResponse(
        ticker=request.ticker, related_tickers=request.related_tickers, model=request.model,
        as_of=as_of.date().isoformat(), last_close=initial,
        horizon=request.horizon, horizon_unit=request.horizon_unit, training_observations=len(prices),
        direction='up' if change > 1e-8 else 'down' if change < -1e-8 else 'flat',
        predicted_change_pct=change, forecasts=points,
        training_data=[TrainingPoint(date=date.date().isoformat(), closes=row.to_dict())
                       for date, row in prices.iterrows()],
        fitted=fitted, diagnostics=info, warnings=messages,
        terminal_log_returns=terminal.tolist(), terminal_returns=terminal_returns.tolist())
