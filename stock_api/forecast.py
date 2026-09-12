"""Public, in-process forecasting interface. No HTTP server is needed."""
from typing import Literal, Sequence

import numpy as np

from stock_api.data import DataError, ProviderError, latest_session, future_sessions, load_prices
from stock_api.models import MODELS
from stock_api.sde import SDE_MODELS, lognormal_statistics
from stock_api.heteroskedastic import RETURN_GP_MODELS
from stock_api.schemas import (ForecastPoint, ForecastRequest, ForecastResponse,
                               FitPoint, SigmaInterval, TrainingPoint, ModelName, ForecastComparison)


class ForecastError(RuntimeError):
    """A model could not produce a finite forecast."""


def predict(
    ticker: str,
    *,
    related_tickers: Sequence[str] | None = None,
    model: ModelName | list[ModelName] = "multitask_gp",
    horizon: int = 7,
    horizon_unit: Literal["calendar", "trading"] = "calendar",
    lookback: int = 120,
    seed: int = 0,
    n_paths: int = 10_000,
) -> ForecastResponse | ForecastComparison:
    """Download completed US-equity closes and forecast a target stock.

    Parameters
    ----------
    ticker : str
        Target Yahoo ticker, for example "AMZN".
    related_tickers : sequence of str, optional
        Up to three related tickers jointly fitted with the target.
    model : str or list of str
        gp, multitask_gp, var, last_price (alias random_walk), gbm_zero_drift,
        gbm_estimated_drift, ou_returns, volatility_gp_returns, or
        heteroskedastic_gp_returns. Only multitask_gp and var accept
        related tickers for direct forecasting.
        A list returns a ForecastComparison fitted to one shared price download.
        Single-ticker models in a comparison use only the target column.
    horizon : int
        Forecast length, 1 through 30 (default 7).
    horizon_unit : {"calendar", "trading"}
        Calendar days after the last completed close, or trading sessions.
    lookback : int
        Number of training sessions, 60 through 252 (default 120).
    seed : int
        Monte Carlo seed (default 0); validation supplies a per-fold RNG.
    n_paths : int
        Monte Carlo ensemble size, 100 through 100000 (default 10000).

    Returns
    -------
    ForecastResponse or ForecastComparison
        Attribute-accessible result; use model_dump() for a dictionary,
        or model_dump_json() for JSON. Its forecasts field contains daily points.

    Raises
    ------
    pydantic.ValidationError
        Invalid arguments (also a ValueError).
    DataError
        Missing/stale prices or no sessions in the requested horizon.
    ProviderError
        The price provider could not return usable data.
    ForecastError
        Numerical fitting failed.
    """
    request = ForecastRequest(ticker=ticker, related_tickers=related_tickers if related_tickers is not None else [],
                              model=model, horizon=horizon, horizon_unit=horizon_unit, lookback=lookback,
                              seed=seed, n_paths=n_paths)
    try:
        return _forecast(request)
    except (DataError, ProviderError):
        raise
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ForecastError("Model fitting failed; try another model or shorter lookback") from exc


def _forecast(request: ForecastRequest) -> ForecastResponse | ForecastComparison:
    as_of = latest_session()
    dates = future_sessions(as_of, request.horizon, request.horizon_unit)
    if not len(dates):
        raise DataError("No market sessions in this calendar horizon; use trading days")
    if isinstance(request.model, list):
        related = request.related_tickers if any(m in ('multitask_gp','var') for m in request.model) else []
        prices = load_prices([request.ticker, *related], request.lookback, as_of)
        results = {}
        for name in request.model:
            selected_related = related if name in ('multitask_gp','var') else []
            single = ForecastRequest(**{**request.model_dump(), 'model':name,
                                        'related_tickers':selected_related})
            try:
                results[name] = _fit_forecast(single, prices[[request.ticker, *selected_related]].copy(), as_of, dates)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                raise ForecastError(f'Model {name!r} failed in forecast comparison') from exc
        return ForecastComparison(ticker=request.ticker, related_tickers=related,
                                  as_of=as_of.date().isoformat(), last_close=float(prices.iloc[-1,0]),
                                  horizon=request.horizon, horizon_unit=request.horizon_unit,
                                  training_data=[TrainingPoint(date=date.date().isoformat(), closes=row.to_dict())
                                                 for date,row in prices.iterrows()], results=results)
    prices = load_prices([request.ticker, *request.related_tickers], request.lookback, as_of)
    return _fit_forecast(request, prices, as_of, dates)


def _fit_forecast(request, prices, as_of, dates):
    kwargs = {'seed':request.seed, 'n_paths':request.n_paths} if request.model in (SDE_MODELS | RETURN_GP_MODELS) else {}
    means, variances, diagnostics = MODELS[request.model](prices.to_numpy(), len(dates), include_fit=True, **kwargs)
    ensemble = diagnostics.pop('_ensemble', None)
    fit_start, fit_means, fit_variances = diagnostics.pop("_fit")
    if not (np.isfinite(means).all() and np.isfinite(variances).all()):
        raise ValueError("Non-finite forecast")
    last = float(prices.iloc[-1, 0])
    statistics = diagnostics.pop('_statistics', None)
    fit_statistics = diagnostics.pop('_fit_statistics', None)
    if statistics is None:
        statistics = lognormal_statistics(means, variances, last)
        # Preserve the existing public Gaussian interval convention.
        with np.errstate(over="raise", invalid="raise"):
            statistics["lower_95"] = np.exp(means-1.96*np.sqrt(variances))
            statistics["upper_95"] = np.exp(means+1.96*np.sqrt(variances))
    # Score the median; predictive intervals include innovation noise.
    with np.errstate(over="raise", invalid="raise"):
        medians, lower, upper = (statistics[k] for k in ('median_price', 'lower_95', 'upper_95'))
        if request.model in ('last_price', 'random_walk'):
            medians = np.full(len(dates), last)
        changes = 100 * (medians / last - 1)
    probabilities = statistics['probability_up']
    points = [ForecastPoint(date=date.date().isoformat(), predicted_close=float(medians[i]),
                            **_uncertainty(means[i], variances[i], statistics, i),
                            lower_95=float(lower[i]), upper_95=float(upper[i]),
                            predicted_change_pct=float(changes[i]), probability_up=float(probabilities[i]),
                            mean_price=float(statistics['mean_price'][i]), median_price=float(medians[i]),
                            price_std=float(statistics['price_std'][i]))
              for i, date in enumerate(dates)]
    warnings = ["Experimental forecasts; accuracy and interval calibration have not been established.",
                "Adjusted closes, completed US market sessions, and XNYS calendar only."]
    if diagnostics.get("converged") is False:
        warnings.append("GP optimizer reached its limit; inspect the fit before relying on it.")
    if diagnostics.get('fallback'):
        warnings.append('OU fallback to IID Gaussian log returns: '+diagnostics['fallback'])
    change = float(changes[-1])
    return ForecastResponse(ticker=request.ticker, related_tickers=request.related_tickers,
                            model=request.model, as_of=as_of.date().isoformat(), last_close=last,
                            horizon=request.horizon, horizon_unit=request.horizon_unit,
                            training_observations=len(prices),
                            direction="up" if change > 1e-8 else "down" if change < -1e-8 else "flat",
                            predicted_change_pct=change, forecasts=points,
                            training_data=[TrainingPoint(date=date.date().isoformat(), closes=row.to_dict())
                                           for date, row in prices.iterrows()],
                            fitted=[FitPoint(date=date.date().isoformat(), predicted_close=float(np.exp(mean)),
                                             **_uncertainty(mean, variance, fit_statistics, i))
                                    for i, (date, mean, variance) in enumerate(zip(prices.index[fit_start:], fit_means, fit_variances))],
                            diagnostics=diagnostics, warnings=warnings,
                            terminal_log_returns=ensemble.terminal_log_returns.tolist() if ensemble else [],
                            terminal_returns=ensemble.terminal_returns.tolist() if ensemble else [])


def _uncertainty(mean, variance, statistics=None, index=0):
    """Use supplied predictive quantiles, otherwise transformed Gaussian bands."""
    if not np.isfinite(mean) or not np.isfinite(variance) or variance < 0:
        raise ValueError("Invalid fitted distribution")
    std = float(np.sqrt(variance))
    if statistics is not None and 'sigma_1_lower' in statistics:
        return {"log_std": std, **{
            f"sigma_{n}": SigmaInterval(lower=float(statistics[f'sigma_{n}_lower'][index]),
                                       upper=float(statistics[f'sigma_{n}_upper'][index]))
            for n in (1, 2, 3)}}
    with np.errstate(over="raise", invalid="raise"):
        return {"log_std": std, **{
            f"sigma_{n}": SigmaInterval(lower=float(np.exp(mean-n*std)),
                                        upper=float(np.exp(mean+n*std)))
            for n in (1, 2, 3)}}
