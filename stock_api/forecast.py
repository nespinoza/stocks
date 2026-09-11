"""Public, in-process forecasting interface. No HTTP server is needed."""
from typing import Literal, Sequence

import numpy as np

from stock_api.data import DataError, ProviderError, latest_session, future_sessions, load_prices
from stock_api.models import MODELS
from stock_api.sde import SDE_MODELS, lognormal_statistics
from stock_api.schemas import (ForecastPoint, ForecastRequest, ForecastResponse,
                               FitPoint, SigmaInterval, TrainingPoint)


class ForecastError(RuntimeError):
    """A model could not produce a finite forecast."""


def predict(
    ticker: str,
    *,
    related_tickers: Sequence[str] | None = None,
    model: Literal["last_price", "random_walk", "gp", "multitask_gp", "var", "gbm_zero_drift", "gbm_estimated_drift", "ou_returns"] = "multitask_gp",
    horizon: int = 7,
    horizon_unit: Literal["calendar", "trading"] = "calendar",
    lookback: int = 120,
    seed: int = 0,
    n_paths: int = 10_000,
) -> ForecastResponse:
    """Download completed US-equity closes and forecast a target stock.

    Parameters
    ----------
    ticker : str
        Target Yahoo ticker, for example "AMZN".
    related_tickers : sequence of str, optional
        Up to three related tickers jointly fitted with the target.
    model : str
        gp, multitask_gp, var, last_price (alias random_walk), gbm_zero_drift,
        gbm_estimated_drift, or ou_returns. Only multitask_gp and var accept
        related tickers for direct forecasting.
    horizon : int
        Forecast length, 1 through 30 (default 7).
    horizon_unit : {"calendar", "trading"}
        Calendar days after the last completed close, or trading sessions.
    lookback : int
        Number of training sessions, 60 through 252 (default 120).
    seed : int
        SDE Monte Carlo seed (default 0); validation supplies a per-fold RNG.
    n_paths : int
        SDE ensemble size, 100 through 100000 (default 10000).

    Returns
    -------
    ForecastResponse
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


def _forecast(request: ForecastRequest) -> ForecastResponse:
    as_of = latest_session()
    dates = future_sessions(as_of, request.horizon, request.horizon_unit)
    if not len(dates):
        raise DataError("No market sessions in this calendar horizon; use trading days")
    prices = load_prices([request.ticker, *request.related_tickers], request.lookback, as_of)
    kwargs = {'seed':request.seed, 'n_paths':request.n_paths} if request.model in SDE_MODELS else {}
    means, variances, diagnostics = MODELS[request.model](prices.to_numpy(), len(dates), include_fit=True, **kwargs)
    ensemble = diagnostics.pop('_ensemble', None)
    fit_start, fit_means, fit_variances = diagnostics.pop("_fit")
    if not (np.isfinite(means).all() and np.isfinite(variances).all()):
        raise ValueError("Non-finite forecast")
    last = float(prices.iloc[-1, 0])
    sigma = np.sqrt(variances)
    # Report the lognormal median; intervals include model observation/innovation noise.
    with np.errstate(over="raise", invalid="raise"):
        medians, lower, upper = np.exp(means), np.exp(means-1.96*sigma), np.exp(means+1.96*sigma)
        if request.model in ('last_price', 'random_walk'):
            medians = np.full(len(dates), last)
        changes = 100 * (medians / last - 1)
    statistics = lognormal_statistics(means, variances, last)
    probabilities = statistics['probability_up']
    points = [ForecastPoint(date=date.date().isoformat(), predicted_close=float(medians[i]),
                            **_uncertainty(means[i], variances[i]),
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
                                             **_uncertainty(mean, variance))
                                    for date, mean, variance in zip(prices.index[fit_start:], fit_means, fit_variances)],
                            diagnostics=diagnostics, warnings=warnings,
                            terminal_log_returns=ensemble.terminal_log_returns.tolist() if ensemble else [],
                            terminal_returns=ensemble.terminal_returns.tolist() if ensemble else [])


def _uncertainty(mean, variance):
    """Gaussian log-space sigma intervals, transformed to price units."""
    if not np.isfinite(mean) or not np.isfinite(variance) or variance < 0:
        raise ValueError("Invalid fitted distribution")
    std = float(np.sqrt(variance))
    with np.errstate(over="raise", invalid="raise"):
        return {"log_std": std, **{
            f"sigma_{n}": SigmaInterval(lower=float(np.exp(mean-n*std)),
                                        upper=float(np.exp(mean+n*std)))
            for n in (1, 2, 3)}}
