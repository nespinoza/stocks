"""Lazy, optional Matplotlib plotting of a stored ForecastResponse."""
from datetime import date, timedelta
from numbers import Integral

import numpy as np


def plot_result(result, *, history_days, forecast_days, sigmas, show_related,
                show_fit, uncertainty, ax, figsize, show):
    for name, value in (("history_days", history_days), ("forecast_days", forecast_days)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer or None")
    sigmas = tuple(sigmas)
    if any(n not in (1, 2, 3) for n in sigmas):
        raise ValueError("sigmas must contain only 1, 2, or 3")
    if uncertainty not in ("bands", "errorbars"):
        raise ValueError("uncertainty must be 'bands' or 'errorbars'")
    if not result.training_data:
        raise ValueError("This result has no stored training data; run predict again")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib; install with pip install 'stock-movement-api[plots]' "
                          "or pip install '.[plots]' from the repository") from exc

    origin = date.fromisoformat(result.as_of)
    start = origin - timedelta(days=int(history_days)) if history_days is not None else date.min
    end = origin + timedelta(days=int(forecast_days)) if forecast_days is not None else date.max
    training = [p for p in result.training_data if start <= date.fromisoformat(p.date) <= origin]
    comparison = hasattr(result, 'results')
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    dates = [date.fromisoformat(p.date) for p in training]
    ax.plot(dates, [p.closes[result.ticker] for p in training], '.-',
            color='black', markersize=3, label=f'{result.ticker} training data')
    if show_related:
        last = result.training_data[-1].closes
        for ticker in result.related_tickers:
            factor = result.last_close / last[ticker]
            ax.plot(dates, [p.closes[ticker] * factor for p in training],
                    alpha=0.55, linewidth=1, label=f'{ticker} training (rebased)',
                    **({'color':'0.6', 'linestyle':':'} if comparison else {}))

    def distribution(points, label, color, linestyle='-'):
        if not points:
            return
        x = [date.fromisoformat(p.date) for p in points]
        y = np.array([p.predicted_close for p in points])
        ax.plot(x, y, marker='.', linestyle=linestyle, color=color, label=label)
        for n in sorted(set(sigmas), reverse=True):
            lower = np.array([getattr(p, f'sigma_{n}').lower for p in points])
            upper = np.array([getattr(p, f'sigma_{n}').upper for p in points])
            if uncertainty == 'bands':
                ax.fill_between(x, lower, upper, color=color, alpha=0.12,
                                label='_nolegend_' if comparison else f'{label} {n}σ')
            else:
                ax.errorbar(x, y, yerr=np.vstack([y-lower, upper-y]), fmt='none',
                            color=color, alpha=0.45, capsize=2,
                            label='_nolegend_' if comparison else f'{label} {n}σ')
    models = result.results.items() if comparison else [(result.model, result)]
    for i, (name, forecast) in enumerate(models):
        fitted = [p for p in forecast.fitted if start <= date.fromisoformat(p.date) <= origin]
        future = [p for p in forecast.forecasts if origin < date.fromisoformat(p.date) <= end]
        color = plt.get_cmap('tab10')(i % 10)
        if show_fit:
            distribution(fitted, f'{name} fit' if comparison else 'Historical fit',
                         color if comparison else 'tab:blue', '--' if comparison else '-')
        distribution(future, f'{name} forecast' if comparison else 'Forecast',
                     color if comparison else 'tab:orange')
    ax.axvline(origin, color='gray', linestyle='--', linewidth=1, label='Forecast origin')
    title_model = 'model comparison' if comparison else result.model
    title = f'{result.ticker} — {title_model} — {result.horizon} {result.horizon_unit} days'
    if comparison and sigmas:
        title += '\n' + ', '.join(f'{n}σ' for n in sorted(set(sigmas))) + f' {uncertainty}'
    ax.set(xlabel='Date', ylabel='Adjusted close (related tickers rebased)', title=title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize='small', ncol=2)
    ax.figure.autofmt_xdate()
    ax.figure.tight_layout()
    if show:
        plt.show()
    return ax
