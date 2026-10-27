"""Synthetic data with known answers, for null and recovery tests of the pipeline."""
import numpy as np
import pandas as pd


def signal_panel(dates=120, assets=30, ic=0.0, volatility=0.08, seed=0):
    """Long panel whose per-date cross-sectional signal/outcome correlation is ic.

    ic=0 gives a pure-noise null; a pipeline must find no skill there.
    """
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((dates, assets))
    noise = rng.standard_normal((dates, assets))
    outcome = volatility*(ic*signal+np.sqrt(1-ic**2)*noise)
    index = pd.MultiIndex.from_product([pd.bdate_range('2000-01-03', periods=dates), range(assets)],
                                       names=['date', 'asset'])
    return pd.DataFrame({'signal': signal.ravel(), 'outcome': outcome.ravel()}, index=index).reset_index()


def gaussian_forecasts(n=500, miscalibration=1.0, seed=0):
    """Gaussian forecasts whose true scale is miscalibration × the predicted scale."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, .01, n)
    std = rng.uniform(.01, .03, n)
    actual = mean+miscalibration*std*rng.standard_normal(n)
    return mean, std, actual


def noise_strategies(periods=1000, configurations=50, volatility=.01, seed=0):
    """Returns of skill-less strategy configurations, for overfitting statistics."""
    rng = np.random.default_rng(seed)
    return volatility*rng.standard_normal((periods, configurations))
