"""Distribution and inference tests use deterministic synthetic data, never validation targets."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

from stock_api import ValidationConfig, ValidationResult, builtin_models, validate
from stock_api.data import calendar
from stock_api.heteroskedastic import (RETURN_GP_MODELS, _objective, fit_latent_returns,
                                      paths_from_returns)


def prices_from_returns(returns):
    return paths_from_returns(100., np.asarray(returns)[None, :])[0, :, None]


@pytest.mark.parametrize('mean_gp', [False, True])
def test_variational_gradient(mean_gp):
    rng = np.random.default_rng(4)
    y = rng.normal(size=12)
    p = np.r_[rng.normal(0,.2,12), rng.normal(-.7, .2, 12), np.log(.8), np.log(5), -.2]
    if mean_gp:
        p = np.r_[p, np.log(.3), np.log(8)]
    numerical = approx_derivative(lambda x: _objective(x, y, mean_gp)[0], p).ravel()
    np.testing.assert_allclose(_objective(p, y, mean_gp)[1], numerical, atol=1e-6, rtol=1e-5)


def test_constant_volatility_and_zero_mean_shrinkage():
    returns = np.random.default_rng(71).normal(0, .02, 180)
    for mean_gp in (False, True):
        p = fit_latent_returns(returns, mean_gp=mean_gp)
        variance = p['scale']**2*np.exp(p['g_mean']+.5*np.diag(p['g_cov']))
        assert .5*.02**2 < variance.mean() < 2*.02**2
        assert variance.max()/variance.min() < 3
        assert np.max(np.abs(p['scale']*p['f_mean'])) < .005


def test_changing_volatility():
    t = np.arange(180)
    sigma = np.exp(-4+.9*np.sin(t/22))
    p = fit_latent_returns(sigma*np.random.default_rng(71).normal(size=len(t)))
    variance = p['scale']**2*np.exp(p['g_mean']+.5*np.diag(p['g_cov']))
    assert np.corrcoef(variance, sigma**2)[0, 1] > .65
    assert variance.max()/variance.min() > 3
    np.testing.assert_array_equal(p['f_mean'], 0)


def test_joint_mean_volatility_and_separate_timescales():
    t = np.arange(180)
    sigma = np.exp(-4+.9*np.sin(t/22))
    mean = .028*np.sin(t/9)
    p = fit_latent_returns(mean+sigma*np.random.default_rng(71).normal(size=len(t)), mean_gp=True)
    variance = p['scale']**2*np.exp(p['g_mean']+.5*np.diag(p['g_cov']))
    assert np.corrcoef(p['f_mean'], mean)[0, 1] > .75
    assert np.corrcoef(variance, sigma**2)[0, 1] > .5
    # Broad separation, not exact recovery from one noisy realization.
    assert p['ellg'] > 1.3*p['ellf']
    assert p['converged'] and p['projected_gradient'] < .02


@pytest.mark.parametrize('name', RETURN_GP_MODELS)
def test_predictive_distribution_and_seed(name):
    model = RETURN_GP_MODELS[name]
    prices = prices_from_returns(np.random.default_rng(31).normal(0, .02, 60))
    mean, var, info = model(prices, 5, seed=4, n_paths=2000, include_fit=True)
    _, _, repeated = model(prices, 5, seed=4, n_paths=2000)
    np.testing.assert_array_equal(info['_ensemble'].paths, repeated['_ensemble'].paths)
    assert not np.array_equal(info['_ensemble'].paths,
                              model(prices, 5, seed=5, n_paths=2000)[2]['_ensemble'].paths)
    stats = info['_statistics']
    assert np.isfinite(var).all() and (var > 0).all()
    assert (stats['price_std'] > 0).all()
    assert (stats['lower_95'] < stats['median_price']).all()
    assert (stats['median_price'] < stats['upper_95']).all()
    assert ((stats['probability_up'] >= 0) & (stats['probability_up'] <= 1)).all()
    for n in (1, 2, 3):
        assert (stats[f'sigma_{n}_lower'] < stats['median_price']).all()
        assert (stats[f'sigma_{n}_upper'] > stats['median_price']).all()
    paths = info['_ensemble'].paths
    np.testing.assert_allclose(stats['lower_95'], np.quantile(paths[:, 1:], .025, axis=0))
    np.testing.assert_allclose(stats['mean_price'], paths[:, 1:].mean(axis=0))
    np.testing.assert_allclose(np.log(paths[:, 1:]).mean(axis=0), mean, atol=1e-12)
    assert paths.shape == (2000, 6)
    assert info['_fit'][0] == 1
    assert len(info['_fit_statistics']['lower_95']) == 60
    if name == 'volatility_gp_returns':
        np.testing.assert_array_equal(stats['median_price'], prices[-1, 0])
        np.testing.assert_array_equal(stats['probability_up'], .5)
        assert info['expected_terminal_log_return'] == 0


def test_price_path_reconstruction():
    returns = np.array([[.01, -.02, .03], [-.05, .04, .01]])
    paths = paths_from_returns(123., returns)
    np.testing.assert_allclose(np.diff(np.log(paths), axis=1), returns, atol=1e-14)
    np.testing.assert_allclose(paths[:, -1], 123*np.exp(returns.sum(axis=1)))


def test_inference_failure_and_input_validation():
    with pytest.raises(ValueError, match='did not converge'):
        fit_latent_returns(np.random.default_rng(4).normal(size=30), maxiter=0)
    for bad in (np.ones(7), np.ones(513), np.array([np.nan]*30)):
        with pytest.raises(ValueError, match='finite daily'):
            fit_latent_returns(bad)


def test_fold_leakage_and_saved_diagnostics(tmp_path):
    index = calendar().sessions_in_range('2025-12-01', '2026-03-31').tz_localize(None)
    values = prices_from_returns(np.random.default_rng(12).normal(0, .02, len(index)-1))
    prices = pd.DataFrame(values, index=index, columns=['AMZN'])
    config = ValidationConfig(reference_date='2026-04-01', months=2, seed=42)
    registry = builtin_models(n_paths=200)
    models = {name: registry[name] for name in RETURN_GP_MODELS}
    result = validate('AMZN', config=config, prices=prices, models=models)
    altered = prices.copy()
    altered.loc[altered.index >= '2026-02-01'] *= 1.7
    changed = validate('AMZN', config=config, prices=altered, models=models)
    for original, later in zip(result.metrics, changed.metrics):
        if original['origin'] == '2026-02-01':
            assert original['diagnostics'] == later['diagnostics']
            assert original['terminal_predicted_log_return'] == later['terminal_predicted_log_return']
            assert original['terminal_actual_log_return'] != later['terminal_actual_log_return']
    assert [d for d in result.distributions if d['origin'] == '2026-02-01'] == [
        d for d in changed.distributions if d['origin'] == '2026-02-01']
    restored = ValidationResult.load(result.save(tmp_path/'return-gp.json'))
    pd.testing.assert_frame_equal(restored.summary(), result.summary())
    for metric in restored.metrics:
        assert metric['diagnostics']['converged']
        assert metric['diagnostics']['time_unit'] == 'trading_days'
        assert '_statistics' not in metric['diagnostics']
    assert {'terminal_return_mae', 'brier_score'}.issubset(result.summary().columns)


@pytest.mark.parametrize('name', RETURN_GP_MODELS)
def test_public_adapter_keeps_empirical_forecast_and_fit_bands(name):
    from stock_api.forecast import _fit_forecast
    from stock_api.schemas import ForecastRequest
    values = prices_from_returns(np.random.default_rng(31).normal(0, .02, 59))
    index = pd.bdate_range(end='2026-05-22', periods=len(values))
    prices = pd.DataFrame(values, index=index, columns=['AMZN'])
    dates = pd.bdate_range('2026-05-26', periods=4)
    request = ForecastRequest(ticker='AMZN', model=name, lookback=60, seed=4, n_paths=200)
    result = _fit_forecast(request, prices, index[-1], dates)
    _, _, info = RETURN_GP_MODELS[name](values, 4, seed=4, n_paths=200, include_fit=True)
    np.testing.assert_allclose([p.lower_95 for p in result.forecasts], info['_statistics']['lower_95'])
    np.testing.assert_allclose([p.sigma_3.upper for p in result.forecasts], info['_statistics']['sigma_3_upper'])
    np.testing.assert_allclose([p.sigma_2.lower for p in result.fitted], info['_fit_statistics']['sigma_2_lower'])
    assert len(result.terminal_log_returns) == 200
    assert '_statistics' not in result.model_dump_json()
