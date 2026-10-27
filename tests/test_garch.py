"""GJR recursion, distribution, exogenous-input and integration checks."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from arch.univariate import ARX, GARCH, StudentsT
from stock_api.garch import (RegressionGJRFit, fit_regression_gjr,
                             regression_gjr_garch, next_variance)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    y = np.empty(600)
    r, e, h = 0., 0., .0003
    for i in range(len(y)):
        h = .00001 + (.08 + .12*(e < 0))*e*e + .8*h
        e = np.sqrt(h)*rng.standard_t(8)*np.sqrt(6/8)
        r = .0003 + .15*r + e
        y[i] = r
    return y


def test_variance_and_residuals_match_reference_library(returns):
    fit = fit_regression_gjr(returns)
    model = ARX(pd.Series(100*returns, name='return'), lags=1, rescale=False)
    model.volatility = GARCH(p=1, o=1, q=1)
    model.distribution = StudentsT()
    reference = model.fit(disp='off', update_freq=0, show_warning=False,
                          options={'maxiter':2000, 'ftol':1e-9})
    np.testing.assert_allclose(fit.parameters, reference.params)
    omega, alpha, gamma, beta, _ = fit.parameters[-5:]
    expected = reference.forecast(horizon=1, reindex=False).residual_variance.iloc[-1, 0]
    assert next_variance(fit.last_residual, fit.last_variance, omega, alpha, gamma, beta) == pytest.approx(expected)
    np.testing.assert_allclose(fit.fitted_mean, returns[1:]-reference.resid.to_numpy()[1:]/100)


def test_path_recursion_uses_simulated_returns_and_asymmetry():
    params = np.array([.1, .2, .02, .1, .2, .7, 8.])
    fit = RegressionGJRFit(params, np.empty(0), np.empty(0), .5, -.3, .04,
                           np.empty(0), np.empty(0), {})
    class FixedInnovations:
        def standard_t(self, nu, size):
            return np.ones(size)*np.sqrt(nu/(nu-2))
    actual = fit.simulate_returns(3, n_paths=2, rng=FixedInnovations())
    r, e, h = .5, -.3, .04
    expected = []
    for _ in range(3):
        h = .02 + (.1+.2*(e < 0))*e*e + .7*h
        e = np.sqrt(h)
        r = .1+.2*r+e
        expected.append(r/100)
    np.testing.assert_allclose(actual, np.tile(expected,(2,1)))
    assert next_variance(-.3,.04,.02,.1,.2,.7) > next_variance(.3,.04,.02,.1,.2,.7)


def test_external_regression_recovery_and_explicit_future_inputs():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(2000,1))
    y = .002 + .007*x[:,0] + rng.normal(0,.01,len(x))
    fit = fit_regression_gjr(y, regressors=x)
    assert fit.diagnostics['regressor_coefficients_log_return'][0] == pytest.approx(.007, abs=.001)
    assert fit.diagnostics['intercept_log_return'] == pytest.approx(.002, abs=.001)
    with pytest.raises(ValueError, match='Future regressors'):
        fit.simulate_returns(3)
    low = fit.simulate_returns(3,n_paths=100,seed=3,future_regressors=np.zeros((3,1)))
    high = fit.simulate_returns(3,n_paths=100,seed=3,future_regressors=np.ones((3,1)))
    assert np.all(high > low)
    scenarios = np.zeros((100,3,1))
    np.testing.assert_array_equal(low,fit.simulate_returns(3,n_paths=100,seed=3,future_regressors=scenarios))
    with pytest.raises(ValueError, match='Future regressors'):
        fit.simulate_returns(3,future_regressors=np.ones((2,1)))


def test_public_distribution_seed_fit_bands_and_missing_price_moments(returns, monkeypatch):
    from stock_api import predict, ForecastResponse
    from stock_api import forecast
    dates = pd.bdate_range(end='2026-08-28', periods=251)
    prices = pd.DataFrame({'AMZN':100*np.exp(np.r_[0,returns[-250:]].cumsum())}, index=dates)
    monkeypatch.setattr(forecast,'latest_session',lambda:dates[-1])
    monkeypatch.setattr(forecast,'load_prices',lambda *a,**k:prices.copy())
    a = predict('AMZN',model='regression_gjr_garch',seed=17,n_paths=2000)
    b = predict('AMZN',model=['last_price','regression_gjr_garch'],seed=17,n_paths=2000)['regression_gjr_garch']
    assert a.model_dump() == b.model_dump()
    assert ForecastResponse.model_validate_json(a.model_dump_json()) == a
    assert all(p.mean_price is None and p.price_std is None for p in a.forecasts)
    samples = np.asarray(a.terminal_log_returns)
    assert a.forecasts[-1].probability_up == pytest.approx((samples > 0).mean())
    assert a.forecasts[-1].lower_95 == pytest.approx(a.last_close*np.exp(np.quantile(samples,.025)))
    assert a.fitted[0].date == str(dates[2].date())
    nu = a.diagnostics['parameters_percent_units']['nu']
    point = a.fitted[0]
    assert point.sigma_2.upper == pytest.approx(point.predicted_close*np.exp(point.log_std*t.ppf(norm.cdf(2),nu)*np.sqrt((nu-2)/nu)))


def test_invalid_inputs_and_failed_fits_are_not_replaced(returns, monkeypatch):
    for y in [returns[:20], np.zeros(80), np.full(80,np.nan)]:
        with pytest.raises(ValueError, match='31 finite'):
            fit_regression_gjr(y)
    with pytest.raises(ValueError, match='Constant regressors'):
        fit_regression_gjr(returns,regressors=np.ones((len(returns),1)))
    original = ARX.fit
    def failed(self,*args,**kwargs):
        result = original(self,*args,**kwargs)
        result.optimization_result.status = 9
        return result
    monkeypatch.setattr(ARX,'fit',failed)
    with pytest.raises(ValueError,match='optimizer failed'):
        fit_regression_gjr(returns)


def test_historical_gjr_preserves_distribution_and_no_future_leakage(returns,tmp_path):
    from stock_api.historical import HistoricalConfig,run_historical_validation
    from stock_api.historical.config import sessions_for
    from stock_api.historical.runner import load_record
    from stock_api.historical.analysis import score
    cfg = HistoricalConfig(target_ticker='AMZN',start_date='2025-09-01',end_date='2025-09-10',
                           models=['last_price','regression_gjr_garch'],lookbacks=[90],n_paths=300)
    dates = sessions_for(cfg)
    prices = pd.DataFrame({'AMZN':100*np.exp(returns[:len(dates)].cumsum())},index=dates)
    run_historical_validation(cfg,output=tmp_path/'a',prices=prices,progress=lambda _:None)
    changed = prices.copy();changed.loc[changed.index>='2025-09-01'] *= 2
    run_historical_validation(cfg,output=tmp_path/'b',prices=changed,progress=lambda _:None)
    a = load_record(next((tmp_path/'a/records').glob('*__regression_gjr_garch__*.json')))
    b = load_record(next((tmp_path/'b/records').glob('*__regression_gjr_garch__*.json')))
    assert a['status'] == b['status'] == 'ok'
    assert a['prediction'] == b['prediction'] and a['realized'] != b['realized']
    assert a['prediction']['path_mean'] is None
    assert a['prediction']['distribution'] == 'conditional_student_t_gjr_simulation'
    assert np.isnan(score(a)['negative_log_score'])


def test_validation_registry_replays_gjr(returns):
    from stock_api import ValidationConfig, validate, builtin_models
    from stock_api.data import calendar
    dates = calendar().sessions_in_range('2025-01-01','2025-09-30').tz_localize(None)
    prices = pd.DataFrame({'AMZN':100*np.exp(returns[:len(dates)].cumsum())},index=dates)
    cfg = ValidationConfig(reference_date='2025-10-01',months=1,train_days=90)
    models = {'regression_gjr_garch':builtin_models(n_paths=300)['regression_gjr_garch']}
    result = validate('AMZN',config=cfg,prices=prices,models=models)
    assert result.rerun(models).predictions == result.predictions


def test_backtest_explicit_gjr_is_reproducible(returns):
    from stock_api.backtest import evaluate
    prices = 100*np.exp(returns[:130].cumsum())[:,None]
    settings = dict(lookback=120,horizon=5,folds=2,models=['regression_gjr_garch'])
    assert evaluate(prices,**settings) == evaluate(prices,**settings)
