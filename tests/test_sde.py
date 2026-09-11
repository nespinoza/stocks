import numpy as np
import pandas as pd
import pytest

from stock_api.sde import (simulate_gbm, euler_maruyama, ItoProcess, gbm_zero_drift,
                           gbm_estimated_drift, ou_returns, lognormal_statistics)
from stock_api import ValidationConfig, validate, builtin_models, ValidationResult
from stock_api.data import calendar


def training(seed=1, n=100):
    rng = np.random.default_rng(seed)
    return (100*np.exp(np.r_[0, np.cumsum(rng.normal(.001, .02, n-1))]))[:, None]


def test_exact_gbm_analytical_moments():
    s0, mu, sigma, t = 100, .001, .025, 7
    paths = simulate_gbm(s0, mu, sigma, steps=t, n_paths=150_000, seed=1)
    expected_mean = s0*np.exp(mu*t)
    expected_var = s0**2*np.exp(2*mu*t)*np.expm1(sigma**2*t)
    assert paths[:, -1].mean() == pytest.approx(expected_mean, abs=.06)
    assert paths[:, -1].var() == pytest.approx(expected_var, rel=.02)
    stats = lognormal_statistics(np.array([np.log(s0)+(mu-.5*sigma**2)*t]), np.array([sigma**2*t]), s0)
    assert stats['mean_price'][0] == pytest.approx(expected_mean)
    assert stats['price_std'][0]**2 == pytest.approx(expected_var)


def test_gbm_drift_and_reproducibility():
    prices = training()
    r = np.diff(np.log(prices[:, 0]))
    mean, variance, info = gbm_estimated_drift(prices, 7, seed=10)
    assert info['mu'] == pytest.approx(r.mean()+.5*r.var(ddof=1))
    assert mean[-1]-np.log(prices[-1, 0]) == pytest.approx(7*r.mean())
    assert variance[-1] == pytest.approx(7*r.var(ddof=1))
    np.testing.assert_equal(info['_ensemble'].paths, gbm_estimated_drift(prices, 7, seed=10)[2]['_ensemble'].paths)
    assert not np.array_equal(info['_ensemble'].paths, gbm_estimated_drift(prices, 7, seed=11)[2]['_ensemble'].paths)
    zero_mean, zero_var, zero_info = gbm_zero_drift(prices, 7)
    assert zero_info['mu'] == 0
    stats = lognormal_statistics(zero_mean, zero_var, prices[-1, 0])
    np.testing.assert_allclose(stats['mean_price'], prices[-1, 0])
    assert (stats['median_price'] < prices[-1, 0]).all()
    assert (stats['probability_up'] < .5).all()


def test_zero_volatility_and_zero_drift():
    paths = simulate_gbm(100, 0, 0, steps=7, n_paths=100, seed=1)
    np.testing.assert_equal(paths, 100)
    mean, variance, info = gbm_zero_drift(np.full((30,1),100), 7)
    stats = lognormal_statistics(mean, variance, 100)
    np.testing.assert_equal(variance, 0)
    np.testing.assert_equal(stats['price_std'], 0)
    np.testing.assert_equal(stats['probability_up'], 0)  # strict >, not >=
    np.testing.assert_allclose(stats['lower_95'],stats['upper_95'])
    growing = simulate_gbm(100, .01, 0, steps=7, n_paths=3)
    np.testing.assert_allclose(growing[0], 100*np.exp(.01*np.arange(8)))


def test_em_brownian_and_subclass():
    class Brownian(ItoProcess):
        def drift(self, x, t):
            return .2
        def diffusion(self, x, t):
            return .4
    paths = Brownian().simulate(1, steps=10, dt=.1, n_paths=30_000, seed=2)
    assert paths[:, -1].mean() == pytest.approx(1.2, abs=.01)
    assert paths[:, -1].var() == pytest.approx(.16, rel=.025)
    np.testing.assert_equal(paths, Brownian().simulate(1, steps=10, dt=.1, n_paths=30_000, seed=2))


def test_em_convergence_to_exact_gbm():
    # Same Brownian paths at two discretizations: a strong-error comparison.
    n, fine = 4000, 256
    mu, sigma, s0 = .04, .3, 100
    z = np.random.default_rng(12).standard_normal((n, fine))
    exact = s0*np.exp(mu-.5*sigma**2+sigma*z.sum(axis=1)/np.sqrt(fine))
    errors = []
    for steps in (4, 256):
        group = fine//steps
        normals = z.reshape(n, steps, group).sum(axis=2)/np.sqrt(group)
        paths = euler_maruyama(s0, lambda x,t:mu*x, lambda x,t:sigma*x,
                               steps=steps, dt=1/steps, n_paths=n, normal_draws=normals)
        errors.append(np.sqrt(np.mean((paths[:, -1]-exact)**2)))
    assert errors[1] < errors[0]/3
    assert paths[:, -1].mean() == pytest.approx(exact.mean(), abs=.1)
    assert paths[:, -1].var() == pytest.approx(exact.var(), rel=.02)


@pytest.mark.parametrize('model', [gbm_zero_drift, gbm_estimated_drift, ou_returns])
def test_predictive_distribution(model):
    prices = training()
    mean, variance, info = model(prices, 7, seed=42, n_paths=20_000)
    stats = lognormal_statistics(mean, variance, prices[-1, 0])
    assert (stats['lower_95'] <= stats['median_price']).all()
    assert (stats['median_price'] <= stats['upper_95']).all()
    assert ((stats['probability_up'] >= 0) & (stats['probability_up'] <= 1)).all()
    ensemble = info['_ensemble']
    assert len(ensemble.terminal_returns) == 20_000
    assert np.mean(ensemble.terminal_returns > 0) == pytest.approx(stats['probability_up'][-1], abs=.015)
    assert np.log(ensemble.paths[:, -1]).mean() == pytest.approx(mean[-1], abs=.004)
    assert np.log(ensemble.paths[:, -1]).var() == pytest.approx(variance[-1], rel=.04)


def test_ou_stability_and_fallback():
    rng = np.random.default_rng(4)
    returns = np.zeros(500)
    for i in range(1, 500):
        returns[i] = .001+.65*returns[i-1]+rng.normal(0,.01)
    prices = (100*np.exp(np.r_[0,np.cumsum(returns)]))[:,None]
    _, _, info = ou_returns(prices, 7)
    assert info['fallback'] is None
    assert 0 < info['ar_phi'] < .98
    assert info['theta'] > 0 and info['sigma'] > 0
    assert ou_returns(prices[:12],7)[2]['fallback'] == 'fewer_than_20_returns'
    assert ou_returns(np.full((60,1),100),7)[2]['fallback'] == 'degenerate_return_history'
    alternating = (100*np.exp(np.cumsum(np.tile([.01,-.01],50))))[:,None]
    assert ou_returns(alternating,7)[2]['fallback'] == 'ar_coefficient_outside_stable_ou_range'


def test_fold_leakage_and_diagnostics(tmp_path):
    index = calendar().sessions_in_range('2025-12-01','2026-03-31').tz_localize(None)
    prices = pd.DataFrame(training(n=len(index)),index=index,columns=['AMZN'])
    config = ValidationConfig(reference_date='2026-04-01',months=2,seed=42)
    registry = builtin_models(n_paths=500)
    selected = {name:registry[name] for name in ['gbm_zero_drift','gbm_estimated_drift','ou_returns','last_price']}
    result = validate('AMZN',config=config,prices=prices,models=selected)
    changed = prices.copy()
    changed.loc[changed.index >= '2026-03-01'] *= 5
    later = validate('AMZN',config=config,prices=changed,models=selected)
    for a,b in zip(result.metrics,later.metrics):
        if a['origin'] == '2026-02-01':
            assert a == b
    assert result.distributions[0] == later.distributions[0]
    assert np.isnan(result.summary().loc['last_price','terminal_direction_accuracy'])
    assert result.summary().loc['last_price','directional_folds'] == 0
    assert result.summary().loc['last_price','terminal_brier_score'] == pytest.approx(.25)
    for m in result.metrics:
        assert m['terminal_log_return_absolute_error'] == pytest.approx(abs(m['terminal_predicted_log_return']-m['terminal_actual_log_return']))
        assert 0 <= m['terminal_brier_score'] <= 1
    path = result.save(tmp_path/'result.json')
    restored = ValidationResult.load(path)
    assert restored.distributions == result.distributions
    pd.testing.assert_frame_equal(restored.summary(),result.summary())


@pytest.mark.parametrize('kwargs', [{'dt':0}, {'n_paths':0}, {'steps':0}])
def test_bad_simulation_settings(kwargs):
    options = {'steps':7,'n_paths':10,'dt':1,**kwargs}
    with pytest.raises(ValueError):
        simulate_gbm(100,0,.01,**options)


def test_active_ou_marginals_match_paths():
    rng = np.random.default_rng(18)
    r = np.zeros(300)
    for i in range(1,len(r)):
        r[i] = .0005+.7*r[i-1]+rng.normal(0,.012)
    prices = (100*np.exp(np.r_[0,np.cumsum(r)]))[:,None]
    mean, variance, info = ou_returns(prices,7,seed=19,n_paths=30_000)
    assert info['fallback'] is None
    log_paths = np.log(info['_ensemble'].paths[:,1:])
    np.testing.assert_allclose(log_paths.mean(axis=0),mean,atol=.002)
    np.testing.assert_allclose(log_paths.var(axis=0),variance,rtol=.035)


def test_parameters_do_not_see_first_fold_targets():
    index = calendar().sessions_in_range('2025-12-01','2026-02-28').tz_localize(None)
    prices = pd.DataFrame(training(n=len(index)),index=index,columns=['AMZN'])
    config = ValidationConfig(reference_date='2026-03-01',months=1)
    models = {name:spec for name,spec in builtin_models(n_paths=100).items() if name.startswith('gbm_') or name=='ou_returns'}
    base = validate('AMZN',config=config,prices=prices,models=models)
    changed = prices.copy()
    changed.loc[changed.index >= '2026-02-01'] *= 2
    other = validate('AMZN',config=config,prices=changed,models=models)
    assert base.distributions == other.distributions
    for a,b in zip(base.predictions,other.predictions):
        assert a['predicted_close'] == b['predicted_close']
        assert a['probability_up'] == b['probability_up']
        assert a['actual_close'] != b['actual_close']
    for a,b in zip(base.metrics,other.metrics):
        assert a['diagnostics'] == b['diagnostics']


def test_production_gbm_does_not_call_em(monkeypatch):
    import stock_api.sde as module
    def fail(*args,**kwargs):
        raise AssertionError('Production GBM must use exact transitions')
    monkeypatch.setattr(module,'euler_maruyama',fail)
    gbm_zero_drift(training(),7)
    gbm_estimated_drift(training(),7)
