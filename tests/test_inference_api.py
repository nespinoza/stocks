"""Public sampler selection, compatibility and posterior response integration."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stock_api import predict, ForecastComparison
from stock_api import forecast
from stock_api.api import app
from stock_api.schemas import ForecastRequest


@pytest.fixture
def snapshot(monkeypatch):
    rng = np.random.default_rng(21)
    frame = pd.DataFrame(np.exp(np.log([100, 200]) + np.cumsum(rng.normal(0, .01, (60, 2)), axis=0)),
                         index=pd.bdate_range(end='2026-05-22', periods=60), columns=['AMZN', 'AAPL'])
    calls = []
    def load(tickers, lookback, as_of):
        calls.append(tickers)
        return frame[tickers].copy()
    monkeypatch.setattr(forecast, 'latest_session', lambda: frame.index[-1])
    monkeypatch.setattr(forecast, 'load_prices', load)
    return frame, calls


@pytest.mark.parametrize('options', [
    {'model': ['gp'], 'inference': 'dynesty'},
    {'model': ['gp', 'gp'], 'inference': ['dynesty']},
    {'model': 'gp', 'inference': ['dynesty']},
    {'model': ['gp'], 'inference': ['dynesty'], 'inference_setup': {}},
    {'model': ['gp'], 'inference': ['dynesty'], 'inference_setup': [{}, {}]},
    {'model': 'gp', 'inference': 'dynesty', 'inference_setup': [{}]},
    {'model': 'gp', 'inference': 'mcmc'},
    {'model': 'var', 'inference': 'dynesty'},
    {'model': 'volatility_gp_returns', 'inference': 'dynesty'},
    {'model': 'last_price', 'inference': 'lbfgsb'},
    {'model': 'gp', 'inference_setup': {'nlive': 30}},
    {'model': 'gp', 'inference': 'lbfgsb', 'inference_setup': {'maxiter': 100}},
    {'model': 'gp', 'inference': 'dynesty', 'inference_setup': {'draws': 30}},
    {'model': 'gp', 'inference': 'dynesty', 'inference_setup': {'nlive': 1}},
    {'model': 'gp', 'inference': 'dynesty', 'inference_setup': {'seed': 4}},
    {'model': 'gp', 'inference': 'dynesty', 'inference_setup': {'priors': {'typo': [1, 2]}}},
    {'model': 'gp', 'inference': 'dynesty', 'inference_setup': {'priors': {'length': [3, 1]}}},
])
def test_invalid_before_download(snapshot, options):
    with pytest.raises(ValueError):
        predict('AMZN', **options)
    assert not snapshot[1]
    assert TestClient(app).post('/predict', json={'ticker': 'AMZN', **options}).status_code == 422
    assert not snapshot[1]


def test_legacy_and_explicit_optimizer(snapshot):
    old = predict('AMZN', model='gp', lookback=60)
    explicit_none = predict('AMZN', model='gp', lookback=60, inference=None, inference_setup=None)
    assert old.model_dump() == explicit_none.model_dump()
    selected = predict('AMZN', model='gp', lookback=60, inference='lbfgsb')
    assert selected.forecasts == old.forecasts
    assert selected.fitted == old.fitted
    assert selected.diagnostics['inference'] == 'lbfgsb'
    comparison = predict('AMZN', model=['gp', 'last_price'], inference=[None, None], lookback=60)
    assert comparison['gp'].model_dump() == old.model_dump()
    assert list(comparison.results) == ['gp', 'last_price']


@pytest.fixture
def stub_samplers(monkeypatch):
    from stock_api import posterior_forecast as adapter
    calls = []
    def sample(problem, config, **kwargs):
        calls.append((problem, config))
        coordinates = np.array([problem.transform(np.full(len(problem.names), .4)),
                                problem.transform(np.full(len(problem.names), .6))])
        return coordinates, np.array([.3, .7]), None, None, {'converged': False, 'method': 'test_sampler'}
    def draw(problem, samples, weights, latents, steps, config, **kwargs):
        assert kwargs == {'include_comparison': False}
        assert problem.old_info == {}  # No old optimizer invoked for initialization.
        rng = np.random.default_rng(config.seed)
        initial = problem.prices.iloc[-1, 0]
        return {'paths': initial * np.exp(np.column_stack([np.zeros(config.predictive_draws),
                    np.cumsum(rng.normal(.001, .01, (config.predictive_draws, steps)), axis=1)])),
                'f': np.zeros((config.predictive_draws, problem.n + steps)),
                'sigma': np.full((config.predictive_draws, problem.n + steps), .01)}
    monkeypatch.setattr(adapter, 'sample_gaussian', sample)
    monkeypatch.setattr(adapter, 'sample_latent', sample)
    monkeypatch.setattr(adapter, 'draw_predictive', draw)
    return calls


def test_repeated_models_settings_and_roundtrip(snapshot, stub_samplers):
    result = predict('AMZN', model=['gp', 'gp', 'gp', 'multitask_gp'],
                     inference=[None, 'dynesty', 'dynesty', 'dynesty'],
                     inference_setup=[None, {'nlive': 20}, {'nlive': 30}, {'priors': {'length': [.7, 100]}}],
                     related_tickers=['AAPL'], n_paths=100, seed=41, lookback=60)
    assert snapshot[1] == [['AMZN', 'AAPL']]
    assert list(result.results) == ['gp:lbfgsb', 'gp:dynesty', 'gp:dynesty#2', 'multitask_gp']
    assert [config.nlive for _, config in stub_samplers] == [20, 30, 150]
    assert stub_samplers[-1][1].priors.length == (.7, 100)
    assert [list(problem.prices.columns) for problem, _ in stub_samplers] == [['AMZN'], ['AMZN'], ['AMZN', 'AAPL']]
    assert all(config.seed == 41 and config.predictive_draws == 100 for _, config in stub_samplers)
    sampled = result['gp:dynesty']
    assert len(sampled.terminal_log_returns) == 100
    assert any('convergence' in text for text in sampled.warnings)
    assert len(sampled.fitted) == len(snapshot[0])
    assert [p.date for p in sampled.fitted] == [d.date().isoformat() for d in snapshot[0].index]
    assert ForecastComparison.model_validate_json(result.model_dump_json()) == result
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    restored = ForecastComparison.model_validate_json(result.model_dump_json())
    ax = restored.plot(show=False)
    lines = {line.get_label(): line for line in ax.lines}
    for name in result.results:
        assert lines[f'{name} fit'].get_linestyle() == '--'
        assert lines[f'{name} fit'].get_color() == lines[f'{name} forecast'].get_color()
        np.testing.assert_allclose(lines[f'{name} fit'].get_ydata(),
                                   [p.predicted_close for p in result[name].fitted])
    plt.close(ax.figure)


@pytest.mark.parametrize('model', ['volatility_gp_returns', 'heteroskedastic_gp_returns'])
def test_return_gp_mcmc_response(snapshot, stub_samplers, model):
    result = predict('AMZN', model=model, inference='mcmc',
                     inference_setup={'chains': 2, 'warmup': 10, 'draws': 20}, n_paths=100)
    assert stub_samplers[0][1].interweave
    assert all(p.mean_price is None and p.price_std is None for p in result.forecasts)
    if model == 'volatility_gp_returns':
        assert result.direction == 'flat'
        assert all(p.probability_up == .5 and p.median_price == result.last_close for p in result.forecasts)
    assert len(result.fitted) == len(snapshot[0]) - 1
    assert [p.date for p in result.fitted] == [d.date().isoformat() for d in snapshot[0].index[1:]]
    if model == 'volatility_gp_returns':
        np.testing.assert_array_equal([p.predicted_close for p in result.fitted], snapshot[0].AMZN.iloc[:-1])
    assert result.diagnostics['inference'] == 'mcmc'
    assert result.model_dump_json()


def test_http_sampler_selection(snapshot, stub_samplers):
    payload = {'ticker': 'AMZN', 'model': ['gp', 'gp'], 'inference': [None, 'dynesty'],
               'inference_setup': [None, {'nlive': 20}], 'n_paths': 100}
    response = TestClient(app).post('/predict', json=payload)
    assert response.status_code == 200
    assert response.json() == predict(**payload).model_dump()


@pytest.mark.parametrize('model,method,setup', [
    ('gp', 'dynesty', {'nlive': 20, 'nested_dlogz': 2., 'nested_maxcall': 1000}),
    ('volatility_gp_returns', 'mcmc', {'warmup': 10, 'draws': 20, 'chains': 2}),
])
def test_real_sampler_smoke_deterministic_no_optimizer(snapshot, monkeypatch, model, method, setup):
    if method == 'dynesty':
        pytest.importorskip('dynesty')
    def fail(*args, **kwargs):
        raise AssertionError('Posterior inference must not call legacy optimizer')
    monkeypatch.setitem(forecast.MODELS, model, fail)
    a = predict('AMZN', model=model, inference=method, inference_setup=setup, n_paths=100, seed=19)
    b = predict('AMZN', model=model, inference=method, inference_setup=setup, n_paths=100, seed=19)
    assert a.model_dump() == b.model_dump()
    paths = np.array(a.terminal_log_returns)
    assert np.isfinite(paths).all() and len(paths) == 100
    terminal = a.forecasts[-1]
    assert terminal.lower_95 == pytest.approx(np.quantile(np.exp(paths) * a.last_close, .025))
    assert terminal.upper_95 == pytest.approx(np.quantile(np.exp(paths) * a.last_close, .975))
    assert terminal.log_std == pytest.approx(paths.std())
    assert a.diagnostics['method'] in ('dynesty_exact_gaussian_marginal_likelihood', 'joint_latent_gaussian_mcmc_collapsed_f')


def test_fitted_evaluates_weighted_posterior_states_individually():
    from types import SimpleNamespace
    from stock_api.posterior_forecast import posterior_fitted
    calls = []
    def conditional(parameters, times):
        calls.append(float(parameters[0]))
        # Nonlinear response distinguishes mixing predictions from predicting
        # at averaged parameters. Zero noise isolates parameter uncertainty.
        return np.full(len(times), parameters[0] ** 2), np.zeros((len(times), len(times)))
    problem = SimpleNamespace(n=3, x=np.arange(3), predict=conditional,
                              prices=pd.DataFrame({'AMZN': [10., 11., 12.]},
                                                  index=pd.bdate_range('2026-01-05', periods=3)))
    settings = SimpleNamespace(seed=4, predictive_draws=1000)
    samples, weights = np.array([[1.], [3.]]), np.array([.8, .2])
    fitted = posterior_fitted(problem, samples, weights, {}, settings, gaussian=True)
    selected = np.random.default_rng(settings.seed + 2).choice(2, size=1000, p=weights)
    logs = samples[selected, 0] ** 2
    assert calls == [1., 3.]
    for point in fitted:
        assert point.predicted_close == pytest.approx(np.median(np.exp(logs)))
        assert point.log_std == pytest.approx(logs.std())
        assert point.sigma_2.lower == pytest.approx(np.exp(1.))
        assert point.sigma_2.upper == pytest.approx(np.exp(9.))


@pytest.mark.parametrize('model', ['volatility_gp_returns', 'heteroskedastic_gp_returns'])
def test_all_return_sampler_combinations_plot_fits_offline(snapshot, stub_samplers, monkeypatch, model):
    result = predict('AMZN', model=[model, model], inference=['mcmc', 'mcmc'],
                     inference_setup=[{'draws': 20}, {'draws': 30}], n_paths=100)
    restored = ForecastComparison.model_validate_json(result.model_dump_json())
    def fail(*args, **kwargs):
        raise AssertionError('Plotting must not refit or download')
    monkeypatch.setattr(forecast, 'load_prices', fail)
    monkeypatch.setattr(forecast, '_fit_forecast', fail)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ax = restored.plot(show=False, history_days=7, sigmas=(2,), uncertainty='errorbars')
    labels = {line.get_label() for line in ax.lines}
    for name in restored.results:
        assert f'{name} fit' in labels and f'{name} forecast' in labels
    plt.close(ax.figure)
    ax = restored.plot(show=False, show_fit=False)
    assert not any(line.get_label().endswith(' fit') for line in ax.lines)
    plt.close(ax.figure)
