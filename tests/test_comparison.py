import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stock_api import predict, ForecastComparison, ForecastResponse
from stock_api import forecast
from stock_api.api import app


@pytest.fixture
def snapshot(monkeypatch):
    index = pd.bdate_range(end='2026-05-22', periods=120)
    rng = np.random.default_rng(41)
    values = np.exp(np.log([100,150,200])+np.cumsum(rng.normal(0,.01,(120,3)),axis=0))
    frame = pd.DataFrame(values,index=index,columns=['AMZN','GOOGL','AAPL'])
    calls = []
    def load(tickers, lookback, as_of):
        calls.append(tickers)
        return frame.loc[:,tickers].iloc[-lookback:].copy()
    monkeypatch.setattr(forecast,'latest_session',lambda:index[-1])
    monkeypatch.setattr(forecast,'load_prices',load)
    return frame, calls


def test_shared_snapshot_and_individual_results(snapshot):
    _, calls = snapshot
    result = predict('AMZN',model=['gp','multitask_gp','last_price'],related_tickers=['GOOGL','AAPL'],lookback=60)
    assert isinstance(result,ForecastComparison)
    assert calls == [['AMZN','GOOGL','AAPL']]
    assert result['gp'].related_tickers == []
    assert result['multitask_gp'].related_tickers == ['GOOGL','AAPL']
    assert len(result.training_data) == 60
    assert list(result.results) == ['gp','multitask_gp','last_price']
    single = predict('AMZN',model='gp',lookback=60)
    assert isinstance(single,ForecastResponse)
    assert single.model_dump() == result['gp'].model_dump()


@pytest.mark.parametrize('models',[[],['gp','gp'],['invalid'],['gp',4]])
def test_invalid_lists_before_download(snapshot, models):
    with pytest.raises(ValueError):
        predict('AMZN',model=models)
    assert snapshot[1] == []


def test_singleton_list_and_http(snapshot):
    result = predict('AMZN',model=['gp'],lookback=60)
    assert isinstance(result,ForecastComparison)
    client = TestClient(app)
    response = client.post('/predict',json={'ticker':'AMZN','model':['gp'],'lookback':60})
    assert response.status_code == 200
    assert response.json() == result.model_dump()
    assert client.post('/predict',json={'ticker':'AMZN','model':[]}).status_code == 422


def test_sde_seed_is_independent_of_model_order(snapshot):
    a = predict('AMZN',model=['gbm_zero_drift','gp'],n_paths=100,seed=23)
    b = predict('AMZN',model=['gp','gbm_zero_drift'],n_paths=100,seed=23)
    assert a['gbm_zero_drift'].model_dump() == b['gbm_zero_drift'].model_dump()


def test_plot_comparison_offline(snapshot, monkeypatch, tmp_path):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = predict('AMZN',model=['gp','gbm_zero_drift'],lookback=60,n_paths=100)
    restored = ForecastComparison.model_validate_json(result.model_dump_json())
    def fail(*args,**kwargs):
        raise AssertionError('Plot must not download or refit')
    monkeypatch.setattr(forecast,'load_prices',fail)
    monkeypatch.setattr(forecast,'_fit_forecast',fail)
    ax = restored.plot(history_days=7,forecast_days=7,sigmas=(1,2),show=False)
    lines = {line.get_label():line for line in ax.lines}
    for name in result.results:
        assert lines[f'{name} fit'].get_linestyle() == '--'
        assert lines[f'{name} forecast'].get_linestyle() == '-'
        assert lines[f'{name} fit'].get_color() == lines[f'{name} forecast'].get_color()
    assert lines['gp forecast'].get_color() != lines['gbm_zero_drift forecast'].get_color()
    assert len(lines['AMZN training data'].get_xdata()) == 6
    assert len(lines['gp forecast'].get_xdata()) == 4
    assert len(ax.collections) == 8
    # Bands retain the actual returned bounds, not a shared/artificial width.
    vertices = ax.collections[1].get_paths()[0].vertices
    selected = [p for p in restored['gp'].fitted if p.date >= '2026-05-15']
    assert vertices[:,1].min() == pytest.approx(min(p.sigma_1.lower for p in selected))
    assert vertices[:,1].max() == pytest.approx(max(p.sigma_1.upper for p in selected))
    ax.figure.savefig(tmp_path/'comparison.png')
    plt.close(ax.figure)
    fig,ax = plt.subplots()
    assert restored.plot(show=False,ax=ax,show_fit=False,uncertainty='errorbars',sigmas=(2,)) is ax
    assert len(ax.containers) == 2
    plt.close(fig)


def test_fit_failure_names_model(snapshot, monkeypatch):
    from stock_api import ForecastError
    def fail(*args, **kwargs):
        raise ValueError('Numerical fit failed')
    monkeypatch.setitem(forecast.MODELS, 'gp', fail)
    with pytest.raises(ForecastError, match="Model 'gp'"):
        predict('AMZN', model=['last_price', 'gp'])
    response = TestClient(app).post('/predict', json={'ticker':'AMZN', 'model':['gp']})
    assert response.status_code == 500
    assert "Model 'gp'" in response.json()['detail']
