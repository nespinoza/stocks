import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stock_api import api, data, forecast
from stock_api.models import MODELS, multitask_gp, random_walk
from stock_api.backtest import evaluate


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    common = rng.normal(0.0005, 0.012, (100, 1))
    returns = common + rng.normal(0, 0.004, (100, 3))
    return np.exp(np.log([100, 150, 200]) + np.cumsum(returns, axis=0))


@pytest.mark.parametrize("name", MODELS)
def test_model_distribution(name, prices):
    mean, variance, _ = MODELS[name](prices, 7)
    assert mean.shape == variance.shape == (7,)
    assert np.isfinite(mean).all() and np.isfinite(variance).all()
    assert (variance > 0).all()


def test_baseline(prices):
    mean, variance, _ = random_walk(prices, 7)
    np.testing.assert_allclose(np.exp(mean), prices[-1, 0])
    np.testing.assert_allclose(variance, variance[0] * np.arange(1, 8))


def test_gp_uses_related_tickers(prices):
    joint = multitask_gp(prices, 7)[0]
    altered = prices.copy()
    altered[-10:, 1:] *= np.linspace(1, 1.4, 10)[:, None]
    assert not np.allclose(joint, multitask_gp(altered, 7)[0], atol=1e-5)
    order = multitask_gp(prices[:, [0, 2, 1]], 7)[0]
    np.testing.assert_allclose(joint, order, atol=1e-4)


@pytest.mark.parametrize("name", MODELS)
def test_constant_series(name):
    mean, variance, _ = MODELS[name](np.ones((60, 2)) * 100, 7)
    np.testing.assert_allclose(np.exp(mean), 100, atol=1e-5)
    assert np.isfinite(variance).all()


def test_sessions():
    # Friday before Memorial Day: skip weekend and Monday holiday.
    days = data.future_sessions(pd.Timestamp("2026-05-22"), 7, "calendar")
    assert [d.day for d in days] == [26, 27, 28, 29]
    assert len(data.future_sessions(pd.Timestamp("2026-05-22"), 7, "trading")) == 7
    assert data.latest_session(pd.Timestamp("2026-05-26T14:00:00Z")) == pd.Timestamp("2026-05-22")


@pytest.fixture
def client(monkeypatch, prices):
    monkeypatch.setattr(forecast, "latest_session", lambda: pd.Timestamp("2026-05-22"))
    monkeypatch.setattr(forecast, "load_prices", lambda tickers, lookback, as_of:
                        pd.DataFrame(prices[-lookback:, :len(tickers)], columns=tickers,
                                     index=data.calendar().sessions_in_range("2025-01-01", "2026-05-22")[-lookback:]))
    return TestClient(api.app)


@pytest.mark.parametrize("name", MODELS)
def test_api(client, name):
    payload = {"ticker":"amzn", "model":name, "lookback":60}
    if name not in ("last_price", "random_walk", "gp", "gbm_zero_drift", "gbm_estimated_drift", "ou_returns"):
        payload["related_tickers"] = ["googl", "aapl"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ticker"] == "AMZN"
    assert len(result["forecasts"]) == 4
    for point in result["forecasts"]:
        assert point["lower_95"] <= point["predicted_close"] <= point["upper_95"]
        assert 0 <= point["probability_up"] <= 1
    assert result["predicted_change_pct"] == result["forecasts"][-1]["predicted_change_pct"]


@pytest.mark.parametrize("payload", [{"ticker":""}, {"ticker":"AAPL", "horizon":0},
    {"ticker":"AAPL", "model":"bad"}, {"ticker":"AAPL", "lookback":999},
    {"ticker":"AAPL", "related_tickers":["B", "C", "D", "E"]}])
def test_validation(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_errors_release_slot(client, monkeypatch):
    for error, status in [(data.DataError("missing prices"), 422),
                          (data.ProviderError("provider down"), 502)]:
        def fail(*args):
            raise error
        monkeypatch.setattr(forecast, "load_prices", fail)
        assert client.post("/predict", json={"ticker":"AAPL"}).status_code == status
    assert api.fit_slot.acquire(blocking=False)
    try:
        assert client.post("/predict", json={"ticker":"AAPL"}).status_code == 503
    finally:
        api.fit_slot.release()


def test_download_alignment(monkeypatch):
    as_of = pd.Timestamp("2026-05-22")
    index = data.calendar().sessions_in_range("2026-01-01", as_of)[-60:]
    raw = pd.DataFrame(np.ones((60, 2))*100, index=index,
                       columns=pd.MultiIndex.from_product([["Close"], ["AMZN", "AAPL"]]))
    monkeypatch.setattr(data.yf, "download", lambda *args, **kwargs: raw)
    frame = data.load_prices(["AAPL", "AMZN"], 60, as_of)
    assert list(frame.columns) == ["AAPL", "AMZN"]
    assert len(frame) == 60
    raw.iloc[-1, 0] = np.nan
    with pytest.raises(data.DataError, match="AMZN"):
        data.load_prices(["AMZN", "AAPL"], 60, as_of)


def test_walk_forward_has_no_future_leakage(monkeypatch, prices):
    windows = []
    def spy(train, steps):
        windows.append(train.copy())
        return random_walk(train, steps)
    monkeypatch.setitem(MODELS, "spy", spy)
    result = evaluate(prices, lookback=60, horizon=7, folds=2)
    np.testing.assert_equal(windows[0], prices[26:86])
    np.testing.assert_equal(windows[1], prices[33:93])
    assert result["spy"]["folds"] == 2


def test_gp_matches_full_joint_conditioning(prices):
    # Conditioning uses the target's covariance only, at the jointly learned timescale.
    prices = prices[-60:]
    mean, variance, info = multitask_gp(prices, 3)
    logs = np.log(prices[:, 0])
    scale = logs.std()
    y = (logs-logs.mean())/scale
    amplitude = info['standardized_amplitudes'][0]
    noise = info['standardized_noises'][0]
    def kernel(a, b):
        d = np.sqrt(3)*np.abs(a[:, None]-b[None, :])/info['length_scale']
        return amplitude**2*(1+d)*np.exp(-d)
    x, future = np.arange(60), np.arange(60,63)
    joint = kernel(x,x)+(noise**2+1e-8)*np.eye(60)
    cross = kernel(x,future)
    expected_mean = cross.T@np.linalg.solve(joint,y)
    expected_var = amplitude**2+noise**2-np.sum(cross*np.linalg.solve(joint,cross),axis=0)
    np.testing.assert_allclose(mean, expected_mean*scale+logs.mean(), atol=1e-9)
    np.testing.assert_allclose(variance, expected_var*scale**2, atol=1e-9)
    _, _, retained = multitask_gp(prices, 3, include_fit=True)
    _, fitted_mean, fitted_variance = retained['_fit']
    cross = kernel(x,x)
    np.testing.assert_allclose(fitted_mean, cross.T@np.linalg.solve(joint,y)*scale+logs.mean(),atol=1e-9)
    expected_var = amplitude**2+noise**2-np.sum(cross*np.linalg.solve(joint,cross),axis=0)
    np.testing.assert_allclose(fitted_variance,expected_var*scale**2,atol=1e-9)


@pytest.mark.parametrize("name", ["random_walk", "multitask_gp", "var"])
def test_public_python_api_matches_http(client, name):
    from stock_api import predict, ForecastResponse

    kwargs = {"ticker": "amzn", "model": name, "lookback": 60}
    if name not in ("last_price", "random_walk", "gp", "gbm_zero_drift", "gbm_estimated_drift", "ou_returns"):
        kwargs["related_tickers"] = ["googl", "aapl"]
    result = predict(**kwargs)
    assert isinstance(result, ForecastResponse)
    assert result.ticker == "AMZN"
    assert result.forecasts[0].predicted_close > 0
    assert result.model_dump() == client.post("/predict", json=kwargs).json()


def test_python_validation():
    from stock_api import predict
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        predict("AMZN", horizon=0)
    with pytest.raises(ValidationError):
        predict("AMZN", related_tickers="AAPL")


def test_python_errors(client, monkeypatch):
    from stock_api import predict, DataError, ProviderError, ForecastError

    for error in [DataError("missing"), ProviderError("unavailable"), ValueError("numerical failure")]:
        def fail(*args):
            raise error
        monkeypatch.setattr(forecast, "load_prices", fail)
        expected = ForecastError if type(error) is ValueError else type(error)
        with pytest.raises(expected):
            predict("AMZN")
    assert client.post("/predict", json={"ticker": "AMZN"}).status_code == 500


@pytest.mark.parametrize("name,offset", [("random_walk", 1), ("multitask_gp", 0), ("var", 6)])
def test_fit_and_sigma_intervals(client, name, offset, prices):
    from stock_api import predict
    result = predict("AMZN", model=name, lookback=60)
    assert len(result.training_data) == 60
    assert len(result.fitted) == 60-offset
    assert result.fitted[0].date == result.training_data[offset].date
    np.testing.assert_allclose([p.closes["AMZN"] for p in result.training_data], prices[-60:, 0])
    for point in [*result.fitted, *result.forecasts]:
        assert point.sigma_3.lower <= point.sigma_2.lower <= point.sigma_1.lower <= point.predicted_close
        assert point.predicted_close <= point.sigma_1.upper <= point.sigma_2.upper <= point.sigma_3.upper
        for n in (1, 2, 3):
            interval = getattr(point, f"sigma_{n}")
            np.testing.assert_allclose(interval.lower, point.predicted_close*np.exp(-n*point.log_std))
            np.testing.assert_allclose(interval.upper, point.predicted_close*np.exp(n*point.log_std))
    if name == "random_walk":
        np.testing.assert_allclose([p.predicted_close for p in result.fitted], prices[-60:-1, 0])


@pytest.mark.parametrize("name", ["random_walk", "multitask_gp", "var"])
def test_retaining_fit_preserves_forecast(name, prices):
    mean, variance, _ = MODELS[name](prices, 7)
    retained_mean, retained_variance, info = MODELS[name](prices, 7, include_fit=True)
    np.testing.assert_allclose(mean, retained_mean, atol=1e-10)
    np.testing.assert_allclose(variance, retained_variance, atol=1e-10)
    assert "_fit" in info


def test_gp_shares_timescale_without_cross_ticker_covariance(prices):
    mean, variance, info = multitask_gp(prices, 7)
    # Reverse auxiliary log-price signs: their smoothness is unchanged, while
    # correlations with the target flip. Independent likelihoods must be invariant.
    reversed_aux = prices.copy()
    reversed_aux[:, 1:] = 10000 / reversed_aux[:, 1:]
    other_mean, other_variance, other_info = multitask_gp(reversed_aux, 7)
    np.testing.assert_allclose(mean, other_mean, atol=1e-5)
    np.testing.assert_allclose(variance, other_variance, atol=1e-7)
    assert len(info['amplitudes']) == 3
    assert len(info['observation_noises']) == 3
    assert info['length_scale'] == pytest.approx(other_info['length_scale'], rel=1e-4)
    assert info['kernel'] == 'matern_3_2'


def test_last_price_exact(client):
    from stock_api import predict
    result = predict('AMZN', model='last_price', horizon_unit='trading', lookback=60)
    assert len(result.forecasts) == 7
    assert all(p.predicted_close == result.last_close for p in result.forecasts)
    assert result.direction == 'flat'


def test_gp_ignores_auxiliary_tickers(prices):
    from stock_api.models import gp
    mean, variance, info = gp(prices, 7, include_fit=True)
    altered = prices.copy()
    altered[:, 1:] *= np.linspace(1, 5, len(prices))[:, None]
    other_mean, other_variance, _ = gp(altered, 7)
    single_mean, single_variance, _ = multitask_gp(prices[:, :1], 7)
    np.testing.assert_allclose(mean, other_mean)
    np.testing.assert_allclose(variance, other_variance)
    np.testing.assert_allclose(mean, single_mean)
    np.testing.assert_allclose(variance, single_variance)
    assert len(info['amplitudes']) == 1
    assert len(info['_fit'][1]) == len(prices)


def test_gp_python_interface(client):
    from stock_api import predict
    result = predict('AMZN', model='gp', lookback=60)
    assert result.model == 'gp'
    assert len(result.fitted) == 60
    assert len(result.diagnostics['amplitudes']) == 1
    with pytest.raises(ValueError, match='only the target'):
        predict('AMZN', model='gp', related_tickers=['AAPL'])
