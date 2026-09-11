"""Plot from a serialized result: no network or refitting required."""
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stock_api import predict, ForecastResponse
from stock_api import forecast


@pytest.fixture
def result(monkeypatch):
    dates = pd.bdate_range(end="2026-05-22", periods=60)
    values = np.exp(np.linspace(4.5, 4.7, 60))
    monkeypatch.setattr(forecast, "latest_session", lambda: dates[-1])
    monkeypatch.setattr(forecast, "load_prices", lambda *args:
                        pd.DataFrame({"AMZN":values, "AAPL":values*2}, index=dates))
    result = predict("AMZN", related_tickers=["AAPL"], model="var", lookback=60)
    def fail(*args, **kwargs):
        raise AssertionError("Plot must not fetch or fit")
    monkeypatch.setattr(forecast, "load_prices", fail)
    monkeypatch.setattr(forecast, "_forecast", fail)
    return ForecastResponse.model_validate_json(result.model_dump_json())


def test_default_plot(result, tmp_path):
    ax = result.plot(show=False)
    lines = {line.get_label(): line for line in ax.lines}
    assert len(lines["AMZN training data"].get_xdata()) == 60
    assert len(lines["Historical fit"].get_xdata()) == 54
    assert len(lines["Forecast"].get_xdata()) == 4
    np.testing.assert_allclose(lines["AAPL training (rebased)"].get_ydata(),
                               lines["AMZN training data"].get_ydata())
    assert len(ax.collections) == 6
    ax.figure.savefig(tmp_path/"forecast.png")
    assert (tmp_path/"forecast.png").stat().st_size > 1000
    plt.close(ax.figure)


def test_zoom_and_controls(result):
    fig, ax = plt.subplots()
    returned = result.plot(history_days=7, forecast_days=4, show_related=False,
                           show_fit=False, sigmas=(1, 3), uncertainty="errorbars", ax=ax, show=False)
    assert returned is ax
    lines = {line.get_label(): line for line in ax.lines}
    assert len(lines["AMZN training data"].get_xdata()) == 6
    assert len(lines["Forecast"].get_xdata()) == 1
    assert "Historical fit" not in lines
    assert "AAPL training (rebased)" not in lines
    assert len(ax.containers) == 2
    plt.close(fig)


def test_hide_forecast_and_uncertainty(result):
    ax = result.plot(forecast_days=0, sigmas=(), show=False)
    assert "Forecast" not in [line.get_label() for line in ax.lines]
    assert not ax.collections
    plt.close(ax.figure)


@pytest.mark.parametrize("kwargs", [{"history_days":-1}, {"forecast_days":1.5},
                                    {"sigmas":(4,)}, {"uncertainty":"bad"}])
def test_invalid_controls(result, kwargs):
    with pytest.raises(ValueError):
        result.plot(**kwargs, show=False)
