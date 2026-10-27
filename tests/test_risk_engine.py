import numpy as np
import pandas as pd
import pytest

from stock_api.allocation import equal_risk_contribution, inverse_volatility, minimum_variance, risk_contributions
from stock_api.engine import month_end_dates, performance, run_backtest
from stock_api.risk import (factor_covariance, garman_klass, har_forecast, ledoit_wolf, shrunk_correlation,
                            ewma_forecast)


def garch_like_returns(n=3000, seed=0):
    """Returns with persistent volatility, so HAR should beat a constant forecast."""
    rng = np.random.default_rng(seed)
    log_vol = np.zeros(n)
    for t in range(1, n):
        log_vol[t] = .98*log_vol[t-1]+.15*rng.standard_normal()
    vol = .01*np.exp(log_vol)
    return pd.Series(vol*rng.standard_normal(n), index=pd.bdate_range('2000-01-03', periods=n)), vol


def test_har_tracks_persistent_volatility_without_lookahead():
    returns, vol = garch_like_returns()
    errors_har, errors_flat = [], []
    for end in range(1600, 2900, 21):
        history = returns.iloc[:end]
        forecast, fit = har_forecast(history**2, history, horizon=21, window=1500)
        truth = np.sum(vol[end:end+21]**2)
        errors_har.append((np.log(forecast)-np.log(truth))**2)
        errors_flat.append((np.log(21*history.var())-np.log(truth))**2)
        assert fit['n'] <= 1500  # targets beyond the decision date are excluded
    assert np.mean(errors_har) < .7*np.mean(errors_flat)


def test_har_ignores_future_data():
    returns, _ = garch_like_returns()
    history = returns.iloc[:2000]
    extended = returns.copy()
    extended.iloc[2000:] *= 50
    assert har_forecast(history**2, history)[0] == pytest.approx(
        har_forecast(extended.iloc[:2000]**2, extended.iloc[:2000])[0])


def test_ewma_and_garman_klass_basics():
    returns = pd.Series(np.full(500, .01))
    assert ewma_forecast(returns, horizon=21) == pytest.approx(21*1e-4)
    assert garman_klass(np.array(100.), np.array(102.), np.array(99.), np.array(100.)) > 0


def test_ledoit_wolf_shrinks_more_with_fewer_observations_and_stays_pd():
    rng = np.random.default_rng(1)
    scales = np.linspace(.5, 2, 40)  # non-spherical truth, so full shrinkage is wrong
    few = rng.standard_normal((30, 40))*scales
    many = rng.standard_normal((3000, 40))*scales
    cov_few, shrink_few = ledoit_wolf(few)
    _, shrink_many = ledoit_wolf(many)
    assert shrink_few > shrink_many
    assert np.linalg.eigvalsh(cov_few).min() > 0
    corr, _ = shrunk_correlation(few)
    assert np.allclose(np.diag(corr), 1)


def test_factor_covariance_recovers_one_factor_structure():
    rng = np.random.default_rng(2)
    beta = rng.uniform(.5, 1.5, 20)
    x = np.outer(rng.standard_normal(5000), beta)+.5*rng.standard_normal((5000, 20))
    model = factor_covariance(x, factors=1)
    truth = np.outer(beta, beta)+.25*np.eye(20)
    assert np.abs(model-truth).max() < .15


def test_risk_parity_equalizes_contributions_and_reduces_to_inverse_vol():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 6))
    cov = a@a.T+np.eye(6)
    contributions = risk_contributions(equal_risk_contribution(cov), cov)
    assert np.allclose(contributions, 1/6, atol=1e-5)
    vols = np.array([.1, .2, .4])
    assert np.allclose(equal_risk_contribution(np.diag(vols**2)), inverse_volatility(vols), atol=1e-6)


def test_minimum_variance_respects_cap_and_beats_equal_weight():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((10, 10))
    cov = a@a.T+.1*np.eye(10)
    weights = minimum_variance(cov, cap=.25)
    assert weights.max() <= .25+1e-9 and weights.sum() == pytest.approx(1)
    equal = np.full(10, .1)
    assert weights@cov@weights <= equal@cov@equal


def test_engine_lag_costs_and_drift():
    dates = pd.bdate_range('2020-01-01', periods=6)
    returns = pd.DataFrame({'A': [0, .10, 0, 0, 0, 0], 'B': [0, 0, 0, .05, 0, 0]}, index=dates, dtype=float)
    decisions = {dates[0]: pd.Series({'A': 1., 'B': 0.}), dates[2]: pd.Series({'A': 0., 'B': 1.})}
    daily, weights = run_backtest(returns, decisions, cost_bps=10, lag=1)
    # Executed at dates[1] close: the +10% on dates[1] is NOT earned (decided at dates[0], filled after).
    assert daily.loc[dates[1], 'gross'] == 0
    assert daily.loc[dates[1], 'cost'] == pytest.approx(.001)
    # Switch decided dates[2], executed dates[3]: dates[3] return still belongs to A (0%).
    assert daily.loc[dates[3], 'gross'] == 0
    assert daily.loc[dates[3], 'turnover'] == pytest.approx(2)
    assert list(weights.index) == [dates[1], dates[3]]


def test_engine_rejects_invalid_weights_and_month_ends():
    dates = pd.bdate_range('2020-01-01', periods=70)
    returns = pd.DataFrame(0., index=dates, columns=['A', 'B'])
    with pytest.raises(ValueError):
        run_backtest(returns, {dates[0]: np.array([.7, .7])})
    assert list(month_end_dates(dates).strftime('%m-%d')) == ['01-31', '02-28', '03-31', '04-07']
    perf = performance(pd.Series(np.tile([.01, -.01], 50), index=pd.bdate_range('2020-01-01', periods=100)))
    assert perf['max_drawdown'] < 0
