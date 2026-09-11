import numpy as np
import pandas as pd
import pytest

from stock_api import (ModelSpec, ValidationConfig, ValidationResult, builtin_models,
                       compare_results, validate)
from stock_api.validation import make_folds
from stock_api.data import calendar


@pytest.fixture
def config():
    return ValidationConfig(reference_date='2027-09-01')


@pytest.fixture
def prices():
    index = calendar().sessions_in_range('2026-07-01', '2027-08-31').tz_localize(None)
    rng = np.random.default_rng(21)
    values = np.exp(np.log([200, 150, 180])+np.cumsum(rng.normal(0, .01, (len(index), 3)), axis=0))
    return pd.DataFrame(values, index=index, columns=['AMZN','GOOGL','AAPL'])


class LastPrice:
    def fit_predict(self, train, future, *, rng):
        return pd.DataFrame({'predicted_close':train.prices.iloc[-1, 0]}, index=future.index)


class RandomModel:
    def fit_predict(self, train, future, *, rng):
        return pd.DataFrame({'predicted_close':train.prices.iloc[-1, 0]*np.exp(rng.normal(0, .01, len(future)))},
                            index=future.index)


def test_monthly_dates(config):
    folds = make_folds(config)
    assert len(folds) == 12
    assert folds[0][0] == pd.Timestamp('2026-09-01')
    assert folds[-1][0] == pd.Timestamp('2027-08-01')
    february = next(f for f in folds if f[0] == pd.Timestamp('2027-02-01'))
    _, train, test = february
    assert train[0] >= pd.Timestamp('2027-01-02')
    assert train[-1] == pd.Timestamp('2027-01-29')
    assert test[0] == pd.Timestamp('2027-02-01')
    assert test[-1] == pd.Timestamp('2027-02-05')
    for origin, train, test in folds:
        assert train.max() < origin <= test.min()
        assert test.max() < pd.Timestamp(config.reference_date)


def test_trading_and_weekly():
    cfg = ValidationConfig(reference_date='2027-09-01', months=2, train_days=30,
                           train_unit='trading', horizon_unit='trading', frequency='weekly')
    folds = make_folds(cfg)
    assert len(folds) >= 6
    assert all(len(train) == 30 and len(test) == 7 for _, train, test in folds)


def test_no_leakage_and_fresh_models(config, prices):
    features = pd.DataFrame({'sentiment':np.arange(len(prices)), 'scheduled':1.0}, index=prices.index)
    calls = []
    class Spy:
        def __init__(self):
            self.used = False
        def fit_predict(self, train, future, *, rng):
            assert not self.used
            self.used = True
            assert train.prices.index.max() < future.index.min()
            assert train.features.index.equals(train.prices.index)
            assert list(future.columns) == ['time', 'scheduled']
            assert 'sentiment' in train.features
            assert list(train.prices.columns) == ['AMZN','GOOGL','AAPL']
            calls.append(train.prices.index.copy())
            value = train.prices.iloc[-1, 0]
            train.prices.iloc[:, :] = 1.0  # Copies protect later folds and scores.
            return pd.DataFrame({'predicted_close':value}, index=future.index)
    original = prices.copy()
    result = validate('AMZN', related_tickers=['GOOGL','AAPL'], config=config, prices=prices,
                      features=features, known_future_features=['scheduled'], models={'spy':ModelSpec(Spy,'1')})
    pd.testing.assert_frame_equal(original, prices)
    assert len(calls) == 12
    assert len(result.metrics) == 12
    assert result.summary().loc['spy','folds'] == 12


def test_metrics_and_snapshot_replay(config, prices, tmp_path):
    result = validate('AMZN', config=config, prices=prices, models={'baseline':ModelSpec(LastPrice,'1')})
    points = pd.DataFrame(result.predictions)
    first = points[points.origin == result.metrics[0]['origin']]
    error = first.predicted_close-first.actual_close
    assert result.metrics[0]['mae'] == pytest.approx(np.abs(error).mean())
    assert result.metrics[0]['rmse'] == pytest.approx(np.sqrt((error**2).mean()))
    assert result.summary().loc['baseline','mean_mae'] == pytest.approx(np.mean([m['mae'] for m in result.metrics]))
    assert result.metrics[0]['coverage_95'] is None
    path = result.save(tmp_path/'baseline.json')
    with pytest.raises(FileExistsError):
        result.save(path)
    restored = ValidationResult.load(path)
    assert restored.model_dump() == result.model_dump()
    new = restored.rerun({'new':ModelSpec(RandomModel,'1')})
    again = restored.rerun({'new':ModelSpec(RandomModel,'1')})
    assert new.predictions == again.predictions
    assert new.experiment_id == result.experiment_id
    combined = compare_results(restored, new)
    assert list(combined.summary().index) == ['baseline','new']
    assert len(restored.metrics) == 12
    with pytest.raises(ValueError, match='overlap'):
        compare_results(restored, restored)
    altered = prices.copy()
    altered.loc['2027-08-04', 'AMZN'] *= 1.01
    different = validate('AMZN', config=config, prices=altered, models={'other':ModelSpec(LastPrice,'1')})
    with pytest.raises(ValueError, match='differ'):
        compare_results(restored, different)


def test_replay_features(config, prices):
    features = pd.DataFrame({'sentiment':np.sin(np.arange(len(prices)))}, index=prices.index)
    result = validate('AMZN', config=config, prices=prices, features=features,
                      models={'a':ModelSpec(LastPrice,'1')})
    assert result.rerun({'b':ModelSpec(LastPrice,'1')}).experiment_id == result.experiment_id


@pytest.mark.parametrize('model', ['random_walk','gp','multitask_gp','var'])
def test_builtin_models(model, prices):
    cfg = ValidationConfig(reference_date='2027-09-01', months=2)
    result = validate('AMZN', related_tickers=['GOOGL','AAPL'], config=cfg, prices=prices, models=[model])
    assert len(result.metrics) == 2
    assert all(0 <= m['coverage_95'] <= 1 for m in result.metrics)
    assert result.summary().loc[model, 'mean_mae'] >= 0


def test_missing_data_and_features(config, prices):
    with pytest.raises(ValueError, match='cover'):
        validate('AMZN', config=config, prices=prices.drop(pd.Timestamp('2027-02-01')))
    late = pd.DataFrame({'sentiment':[.3]}, index=[pd.Timestamp('2027-01-01')])
    with pytest.raises(ValueError, match='lack past'):
        validate('AMZN', config=config, prices=prices, features=late)


def test_future_cutoff_without_download(config, monkeypatch):
    import stock_api.validation as module
    monkeypatch.setattr(module, 'latest_session', lambda:pd.Timestamp('2026-09-09'))
    def fail(*args):
        raise AssertionError('Must reject future observations before downloading')
    monkeypatch.setattr(module, 'load_prices', fail)
    with pytest.raises(ValueError, match='not yet available'):
        validate('AMZN', config=config)


def test_invalid_predictions(config, prices):
    class Broken:
        def fit_predict(self, train, future, *, rng):
            return pd.DataFrame({'predicted_close':np.nan}, index=future.index)
    with pytest.raises(ValueError, match="broken.*2026-09-01"):
        validate('AMZN', config=config, prices=prices, models={'broken':ModelSpec(Broken,'1')})


def test_later_prices_do_not_change_earlier_fold(config, prices):
    base = validate('AMZN', config=config, prices=prices, models=['var'])
    changed = prices.copy()
    changed.loc[changed.index >= '2027-01-01'] *= 10
    second = validate('AMZN', config=config, prices=changed, models=['var'])
    assert base.predictions[:4] == second.predictions[:4]


def test_plot(config, prices, tmp_path):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = validate('AMZN', config=config, prices=prices, models=['random_walk','var'])
    ax = result.plot(show=False)
    assert len(ax.lines) == 2
    assert all(len(line.get_xdata()) == 12 for line in ax.lines)
    ax.figure.savefig(tmp_path/'errors.png')
    assert (tmp_path/'errors.png').stat().st_size > 1000
    plt.close(ax.figure)


def test_flat_direction_ignores_log_roundoff(config, prices):
    from stock_api.validation import _direction
    last = 252.39999389648438
    assert _direction(np.exp(np.log(last)), last) == 0
    assert _direction(last*1.01, last) == 1
    assert _direction(last*.99, last) == -1
    run = validate('AMZN', config=config, prices=prices, models=['random_walk'])
    assert np.isnan(run.summary().loc['random_walk', 'direction_accuracy'])


def test_tampered_snapshot_rejected(config, prices, tmp_path):
    import json
    result = validate('AMZN', config=config, prices=prices, models=['random_walk'])
    document = result.model_dump()
    document['snapshots']['prices'][0]['AMZN'] += 1
    path = tmp_path/'modified.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='fingerprint'):
        ValidationResult.load(path)


def test_normalization_units_and_scale(config, prices):
    from stock_api.validation import _normalized_errors
    metrics = _normalized_errors(np.array([100., 200.]), np.array([110., 180.]))
    assert metrics['mean_actual_price'] == 150
    assert metrics['nmae_pct'] == 10
    assert metrics['nrmse_pct'] == pytest.approx(100*np.sqrt(250)/150)
    base = validate('AMZN', config=config, prices=prices, models={'last':ModelSpec(LastPrice,'1')})
    scaled = validate('AMZN', config=config, prices=prices*10, models={'last':ModelSpec(LastPrice,'1')})
    for column in ['mean_nmae_pct','pooled_nmae_pct','mean_nrmse_pct','pooled_nrmse_pct']:
        assert base.summary().loc['last',column] == pytest.approx(scaled.summary().loc['last',column])
    assert scaled.summary().loc['last','mean_mae'] == pytest.approx(10*base.summary().loc['last','mean_mae'])
    assert base.summary().loc['last','mean_nmae_pct'] == pytest.approx(np.mean([m['nmae_pct'] for m in base.metrics]))


def test_older_results_get_normalized_metrics(config, prices, tmp_path):
    result = validate('AMZN', config=config, prices=prices, models=['random_walk'])
    expected = result.summary()
    for metric in result.metrics:
        for key in ['mean_actual_price','nmae_pct','nrmse_pct']:
            metric.pop(key)
    path = result.save(tmp_path/'old.json')
    loaded = ValidationResult.load(path)
    pd.testing.assert_frame_equal(expected, loaded.summary())


def test_default_suite_includes_day_zero_baseline(prices):
    config = ValidationConfig(reference_date='2027-09-01', months=1)
    result = validate('AMZN', config=config, prices=prices)
    assert set(result.model_metadata) == {'last_price','gp','multitask_gp','var','gbm_zero_drift','gbm_estimated_drift','ou_returns'}
    assert result.model_metadata['multitask_gp']['version'] == '2'
    fold = result.folds[0]
    day_zero = prices.loc[fold['train_dates'][-1], 'AMZN']
    baseline = [p for p in result.predictions if p['model'] == 'last_price']
    assert all(p['predicted_close'] == day_zero for p in baseline)
