"""Leakage-aware rolling-origin validation and portable experiment records."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
from typing import Callable, Literal, Protocol
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from stock_api.data import calendar, latest_session, load_prices
from stock_api.models import MODELS
from stock_api.sde import SDE_MODELS, lognormal_statistics


class ValidationConfig(BaseModel):
    reference_date: date
    months: int = Field(default=12, ge=1, le=120)
    train_days: int = Field(default=30, ge=8, le=2000)
    horizon: int = Field(default=7, ge=1, le=365)
    train_unit: Literal['calendar', 'trading'] = 'calendar'
    horizon_unit: Literal['calendar', 'trading'] = 'calendar'
    frequency: Literal['monthly', 'weekly'] = 'monthly'
    min_train_observations: int = Field(default=10, ge=8)
    seed: int = Field(default=0, ge=0)


@dataclass
class TrainingData:
    """Only observations available before the forecast origin; target is prices column 0.

    Features are indexed by availability date, not news event date. Models must
    fit all scalers, feature selection and other learned transforms here.
    """
    prices: pd.DataFrame
    features: pd.DataFrame


class ValidationModel(Protocol):
    def fit_predict(self, train: TrainingData, future: pd.DataFrame, *,
                    rng: np.random.Generator) -> pd.DataFrame:
        """Return predicted_close, optionally lower_95/upper_95, at future.index."""
        ...


@dataclass
class ModelSpec:
    factory: Callable[[], ValidationModel]
    version: str
    parameters: dict = field(default_factory=dict)


@dataclass
class _Builtin:
    name: str
    n_paths: int = 10_000

    def fit_predict(self, train, future, *, rng):
        kwargs = {}
        if self.name in ('gp', 'multitask_gp'):
            kwargs = {'train_times': train.features['time'].to_numpy(),
                      'future_times': future['time'].to_numpy()}
        if self.name in SDE_MODELS:
            kwargs.update(rng=rng, n_paths=self.n_paths)
        mean, variance, diagnostics = MODELS[self.name](train.prices.to_numpy(), len(future), **kwargs)
        std = np.sqrt(variance)
        predicted = (np.full(len(future), train.prices.iloc[-1, 0])
                     if self.name in ('last_price', 'random_walk') else np.exp(mean))
        statistics = lognormal_statistics(mean, variance, float(train.prices.iloc[-1, 0]))
        result = pd.DataFrame({'predicted_close':predicted, **statistics}, index=future.index)
        ensemble = diagnostics.pop('_ensemble', None)
        result.attrs['diagnostics'] = diagnostics
        if ensemble:
            result.attrs['terminal_log_returns'] = ensemble.terminal_log_returns.tolist()
        return result


def builtin_models(*, n_paths=10_000):
    """Fresh registry; callers may add ModelSpec entries for arbitrary regressors."""
    return {name: ModelSpec(lambda name=name: _Builtin(name, n_paths), version='2' if name == 'multitask_gp' else '1',
                           parameters={'implementation_sha256': _hash(inspect.getsource(MODELS[name]) +
                                       (inspect.getsource(sys.modules['stock_api.sde']) if name in SDE_MODELS else '')),
                                       **({'n_paths':n_paths} if name in SDE_MODELS else {})})
            for name in MODELS}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _records(frame):
    return [{'date':str(day.date()), **{column:float(value) for column, value in row.items()}}
            for day, row in frame.iterrows()]


def _frame(value, name, positive=False):
    if not isinstance(value, pd.DataFrame) or not isinstance(value.index, pd.DatetimeIndex):
        raise ValueError(f'{name} must be a DataFrame with a DatetimeIndex')
    frame = value.copy(deep=True)
    if frame.index.tz is not None:
        raise ValueError(f'{name} must use timezone-naive daily availability dates')
    if not frame.index.equals(frame.index.normalize()) or frame.index.has_duplicates:
        raise ValueError(f'{name} requires unique midnight daily dates')
    if frame.columns.has_duplicates or any(not isinstance(c, str) for c in frame.columns):
        raise ValueError(f'{name} requires unique string column names')
    frame = frame.sort_index().astype(float)
    if not np.isfinite(frame.to_numpy()).all() or (positive and (frame.to_numpy() <= 0).any()):
        raise ValueError(f'{name} has nonfinite or invalid values; prepare data without future filling')
    return frame


def make_folds(config: ValidationConfig):
    """Monthly first-day or weekly Monday origins; all test dates precede cutoff."""
    end = pd.Timestamp(config.reference_date)
    start = end - pd.DateOffset(months=config.months)
    origins = pd.date_range(start, end, freq='MS' if config.frequency == 'monthly' else 'W-MON', inclusive='left')
    padding = config.train_days*3+30 if config.train_unit == 'trading' else config.train_days+10
    sessions = calendar().sessions_in_range(start-pd.Timedelta(days=padding), end).tz_localize(None)
    folds = []
    for origin in origins:
        before = sessions[sessions < origin]
        train = (before[-config.train_days:] if config.train_unit == 'trading' else
                 before[before >= origin-pd.Timedelta(days=config.train_days)])
        after = sessions[(sessions >= origin) & (sessions < end)]
        if config.horizon_unit == 'calendar':
            if origin+pd.Timedelta(days=config.horizon) > end:
                continue
            test = after[after < origin+pd.Timedelta(days=config.horizon)]
        else:
            test = after[:config.horizon]
            if len(test) != config.horizon:
                continue
        if len(train) < config.min_train_observations:
            raise ValueError(f'Insufficient training sessions at {origin.date()}; increase train_days')
        if len(test):
            folds.append((origin, train, test))
    if not folds:
        raise ValueError('No complete folds in the requested evaluation window')
    return folds


class ValidationResult(BaseModel):
    schema_version: int = 1
    run_id: str
    created_at: str
    experiment_id: str
    config: dict
    model_metadata: dict
    environment: dict
    snapshots: dict
    folds: list[dict]
    predictions: list[dict]
    metrics: list[dict]
    distributions: list[dict] = Field(default_factory=list)

    def rerun(self, models):
        """Evaluate new models on this exact saved data/configuration, without downloads."""
        self._check_identity()
        def restore(records):
            frame = pd.DataFrame(records).set_index('date')
            frame.index = pd.to_datetime(frame.index)
            return frame
        settings = self.config
        features = restore(self.snapshots['aligned_features'])[self.snapshots['feature_columns']]
        return validate(settings['ticker'], related_tickers=settings['related_tickers'],
                        config=ValidationConfig(**{k:v for k,v in settings.items() if k in ValidationConfig.model_fields}),
                        prices=restore(self.snapshots['prices']), features=features,
                        known_future_features=settings['known_future_features'], models=models)

    def summary(self):
        """One row per model. mean_mae gives each fold equal weight."""
        metrics = pd.DataFrame(self.metrics)
        points = pd.DataFrame(self.predictions)
        rows = []
        for model, group in metrics.groupby('model', sort=False):
            p = points[points.model == model]
            error = p.predicted_close - p.actual_close
            normalized = _normalized_errors(p.actual_close.to_numpy(), p.predicted_close.to_numpy())
            rows.append({'model':model, 'folds':len(group), 'observations':len(p),
                         'mean_mae':float(group.mae.mean()), 'pooled_mae':float(error.abs().mean()),
                         'mean_rmse':float(group.rmse.mean()), 'pooled_rmse':float(np.sqrt(np.mean(error**2))),
                         'mean_nmae_pct':float(group.nmae_pct.mean()),
                         'pooled_nmae_pct':normalized['nmae_pct'],
                         'mean_nrmse_pct':float(group.nrmse_pct.mean()),
                         'pooled_nrmse_pct':normalized['nrmse_pct'],
                         'mean_mape_pct':float(group.mape_pct.mean()),
                         'direction_accuracy':pd.to_numeric(group.direction_correct).mean(),
                         'terminal_direction_accuracy':pd.to_numeric(group.direction_correct).mean(),
                         'directional_folds':int(group.direction_correct.notna().sum()),
                         'terminal_log_return_mae':pd.to_numeric(group.terminal_log_return_absolute_error).mean(),
                         'terminal_log_return_mae_pct':100*pd.to_numeric(group.terminal_log_return_absolute_error).mean(),
                         'terminal_brier_score':pd.to_numeric(group.terminal_brier_score).mean(),
                         'mean_probability_up':pd.to_numeric(group.terminal_probability_up).mean(),
                         'realized_up_frequency':pd.to_numeric(group.terminal_actual_up).mean(),
                         'coverage_95':pd.to_numeric(group.coverage_95).mean()})
        return pd.DataFrame(rows).set_index('model')

    def save(self, path):
        """Save an immutable JSON record; refuse to overwrite an existing file."""
        self._check_identity()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump_json(indent=2)
        with path.open('x') as stream:
            stream.write(payload)
        return path

    @classmethod
    def load(cls, path):
        result = cls.model_validate_json(Path(path).read_text())
        if result.schema_version != 1:
            raise ValueError('Unsupported validation result schema')
        result._check_identity()
        # Older records retain individual predictions, so no model rerun is needed.
        for metric in result.metrics:
            points = [p for p in result.predictions
                      if p['model'] == metric['model'] and p['origin'] == metric['origin']]
            if not {'mean_actual_price', 'nmae_pct', 'nrmse_pct'}.issubset(metric):
                points = [p for p in result.predictions
                          if p['model'] == metric['model'] and p['origin'] == metric['origin']]
                metric.update(_normalized_errors(np.array([p['actual_close'] for p in points]),
                                                  np.array([p['predicted_close'] for p in points])))
            fold = next(f for f in result.folds if f['origin'] == metric['origin'])
            last = next(p[result.config['ticker']] for p in result.snapshots['prices']
                        if p['date'] == fold['train_dates'][-1])
            metric.update(_terminal_metrics(points[-1]['predicted_close'], points[-1]['actual_close'],
                                             last, points[-1].get('probability_up')))
        return result

    def _check_identity(self):
        expected = _hash({'config':self.config, 'snapshots':self.snapshots, 'folds':self.folds})
        if self.experiment_id != expected:
            raise ValueError('Experiment fingerprint does not match its data, configuration or folds')

    def plot(self, *, metric='nmae_pct', ax=None, show=True):
        """Plot errors by forecast origin; returns a Matplotlib Axes."""
        if metric not in ('mae', 'rmse', 'nmae_pct', 'nrmse_pct', 'mape_pct', 'direction_correct', 'coverage_95'):
            raise ValueError('Unsupported fold metric')
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))
        frame = pd.DataFrame(self.metrics)
        for name, group in frame.groupby('model', sort=False):
            group = group.sort_values('origin')
            ax.plot(pd.to_datetime(group.origin), pd.to_numeric(group[metric]), 'o-', label=name)
        labels = {'nmae_pct':'MAE / mean observed price (%)',
                  'nrmse_pct':'RMSE / mean observed price (%)',
                  'mae':'MAE (price units)', 'rmse':'RMSE (price units)',
                  'mape_pct':'MAPE (%)', 'direction_correct':'Correct direction (0 or 1)',
                  'coverage_95':'95% interval coverage (fraction)'}
        ax.set(xlabel='Forecast origin (training window ends before this date)', ylabel=labels[metric],
               title='Walk-forward validation')
        ax.grid(alpha=0.25)
        ax.legend()
        ax.figure.autofmt_xdate()
        ax.figure.tight_layout()
        if show:
            plt.show()
        return ax


def validate(ticker: str, *, config: ValidationConfig, related_tickers=(), prices=None,
             features=None, known_future_features=(), models=None) -> ValidationResult:
    """Run fresh models on each fold; default models are all built-ins.

    prices: daily positive adjusted closes with columns [target, related...].
    features: numeric daily regressors indexed by when they became available;
      reindexing uses past values only. 'time' is reserved (days since epoch).
    known_future_features: explicit names safe to disclose at forecast time
      (e.g. a scheduled event). Never mark realized future sentiment as known.
    models: names of built-ins, or {name: ModelSpec(factory, version, parameters)}.
    Each factory MUST return a fresh instance and use the supplied RNG for randomness.
    A model gets copies of training data and only known future features, never test prices.
    """
    if not isinstance(config, ValidationConfig):
        raise TypeError('config must be a ValidationConfig')
    tickers = list(dict.fromkeys([ticker.strip().upper(), *[t.strip().upper() for t in related_tickers]]))
    if not tickers[0]:
        raise ValueError('ticker cannot be empty')
    folds = make_folds(config)
    required = folds[0][1]
    for _, train, test in folds:
        required = required.union(train).union(test)
    if prices is None:
        if required[-1] > latest_session():
            raise ValueError('Requested test observations are not yet available; choose an earlier reference_date or supply historical/synthetic prices')
        sessions = calendar().sessions_in_range(required[0], required[-1])
        prices = load_prices(tickers, len(sessions), required[-1])
    prices = _frame(prices, 'prices', positive=True)
    missing = set(tickers)-set(prices.columns)
    if missing or not required.isin(prices.index).all():
        raise ValueError(f'Prices must cover all requested sessions and tickers; missing columns: {sorted(missing)}')
    prices = prices.loc[required, tickers]
    known = list(known_future_features)
    if len(set(known)) != len(known):
        raise ValueError('known_future_features contains duplicates')
    if features is None:
        features = pd.DataFrame(index=required)
    else:
        features = _frame(features, 'features')
    if 'time' in features.columns or not set(known).issubset(features.columns):
        raise ValueError("'time' is reserved; known_future_features must name provided columns")
    # Retain only feature values that existed by each date, with no backward filling.
    source_features = features.copy()
    features = features.reindex(features.index.union(required)).sort_index().ffill().loc[required]
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Features lack past values for the start of the training window')
    features['time'] = (required-pd.Timestamp('1970-01-01')).days.astype(float)
    registry = builtin_models()
    if models is None:
        selected = {name:spec for name,spec in registry.items() if name != 'random_walk'}
    elif isinstance(models, dict):
        selected = models
    else:
        names = [models] if isinstance(models, str) else list(models)
        if len(set(names)) != len(names) or any(name not in registry for name in names):
            raise ValueError('Choose unique registered model names, or pass a ModelSpec mapping')
        selected = {name:registry[name] for name in names}
    if not selected or any(not isinstance(spec, ModelSpec) or not spec.version for spec in selected.values()):
        raise ValueError('Provide at least one ModelSpec with an explicit version')
    if any(not isinstance(name, str) or not name for name in selected):
        raise ValueError('Model names must be nonempty strings')
    fold_records = [{'origin':str(origin.date()), 'train_dates':[str(d.date()) for d in train],
                     'test_dates':[str(d.date()) for d in test]} for origin, train, test in folds]
    settings = {**config.model_dump(mode='json'), 'ticker':tickers[0], 'related_tickers':tickers[1:],
                'known_future_features':known, 'calendar':'XNYS'}
    snapshots = {'prices':_records(prices), 'aligned_features':_records(features),
                 'feature_columns':list(source_features.columns)}
    identity = _hash({'config':settings, 'snapshots':snapshots, 'folds':fold_records})
    predictions, metrics, distributions = [], [], []
    metadata = {}
    for name, spec in selected.items():
        try:
            source = inspect.getsource(spec.factory)
        except (OSError, TypeError):
            source = None
        metadata[name] = {'version':spec.version, 'parameters':spec.parameters,
                          'factory_source_sha256':_hash(source) if source else None}
        for origin, train_dates, test_dates in folds:
            seed = int(_hash([config.seed, str(origin.date())])[:16], 16)
            train = TrainingData(prices.loc[train_dates].copy(deep=True), features.loc[train_dates].copy(deep=True))
            future = features.loc[test_dates, ['time', *known]].copy(deep=True)
            try:
                output = spec.factory().fit_predict(train, future, rng=np.random.default_rng(seed))
                output = _validate_output(output, test_dates)
            except Exception as exc:
                raise ValueError(f'Model {name!r} failed at fold {origin.date()}: {exc}') from exc
            actual = prices.loc[test_dates, tickers[0]].to_numpy()
            predicted = output.predicted_close.to_numpy()
            error = predicted-actual
            last = float(prices.loc[train_dates[-1], tickers[0]])
            has_intervals = 'lower_95' in output
            covered = ((actual >= output.lower_95) & (actual <= output.upper_95)) if has_intervals else None
            metrics.append({'model':name, 'origin':str(origin.date()), 'seed':seed,
                            **_terminal_metrics(predicted[-1], actual[-1], last,
                                                float(output.probability_up.iloc[-1]) if 'probability_up' in output else None),
                            'diagnostics':output.attrs.get('diagnostics', {}),
                            **_normalized_errors(actual, predicted),
                            'mae':float(np.mean(np.abs(error))), 'rmse':float(np.sqrt(np.mean(error**2))),
                            'mape_pct':float(np.mean(np.abs(error)/actual)*100),
                            'coverage_95':float(covered.mean()) if has_intervals else None})
            if 'terminal_log_returns' in output.attrs:
                distributions.append({'model':name, 'origin':str(origin.date()),
                                      'terminal_log_returns':output.attrs['terminal_log_returns']})
            for i, day in enumerate(test_dates):
                predictions.append({'model':name, 'origin':str(origin.date()), 'date':str(day.date()),
                                    'actual_close':float(actual[i]), 'predicted_close':float(predicted[i]),
                                    'probability_up':float(output.probability_up.iloc[i]) if 'probability_up' in output else None,
                                    **{column:float(output[column].iloc[i]) for column in
                                       ('mean_price','median_price','price_std') if column in output},
                                    'lower_95':float(output.lower_95.iloc[i]) if has_intervals else None,
                                    'upper_95':float(output.upper_95.iloc[i]) if has_intervals else None})
    environment = {pkg:importlib.metadata.version(pkg) for pkg in ('numpy','scipy','pandas','pydantic','exchange-calendars')}
    environment['python'] = sys.version
    return ValidationResult(run_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
                            experiment_id=identity, config=settings, model_metadata=metadata,
                            environment=environment, snapshots=snapshots, folds=fold_records,
                            predictions=predictions, metrics=metrics, distributions=distributions)


def _normalized_errors(actual, predicted):
    mean_price = float(np.mean(actual))
    error = predicted-actual
    return {'mean_actual_price':mean_price,
            'nmae_pct':float(100*np.mean(np.abs(error))/mean_price),
            'nrmse_pct':float(100*np.sqrt(np.mean(error**2))/mean_price)}


def _direction(value, last):
    # Roundoff in exp(log(last)) must not turn a flat baseline into a directional bet.
    return 0 if abs(value/last-1) <= 1e-10 else int(np.sign(value-last))


def _terminal_metrics(predicted, actual, last, probability):
    predicted_return, actual_return = float(np.log(predicted/last)), float(np.log(actual/last))
    direction = _direction(predicted, last)
    correct = bool(direction == _direction(actual, last)) if direction else None
    event = bool(actual > last)
    return {'direction_correct':correct, 'terminal_direction_correct':correct,
            'terminal_predicted_log_return':predicted_return, 'terminal_actual_log_return':actual_return,
            'terminal_log_return_absolute_error':abs(predicted_return-actual_return),
            'terminal_probability_up':probability, 'terminal_actual_up':event,
            'terminal_brier_score':float((probability-int(event))**2) if probability is not None else None}


def _validate_output(output, dates):
    if not isinstance(output, pd.DataFrame) or not output.index.equals(dates):
        raise ValueError('Predictions must be a DataFrame indexed exactly by future.index')
    if output.columns.has_duplicates or 'predicted_close' not in output:
        raise ValueError('Predictions require unique columns including predicted_close')
    if ('lower_95' in output) != ('upper_95' in output):
        raise ValueError('Provide both lower_95 and upper_95 or neither')
    columns = ['predicted_close'] + (['lower_95','upper_95'] if 'lower_95' in output else [])
    extras = [c for c in ('probability_up','mean_price','median_price','price_std') if c in output]
    attrs = output.attrs.copy()
    extra_values = output[extras].astype(float)
    if not np.isfinite(extra_values.to_numpy()).all():
        raise ValueError('Predictive statistics must be finite')
    if 'probability_up' in extras and not extra_values.probability_up.between(0, 1).all():
        raise ValueError('probability_up must be between 0 and 1')
    if 'price_std' in extras and (extra_values.price_std < 0).any():
        raise ValueError('price_std must be nonnegative')
    output = output[columns].astype(float)
    if not np.isfinite(output.to_numpy()).all() or (output.to_numpy() <= 0).any():
        raise ValueError('Predictions and bounds must be finite positive prices')
    if 'lower_95' in output and ((output.lower_95 > output.predicted_close) | (output.upper_95 < output.predicted_close)).any():
        raise ValueError('Prediction bounds must contain predicted_close')
    output = pd.concat([output, extra_values], axis=1)
    output.attrs = attrs
    return output


def compare_results(*results: ValidationResult) -> ValidationResult:
    """Merge compatible experiments, keeping originals immutable; reject name collisions."""
    if not results:
        raise ValueError('Provide at least one saved or in-memory result')
    for result in results:
        result._check_identity()
    combined = results[0].model_copy(deep=True)
    for result in results[1:]:
        if result.experiment_id != combined.experiment_id:
            raise ValueError('Experiments differ in data, folds, configuration or seed; rerun on the saved snapshot')
        if set(result.model_metadata) & set(combined.model_metadata):
            raise ValueError('Model names overlap; use distinct versioned names when comparing implementations')
        combined.model_metadata.update(result.model_metadata)
        combined.predictions.extend(result.predictions)
        combined.metrics.extend(result.metrics)
        combined.distributions.extend(result.distributions)
    combined.run_id = str(uuid.uuid4())
    combined.created_at = datetime.now(timezone.utc).isoformat()
    combined.environment = {'source_runs':[{ 'run_id':r.run_id, 'environment':r.environment} for r in results]}
    return combined
