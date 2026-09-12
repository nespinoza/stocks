"""Adapt one immutable historical fold to the existing detailed GP diagnostics."""
import json
from pathlib import Path
import pandas as pd
from stock_api.validation import ValidationResult,_hash,_records
from stock_api.diagnostics import diagnose_fold,DiagnosticConfig
from .config import HistoricalConfig,fold_dates,make_origins
from .runner import sha,unit_seed
from .inference import GP_MODELS


def diagnostic_input(output,origin,lookback):
    root=Path(output);meta=json.loads((root/'experiment-config.json').read_text())
    config=HistoricalConfig.model_validate(meta['config'])
    if origin not in make_origins(config) or lookback not in config.lookbacks:raise ValueError('Fold is not in frozen experiment')
    if sha(root/'prices.csv')!=meta['snapshot_sha256']:raise ValueError('Snapshot checksum mismatch')
    prices=pd.read_csv(root/'prices.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    train,test=fold_dates(config,origin,lookback);prices=prices.loc[train.union(test)]
    if prices.isna().any().any():raise ValueError('Selected diagnostic fold contains missing ticker data')
    settings={'ticker':config.target_ticker,'related_tickers':config.auxiliary_tickers,'train_days':lookback,
              'horizon':config.forecast_horizon_days,'horizon_unit':'calendar','seed':config.seed,
              'source_historical_experiment':meta['experiment_id']}
    folds=[{'origin':origin,'train_dates':[str(x.date()) for x in train],'test_dates':[str(x.date()) for x in test]}]
    snapshots={'prices':_records(prices),'aligned_features':[],'feature_columns':[]}
    identity=_hash({'config':settings,'snapshots':snapshots,'folds':folds})
    return ValidationResult(run_id='historical-diagnostic-'+identity[:12],created_at=meta['created_at'],
        experiment_id=identity,config=settings,model_metadata={},environment=meta['implementation'],snapshots=snapshots,
        folds=folds,predictions=[],metrics=[])


def diagnose_historical_fold(output,origin,lookback,models,*,config=None):
    if not set(models)<=GP_MODELS:raise ValueError('Detailed diagnostics are available for GP models only')
    run=diagnostic_input(output,origin,lookback)
    meta=json.loads((Path(output)/'experiment-config.json').read_text())
    historical=HistoricalConfig.model_validate(meta['config'])
    settings=config or historical.inference.model_copy(update={'profile_surfaces':True,'multistart':True,'corner_plots':True,'latent_plots':True})
    result=None
    for model in models:
        selected=settings.model_copy(update={'seed':unit_seed(historical,origin,model,lookback),
                                             'predictive_draws':historical.n_paths})
        result=diagnose_fold(run,origin,models=[model],config=selected,output=Path(output)/'diagnostics',resume=True)
    return result
