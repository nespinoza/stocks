"""Compare new GJR fits with archived GP predictions, without downloading/refitting GP.

Run from the repository root. The primary comparison is declared before fitting:
365-calendar-day GJR versus archived 30-calendar-day heteroskedastic GP, on the
same weekly origins in September 2025–August 2026. Other windows are sensitivities.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_api.historical import HistoricalConfig, run_historical_validation
from stock_api.historical.config import make_origins
from stock_api.historical.runner import sha, load_record, atomic_json, atomic_bytes
from stock_api.historical.analysis import score, block_comparison, holm


GJR = 'regression_gjr_garch'
GP = 'heteroskedastic_gp_returns'
LOOKBACKS = [60, 90, 180, 365, 730]
METRICS = ['mae', 'nmae_pct', 'terminal_return_mae', 'brier', 'coverage_95', 'interval_width_95', 'wis']
LOSSES = ['mae', 'terminal_return_mae', 'brier', 'wis']


def scored(record):
    row = score(record)
    row['variant'] = f"{record['model']}:{record['lookback']}d"
    row['training_observations'] = len(record['train_dates'])
    if record['status'] == 'ok':
        p, y = record['prediction'], record['realized']['terminal_log_return']
        w = .5*abs(y-p['median_log_return'])
        for level, interval in p['intervals'].items():
            alpha = 1-float(level)
            lo, hi = interval['return_lower'], interval['return_upper']
            w += alpha/2*(hi-lo+2/alpha*(max(lo-y,0)+max(y-hi,0)))
        row['wis'] = w/(len(p['intervals'])+.5)
    row['error_message'] = record.get('error', {}).get('message')
    if record.get('prediction'):
        diagnostics = record['prediction']['diagnostics']
        row['variance_persistence'] = diagnostics.get('variance_persistence')
        row['student_t_df'] = diagnostics.get('parameters_percent_units', {}).get('nu')
    return row


def markdown_table(frame):
    # Avoid an extra tabulate dependency for report-only formatting.
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in frame.itertuples(index=False, name=None)]])


def compare(source, output, config):
    origins = make_origins(config)
    source_meta = json.loads((source/'experiment-config.json').read_text())
    prices = pd.read_csv(source/'prices.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    assert sha(source/'prices.csv') == source_meta['snapshot_sha256']
    records, provenance = [], []
    for origin in origins:
        for model, windows, directory in [(GJR, LOOKBACKS, output),
                                          (GP, [30,60,90,180], source),
                                          ('last_price', [90], source)]:
            for lookback in windows:
                path = directory/'records'/f'{origin}__{model}__{lookback}.json'
                meta = json.loads((directory/'experiment-config.json').read_text())
                record = load_record(path, meta['experiment_id'])
                if record.get('realized'):
                    actual = prices.loc[pd.DatetimeIndex(record['forecast_dates']), 'AMZN'].to_numpy()
                    np.testing.assert_array_equal(record['realized']['prices'], actual)
                records.append(record)
                provenance.append({'unit_id':record['unit_id'], 'source':str(path), 'sha256':sha(path),
                                   'experiment_id':record['experiment_id']})
    # Every candidate uses the same target dates and starting close at each origin.
    for origin in origins:
        group = [r for r in records if r['origin'] == origin]
        assert len({tuple(r['forecast_dates']) for r in group}) == 1
        assert len({r.get('starting_price') for r in group}) == 1
    frame = pd.DataFrame([scored(r) for r in records])
    summaries = []
    for variant, g in frame.groupby('variant', sort=False):
        successful = g[g.status == 'ok']
        summaries.append({'variant':variant, 'successful':len(successful), 'failed':len(g)-len(successful),
            'median_runtime_seconds':g.runtime_seconds.median(),
            'min_training_prices':g.training_observations.min(), 'max_training_prices':g.training_observations.max(),
            **{metric:successful[metric].mean() for metric in METRICS}})
    summary = pd.DataFrame(summaries)
    comparisons = []
    for window in LOOKBACKS:
        candidate = frame[(frame.model == GJR)&(frame.lookback == window)&(frame.status == 'ok')].set_index('origin')
        for reference in [f'{GP}:{n}d' for n in [30,60,90,180]]+['last_price:90d']:
            benchmark = frame[(frame.variant == reference)&(frame.status == 'ok')].set_index('origin')
            pairs = candidate.join(benchmark, how='inner', lsuffix='_candidate', rsuffix='_reference')
            for loss in LOSSES:
                delta = pairs[loss+'_candidate']-pairs[loss+'_reference']
                stats = block_comparison(delta,origins,block_weeks=8,replicates=2000,seed=config.seed)
                comparisons.append({'candidate':f'{GJR}:{window}d','reference':reference,'loss':loss,
                    'candidate_mean':pairs[loss+'_candidate'].mean(),
                    'reference_mean':pairs[loss+'_reference'].mean(), **stats})
    paired = pd.DataFrame(comparisons)
    paired['holm_p_value'] = holm(paired.p_value, total_tests=len(LOOKBACKS)*5*len(LOSSES))
    for name, table in [('scores',frame),('summary',summary),('paired-comparisons',paired)]:
        atomic_bytes(output/f'{name}.csv',table.to_csv(index=False).encode())
    atomic_json(output/'comparison-provenance.json',provenance)
    display = summary[['variant','successful','failed','mae','nmae_pct','terminal_return_mae','brier','coverage_95','wis']].copy()
    display['terminal_return_mae'] *= 100
    display['coverage_95'] *= 100
    display['wis'] *= 100
    for col in display.columns[3:]:
        display[col] = display[col].map(lambda v:f'{v:.4f}')
    display = display.rename(columns={'mae':'MAE $','nmae_pct':'NMAE %','terminal_return_mae':'Terminal MAE log pp',
                                      'coverage_95':'95% coverage %','wis':'WIS log pp'})
    primary = paired[(paired.candidate == f'{GJR}:365d')&(paired.reference == f'{GP}:30d')].copy()
    primary_display = primary[['loss','paired_origins','candidate_mean','reference_mean','mean_loss_difference','lower95','upper95','holm_p_value']].copy()
    for col in primary_display.columns[2:]:
        primary_display[col] = primary_display[col].map(lambda v:f'{v:.6f}' if pd.notna(v) else 'unavailable')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for ax, loss in zip(axes,['mae','brier','wis']):
        for variant in [f'{GJR}:365d',f'{GP}:30d','last_price:90d']:
            subset = frame[frame.variant == variant].sort_values('origin')
            ax.plot(pd.to_datetime(subset.origin),subset[loss].cumsum(),label=variant)
        ax.set_title(f'Cumulative {loss} (lower is better)')
        ax.tick_params(axis='x',rotation=30)
    axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    report = f'''# Regression–GJR-GARCH versus the saved Amazon GP forecasts

Period: September 2025–August 2026. {len(origins)} weekly origins, {origins[0]}–{origins[-1]}, with the original seven-calendar-day horizon. This is a retrospective development comparison, not an untouched test set.

## Model and fixed comparison design

The new model jointly estimates an intercept and AR(1) regression on daily Amazon log returns, with GJR-GARCH(1,1) conditional variance and standardized Student-t innovations. No GOOGL/AAPL, market index, or other external predictors enter this comparison. The lower-level API supports additional explicitly aligned regressors and future regressor scenarios. This benchmark tests the autoregressive version; it cannot measure the value of future external data.

The **primary choice was fixed before fitting: GJR with 365 calendar days of history versus the archived 30-calendar-day heteroskedastic GP**, the previously identified reliable GP setting. This is a comparison of practical configurations with different training lengths and inference, not a controlled model-family comparison. GJR 60/90/180-day windows provide same-history comparisons against the saved GP; 730 days is a longer-history sensitivity. GJR requires at least 31 daily returns, so the 30-calendar-day window is not estimated. These are calendar-day windows, not the public API's trading-session lookback.

Both use 2,000 predictive paths. GJR forecasts condition on jointly estimated maximum-likelihood parameters; the archived GP integrates hyperparameter and latent uncertainty using MCMC. Every future GJR path recursively updates its lagged return, residual sign, and variance. No Gaussian approximation or square-root-of-time interval scaling is used. Price means and standard deviations are omitted because exponentiated Student-t log returns do not have finite population moments; forecast medians and quantiles remain usable.

The immutable original price snapshot and GP predictions are reused. GP fits were not rerun, and the existing September 2026 archive was not modified. The new experiment has separate source/environment fingerprints and checkpoints. `{source_meta['snapshot_sha256']}` identifies the original price CSV. [Input-record provenance](comparison-provenance.json) links every score to its checksummed record.

## Results

All aggregate rows below use their own successful fits. Read failure counts before comparing rows; the paired tests below use only matching successful dates. MAE is the equal-origin mean of daily median-price errors. Terminal MAE and WIS use log-return percentage points. Coverage is terminal 95% coverage, not simultaneous path coverage. Smaller MAE, Brier and WIS are better.

{markdown_table(display)}

## Primary paired comparison

Differences are **GJR minus GP**, so negative favors GJR. All values in this table retain raw units: dollars for MAE, dimensionless log returns for terminal MAE/WIS, and probability-squared for Brier. Confidence intervals are marginal eight-week circular moving-block bootstrap intervals, not simultaneous intervals. Holm p-values cover all 100 planned contrasts (five GJR windows × five references × four losses). An interval spanning zero does not support a directional conclusion. Comparisons with fewer than 20 successful pairs have no inferential interval.

{markdown_table(primary_display)}

![Primary configurations and persistence](comparison.png)

## Limits and operational interpretation

Fitting errors are preserved as failures without fallback or outcome-driven retuning. GP convergence failures remain excluded exactly as in the saved experiment. Check [per-origin scores](scores.csv) for failure reasons, training sizes, fitted variance persistence and Student-t degrees of freedom. A numerically successful GARCH fit can still sit near a parameter boundary; optimizer convergence is not evidence that parameters are well identified. The 60/90-day fits are particularly short for estimating both regression and volatility.

The held-out outcomes here were already inspected in earlier repository analyses. The period has roughly one year of weekly episodes, with holiday-related overlap; it cannot establish long-run generalization. Simulation error remains, particularly in tail estimates. The point, probability and interval metrics should be assessed separately, and differences in parameter uncertainty and training history should not be attributed solely to the choice of GARCH versus GP.

Full [summary](summary.csv), [paired comparisons](paired-comparisons.csv), and [frozen comparison configuration](comparison-config.json) are included. Reproduce/resume without downloading:

```bash
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python examples/gjr_amzn_comparison.py
```
'''
    atomic_bytes(output/'REPORT.md',report.encode())
    print(display.to_string(index=False))
    print('\nPRIMARY\n'+primary_display.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('historical_validation/AMZN/2016-2026-v1'))
    parser.add_argument('--output',type=Path,default=Path('historical_validation/AMZN/gjr-2025-2026-v1'))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('The original archive cannot be an output directory')
    source = json.loads((args.source/'experiment-config.json').read_text())
    if sha(args.source/'prices.csv') != source['snapshot_sha256']:
        raise ValueError('Original price snapshot checksum mismatch')
    config = HistoricalConfig(target_ticker='AMZN',start_date='2025-09-01',end_date='2026-09-01',
        models=['last_price',GJR],lookbacks=LOOKBACKS,n_paths=2000,seed=20261018)
    comparison = {'schema_version':1,'source_experiment_id':source['experiment_id'],
        'source_prices_sha256':source['snapshot_sha256'],'source':str(args.source.resolve()),
        'new_config':config.model_dump(mode='json'),'primary_candidate':f'{GJR}:365d',
        'primary_reference':f'{GP}:30d','planned_tests':100,'script_sha256':sha(Path(__file__)),
        'mean_regressors':['intercept','previous_target_log_return'],
        'source_gp_lookbacks':[30,60,90,180],'persistence_reference_lookback':90}
    path = args.output/'comparison-config.json'
    if path.exists():
        if json.loads(path.read_text()) != comparison:
            raise ValueError('Comparison design changed; use a new output directory')
    else:
        atomic_json(path,comparison)
    prices = None if (args.output/'experiment-config.json').exists() else pd.read_csv(
        args.source/'prices.csv',index_col=0,parse_dates=True,float_precision='round_trip')[['AMZN']]
    def progress(value):
        if isinstance(value,dict) and (value['completed']%50 == 0 or value['completed'] == value['total']):
            print(f"Completed {value['completed']}/{value['total']}; failures {value['failures']}",flush=True)
    run_historical_validation(config,output=args.output,prices=prices,workers=1,progress=progress)
    compare(args.source,args.output,config)


if __name__ == '__main__':
    main()
