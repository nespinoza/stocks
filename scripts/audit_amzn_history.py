"""Read-only audit of saved AMZN forecasts; write review artifacts separately.

Run from the repo root with: .conda-env/bin/python scripts/audit_amzn_history.py
No downloads, model fitting, or changes to the original experiment are performed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stock_api.historical.config import HistoricalConfig, fold_dates, sessions_for
from stock_api.historical.runner import load_record, sha, unit_id, unit_seed, units
from stock_api.historical.analysis import score, add_pairs, block_comparison, holm


def plot_review(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = pd.read_csv(out / 'comparison.csv')
    table['skill'] = 100 * (1 - table.mae / table.baseline_mae)
    table['coverage'] = 100 * table.coverage_95
    totals = pd.read_csv(out / 'persistence-on-failure.csv')
    table = table.merge(totals[['model', 'lookback', 'failures_replaced']], on=['model', 'lookback'])
    table['failure_pct'] = 100 * table.failures_replaced / (table.successful + table.failures_replaced)
    names = ['last_price', 'gbm_zero_drift', 'gbm_estimated_drift', 'ou_returns',
             'gp', 'multitask_gp', 'var', 'volatility_gp_returns', 'heteroskedastic_gp_returns']
    labels = ['Last price', 'GBM zero drift', 'GBM estimated drift', 'OU returns',
              'Price GP', 'Shared-timescale GP', 'VAR', 'Volatility return GP', 'Mean + volatility return GP']
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
    for ax, (metric, title, cmap, limits) in zip(axes, [
        ('skill', 'Price MAE improvement vs persistence (%)', 'RdYlGn', (-35, 35)),
        ('coverage', 'Terminal 95% interval coverage (%)', 'Blues', (60, 100)),
        ('failure_pct', 'Fits failing convergence checks (%)', 'OrRd', (0, 65)),
    ]):
        values = table.pivot(index='model', columns='lookback', values=metric).loc[names, [30, 60, 90, 180]].to_numpy()
        ax.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1], aspect='auto')
        for (i, j), value in np.ndenumerate(values):
            text = f'{value:+.1f}' if metric == 'skill' else f'{value:.1f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=10,
                    color='white' if (metric == 'failure_pct' and value > 40) or (metric == 'coverage' and value > 90) else 'black')
        ax.set_xticks(range(4), [30, 60, 90, 180])
        ax.set_yticks(range(len(names)), labels)
        ax.set_xlabel('Lookback (calendar days)')
        ax.set_title(title, fontsize=10, pad=12)
    fig.suptitle('AMZN weekly forecasts, January 2016–August 2026', fontsize=15)
    fig.text(.02, .015, 'Error improvements use matching successful origins; coverage excludes failed fits. Failures are shown separately.\n'
             'No model improves the planned losses significantly after the existing 96-test Holm correction.', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path('historical_validation/AMZN/2016-2026-v1'))
    parser.add_argument('--output', type=Path, default=Path('reports/amzn-review'))
    args = parser.parse_args()
    root, out = args.experiment, args.output
    meta = json.loads((root / 'experiment-config.json').read_text())
    config = HistoricalConfig.model_validate(meta['config'])
    assert sha(root / 'prices.csv') == meta['snapshot_sha256']
    prices = pd.read_csv(root / 'prices.csv', index_col=0, parse_dates=True, float_precision='round_trip')
    assert prices.index.is_unique and prices.index.is_monotonic_increasing
    assert prices.index.equals(sessions_for(config))
    assert np.isfinite(prices.to_numpy()).all() and (prices.to_numpy() > 0).all()
    expected = units(config)
    assert {p.stem for p in (root / 'records').glob('*.json')} == {unit_id(*u) for u in expected}
    rows = []
    train_counts = {lookback: [] for lookback in config.lookbacks}
    for i, unit in enumerate(expected):
        origin, model, lookback = unit
        key = unit_id(*unit)
        r = load_record(root / 'records' / (key + '.json'), meta['experiment_id'])
        assert r['unit_id'] == key and (r['origin'], r['model'], r['lookback']) == unit
        assert r['configuration_hash'] == meta['configuration_hash']
        assert r['seed'] == unit_seed(config, *unit)
        train, future = fold_dates(config, origin, lookback)
        assert r['train_dates'] == [str(d.date()) for d in train]
        assert r['forecast_dates'] == [str(d.date()) for d in future]
        assert train.max() < pd.Timestamp(origin) <= future.min()
        columns = [config.target_ticker, *config.auxiliary_tickers] if model in ('multitask_gp', 'var') else [config.target_ticker]
        training = prices.loc[train, columns]
        assert hashlib.sha256(training.to_csv().encode()).hexdigest() == r['training_sha256']
        assert r['starting_price'] == float(training.iloc[-1, 0])
        actual = prices.loc[future, config.target_ticker].to_numpy()
        np.testing.assert_array_equal(actual, r['realized']['prices'])
        np.testing.assert_allclose(np.log(actual[-1] / r['starting_price']), r['realized']['terminal_log_return'], atol=1e-14)
        if model == 'last_price':
            train_counts[lookback].append(len(train))
        row = score(r)
        if r['status'] == 'ok':
            p = r['prediction']
            y = r['realized']['terminal_log_return']
            # Proper interval scores on terminal log returns; lower is better.
            # WIS uses the existing four central intervals, without refitting.
            weighted = .5 * abs(y - p['median_log_return'])
            for level, interval in p['intervals'].items():
                alpha = 1 - float(level)
                lo, hi = interval['return_lower'], interval['return_upper']
                interval_score = hi - lo + 2 / alpha * (max(lo - y, 0) + max(y - hi, 0))
                weighted += alpha / 2 * interval_score
                if level == '0.95':
                    row['interval_score_95'] = interval_score
            row['wis'] = weighted / (len(p['intervals']) + .5)
        rows.append(row)
        if (i + 1) % 4000 == 0:
            print(f'Verified {i + 1}/{len(expected)} records', flush=True)
    frame = add_pairs(pd.DataFrame(rows))
    saved = pd.read_csv(root / 'summaries/forecast-scores.csv').set_index('unit_id').sort_index()
    rebuilt = frame.set_index('unit_id').sort_index()
    assert saved.index.equals(rebuilt.index)
    assert saved.status.equals(rebuilt.status)
    numeric = [c for c in rebuilt if c in saved and pd.api.types.is_numeric_dtype(rebuilt[c])]
    np.testing.assert_allclose(saved[numeric].to_numpy(dtype=float), rebuilt[numeric].to_numpy(dtype=float), rtol=1e-10, atol=1e-12, equal_nan=True)
    good = frame[frame.status == 'ok'].copy()
    metrics = ['mae', 'nmae_pct', 'terminal_return_mae', 'brier', 'coverage_50', 'coverage_68', 'coverage_90', 'coverage_95', 'interval_width_95', 'wis', 'interval_score_95']
    baseline = good[good.model == 'last_price'].set_index(['origin', 'lookback'])
    paired_rows, common_rows, deployment_rows, yearly_rows = [], [], [], []
    for (model, lookback), g in good.groupby(['model', 'lookback']):
        base = baseline.loc[list(zip(g.origin, g.lookback))]
        paired_rows.append({'model': model, 'lookback': lookback, 'successful': len(g),
                            **{m: g[m].mean() for m in metrics},
                            **{'baseline_' + m: base[m].mean() for m in metrics}})
        # Hypothetical operational policy: use persistence on failed fits.
        # It is a post-hoc sensitivity analysis, not an implemented policy.
        all_base = baseline.xs(lookback, level='lookback')
        used = g.set_index('origin').reindex(all_base.index)
        deployment_rows.append({'model': model, 'lookback': lookback,
            'failures_replaced': len(all_base) - len(g),
            **{m: used[m].fillna(all_base[m]).mean() for m in metrics}})
        for year, y in g.groupby('year'):
            ybase = baseline.loc[list(zip(y.origin, y.lookback))]
            yearly_rows.append({'model': model, 'lookback': lookback, 'year': year, 'successful': len(y),
                **{m: y[m].mean() for m in metrics},
                **{'delta_' + m: y[m].mean() - ybase[m].mean() for m in metrics}})
    for lookback, g in good.groupby('lookback'):
        counts = g.groupby('origin').model.nunique()
        common = g[g.origin.isin(counts[counts == len(config.models)].index)]
        for model, subset in common.groupby('model'):
            common_rows.append({'model': model, 'lookback': lookback, 'common_origins': len(subset),
                                **{m: subset[m].mean() for m in metrics}})
    # Recompute the repository's planned comparisons and family correction.
    planned = []
    for (model, lookback), g in good.groupby(['model', 'lookback']):
        if model == 'last_price':
            continue
        for loss in ('mae', 'terminal_return_mae', 'brier'):
            planned.append({'model': model, 'lookback': lookback, 'loss': loss,
                **block_comparison(g.set_index('origin')['delta_' + loss], meta['origins'],
                    config.bootstrap_block_weeks, config.bootstrap_replicates, config.seed)})
    planned = pd.DataFrame(planned)
    planned['holm_p_value'] = holm(planned.p_value, 96)
    stored = pd.read_csv(root / 'summaries/paired-comparisons.csv')
    keys = ['model', 'lookback', 'loss']
    numeric = planned.select_dtypes(include='number').columns.difference(['lookback'])
    np.testing.assert_allclose(planned.set_index(keys).sort_index()[numeric], stored.set_index(keys).sort_index()[numeric], atol=1e-12, rtol=1e-10, equal_nan=True)
    source_changes = [name for name, value in meta['implementation']['source_hashes'].items()
                      if sha(Path('stock_api') / name) != value]
    report_meta = json.loads((root / 'report-manifest.json').read_text())
    report_changes = [name for name, value in report_meta['report_source_sha256'].items()
                      if sha(Path('stock_api/historical') / name) != value]
    audit = {'experiment': str(root), 'experiment_id': meta['experiment_id'],
        'snapshot_sha256': meta['snapshot_sha256'], 'snapshot_rows': len(prices),
        'snapshot_start': str(prices.index.min().date()), 'snapshot_end': str(prices.index.max().date()),
        'verified_records': len(frame), 'status_counts': frame.status.value_counts().to_dict(),
        'record_checksums_dates_seeds_training_and_realizations_verified': True,
        'saved_scores_and_96_planned_comparisons_reproduced': True,
        'changed_inference_sources': source_changes, 'changed_report_sources': report_changes,
        'training_sessions': {str(k): {'min': min(v), 'max': max(v), 'median': float(np.median(v))} for k, v in train_counts.items()},
        'significant_improvements_holm_05': int(((planned.holm_p_value < .05) & (planned.mean_loss_difference < 0)).sum()),
        'significant_deteriorations_holm_05': int(((planned.holm_p_value < .05) & (planned.mean_loss_difference > 0)).sum()),
        'summed_unit_runtime_hours': float(frame.runtime_seconds.sum() / 3600),
        'scope': 'Offline consistency audit; no independent re-download or model refit. WIS/common panels/failure replacement are post-hoc descriptive analyses.'}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    for name, data in [('comparison', paired_rows), ('common-origins', common_rows),
                       ('persistence-on-failure', deployment_rows), ('yearly', yearly_rows)]:
        pd.DataFrame(data).to_csv(out / (name + '.csv'), index=False)
    plot_review(out)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
