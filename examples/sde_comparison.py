"""Run the seven-model SDE benchmark on one common historical price snapshot.

python examples/sde_comparison.py --reference-date 2026-09-01
Reuse the downloaded CSV with --prices PATH to run without network access.
"""
import argparse
from pathlib import Path
import uuid

import pandas as pd

from stock_api import ValidationConfig, builtin_models, validate
from stock_api.data import calendar, latest_session, load_prices
from stock_api.validation import make_folds


MODEL_NAMES = ['last_price','gp','multitask_gp','var','gbm_zero_drift','gbm_estimated_drift','ou_returns']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-date', default='2026-09-01')
    parser.add_argument('--lookbacks', type=int, nargs='+', default=[30,60,90,180])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-paths', type=int, default=10_000)
    parser.add_argument('--prices', help='Existing daily adjusted-close CSV with date index')
    parser.add_argument('--output', help='New output directory; existing directories are refused')
    args = parser.parse_args()
    configs = [ValidationConfig(reference_date=args.reference_date, train_days=days, seed=args.seed)
               for days in args.lookbacks]
    folds = [fold for cfg in configs for fold in make_folds(cfg)]
    start = min(train.min() for _,train,_ in folds)
    end = max(test.max() for _,_,test in folds)
    if args.prices:
        prices = pd.read_csv(args.prices,index_col=0,parse_dates=True,float_precision='round_trip')
    else:
        if end > latest_session():
            parser.error('Reference date requires future prices')
        sessions = calendar().sessions_in_range(start,end)
        prices = load_prices(['AMZN','GOOGL','AAPL'],len(sessions),end)
    root = Path(args.output or f'validation_runs/sde-{args.reference_date}-{uuid.uuid4()}')
    root.mkdir(parents=True,exist_ok=False)
    prices.to_csv(root/'prices.csv',index_label='date')
    registry = builtin_models(n_paths=args.n_paths)
    models = {name:registry[name] for name in MODEL_NAMES}
    summaries = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for config in configs:
        run = validate('AMZN',related_tickers=['GOOGL','AAPL'],config=config,prices=prices,models=models)
        run.save(root/f'lookback-{config.train_days}.json')
        summary = run.summary()
        summary.to_csv(root/f'lookback-{config.train_days}.csv')
        summary = summary.assign(lookback_days=config.train_days)
        summaries.append(summary)
        pd.DataFrame(run.metrics).drop(columns=['diagnostics']).to_csv(root/f'folds-{config.train_days}.csv',index=False)
        ax = run.plot(show=False)
        ax.set_title(f'AMZN: {config.train_days}-calendar-day lookback, 7-calendar-day forecast')
        ax.figure.savefig(root/f'lookback-{config.train_days}.png',dpi=150)
        plt.close(ax.figure)
        print(f'LOOKBACK {config.train_days}',flush=True)
        print(summary[['mean_mae','mean_nmae_pct','terminal_log_return_mae_pct',
                       'terminal_direction_accuracy','coverage_95','terminal_brier_score']].to_string(),flush=True)
    pd.concat(summaries).to_csv(root/'comparison.csv')
    print(f'Saved comparison to {root}',flush=True)


if __name__ == '__main__':
    main()
