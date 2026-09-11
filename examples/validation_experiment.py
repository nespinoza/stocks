"""Run a monthly experiment, save its snapshot/metrics, and render a plot.

python examples/validation_experiment.py --reference-date 2026-09-01
python examples/validation_experiment.py --reference-date 2026-09-01 --synthetic
"""
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from stock_api import ValidationConfig, validate
from stock_api.data import calendar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-date', default=date.today().replace(day=1).isoformat())
    parser.add_argument('--ticker', default='AMZN')
    parser.add_argument('--related', nargs='*', default=['GOOGL', 'AAPL'])
    parser.add_argument('--models', nargs='+', default=['last_price', 'gp', 'multitask_gp', 'var', 'gbm_zero_drift', 'gbm_estimated_drift', 'ou_returns'])
    parser.add_argument('--train-days', type=int, default=30)
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--months', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='validation_runs')
    parser.add_argument('--synthetic', action='store_true', help='Demonstration data, not observed stock prices')
    args = parser.parse_args()
    config = ValidationConfig(reference_date=args.reference_date, months=args.months,
                              train_days=args.train_days, horizon=args.horizon, seed=args.seed)
    prices = None
    if args.synthetic:
        end = pd.Timestamp(args.reference_date)
        start = end-pd.DateOffset(months=args.months)-pd.Timedelta(days=args.train_days+30)
        dates = calendar().sessions_in_range(start, end).tz_localize(None)
        tickers = list(dict.fromkeys([args.ticker.upper(), *[t.upper() for t in args.related]]))
        rng = np.random.default_rng(args.seed)
        common = rng.normal(0.0003, 0.012, (len(dates), 1))
        logs = np.log(np.linspace(150, 250, len(tickers))) + np.cumsum(
            common+rng.normal(0, 0.007, (len(dates), len(tickers))), axis=0)
        prices = pd.DataFrame(np.exp(logs), index=dates, columns=tickers)
    result = validate(args.ticker, related_tickers=args.related, config=config,
                      prices=prices, models=args.models)
    label = 'synthetic' if args.synthetic else 'yahoo'
    prefix = Path(args.output)/f'{label}-{args.reference_date}-{result.run_id}'
    result.save(prefix.with_suffix('.json'))
    summary = result.summary()
    summary.to_csv(prefix.with_suffix('.csv'))
    import matplotlib
    matplotlib.use('Agg')
    ax = result.plot(show=False)
    ax.set_title(f'{args.ticker}: monthly walk-forward validation ({label} data)')
    ax.figure.savefig(prefix.with_suffix('.png'), dpi=150)
    print(summary.to_string())
    print(f'Saved {label} results to {prefix}.[json,csv,png]')


if __name__ == '__main__':
    main()
