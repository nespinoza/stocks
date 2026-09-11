"""Small chronological evaluation; python -m stock_api.backtest --help."""
import argparse
import json

import numpy as np

from stock_api.schemas import ForecastRequest
from stock_api.data import latest_session, load_prices
from stock_api.models import MODELS
from stock_api.validation import _direction


def evaluate(prices, lookback=120, horizon=7, folds=5):
    if len(prices) < lookback + folds * horizon:
        raise ValueError("Not enough observations for disjoint evaluation windows")
    results = {}
    for name, model in MODELS.items():
        if name == 'random_walk':  # Legacy alias of last_price; score only once.
            continue
        errors, directions, coverage = [], [], []
        for start in range(len(prices)-folds*horizon, len(prices), horizon):
            train = prices[start-lookback:start]
            truth = prices[start:start+horizon, 0]
            mean, variance, _ = model(train, horizon)
            median = np.exp(mean)
            errors.extend(np.abs(median-truth).tolist())
            direction = _direction(median[-1], train[-1, 0])
            if direction:
                directions.append(bool(direction == _direction(truth[-1], train[-1, 0])))
            coverage.extend(((np.log(truth) >= mean-1.96*np.sqrt(variance)) &
                             (np.log(truth) <= mean+1.96*np.sqrt(variance))).tolist())
        accuracy = float(np.mean(directions)) if directions else None
        results[name] = {"mae":float(np.mean(errors)), "endpoint_direction_accuracy":accuracy,
                         "terminal_direction_accuracy":accuracy,
                         "interval_coverage_95":float(np.mean(coverage)), "folds":folds}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--related", nargs="*", default=[])
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=7, help="Trading sessions per fold")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    request = ForecastRequest(ticker=args.ticker, related_tickers=args.related,
                              lookback=args.lookback, horizon=args.horizon)
    if not 1 <= args.folds <= 20:
        parser.error("folds must be between 1 and 20")
    prices = load_prices([request.ticker, *request.related_tickers],
                         args.lookback + args.horizon * args.folds, latest_session())
    print(json.dumps(evaluate(prices.to_numpy(), args.lookback, args.horizon, args.folds), indent=2))


if __name__ == "__main__":
    main()
