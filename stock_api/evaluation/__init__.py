"""Forecast, signal and backtest evaluation tools (see claude_report.md §D)."""
from stock_api.evaluation.stats import diebold_mariano, hac_mean_test, long_run_variance
from stock_api.evaluation.calibration import (berkowitz, christoffersen, kupiec, pit_from_quantiles,
                                              pit_from_samples, qlike)
from stock_api.evaluation.signals import information_coefficient, ic_summary, neutralize, quantile_spread
from stock_api.evaluation.attribution import factor_regression, load_french_factors, parse_french_csv
from stock_api.evaluation.overfitting import (deflated_sharpe, expected_max_sharpe, pbo_cscv,
                                              probabilistic_sharpe, sharpe)

__all__ = ['diebold_mariano', 'hac_mean_test', 'long_run_variance', 'berkowitz', 'christoffersen',
           'kupiec', 'pit_from_quantiles', 'pit_from_samples', 'qlike', 'information_coefficient',
           'ic_summary', 'neutralize', 'quantile_spread', 'factor_regression', 'load_french_factors',
           'parse_french_csv', 'deflated_sharpe', 'expected_max_sharpe', 'pbo_cscv',
           'probabilistic_sharpe', 'sharpe']
