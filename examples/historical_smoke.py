"""Two real-data origins, all nine models, smoke-only inference settings; no download."""
from pathlib import Path
import json
import argparse
import pandas as pd
from stock_api import ValidationResult
from stock_api.historical import HistoricalConfig,run_historical_validation,generate_report,status
from stock_api.diagnostics import DiagnosticConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production',action='store_true',help='One origin with the frozen full production sampling budgets')
    args=parser.parse_args()
    source=Path('validation_runs/return-gp-2026-09-01-benchmarks/lookback-180.json')
    saved=ValidationResult.load(source)
    prices=pd.DataFrame(saved.snapshots['prices']).set_index('date');prices.index=pd.to_datetime(prices.index)
    config=HistoricalConfig(target_ticker='AMZN',auxiliary_tickers=['GOOGL','AAPL'],start_date='2025-11-03',end_date='2025-11-18',
        lookbacks=[30],n_paths=200,min_bin_count=2,bootstrap_replicates=100,
        inference=DiagnosticConfig(profile_surfaces=False,multistart=False,corner_plots=False,latent_plots=False,
            nlive=40,nested_maxcall=10000,interweave=True,warmup=100,draws=200,chains=4))
    root=Path('historical_validation/smoke/AMZN-two-origins-v2')
    if args.production:
        config=HistoricalConfig(target_ticker='AMZN',auxiliary_tickers=['GOOGL','AAPL'],start_date='2025-11-03',end_date='2025-11-11',
                                lookbacks=[30],min_bin_count=2,bootstrap_replicates=100)
        root=Path('historical_validation/smoke/AMZN-production-one-origin-v2')
    root.mkdir(parents=True,exist_ok=True)
    config_path=root/'smoke-config.json'
    if not config_path.exists():config_path.write_text(config.model_dump_json(indent=2)+'\n')
    run_historical_validation(config,output=root,prices=None if (root/'experiment-config.json').exists() else prices,workers=1)
    print(generate_report(root));print(json.dumps(status(root,verify=True),indent=2))
    print('SMOKE ONLY: one production-budget origin' if args.production else 'SMOKE ONLY: reduced sampling budgets deliberately exercise failure handling; not scientific evidence.')


if __name__=='__main__':main()
