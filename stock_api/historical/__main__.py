"""Noninteractive numerical CLI; requires no Codex process or API."""
import os
# Set before numerical modules are imported by main; runner also limits loaded BLAS.
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(name,'1')
import argparse
import json
from pathlib import Path
import pandas as pd
from . import HistoricalConfig,run_historical_validation,generate_report,status,diagnose_historical_fold


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    run=sub.add_parser('run');run.add_argument('--config',required=True);run.add_argument('--output',required=True)
    run.add_argument('--workers',type=int,default=1);run.add_argument('--max-units',type=int)
    run.add_argument('--prices-csv');run.add_argument('--retry-failures',action='store_true');run.add_argument('--report',action='store_true')
    for name in ['status','report']:
        item=sub.add_parser(name);item.add_argument('--output',required=True)
        if name=='status':item.add_argument('--verify',action='store_true')
    diag=sub.add_parser('diagnose');diag.add_argument('--output',required=True);diag.add_argument('--origin',required=True)
    diag.add_argument('--lookback',type=int,required=True);diag.add_argument('--model',action='append',required=True)
    args=parser.parse_args()
    if args.action=='run':
        config=HistoricalConfig.model_validate_json(Path(args.config).read_text())
        prices=pd.read_csv(args.prices_csv,index_col=0,parse_dates=True) if args.prices_csv else None
        root=run_historical_validation(config,output=args.output,prices=prices,workers=args.workers,max_units=args.max_units,retry_failures=args.retry_failures)
        if args.report:print(generate_report(root))
        print(json.dumps(status(root),indent=2))
    elif args.action=='status':print(json.dumps(status(args.output,verify=args.verify),indent=2))
    elif args.action=='report':print(generate_report(args.output))
    else:print(diagnose_historical_fold(args.output,args.origin,args.lookback,args.model))


if __name__=='__main__':main()
