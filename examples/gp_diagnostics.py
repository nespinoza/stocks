"""Diagnose one saved origin, never rerun validation or download prices."""
import argparse
from pathlib import Path
from stock_api import ValidationResult
from stock_api.diagnostics import DiagnosticConfig,diagnose_fold


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saved-runs',default='validation_runs/return-gp-2026-09-01-benchmarks')
    p.add_argument('--origin',default='2025-11-01')
    p.add_argument('--lookbacks',type=int,nargs='+',default=[30,60,90,180])
    p.add_argument('--models',nargs='+',default=['gp','multitask_gp','volatility_gp_returns','heteroskedastic_gp_returns'])
    p.add_argument('--config',help='DiagnosticConfig JSON file')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--output',default='diagnostics')
    args=p.parse_args()
    config=DiagnosticConfig.model_validate_json(Path(args.config).read_text()) if args.config else DiagnosticConfig()
    for days in args.lookbacks:
        run=ValidationResult.load(Path(args.saved_runs)/f'lookback-{days}.json')
        diagnose_fold(run,args.origin,models=args.models,config=config,output=args.output,resume=args.resume,progress=lambda msg:print(msg,flush=True))


if __name__=='__main__':main()
