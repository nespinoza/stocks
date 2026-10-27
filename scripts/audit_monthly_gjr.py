"""Audit the frozen 100-stock shortlist with the repository's GJR model.

The shortlist is never reranked using these results. Checkpoints contain scalar
scores and diagnostics, not a claim of an independently selected live portfolio.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stock_api.garch import regression_gjr_garch
from scripts.screen_monthly_stocks import ROOT, block_test, holm


def main():
    selected = pd.read_csv(ROOT/'top-100.csv')
    primary = pd.read_csv(ROOT/'monthly-predictions.csv')
    primary = primary[primary.ticker.isin(selected.ticker)&(primary.month>='2024-01-01')]
    target = ROOT/'gjr-records'; target.mkdir(exist_ok=True)
    design = {'model':'regression_gjr_garch','lookback_calendar_days':365,'n_paths':2000,
              'evaluation_months':['2024-01','2026-09'],'seed':20261018,
              'shortlist_sha256':hashlib.sha256((ROOT/'top-100.csv').read_bytes()).hexdigest(),
              'model_sha256':hashlib.sha256(Path('stock_api/garch.py').read_bytes()).hexdigest()}
    design_path = ROOT/'gjr-audit-design.json'
    if design_path.exists():
        if json.loads(design_path.read_text())!=design: raise ValueError('GJR audit design changed')
    else: design_path.write_text(json.dumps(design,indent=2))
    records=[]
    for k, ticker in enumerate(selected.ticker,1):
        daily = pd.read_csv(ROOT/'data'/f'{ticker}.csv',index_col=0,parse_dates=True)
        for row in primary[primary.ticker==ticker].itertuples():
            checkpoint = target/f'{ticker}-{row.month}.json'
            if checkpoint.exists(): records.append(json.loads(checkpoint.read_text())); continue
            started = time.perf_counter()
            record={'ticker':ticker,'month':row.month,'status':'failed'}
            try:
                month = pd.Timestamp(row.month).to_period('M')
                origin = month.start_time
                train = daily.loc[(daily.index>=origin-pd.Timedelta(days=365))&(daily.index<origin),'Adj Close']
                test = daily.loc[daily.index.to_period('M')==month,'Adj Close']
                if len(train)<32 or len(test)<15: raise ValueError('Insufficient sessions')
                seed = int(hashlib.sha256(f"20261018:{ticker}:{row.month}".encode()).hexdigest()[:8],16)
                _,_,info = regression_gjr_garch(train.to_numpy()[:,None],len(test),seed=seed,n_paths=2000)
                ensemble = info.pop('_ensemble'); info.pop('_statistics')
                increments = np.diff(np.log(ensemble.paths),axis=1)
                terminal = increments.sum(axis=1)
                actual = float(np.log(test.iloc[-1]/train.iloc[-1]))
                if not np.isclose(actual,row.realized_return,atol=1e-10): raise ValueError('Target differs from monthly screen')
                probability = float((terminal>0).mean())
                logrv_forecast = float(np.log((increments**2).sum(axis=1)).mean())
                record.update(status='ok', probability_up=probability,
                    brier=(probability-float(actual>0))**2, baseline_brier=row.baseline_brier,
                    correct=float((probability>=.5)==(actual>0)), baseline_correct=row.baseline_correct,
                    logrv_error=float((np.log(row.rv)-logrv_forecast)**2),baseline_logrv_error=row.baseline_logrv_error,
                    train_sessions=len(train),forecast_sessions=len(test),diagnostics=info)
            except Exception as exc: record['error']=str(exc)
            record['runtime_seconds']=time.perf_counter()-started
            checkpoint.write_text(json.dumps(record,indent=2))
            records.append(record)
        if k%10==0: print(f'{k}/100 stocks; {len(records)} records; {sum(r["status"]!="ok" for r in records)} failures',flush=True)
    scores = pd.DataFrame(records)
    scores.drop(columns=['diagnostics'],errors='ignore').to_csv(ROOT/'gjr-monthly-scores.csv',index=False)
    rows=[]
    for ticker, f in scores.groupby('ticker'):
        good=f[f.status=='ok']
        p,lo,hi=block_test(good.baseline_brier-good.brier,20261018) if len(good)>=20 else (1.,np.nan,np.nan)
        rows.append({'ticker':ticker,'successful':len(good),'failed':len(f)-len(good),
            'gjr_brier_skill_pct':100*(1-good.brier.mean()/good.baseline_brier.mean()),
            'gjr_accuracy_pct':100*good.correct.mean(),
            'gjr_volatility_skill_pct':100*(1-good.logrv_error.mean()/good.baseline_logrv_error.mean()),
            'p_value':p,'lower95':lo,'upper95':hi})
    summary = pd.DataFrame(rows);summary['holm_p_value']=holm(summary.p_value)
    summary.to_csv(ROOT/'gjr-stock-summary.csv',index=False)
    good=scores[scores.status=='ok']
    aggregate={'completed_utc':datetime.now(timezone.utc).isoformat(),'records':len(scores),
        'successful':len(good),'failures':len(scores)-len(good),
        'brier':good.brier.mean(),'baseline_brier':good.baseline_brier.mean(),
        'brier_skill_pct':100*(1-good.brier.mean()/good.baseline_brier.mean()),
        'accuracy_pct':100*good.correct.mean(),'baseline_accuracy_pct':100*good.baseline_correct.mean(),
        'volatility_skill_pct':100*(1-good.logrv_error.mean()/good.baseline_logrv_error.mean()),
        'positive_stock_brier_skill':int((summary.gjr_brier_skill_pct>0).sum()),
        'holm_confirmed':int(((summary.holm_p_value<.05)&(summary.gjr_brier_skill_pct>0)).sum()),
        'parameter_uncertainty':'Plug-in MLE; omitted',
        'median_runtime_seconds':good.runtime_seconds.median()}
    (ROOT/'gjr-summary.json').write_text(json.dumps(aggregate,indent=2))
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__': main()
