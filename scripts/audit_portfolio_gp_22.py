#!/usr/bin/env python3
"""Independent reconstruction of saved holdings, costs, chronology and GP moments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/portfolio_gp_22_2026-10-18'


def audit(out=OUT):
    # Permit independently auditing another output directory with the same frozen design.
    OUT=Path(out)
    c=json.loads((OUT/'config.yaml').read_text())
    tickers=(OUT/'tickers.txt').read_text().splitlines()
    assert len(tickers)==len(set(tickers))==22
    close=pd.read_csv(OUT/'data/adjusted_close.csv',index_col=0)
    opening=pd.read_csv(OUT/'data/adjusted_open.csv',index_col=0)
    w=pd.read_csv(OUT/'monthly_weights.csv'); m=pd.read_csv(OUT/'monthly_performance.csv')
    d=pd.read_csv(OUT/'portfolio_daily.csv'); f=pd.read_csv(OUT/'monthly_forecasts.csv')
    b=pd.read_csv(OUT/'benchmark_comparison.csv'); s=pd.read_csv(OUT/'schedule.csv')
    assert len(f)==528 and len(m)==24*6
    assert f[['predicted_mean','predicted_std','realized_log_return']].notna().all().all()
    assert (pd.to_datetime(f.training_end)<=pd.to_datetime(f.forecast_origin)).all()
    assert (pd.to_datetime(f.forecast_origin)<pd.to_datetime(f.execution_date)).all()
    assert (pd.to_datetime(f.training_start)>=pd.Timestamp(c['training_start'])).all()
    assert (pd.to_datetime(f[f.period==0].training_end)<=pd.Timestamp(c['training_end'])).all()
    dates=list(close.index); assert len(d.date.unique())==128
    max_budget_error=0.; max_equity_error=0.; min_eigenvalue=float('inf')
    for strategy, months in m.groupby('strategy',sort=False):
        symbols=['SPY'] if strategy=='SPY' else tickers
        qty=np.zeros(len(symbols)); cash=c['initial_dollars']; total_fees=0.
        curve=d[d.strategy==strategy].set_index('date')
        for _,month in months.iterrows():
            period=int(month.period); execution=month.execution_date
            weights=w[(w.strategy==strategy)&(w.period==period)].set_index('ticker').loc[symbols]
            target=weights.weight.to_numpy()
            assert abs(target.sum()-1)<1e-8 and target.min()>=-1e-9
            if strategy!='SPY': assert target.max()<=c['weight_cap']+1e-8
            op=opening.loc[execution,symbols].to_numpy(); old=qty*op
            wealth=old.sum()+cash
            np.testing.assert_allclose(wealth,month.start_equity,atol=1e-8)
            np.testing.assert_allclose(old/wealth,weights.pretrade_weight.to_numpy(),atol=1e-8)
            dollars=(wealth-month.cost)*target
            bought=np.maximum(dollars-old,0).sum(); sold=np.maximum(old-dollars,0).sum()
            np.testing.assert_allclose([bought,sold],[month.bought_dollars,month.sold_dollars],atol=1e-8)
            np.testing.assert_allclose(month.cost,month.cost_bps/10000*(bought+sold),atol=1e-8)
            max_budget_error=max(max_budget_error,abs(dollars.sum()+month.cost-wealth))
            qty=dollars/op; cash=0.; total_fees+=month.cost
            prices=opening.loc[month.holding_end,symbols].to_numpy() if period<5 else close.loc[month.holding_end,symbols].to_numpy()
            np.testing.assert_allclose(qty@prices,month.end_equity,atol=1e-8)
            for date in dates:
                if execution<=date<=s.iloc[period].target_end:
                    actual=qty@close.loc[date,symbols].to_numpy()
                    error=abs(actual-curve.loc[date,'net_equity']); max_equity_error=max(max_equity_error,error)
                    assert error<1e-8
        metrics=b[b.strategy==strategy].iloc[0]
        np.testing.assert_allclose(total_fees,metrics.costs,atol=1e-9)
        np.testing.assert_allclose(np.prod(1+months.net_return)-1,metrics.net_return,atol=1e-10)
        np.testing.assert_allclose(np.prod(1+months.gross_return)-1,metrics.gross_return,atol=1e-10)
        np.testing.assert_allclose(curve.net_equity.iloc[-1],metrics.net_ending_dollars,atol=1e-9)
    max_correction=0.
    for path in sorted((OUT/'predictive').glob('*.npz')):
        z=np.load(path); cov=z['covariance']; eig=np.linalg.eigvalsh(cov)
        min_eigenvalue=min(min_eigenvalue,eig.min()); assert eig.min()>-1e-10
        np.testing.assert_allclose(z['samples'].mean(axis=0),z['mu'],atol=1e-12)
        np.testing.assert_allclose(np.cov(z['samples'],rowvar=False),cov,atol=1e-12)
        np.testing.assert_allclose(np.diag(z['correlation']),1,atol=1e-12)
        max_correction=max(max_correction,np.abs(z['mc_variance_correction']-1).max())
    audit_result=dict(status='PASS',strategies=24,months_per_strategy=6,gp_fits=528,stock_month_forecasts=528,
        predictive_joint_distributions=24,daily_validation_sessions=127,minimum_predictive_covariance_eigenvalue=float(min_eigenvalue),
        maximum_budget_error_dollars=float(max_budget_error),maximum_reconstructed_daily_equity_error_dollars=float(max_equity_error),
        maximum_mc_standard_deviation_correction=float(max_correction),
        checks=['22 unique tickers','training cutoffs','portfolio constraints','drifted pretrade weights','buys and sells',
                'fee equation','self-financing dollar budget','daily equity reconstruction','monthly compounding','covariance PSD','Monte Carlo moments'])
    (OUT/'audit.json').write_text(json.dumps(audit_result,indent=2)+'\n')
    print(json.dumps(audit_result,indent=2))
    return audit_result

if __name__=='__main__': audit()
