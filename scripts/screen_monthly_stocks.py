"""Research shortlist: chronological monthly forecasts, frozen selection, later audit.

Ranks current eligible stocks on 2017-2023 only. Reports 2024-2026 separately;
current constituent/price filters imply survivorship and selection bias.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path('reports/monthly-stock-screen-2026-10-18')
SETTINGS = {
    'version':1, 'as_of':'2026-10-16', 'completed_month':'2026-09-30',
    'universe':'IWB issuer holdings, 2026-10-15; USD equities',
    'price_floor_exclusive':10, 'median_daily_dollar_volume_floor':10_000_000,
    'liquidity_sessions':63, 'training_months':60, 'ridge_penalty':60.,
    'features':['last_month_log_return','last_3_month_log_return',
                'prior_2_to_12_month_log_return','last_month_log_realized_variance'],
    'selection_start':'2017-01-31', 'selection_end':'2023-12-31',
    'audit_start':'2024-01-31', 'audit_end':'2026-09-30',
    'rank':'minimum Brier skill vs rolling up-frequency across 2017-2020 and 2021-2023; full selection skill breaks ties',
    'number_selected':100, 'bootstrap_blocks_months':6, 'bootstrap_replicates':4000,
    'seed':20261018, 'share_classes':'one per normalized issuer name, highest recent dollar volume',
}


def freeze_design():
    path = ROOT/'screen-design.json'
    if path.exists():
        if json.loads(path.read_text()) != SETTINGS:
            raise ValueError('Screen design differs from its frozen settings')
    else:
        path.write_text(json.dumps(SETTINGS, indent=2)+'\n')


def monthly_data(daily):
    prices = daily['Adj Close']
    returns = np.log(prices).diff()
    monthly = pd.DataFrame({
        'return':np.log(prices.resample('ME').last()).diff(),
        'rv':returns.pow(2).resample('ME').sum(min_count=15),
        'sessions':prices.resample('ME').count(),
    }).loc[:SETTINGS['completed_month']]
    monthly['lag1'] = monthly['return'].shift(1)
    monthly['lag3'] = monthly['return'].rolling(3).sum().shift(1)
    monthly['momentum'] = monthly['return'].shift(2).rolling(11).sum()
    monthly['logrv'] = np.log(monthly.rv).shift(1)
    return monthly


def fit_predict(x_train, y_train, x_future):
    """Training-only scaling and an unpenalized intercept; no hyperparameter search."""
    center, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale>1e-12, scale, 1.)
    x = (x_train-center)/scale
    coefficient = np.linalg.solve(x.T@x+SETTINGS['ridge_penalty']*np.eye(x.shape[1]),
                                  x.T@(y_train-y_train.mean()))
    return float(y_train.mean()+((x_future-center)/scale)@coefficient)


def forecasts(monthly):
    records = []
    x = monthly[['lag1','lag3','momentum','logrv']].to_numpy()
    r = monthly['return'].to_numpy()
    rv = monthly.rv.to_numpy()
    for i, date in enumerate(monthly.index):
        if date < pd.Timestamp(SETTINGS['selection_start']): continue
        train = slice(i-SETTINGS['training_months'], i)
        if i < SETTINGS['training_months'] or not np.isfinite(x[train]).all():
            raise ValueError(f'Insufficient complete training features before {date.date()}')
        if not np.isfinite(x[i]).all() or not np.isfinite(r[train]).all() or not np.isfinite(rv[train]).all():
            raise ValueError('Nonfinite model inputs')
        y = (r[train]>0).astype(float)
        baseline = (y.sum()+1)/(len(y)+2)  # Laplace smoothing, trained on past only.
        probability = np.clip(fit_predict(x[train],y,x[i]) + baseline-y.mean(), .01,.99)
        predicted = fit_predict(x[train],r[train],x[i])
        mean_baseline = r[train].mean()
        # Separate volatility forecast: monthly AR(1) for log realized variance.
        logv = np.log(rv[train]); lagv = x[train,3:4]
        forecast_logv = fit_predict(lagv,logv,x[i,3:4])
        actual = float(r[i]); up = float(actual>0)
        records.append({'month':str(date.date()), 'realized_return':actual,
            'probability_up':float(probability),'baseline_probability_up':float(baseline),
            'brier':float((probability-up)**2),'baseline_brier':float((baseline-up)**2),
            'correct':float((probability>=.5)==bool(up)),
            'baseline_correct':float((baseline>=.5)==bool(up)),
            'predicted_return':predicted,'mean_baseline':float(mean_baseline),
            'squared_error':float((predicted-actual)**2),
            'baseline_squared_error':float((mean_baseline-actual)**2),
            'rv':float(rv[i]), 'logrv_error':float((np.log(rv[i])-forecast_logv)**2),
            'baseline_logrv_error':float((np.log(rv[i])-logv.mean())**2)})
    return pd.DataFrame(records)


def summarize(f):
    return {'months':len(f),'brier':f.brier.mean(),'baseline_brier':f.baseline_brier.mean(),
        'brier_skill_pct':100*(1-f.brier.mean()/f.baseline_brier.mean()),
        'accuracy_pct':100*f.correct.mean(),'baseline_accuracy_pct':100*f.baseline_correct.mean(),
        'return_r2_pct':100*(1-f.squared_error.mean()/f.baseline_squared_error.mean()),
        'volatility_skill_pct':100*(1-f.logrv_error.mean()/f.baseline_logrv_error.mean()),
        'monthly_volatility_pct':100*f.realized_return.std(ddof=1)}


def block_test(delta, seed):
    """One-sided improvement test and marginal CI, circular six-month blocks."""
    values = np.asarray(delta); n = len(values)
    rng = np.random.default_rng(seed)
    starts = rng.integers(n,size=(SETTINGS['bootstrap_replicates'],int(np.ceil(n/6))))
    indices = ((starts[:,:,None]+np.arange(6))%n).reshape(len(starts),-1)[:,:n]
    samples = values[indices].mean(axis=1)
    observed = values.mean()
    p = (1+np.count_nonzero(samples-observed >= observed))/(len(samples)+1)
    return float(p), float(np.quantile(samples,.025)), float(np.quantile(samples,.975))


def holm(pvalues):
    p = np.asarray(pvalues); order=np.argsort(p); output=np.empty(len(p))
    output[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return output


def main():
    freeze_design()
    data = ROOT/'data'
    universe = pd.read_csv(data/'universe.csv')
    rows, excluded, predictions, provenance = [], [], [], []
    expected_months = pd.date_range(SETTINGS['selection_start'],SETTINGS['audit_end'],freq='ME')
    for _, stock in universe.iterrows():
        ticker = stock.ticker
        try:
            if stock.Price <= SETTINGS['price_floor_exclusive']: raise ValueError('Issuer price is not above $10')
            path = data/f'{ticker}.csv'
            if not path.exists(): raise ValueError('No downloaded history')
            metadata = json.loads((data/f'{ticker}.json').read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError('History checksum mismatch')
            daily = pd.read_csv(path,index_col=0,parse_dates=True).sort_index()
            if daily.index.has_duplicates: raise ValueError('Duplicate daily dates')
            if str(daily.index[-1].date()) != SETTINGS['as_of']: raise ValueError('Stale or unexpected last trading date')
            price = float(daily.Close.iloc[-1])
            if not np.isfinite(price) or price <=10: raise ValueError('Latest close is not above $10')
            liquidity = float((daily.Close*daily.Volume).iloc[-63:].median())
            if liquidity<SETTINGS['median_daily_dollar_volume_floor']: raise ValueError('Median daily dollar volume below $10 million')
            if daily.index[0]>pd.Timestamp('2010-12-31'): raise ValueError('Insufficient history for fixed 2017-2026 evaluation')
            if (daily['Adj Close']<=0).any() or not np.isfinite(daily['Adj Close']).all():
                raise ValueError('Invalid adjusted prices')
            # Do not silently repair potentially bad prices or large corporate events.
            if np.log(daily['Adj Close']).diff().abs().max()>1:
                raise ValueError('Adjusted daily log-return jump above 1; manual corporate-action review required')
            monthly = monthly_data(daily)
            if not expected_months.isin(monthly.index).all() or (monthly.loc[expected_months,'sessions']<15).any():
                raise ValueError('Missing or incomplete evaluation month')
            f = forecasts(monthly)
            if len(f)!=len(expected_months): raise ValueError('Incomplete forecast record')
            discovery = f[f.month<=SETTINGS['selection_end']]
            early = discovery[discovery.month<'2021-01-01']
            late = discovery[discovery.month>='2021-01-01']
            audit = f[f.month>=SETTINGS['audit_start']]
            discovery_stats, audit_stats = summarize(discovery), summarize(audit)
            early_skill, late_skill = summarize(early)['brier_skill_pct'],summarize(late)['brier_skill_pct']
            p,lo,hi = block_test(audit.baseline_brier-audit.brier, SETTINGS['seed'])
            issuer = re.sub(r'\s+(CLASS|CL)\s+[A-Z0-9-]+.*$', '', stock.Name)
            rows.append({'ticker':ticker,'name':stock.Name,'sector':stock.Sector,'issuer':issuer,
                'price_usd':price,'price_date':SETTINGS['as_of'],'issuer_price_usd':stock.Price,
                'median_daily_dollar_volume':liquidity,'first_price_date':str(daily.index[0].date()),
                'selection_score':min(early_skill,late_skill),'early_skill_pct':early_skill,
                'late_skill_pct':late_skill, **{f'selection_{k}':v for k,v in discovery_stats.items()},
                **{f'audit_{k}':v for k,v in audit_stats.items()},'audit_p_value':p,
                'audit_brier_improvement_lower95':lo,'audit_brier_improvement_upper95':hi,
                'history_source':metadata['source']})
            predictions.append(f.assign(ticker=ticker))
            provenance.append(metadata)
        except Exception as exc:
            excluded.append({'ticker':ticker,'reason':str(exc)})
    eligible = pd.DataFrame(rows)
    if len(eligible)<100: raise ValueError(f'Only {len(eligible)} eligible stocks; cannot produce 100')
    # Resolve duplicate share classes using liquidity, never forecast performance.
    eligible = eligible.sort_values('median_daily_dollar_volume',ascending=False)
    duplicate = eligible.duplicated('issuer')
    excluded.extend({'ticker':t,'reason':'Less-liquid share class of same normalized issuer'} for t in eligible.loc[duplicate,'ticker'])
    eligible = eligible.loc[~duplicate].copy()
    eligible['audit_holm_p_value'] = holm(eligible.audit_p_value)
    eligible = eligible.sort_values(['selection_score','selection_brier_skill_pct','ticker'],ascending=[False,False,True])
    eligible.insert(0,'rank',np.arange(1,len(eligible)+1))
    selected = eligible.head(100).copy()
    selected['audit_status'] = np.where(selected.audit_brier_skill_pct>0,'positive recent skill','no positive recent skill')
    selected['multiple_testing_confirmed'] = (selected.audit_holm_p_value<.05)&(selected.audit_brier_skill_pct>0)
    pred = pd.concat(predictions,ignore_index=True)
    pred = pred[pred.ticker.isin(eligible.ticker)]
    eligible.to_csv(ROOT/'all-ranked-stocks.csv',index=False)
    selected.to_csv(ROOT/'top-100.csv',index=False)
    pd.DataFrame(excluded).to_csv(ROOT/'excluded-stocks.csv',index=False)
    pred.to_csv(ROOT/'monthly-predictions.csv',index=False)
    (ROOT/'tickers.txt').write_text('\n'.join(selected.ticker)+'\n')
    (ROOT/'data-provenance.json').write_text(json.dumps(provenance,indent=2))
    audit = pred[(pred.ticker.isin(selected.ticker))&(pred.month>=SETTINGS['audit_start'])]
    matrix = audit.pivot(index='month',columns='ticker',values='realized_return')
    correlation = matrix.corr()
    correlation.to_csv(ROOT/'selected-monthly-correlations.csv')
    mask = ~np.eye(len(correlation),dtype=bool)
    sector_counts = selected.groupby('sector').size().sort_values(ascending=False)
    summary = {'created_utc':datetime.now(timezone.utc).isoformat(),'issuer_equities':len(universe),
        'eligible_issuers':len(eligible),'excluded':len(excluded),'selected':len(selected),
        'minimum_selected_price':float(selected.price_usd.min()),
        'positive_selection_both_blocks':int((selected.selection_score>0).sum()),
        'positive_audit_skill':int((selected.audit_brier_skill_pct>0).sum()),
        'positive_selection_and_audit':int(((selected.selection_score>0)&(selected.audit_brier_skill_pct>0)).sum()),
        'holm_confirmed_selected':int(selected.multiple_testing_confirmed.sum()),
        'selected_pooled_audit':summarize(audit),'sector_counts':sector_counts.to_dict(),
        'mean_pairwise_monthly_correlation':float(correlation.to_numpy()[mask].mean()),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(selected[['rank','ticker','price_usd','selection_score','audit_brier_skill_pct']].head(20).to_string(index=False))


if __name__ == '__main__': main()
