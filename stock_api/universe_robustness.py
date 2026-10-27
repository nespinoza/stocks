"""Frozen sampling/benchmark/statistical rules; unavailable history is never a filter."""
import numpy as np
import pandas as pd

SECTOR_QUOTAS={'Information Technology':4,'Health Care':3,'Financials':3,
    'Industrials':2,'Consumer Discretionary':2,'Communication Services':2,
    'Energy':1,'Consumer Staples':1,'Utilities':1,'Materials or Real Estate':1}

class HistoricalDataUnavailable(ValueError):
    """A missing observation must not silently remove a historical candidate."""


def eligible_universe(members,prices,sessions,selection_timestamp,price_floor=10.,dollar_volume_floor=10_000_000.):
    """Build a PIT pool from audited security-ID membership and historical *raw* quotes.

    Input membership must already represent the declared historical parent universe.
    Required fields: security_id, issuer_id, ticker, sector, available_at, first_trade.
    Prices: raw_close/raw_volume (contemporaneous units), total_return_close.
    Unknown histories fail the entire pool; known recent listings can be excluded
    solely because their pre-selection trading histories are too short.
    """
    cutoff=pd.Timestamp(selection_timestamp)
    if cutoff.tzinfo is None: raise ValueError('Selection timestamp must have a timezone')
    available=pd.to_datetime(members.available_at,utc=True)
    if (available>cutoff).any(): raise HistoricalDataUnavailable('Membership or sector became available after selection')
    if members.security_id.duplicated().any(): raise ValueError('Duplicate permanent security IDs')
    calendar=pd.DatetimeIndex(sessions)
    calendar=calendar[calendar<=cutoff.tz_convert('America/New_York').tz_localize(None).normalize()]
    if len(calendar)<253: raise HistoricalDataUnavailable('Need 253 prior sessions for all benchmark inputs')
    required=calendar[-253:]; rows=[]; exclusions=[]
    for m in members.sort_values('security_id').itertuples():
        if pd.Timestamp(m.first_trade)>required[0]:
            exclusions.append(dict(security_id=m.security_id,ticker=m.ticker,reason='known listing date: fewer than 253 prior sessions')); continue
        if m.security_id not in prices: raise HistoricalDataUnavailable(f'Missing historical price record: {m.security_id} ({m.ticker})')
        f=prices[m.security_id].reindex(required)
        cols=['raw_close','raw_volume','total_return_close']
        if f[cols].isna().any().any() or not np.isfinite(f[cols]).all().all():
            raise HistoricalDataUnavailable(f'Missing/nonfinite pre-selection observations: {m.security_id}')
        if (f[['raw_close','total_return_close']]<=0).any().any() or (f.raw_volume<0).any(): raise HistoricalDataUnavailable('Invalid historical quotes')
        price=float(f.raw_close.iloc[-1]); dollar_volume=float((f.raw_close*f.raw_volume).iloc[-63:].median())
        if price<=price_floor or dollar_volume<dollar_volume_floor:
            exclusions.append(dict(security_id=m.security_id,ticker=m.ticker,reason='historical price/liquidity threshold')); continue
        momentum=float(np.log(f.total_return_close.iloc[-22]/f.total_return_close.iloc[-253]))
        rows.append(dict(security_id=m.security_id,issuer_id=m.issuer_id,ticker=m.ticker,sector=m.sector,
            selection_timestamp=cutoff.isoformat(),membership_available_at=str(m.available_at),historical_price=price,
            median_dollar_volume_63=dollar_volume,momentum_12_2=momentum,momentum_start=str(required[-253].date()),
            momentum_end=str(required[-22].date()),history_sessions=len(f)))
    eligible=pd.DataFrame(rows)
    if eligible.empty: raise HistoricalDataUnavailable('No eligible stocks')
    # One share class per issuer, chosen by pre-selection liquidity only.
    eligible=eligible.sort_values(['median_dollar_volume_63','security_id'],ascending=[False,True]).drop_duplicates('issuer_id')
    return eligible.sort_values('security_id').reset_index(drop=True),pd.DataFrame(exclusions)


def sample_portfolios(eligible,window,seed=20261019):
    """Independent draws by SeedSequence; overlap is allowed across portfolio IDs."""
    pool=eligible.sort_values('security_id').copy()
    if pool.security_id.duplicated().any() or pool.issuer_id.duplicated().any(): raise ValueError('One security per issuer required')
    if window not in ('A','B'): raise ValueError('Unknown window')
    pool['stratum']=pool.sector.replace({'Materials':'Materials or Real Estate','Real Estate':'Materials or Real Estate'})
    top=pool.sort_values(['momentum_12_2','security_id'],ascending=[False,True]).head(int(np.ceil(.2*len(pool))))
    tech=pool[pool.sector=='Information Technology']
    if min(len(top),len(tech))<20: raise HistoricalDataUnavailable('Fewer than 20 candidates in technology or top-quintile momentum')
    rows=[]
    for group,number in [('A',0),('B',1),('C',2)]:
        for j in range(10):
            entropy=[seed,ord(window)-ord('A'),number,j]
            rng=np.random.default_rng(np.random.SeedSequence(entropy)); selected=[]
            pools=[(pool[pool.stratum==sector],quota) for sector,quota in SECTOR_QUOTAS.items()] if group=='A' else [(tech if group=='B' else top,20)]
            for candidates,count in pools:
                if len(candidates)<count: raise HistoricalDataUnavailable(f'Insufficient sector quota: {group}, {count}, {len(candidates)}')
                selected.extend(candidates.iloc[rng.choice(len(candidates),count,replace=False)].to_dict('records'))
            assert len(selected)==len({r['security_id'] for r in selected})==20
            for r in selected: rows.append(dict(window=window,group=group,portfolio_id=f'{group}{j+1:02d}',seed_entropy=str(entropy),**r))
    return pd.DataFrame(rows)


def capped_proportional(scores,cap=.2):
    """Water-fill nonnegative scores into fully invested, capped allocations."""
    scores=np.asarray(scores,float)
    if not np.isfinite(scores).all() or (scores<0).any() or len(scores)*cap<1-1e-12: raise ValueError('Invalid scores or infeasible cap')
    if scores.sum()==0: scores=np.ones(len(scores))
    weights=np.zeros(len(scores)); active=np.ones(len(scores),bool); budget=1.
    while active.any():
        values=scores[active]
        if values.sum()==0: values=np.ones(len(values))
        proposed=budget*values/values.sum(); indices=np.flatnonzero(active)
        capped=proposed>cap+1e-14
        if not capped.any(): weights[indices]=proposed; break
        fixed=indices[capped]; weights[fixed]=cap; active[fixed]=False; budget=1-weights.sum()
    return weights


def benchmark_weights(history,strategy,cap=.2):
    """Caller supplies only adjusted-close history through the forecast origin."""
    p=np.asarray(history,float)
    if p.ndim!=2 or len(p)<253 or not np.isfinite(p).all() or (p<=0).any(): raise HistoricalDataUnavailable('Benchmark requires complete prior prices')
    if strategy=='equal': scores=np.ones(p.shape[1])
    elif strategy=='inverse_volatility':
        volatility=np.diff(np.log(p[-64:]),axis=0).std(axis=0,ddof=1)
        if (volatility<=0).any(): raise HistoricalDataUnavailable('Constant historical return series')
        scores=1/volatility
    elif strategy=='momentum':
        momentum=np.log(p[-22]/p[-253])
        scores=pd.Series(momentum).rank(method='average').to_numpy()  # all positive, no forecast tuning
    else: raise ValueError('Unknown benchmark')
    return capped_proportional(scores,cap)


def synchronous_paired_bootstrap(gp,benchmark,seed=20261019,replicates=2000,block=3):
    """Resample shared holding-month blocks across ALL overlapping portfolios.

    Conditional on frozen universes and observed allocation paths. It does not
    refit strategies inside bootstrap paths or estimate independence across eras.
    Rows are aligned holding months, columns portfolio IDs. Do not resample each
    portfolio independently or treat ticker-months as independent observations.
    """
    gp,benchmark=np.asarray(gp,float),np.asarray(benchmark,float)
    if gp.shape!=benchmark.shape or gp.ndim!=2 or gp.shape[0]<2: raise ValueError('Need aligned month × portfolio matrices')
    if not np.isfinite(gp).all() or not np.isfinite(benchmark).all() or (gp< -1).any() or (benchmark< -1).any(): raise ValueError('Missing or invalid returns')
    n=gp.shape[0]; rng=np.random.default_rng(seed)
    starts=rng.integers(n,size=(replicates,int(np.ceil(n/block))))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(replicates,-1)[:,:n]
    excess=np.prod(1+gp[idx],axis=1)-np.prod(1+benchmark[idx],axis=1)
    observed=np.prod(1+gp,axis=0)-np.prod(1+benchmark,axis=0)
    result={}
    for label,fn in [('mean_excess',lambda x:np.mean(x,axis=-1)),('median_excess',lambda x:np.median(x,axis=-1)),('win_fraction',lambda x:np.mean(x>0,axis=-1))]:
        lo,hi=np.quantile(fn(excess),[.025,.975]); result[label]=dict(estimate=float(fn(observed)),lower95=float(lo),upper95=float(hi))
    return result
