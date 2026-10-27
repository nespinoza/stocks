import numpy as np
import pandas as pd
import pytest
from stock_api.universe_robustness import eligible_universe,sample_portfolios,benchmark_weights,capped_proportional,synchronous_paired_bootstrap,HistoricalDataUnavailable,SECTOR_QUOTAS


def candidate_fixture():
    rows=[]
    for k,sector in enumerate(list(SECTOR_QUOTAS)[:-1]+['Materials','Real Estate']):
        for i in range(30):
            id=f'{k:02d}{i:03d}'
            rows.append(dict(security_id=id,issuer_id=id,ticker='TEST'+id,sector=sector,momentum_12_2=(k*30+i)/330))
    return pd.DataFrame(rows)


def test_reproducible_60_draws_and_sector_quotas():
    f=candidate_fixture(); a=sample_portfolios(f,'A'); b=sample_portfolios(f,'B')
    pd.testing.assert_frame_equal(a,sample_portfolios(f.sample(frac=1,random_state=3),'A'))
    assert len(a)==len(b)==600 and a.portfolio_id.nunique()==b.portfolio_id.nunique()==30
    assert not a.security_id.equals(b.security_id)
    for (_,group),g in a.groupby(['portfolio_id','group']):
        assert len(g)==g.security_id.nunique()==20
        if group=='A': assert g.stratum.value_counts().to_dict()==SECTOR_QUOTAS
        if group=='B': assert (g.sector=='Information Technology').all()
        if group=='C': assert g.momentum_12_2.min()>=f.momentum_12_2.quantile(.8)


def eligibility_fixture():
    idx=pd.bdate_range(end='2025-07-16',periods=253)
    members=pd.DataFrame([dict(security_id='stable1',issuer_id='issuer1',ticker='TEST',sector='Information Technology',available_at='2025-07-15T00:00:00Z',first_trade='2000-01-01')])
    prices={'stable1':pd.DataFrame(dict(raw_close=np.arange(253)+20.,raw_volume=np.ones(253)*1e6,total_return_close=np.arange(253)+20.),index=idx)}
    return idx,members,prices


def test_future_prices_do_not_change_eligibility_or_momentum():
    idx,members,prices=eligibility_fixture(); date='2025-07-16T16:00:00-04:00'
    before,_=eligible_universe(members,prices,idx,date)
    future=pd.bdate_range('2025-07-17',periods=20)
    prices['stable1']=pd.concat([prices['stable1'],pd.DataFrame(1e10,index=future,columns=prices['stable1'].columns)])
    after,_=eligible_universe(members,prices,idx.append(future),date)
    pd.testing.assert_frame_equal(before,after)
    assert before.momentum_12_2.iloc[0]==pytest.approx(np.log((20+231)/20))


def test_missing_history_is_fatal_not_a_survivor_filter():
    idx,members,prices=eligibility_fixture()
    with pytest.raises(HistoricalDataUnavailable,match='Missing historical'): eligible_universe(members,{},idx,'2025-07-16T16:00:00-04:00')
    prices['stable1'].iloc[0,0]=np.nan
    with pytest.raises(HistoricalDataUnavailable,match='Missing/nonfinite'): eligible_universe(members,prices,idx,'2025-07-16T16:00:00-04:00')
    members.available_at='2026-10-19T00:00:00Z'
    with pytest.raises(HistoricalDataUnavailable,match='after selection'): eligible_universe(members,prices,idx,'2025-07-16T16:00:00-04:00')


def test_benchmark_caps_and_nonnegative_full_investment():
    rng=np.random.default_rng(34); prices=100*np.exp(np.cumsum(rng.normal(0,.01,(280,20)),axis=0))
    for name in ['equal','inverse_volatility','momentum']:
        w=benchmark_weights(prices,name); assert w.sum()==pytest.approx(1); assert w.min()>=0 and w.max()<=.2+1e-12
    w=capped_proportional([1e8]+[1]*19); assert w[0]==pytest.approx(.2) and w.sum()==pytest.approx(1)
    w=capped_proportional([1,0,0,0,0,0]); assert w.sum()==pytest.approx(1) and w.max()<=.2+1e-12


def test_shared_month_resampling_does_not_invent_independence():
    rng=np.random.default_rng(3); gp=rng.normal(.01,.04,(15,1)); ew=rng.normal(.005,.04,(15,1))
    one=synchronous_paired_bootstrap(gp,ew,replicates=200)
    repeated=synchronous_paired_bootstrap(np.tile(gp,(1,30)),np.tile(ew,(1,30)),replicates=200)
    for metric in one:
        for key in one[metric]: assert one[metric][key]==pytest.approx(repeated[metric][key])
