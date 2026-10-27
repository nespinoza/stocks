import numpy as np
import pandas as pd
import pytest
from stock_api.macro_news import news_vector,classify_statement,ridge_posterior,ar_forecast,block_difference_ci,NEWS
from stock_api.portfolio import optimize_weights,trade,shrunk_covariance


def test_publication_cutoff_and_no_future_news():
    marker={'identifier':'coverage','source':'Archive coverage marker','availability_timestamp':'2020-03-15T12:00:00Z'}
    event={'identifier':'future','source':'Federal Reserve FOMC statement','availability_timestamp':'2020-03-17T12:00:00Z','features':dict(zip(NEWS,[1,1,1,1,1]))}
    before,ids=news_vector([marker,event],pd.Timestamp('2020-03-16'))
    np.testing.assert_array_equal(before,np.zeros(5)); assert 'future' not in ids
    after,ids=news_vector([marker,event],pd.Timestamp('2020-03-17'))
    assert after[0]==1 and 'future' in ids
    with pytest.raises(ValueError): news_vector([],pd.Timestamp('2020-03-17'))


def test_transfer_prior_is_psd_and_updateable():
    rng=np.random.default_rng(1); x=rng.normal(size=(50,4)); y=x@np.ones((4,3))*.01+rng.normal(0,.02,(50,3))
    prior=ridge_posterior(x,y,6)
    opposite=ridge_posterior(np.tile(x,(10,1)),-np.tile(y,(10,1)),6,prior)
    assert opposite['mean'].mean()<0
    assert all(np.linalg.eigvalsh(v).min()>0 for v in opposite['covariances'])


def test_ex_ante_ar_and_block_bootstrap():
    changes=np.arange(24).reshape(12,2)*.01
    forecast=ar_forecast(changes[:6]); changed=changes.copy(); changed[6:]=1e8
    np.testing.assert_array_equal(forecast,ar_forecast(changed[:6]))
    avg,lo,hi=block_difference_ci(np.ones(20)*.03,replicates=100)
    assert abs(avg-.03)<1e-12 and abs(lo-.03)<1e-12 and abs(hi-.03)<1e-12


def test_classifier_is_mechanism_proxy():
    features=classify_statement('The Committee decided to raise the target range. Inflation remains elevated. The war in Ukraine creates risks. Trade policy uncertainty increased.')
    assert features['policy_direction']==1 and features['conflict_attention']==1 and features['trade_attention']==1
    assert classify_statement('Forward guidance moves inflation toward its target.')['conflict_attention']==0
    assert classify_statement('The Committee decided to increase its holdings of Treasury securities.')['policy_direction']==0


def test_optimizer_accounting_with_augmented_mean():
    rng=np.random.default_rng(3); returns=rng.normal(0,.01,(30,22)); cov,_=shrunk_covariance(returns)
    w,_=optimize_weights(rng.normal(0,.002,22),cov*21,np.ones(22)/22)
    assert abs(w.sum()-1)<1e-8 and w.min()>=0 and w.max()<=.2+1e-8
    dollars,fee,buy,sell=trade(2000,np.ones(22)*2000/22,w,.001)
    assert abs(dollars.sum()+fee-2000)<1e-8 and abs(fee-.001*(buy+sell))<1e-8
