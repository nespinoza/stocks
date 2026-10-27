"""Universe and unavailable-prior regression tests for the replication."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from run_macro_news_study import align_priors,periods_for
from prepare_growth20_study import TICKERS
from stock_api.macro_news import ridge_posterior
import pandas as pd

def test_exact_user_universe():
    assert TICKERS=='MXL DELL AXTI BAND SAIH SILC PENG MU AEHR RNG QMCO MRVL AMD FATN INTC HPE SIMO OKTA STX FTNT'.split()
    assert len(TICKERS)==len(set(TICKERS))==20

def test_absent_assets_receive_exact_weak_prior():
    rng=np.random.default_rng(19); x=rng.normal(size=(10,3)); y=rng.normal(size=(10,2))
    post=ridge_posterior(x,y)
    aligned=align_priors({'descriptive':post},['MXL','DELL'],['DELL','FATN','MXL'])['descriptive']
    np.testing.assert_array_equal(aligned['available'],[True,False,True])
    np.testing.assert_array_equal(aligned['mean'][:,0],post['mean'][:,1])
    target=rng.normal(size=(10,3)); weak=ridge_posterior(x,target); transferred=ridge_posterior(x,target,prior=aligned)
    np.testing.assert_array_equal(weak['mean'][:,1],transferred['mean'][:,1])
    np.testing.assert_array_equal(weak['covariances'][1],transferred['covariances'][1])

def test_partial_start_retains_chronological_origin():
    idx=pd.bdate_range('2025-04-08','2025-07-31')
    periods=periods_for(idx)
    assert periods[0]['origin']==pd.Timestamp('2025-04-16')
    for s in periods:
        assert idx[0]<=s['origin']<s['execution']<=s['target_end']
