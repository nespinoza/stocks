"""Calendar boundaries, scale-free ranking and chronology for the watchlist GP."""
import numpy as np
import pandas as pd
import pytest

from scripts import rank_monthly_watchlist_gp as screen


def test_calendar_month_folds_cover_requested_period_without_spillover():
    assert len(screen.periods())==14
    assert str(screen.periods()[0])=='2025-08'
    assert str(screen.periods()[-1])=='2026-09'
    for month in screen.periods():
        train,test=screen.fold_dates(month)
        assert train.min()>=month.start_time-pd.Timedelta(30,unit='D')
        assert train.max()<month.start_time<=test.min()
        assert all(test.to_period('M')==month)
        assert len(train)>=9 and 18<=len(test)<=23
        assert not train.intersection(test).size
    train,test=screen.fold_dates('2025-09')
    assert str(train[-1].date())=='2025-08-29'
    assert str(test[0].date())=='2025-09-02'  # Labor Day excluded.
    assert str(test[-1].date())=='2025-09-30'


def test_normalized_error_is_scale_independent_and_differs_from_endpoint():
    actual=np.array([100.,110.,120.]);predicted=np.array([105.,105.,120.])
    first=screen.score_paths(actual,predicted,100.)
    scaled=screen.score_paths(actual*10,predicted*10,1000.)
    assert first['nmae_pct']==pytest.approx(100*(10/3)/110)
    assert scaled['nmae_pct']==pytest.approx(first['nmae_pct'])
    assert scaled['mae_usd']==pytest.approx(10*first['mae_usd'])
    assert first['month_end_ape_pct']==0
    assert first['baseline_nmae_pct']==pytest.approx(100*10/110)


def test_actual_gp_forecast_excludes_target_month_and_replays_seed(monkeypatch):
    monkeypatch.setitem(screen.SETTINGS,'n_paths',200)
    train,test=screen.fold_dates('2025-08')
    dates=train.union(test)
    values=100*np.exp(np.random.default_rng(71).normal(0,.015,len(dates)).cumsum())
    prices=pd.Series(values,index=dates)
    first=screen.execute('TEST','2025-08',prices,'test-experiment')
    changed=prices.copy();changed.loc[test]*=1.5
    second=screen.execute('TEST','2025-08',changed,'test-experiment')
    assert first['status']==second['status']=='ok'
    assert first['prediction']==second['prediction']
    assert first['training_sha256']==second['training_sha256']
    assert first['scores']!=second['scores']
