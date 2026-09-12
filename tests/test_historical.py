"""Historical orchestration tests use small deterministic local data, never downloads."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from stock_api.historical import HistoricalConfig,run_historical_validation,generate_report,status
from stock_api.historical.config import make_origins,sessions_for,fold_dates
from stock_api.historical import runner
from stock_api.historical.analysis import score,add_pairs,block_comparison,reliability_bins,confidence_bins
from stock_api.historical.diagnostic import diagnostic_input
from stock_api.diagnostics import DiagnosticConfig


@pytest.fixture
def config():
    return HistoricalConfig(target_ticker='XYZ',auxiliary_tickers=['QQQ'],start_date='2025-01-06',end_date='2025-01-25',
        lookbacks=[30],models=['last_price','var'],n_paths=100,data_vintage='synthetic',bootstrap_replicates=100,min_bin_count=2)


@pytest.fixture
def prices(config):
    dates=sessions_for(config);rng=np.random.default_rng(17)
    return pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0,.01,(len(dates),2)),axis=0)),index=dates,columns=['XYZ','QQQ'])


def records(root):return [runner.load_record(p) for p in sorted((root/'records').glob('*.json'))]


def test_generic_origins_and_holiday(config):
    assert make_origins(config)==['2025-01-06','2025-01-13']
    c=config.model_copy(update={'start_date':pd.Timestamp('2025-01-20').date(),'end_date':pd.Timestamp('2025-02-05').date()})
    assert make_origins(c)[0]=='2025-01-21'
    assert len(make_origins(c))==len(set(make_origins(c)))
    c=HistoricalConfig.model_validate({**config.model_dump(),'target_ticker':' nvda ','auxiliary_tickers':['aapl']})
    assert c.target_ticker=='NVDA' and c.auxiliary_tickers==['AAPL']


def test_checkpoint_resume_skip(config,prices,tmp_path,monkeypatch):
    run_historical_validation(config,output=tmp_path,prices=prices,max_units=1,progress=lambda _:None)
    first=next((tmp_path/'records').glob('*.json'));data=first.read_bytes();stamp=first.stat().st_mtime_ns
    run_historical_validation(config,output=tmp_path,progress=lambda _:None)
    assert first.read_bytes()==data and first.stat().st_mtime_ns==stamp
    assert status(tmp_path)['pending']==0
    monkeypatch.setattr(runner,'forecast',lambda *a,**k:pytest.fail('Completed forecast rerun'))
    run_historical_validation(config,output=tmp_path,progress=lambda _:None)
    assert len(records(tmp_path))==4


def test_checkpoint_corruption_rejected(config,prices,tmp_path):
    run_historical_validation(config,output=tmp_path,prices=prices,max_units=1,progress=lambda _:None)
    path=next((tmp_path/'records').glob('*.json'));w=json.loads(path.read_text());w['record']['seed']+=1;path.write_text(json.dumps(w))
    with pytest.raises(ValueError,match='checksum'):run_historical_validation(config,output=tmp_path,progress=lambda _:None)


def test_snapshot_and_config_protected(config,prices,tmp_path):
    run_historical_validation(config,output=tmp_path,prices=prices,max_units=1,progress=lambda _:None)
    with pytest.raises(ValueError,match='configuration'):run_historical_validation(config.model_copy(update={'seed':2}),output=tmp_path,progress=lambda _:None)
    with (tmp_path/'prices.csv').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='checksum'):run_historical_validation(config,output=tmp_path,progress=lambda _:None)


def test_no_future_leakage(config,prices,tmp_path,monkeypatch):
    captured=[];original=runner.forecast
    def spy(train,steps,model,c,seed):
        captured.append(train.index.max());assert train.index.max()<pd.Timestamp('2025-01-06')
        return original(train,steps,model,c,seed)
    monkeypatch.setattr(runner,'forecast',spy)
    c=config.model_copy(update={'end_date':pd.Timestamp('2025-01-14').date()})
    run_historical_validation(c,output=tmp_path/'a',prices=prices,progress=lambda _:None)
    changed=prices.copy();changed.loc[changed.index>=pd.Timestamp('2025-01-06')]*=4
    run_historical_validation(c,output=tmp_path/'b',prices=changed,progress=lambda _:None)
    for a,b in zip(records(tmp_path/'a'),records(tmp_path/'b')):
        assert a['prediction']==b['prediction'] and a['seed']==b['seed']
        assert a['realized']!=b['realized']
    assert captured


def test_missing_auxiliary_only_affects_multivariate(config,prices,tmp_path):
    prices.loc[prices.index[15],'QQQ']=np.nan
    # Pick a date guaranteed in first fold's training window.
    train,_=fold_dates(config,make_origins(config)[0],30);prices.loc[train[-1],'QQQ']=np.nan
    run_historical_validation(config,output=tmp_path,prices=prices,progress=lambda _:None)
    result=records(tmp_path)
    assert all(r['status']=='ok' for r in result if r['model']=='last_price')
    assert any(r['status']=='data_failed' for r in result if r['model']=='var')


def test_holdout_not_saved_or_scored(config,prices,tmp_path):
    c=config.model_copy(update={'untouched_holdout_start':pd.Timestamp('2025-01-14').date()})
    run_historical_validation(c,output=tmp_path,prices=prices,progress=lambda _:None)
    saved=pd.read_csv(tmp_path/'prices.csv',index_col=0)
    assert saved.index.max()<'2025-01-14'
    assert all(max(r['forecast_dates'])<'2025-01-14' for r in records(tmp_path))


def test_seed_worker_independence(config,prices,tmp_path):
    run_historical_validation(config,output=tmp_path/'one',prices=prices,workers=1,progress=lambda _:None)
    run_historical_validation(config,output=tmp_path/'two',prices=prices,workers=2,progress=lambda _:None)
    for a,b in zip(records(tmp_path/'one'),records(tmp_path/'two')):
        assert a['seed']==b['seed'] and a['prediction']==b['prediction']
    assert len({r['seed'] for r in records(tmp_path/'one')})==4


def test_failure_resume_control(config,prices,tmp_path,monkeypatch):
    original=runner.forecast
    def fail(*a,**kw):raise RuntimeError('injected failure')
    monkeypatch.setattr(runner,'forecast',fail)
    run_historical_validation(config,output=tmp_path,prices=prices,progress=lambda _:None)
    assert all(r['status']=='error' for r in records(tmp_path))
    monkeypatch.setattr(runner,'forecast',original)
    run_historical_validation(config,output=tmp_path,progress=lambda _:None)
    assert all(r['status']=='error' for r in records(tmp_path))
    run_historical_validation(config,output=tmp_path,retry_failures=True,progress=lambda _:None)
    assert all(r['status']=='ok' for r in records(tmp_path))
    assert len(list((tmp_path/'failures').rglob('*.json')))==4


def test_report_no_refit_and_deterministic(config,prices,tmp_path,monkeypatch):
    run_historical_validation(config,output=tmp_path,prices=prices,progress=lambda _:None)
    monkeypatch.setattr(runner,'forecast',lambda *a,**kw:pytest.fail('Report fitted a model'))
    path=generate_report(tmp_path);first=path.read_bytes()
    summary=(tmp_path/'summaries/ticker-reliability.json').read_bytes()
    generate_report(tmp_path)
    assert path.read_bytes()==first and (tmp_path/'summaries/ticker-reliability.json').read_bytes()==summary
    f=pd.read_csv(tmp_path/'summaries/forecast-scores.csv')
    assert f[f.model=='last_price'].direction_correct.isna().all()
    assert (f[f.model=='last_price'].delta_terminal_return_mae==0).all()
    assert len(json.loads((tmp_path/'summaries/ticker-reliability.json').read_text())['models'])==2


def test_diagnostic_bridge_exact_fold(config,prices,tmp_path):
    run_historical_validation(config,output=tmp_path,prices=prices,max_units=1,progress=lambda _:None)
    run=diagnostic_input(tmp_path,make_origins(config)[0],30);run._check_identity()
    assert run.config['ticker']=='XYZ' and run.config['related_tickers']==['QQQ']
    assert run.folds[0]['train_dates']==records(tmp_path)[0]['train_dates']


def test_production_gp_does_not_fit_old_optimizer(config,prices,monkeypatch):
    from stock_api.historical.inference import forecast
    import stock_api.diagnostics.problems as problems
    monkeypatch.setattr(problems,'fit_latent_returns',lambda *a,**kw:pytest.fail('Old two-day-floor fit was invoked'))
    monkeypatch.setitem(problems.MODELS,'gp',lambda *a,**kw:pytest.fail('Old GP fit was invoked'))
    settings=DiagnosticConfig(profile_surfaces=False,multistart=False,corner_plots=False,latent_plots=False,
                nlive=20,nested_maxcall=100,chains=2,warmup=10,draws=20,interweave=True)
    c=config.model_copy(update={'inference':settings})
    train=prices.iloc[:12]
    for model in ['gp','volatility_gp_returns','heteroskedastic_gp_returns']:
        result=forecast(train,2,model,c,14)
        assert len(result['terminal_samples'])==100
        assert result['diagnostics']['method']
        if model=='volatility_gp_returns':assert result['probability_up']==.5 and result['median_log_return']==0


def test_block_comparison_dependence_and_small_samples():
    index=pd.date_range('2020-01-06',periods=100,freq='W-MON').strftime('%Y-%m-%d').tolist()
    values=pd.Series(np.repeat(np.arange(10),10)-5.,index=index)
    a=block_comparison(values,index,8,300,42);b=block_comparison(values,index,8,300,42)
    assert a==b and a['lower95']<a['upper95'] and a['paired_origins']==100
    assert block_comparison(values.iloc[:3],index,8,300,42)['p_value'] is None


def test_calibration_sparse_merging_and_tied_confidence():
    f=pd.DataFrame({'probability_up':[.01,.02,.6,.7,.95],'actual_up':[0,1,1,1,0],
       'realized_return':[0,.1,.2,.1,-.2],'terminal_return_mae':[.1]*5})
    b=reliability_bins(f,3)
    assert b['count'].sum()==5 and len(b)==1
    f=f.assign(confidence=.0,direction_correct=np.nan,abs_z=1.,signed_realized_return=np.nan,brier=.25,coverage_95=1)
    assert len(confidence_bins(f))==1


def test_normalized_scale_invariance(config,prices,tmp_path):
    c=config.model_copy(update={'models':['last_price'],'end_date':pd.Timestamp('2025-01-14').date()})
    run_historical_validation(c,output=tmp_path/'a',prices=prices,progress=lambda _:None)
    run_historical_validation(c,output=tmp_path/'b',prices=prices*100,progress=lambda _:None)
    a,b=score(records(tmp_path/'a')[0]),score(records(tmp_path/'b')[0])
    assert a['nmae_pct']==pytest.approx(b['nmae_pct']) and a['terminal_return_mae']==pytest.approx(b['terminal_return_mae'])


def test_atomic_failure_preserves_existing_checkpoint(tmp_path,monkeypatch):
    path=tmp_path/'unit.json';runner.atomic_json(path,{'old':1});original=path.read_bytes()
    def interrupt(*args):raise OSError('simulated interruption before atomic rename')
    monkeypatch.setattr(runner.os,'replace',interrupt)
    with pytest.raises(OSError):runner.atomic_json(path,{'new':2})
    assert path.read_bytes()==original and list(tmp_path.iterdir())==[path]


def test_all_failures_report_and_pending_counts(config,prices,tmp_path,monkeypatch):
    def fail(*a,**kw):raise ValueError('bad model')
    monkeypatch.setattr(runner,'forecast',fail)
    run_historical_validation(config,output=tmp_path,prices=prices,max_units=1,progress=lambda _:None)
    generate_report(tmp_path)
    summary=pd.read_csv(tmp_path/'summaries/overall.csv')
    assert summary.failures.sum()==1 and summary.pending.sum()==3 and summary.successful.sum()==0


def test_unconverged_predictions_preserved_but_not_scored(config,prices,tmp_path,monkeypatch):
    original=runner.forecast
    def unconverged(*args):
        p=original(*args);p['diagnostics']['converged']=False;return p
    monkeypatch.setattr(runner,'forecast',unconverged)
    run_historical_validation(config,output=tmp_path,prices=prices,max_units=1,progress=lambda _:None)
    r=records(tmp_path)[0]
    assert r['status']=='inference_failed' and r['prediction'] and r['realized']
    assert 'mae' not in score(r)


def test_experiment_lock_rejects_concurrent_runner(tmp_path):
    with runner.experiment_lock(tmp_path):
        with pytest.raises(RuntimeError,match='lock'):
            with runner.experiment_lock(tmp_path):pass


def test_holm_accounts_for_unavailable_planned_tests():
    from stock_api.historical.analysis import holm
    np.testing.assert_allclose(holm([.01,.03,np.nan],total_tests=96)[:2],[.96,1.])


def test_next_local_schedule_time_without_scheduling():
    from scripts.start_at_local_time import next_start
    from datetime import datetime
    assert next_start('02:30',datetime(2026,9,12,1))==datetime(2026,9,12,2,30)
    assert next_start('02:30',datetime(2026,9,12,3))==datetime(2026,9,13,2,30)


def test_degenerate_forecast_has_no_continuous_pit(config,prices,tmp_path):
    c=config.model_copy(update={'models':['last_price','gbm_zero_drift'],'end_date':pd.Timestamp('2025-01-14').date()})
    run_historical_validation(c,output=tmp_path,prices=prices*0+100.,progress=lambda _:None)
    r=next(r for r in records(tmp_path) if r['model']=='gbm_zero_drift')
    s=score(r)
    assert np.isnan(s['pit']) and np.isnan(s['negative_log_score']) and np.isnan(s['z'])
    assert s['brier']==0 and s['coverage_95']==1
