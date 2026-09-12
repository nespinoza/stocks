"""Atomic per-unit checkpoints, deterministic workers, immutable data snapshots."""
from __future__ import annotations
import contextlib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import traceback
import warnings
import threading
import numpy as np
import pandas as pd
from .config import HistoricalConfig,make_origins,fold_dates,sessions_for
from .inference import forecast

VERSION='historical-v1'


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [plain(v) for v in value]
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def encoded(value):return json.dumps(plain(value),sort_keys=True,allow_nan=False,separators=(',',':')).encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def atomic_json(path,value):atomic_bytes(path,encoded(value)+b'\n')


@contextlib.contextmanager
def experiment_lock(root):
    import fcntl
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.runner.lock').open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another runner holds this experiment lock') from None
        stream.seek(0);stream.truncate();stream.write(str(os.getpid()));stream.flush()
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def implementation():
    package=Path(__file__).resolve().parents[1]
    paths=[package/'models.py',package/'sde.py',package/'heteroskedastic.py',
           *sorted((package/'diagnostics').glob('*.py')),
           *[package/'historical'/name for name in ['config.py','runner.py','inference.py']]]
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','pydantic','dynesty','exchange-calendars','yfinance','threadpoolctl']}
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=package,text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):commit=None
    return {'version':VERSION,'git_commit':commit,'python':platform.python_version(),'packages':versions,
            'source_hashes':{str(p.relative_to(package)):sha(p) for p in paths}}


def unit_seed(config,origin,model,lookback):
    # No prices, realized values, worker IDs, or whole-snapshot hash enter the RNG.
    return int(digest([config.seed,config.target_ticker,config.auxiliary_tickers,origin,model,lookback,config.forecast_horizon_days])[:8],16)


def unit_id(origin,model,lookback):return f'{origin}__{model}__{lookback}'


def units(config):
    return [(origin,model,lookback) for origin in make_origins(config) for lookback in config.lookbacks for model in config.models]


def download(config):
    import yfinance as yf
    if config.data_vintage!='retrospective_adjusted':raise ValueError('Synthetic/PIT data must be supplied explicitly')
    dates=sessions_for(config);tickers=[config.target_ticker,*config.auxiliary_tickers]
    if dates[-1]>=pd.Timestamp.now(tz='UTC').tz_localize(None).normalize():raise ValueError('Requested prices may not be complete yet')
    raw=yf.download(tickers,start=str(dates[0].date()),end=str(config.evaluation_end),
                    auto_adjust=True,progress=False,threads=False,timeout=30,multi_level_index=True)
    if raw is None or raw.empty:raise ValueError('Provider returned no data; no units were started')
    close=raw['Close']
    if isinstance(close,pd.Series):close=close.to_frame(tickers[0])
    return close


def prepare(config,root,prices=None):
    root=Path(root);manifest_path=root/'experiment-config.json'
    current=implementation()
    if manifest_path.exists():
        meta=json.loads(manifest_path.read_text())
        if meta['config']!=config.model_dump(mode='json'):raise ValueError('Frozen configuration differs; use a new output directory')
        if meta['implementation']!=current:raise ValueError('Inference code/environment differs; restore it or use a new experiment')
        if sha(root/'prices.csv')!=meta['snapshot_sha256']:raise ValueError('Snapshot checksum mismatch')
        if prices is not None:raise ValueError('Resume reuses saved prices; do not supply a new frame')
        return meta,pd.read_csv(root/'prices.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    frame=download(config) if prices is None else prices.copy(deep=True)
    if not isinstance(frame.index,pd.DatetimeIndex) or frame.index.has_duplicates:raise ValueError('Prices require unique dated rows')
    if frame.index.tz is not None:frame.index=frame.index.tz_localize(None)
    if not frame.index.equals(frame.index.normalize()):raise ValueError('Daily midnight timestamps required')
    tickers=[config.target_ticker,*config.auxiliary_tickers]
    if not set(tickers)<=set(frame.columns):raise ValueError('Missing configured ticker columns')
    frame=frame.reindex(index=sessions_for(config),columns=tickers).astype(float)
    # Missing/invalid observations remain missing: record affected units explicitly.
    frame=frame.where(np.isfinite(frame)&(frame>0))
    atomic_bytes(root/'prices.csv',frame.to_csv().encode())
    snapshot_hash=sha(root/'prices.csv')
    identity=digest({'config':config.model_dump(mode='json'),'snapshot':snapshot_hash,'implementation':current})
    meta={'schema_version':1,'experiment_id':identity,'configuration_hash':digest(config.model_dump(mode='json')),
          'created_at':datetime.now(timezone.utc).isoformat(),'config':config.model_dump(mode='json'),
          'implementation':current,'snapshot_sha256':snapshot_hash,'total_units':len(units(config)),
          'origins':make_origins(config),'data_quality':{'rows':len(frame),'missing_by_ticker':frame.isna().sum().to_dict(),
          'first_valid':{t:str(frame[t].first_valid_index()) for t in tickers},'last_valid':{t:str(frame[t].last_valid_index()) for t in tickers}},
          'data_limitation':'Retrospective adjusted Yahoo prices are not a point-in-time vintage archive' if config.data_vintage=='retrospective_adjusted' else config.data_vintage,
          'origin_convention':'Before open of first exchange session each week; train strictly before origin; horizon [origin, origin + calendar days). Holidays can produce overlap; inference uses serially dependent episodes.'}
    atomic_json(manifest_path,meta)
    return meta,frame


def load_record(path,experiment_id=None):
    wrapper=json.loads(Path(path).read_text());record=wrapper['record']
    if wrapper['sha256']!=digest(record):raise ValueError(f'Checkpoint checksum mismatch: {path}')
    if experiment_id and record['experiment_id']!=experiment_id:raise ValueError('Checkpoint experiment mismatch')
    return record


def save_record(path,record):atomic_json(path,{'sha256':digest(record),'record':record})


@contextlib.contextmanager
def heartbeat(root,meta,completed,current,start,progress):
    stop=threading.Event()
    def publish():
        values=list(completed.values())
        info={'ticker':meta['config']['target_ticker'],'completed':len(values),'total':meta['total_units'],
              'failures':sum(s!='ok' for s in values),'current':list(current),
              'fraction':len(values)/max(meta['total_units'],1),'elapsed_seconds':time.monotonic()-start,
              'heartbeat_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()}
        atomic_json(root/'progress.json',info)
        progress(json.dumps(info),flush=True) if progress is print else progress(info)
    def loop():
        while not stop.wait(30):publish()
    thread=threading.Thread(target=loop,daemon=True);thread.start()
    try:yield publish
    finally:stop.set();thread.join();publish()


def _execute(config_dict,meta,frame,unit):
    config=HistoricalConfig.model_validate(config_dict);origin,model,lookback=unit
    start=time.monotonic();train_dates,test_dates=fold_dates(config,origin,lookback)
    seed=unit_seed(config,origin,model,lookback)
    record={'schema_version':1,'unit_id':unit_id(*unit),'experiment_id':meta['experiment_id'],
            'configuration_hash':meta['configuration_hash'],'target_ticker':config.target_ticker,
            'auxiliary_tickers':config.auxiliary_tickers,'model':model,'lookback':lookback,'origin':origin,
            'horizon_days':config.forecast_horizon_days,'seed':seed,'inference_version':VERSION,
            'git_commit':meta['implementation']['git_commit'],'inference_mode':config.inference_mode,
            'train_dates':[str(x.date()) for x in train_dates],'forecast_dates':[str(x.date()) for x in test_dates],
            'status':'error','prediction':None,'realized':None}
    try:
        columns=[config.target_ticker,*config.auxiliary_tickers] if model in ['multitask_gp','var'] else [config.target_ticker]
        train=frame.reindex(train_dates)[columns].copy()
        if len(train)<config.min_train_observations or train.isna().any().any() or not len(test_dates):
            record['status']='data_failed';raise ValueError('Insufficient or missing training observations; no filling applied')
        record['training_sha256']=hashlib.sha256(train.to_csv().encode()).hexdigest()
        record['training_volatility']=float(np.diff(np.log(train.iloc[:,0])).std(ddof=1))
        record['training_return']=float(np.log(train.iloc[-1,0]/train.iloc[0,0]))
        record['starting_price']=float(train.iloc[-1,0])
        # Inference sees only this copied training window and a step count.
        prediction=forecast(train,len(test_dates),model,config,seed)
        record['prediction']=prediction
        record['status']='ok' if prediction['diagnostics'].get('converged',True) else 'inference_failed'
        actual=frame.reindex(test_dates)[config.target_ticker].to_numpy()
        if not np.isfinite(actual).all():
            record['status']='data_failed';raise ValueError('Missing held-out observations; prediction preserved but cannot score')
        realized=np.log(actual[-1]/record['starting_price'])
        all_returns=np.diff(np.log(np.r_[record['starting_price'],actual]))
        record['realized']={'prices':actual.tolist(),'terminal_log_return':float(realized),
                            'direction':int(np.sign(realized)),'positive':bool(realized>0),
                            'horizon_realized_volatility':float(np.sqrt(np.mean(all_returns**2)))}
    except Exception as exc:
        record['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    record['runtime_seconds']=time.monotonic()-start
    return plain(record)


def run_historical_validation(config=None,*,output,prices=None,workers=1,max_units=None,retry_failures=False,progress=print,**settings):
    """Run/resume frozen units. Full-run execution is always an explicit call."""
    config=config or HistoricalConfig(**settings)
    if not isinstance(config,HistoricalConfig):config=HistoricalConfig.model_validate(config)
    if workers<1:raise ValueError('workers must be positive')
    if max_units is not None and max_units<1:raise ValueError('max_units must be positive')
    root=Path(output);start=time.monotonic()
    # threadpoolctl also handles callers who imported NumPy before the CLI.
    from threadpoolctl import threadpool_limits
    with experiment_lock(root),threadpool_limits(limits=1):
        meta,frame=prepare(config,root,prices)
        completed={};pending=[]
        for unit in units(config):
            key=unit_id(*unit);path=root/'records'/f'{key}.json'
            if path.exists():
                record=load_record(path,meta['experiment_id']);completed[key]=record['status']
                if record['status']=='ok' or not retry_failures:continue
            pending.append(unit)
        if max_units is not None:pending=pending[:max_units]
        def finish(record):
            key=record['unit_id'];path=root/'records'/f'{key}.json'
            if path.exists():
                old=load_record(path,meta['experiment_id'])
                if old['status']=='ok':raise RuntimeError('Refusing to overwrite successful checkpoint')
                save_record(root/'failures'/key/(digest(old)+'.json'),old)
            save_record(path,record);completed[key]=record['status']
            info={'ticker':config.target_ticker,'completed':len(completed),'total':meta['total_units'],
                  'failures':sum(s!='ok' for s in completed.values()),'current':key,
                  'fraction':len(completed)/max(meta['total_units'],1),'elapsed_seconds':time.monotonic()-start}
            atomic_json(root/'progress.json',info)
            progress(json.dumps(info),flush=True) if progress is print else progress(info)
        progress(f'Experiment {meta["experiment_id"]}: {len(completed)}/{meta["total_units"]} saved; {len(pending)} units queued')
        current=[]
        with heartbeat(root,meta,completed,current,start,progress) as publish:
            if workers==1:
                for unit in pending:
                    current[:]=[unit_id(*unit)];publish()
                    finish(_execute(config.model_dump(mode='json'),meta,frame,unit))
            else:
                import multiprocessing
                # Bounded in-flight work: at most workers snapshots/tasks at a time.
                with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_worker_init) as pool:
                    iterator=iter(pending);active={}
                    for unit in iterator:
                        active[pool.submit(_execute,config.model_dump(mode='json'),meta,frame,unit)]=unit
                        if len(active)==workers:break
                    current[:]=[unit_id(*u) for u in active.values()];publish()
                    while active:
                        ready,_=wait(active,return_when=FIRST_COMPLETED)
                        for future in ready:
                            active.pop(future);finish(future.result())
                            unit=next(iterator,None)
                            if unit is not None:active[pool.submit(_execute,config.model_dump(mode='json'),meta,frame,unit)]=unit
                        current[:]=[unit_id(*u) for u in active.values()]
            current.clear()
    return root


def _worker_init():
    from threadpoolctl import threadpool_limits
    global _thread_limit
    _thread_limit=threadpool_limits(limits=1)


def status(root,*,verify=False):
    root=Path(root);meta=json.loads((root/'experiment-config.json').read_text());counts={};runtime=0.
    heartbeat_info=json.loads((root/'progress.json').read_text()) if (root/'progress.json').exists() else None
    if not verify:
        completed=(heartbeat_info or {}).get('completed',0)
        return {'output':str(root.resolve()),'experiment_id':meta['experiment_id'],'total':meta['total_units'],
                'completed':completed,'pending':meta['total_units']-completed,'latest_progress':heartbeat_info,
                'note':'Lightweight heartbeat view; use --verify to scan and validate all checkpoints'}
    for path in (root/'records').glob('*.json'):
        record=load_record(path,meta['experiment_id']);counts[record['status']]=counts.get(record['status'],0)+1
        runtime+=record['runtime_seconds']
    return {'output':str(root.resolve()),'experiment_id':meta['experiment_id'],'total':meta['total_units'],'latest_progress':heartbeat_info,
            'completed':sum(counts.values()),'pending':meta['total_units']-sum(counts.values()),
            'status_counts':counts,'summed_unit_runtime_hours':runtime/3600}
