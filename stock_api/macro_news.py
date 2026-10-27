"""Point-in-time macro/news adapters and regularized incremental return models."""
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import zipfile
from io import BytesIO
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

FACTORS=['rates','inflation_expectations','inflation','oil','unemployment','volatility']
SERIES=['DGS2','T10YIE','CPIAUCSL','DCOILWTICO','UNRATE','VIXCLS']
NEWS=['policy_direction','inflation_attention','growth_weakness','trade_attention','conflict_attention']


def macro_states(out):
    index=pd.read_csv(out/'data/macro/snapshot_index.csv')
    rows=[]; provenance=[]
    for (window,origin),group in index.groupby(['window','origin'],sort=True):
        row={'window':window,'origin':origin}
        for _,item in group.iterrows():
            path=out/item.path
            if not path.exists(): raise ValueError(f'Missing macro snapshot {path}')
            f=pd.read_csv(path,index_col=0,parse_dates=True)
            expected=item.series+'_'+item.vintage.replace('-','')
            if f.columns[0]!=expected: raise ValueError(f'Unverified ALFRED vintage header {path}: {f.columns[0]}')
            values=pd.to_numeric(f.iloc[:,0],errors='coerce').dropna()
            if values.empty: raise ValueError(f'Empty ALFRED vintage {path}')
            obs=values.index[-1]; value=float(values.iloc[-1])
            factor=FACTORS[SERIES.index(item.series)]
            if item.series=='CPIAUCSL':
                lag=obs-pd.DateOffset(years=1)
                if lag not in values.index: raise ValueError(f'Need CPI previous-year observation {path}: {lag}')
                transformed=100*(value/values.loc[lag]-1)
            elif item.series=='DCOILWTICO': transformed=np.arcsinh(value/50.)
            elif item.series=='VIXCLS': transformed=np.log(value)
            else: transformed=value
            row[factor]=float(transformed)
            assert pd.Timestamp(item.vintage)<pd.Timestamp(origin)
            assert obs<=pd.Timestamp(item.vintage)
            meta=json.loads(path.with_suffix('.csv.json').read_text())
            provenance.append(dict(window=window,origin=origin,factor=factor,series=item.series,observation_date=str(obs.date()),
                publication_timestamp=None,availability_bound=pd.Timestamp(item.vintage+' 23:59:59').tz_localize('America/New_York').isoformat(),
                vintage=item.vintage,value=value,transformed_value=float(transformed),
                transformation={'CPIAUCSL':'100*(latest CPI / same-month prior-year CPI - 1)','DCOILWTICO':'asinh(WTI/50)','VIXCLS':'log(VIX)'}.get(item.series,'level, percentage points'),
                source=meta['url'],sha256=meta['sha256'],publication_precision='unknown; verified available by vintage day, used only at later origin'))
        rows.append(row)
    states=pd.DataFrame(rows); states.to_csv(out/'data/macro/states.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'data/macro/observation_provenance.csv',index=False)
    return states


def classify_statement(text):
    """Frozen dictionary mechanism proxy; NOT an LLM or calibrated surprise."""
    t=text.lower()
    tighten=bool(re.search(r'\b(?:raise|increase) the target range\b',t))
    ease=bool(re.search(r'\b(?:lower|reduce) the target range\b',t))
    return dict(policy_direction=int(tighten)-int(ease),
        inflation_attention=int(bool(re.search(r'\binflation\b.{0,80}\b(?:elevated|high|risen|increas\w*)\b|\b(?:elevated|high) inflation\b',t))),
        growth_weakness=int(bool(re.search(r'(?:growth|activity|employment|job gains).{0,55}(?:slow|weak|declin|contract)|(?:slow|weak|declin|contract).{0,45}(?:growth|activity|employment)',t))),
        trade_attention=int(bool(re.search(r'tariff|trade polic|trade tension|trade disruption',t))),
        conflict_attention=int(bool(re.search(r'\b(?:ukrain\w*|russia\w*|war|middle east|conflict\w*)\b',t))))


def parse_news(out,c):
    records=[]; coverage=[]
    for path in sorted((out/'data/news/fed').glob('monetary*.htm')):
        soup=BeautifulSoup(path.read_text(encoding='utf-8-sig',errors='replace'),'html.parser')
        title=soup.find('h3',class_='title') or soup.find('h3')
        title=title.get_text(' ',strip=True) if title else ''
        article=soup.find(id='article') or soup
        text=article.get_text(' ',strip=True)
        if 'issues FOMC statement' not in title: continue
        date=re.search(r'(\d{8})',path.name)[1]
        day=datetime.strptime(date,'%Y%m%d')
        match=re.search(r'For release at (\d{1,2}):(\d{2})\s*([ap])\.?m',text,re.I)
        if match:
            hour=int(match[1])%12+(12 if match[3].lower()=='p' else 0)
            published=day.replace(hour=hour,minute=int(match[2]),tzinfo=ZoneInfo('America/New_York'))
            precision='minute'
        else:
            published=day.replace(hour=23,minute=59,tzinfo=ZoneInfo('America/New_York')); precision='date only; end-of-day bound'
        # Exclude navigation/related-article text when a statement body boundary exists.
        start=text.find(title)
        if start>=0: text=text[start+len(title):]
        stop=text.find('Voting for')
        if stop>=0: text=text[:stop]
        features=classify_statement(text)
        meta=json.loads(path.with_suffix('.htm.json').read_text())
        records.append(dict(identifier=path.stem,publication_timestamp=published.isoformat() if precision=='minute' else None,
            publication_date=day.date().isoformat(),availability_timestamp=(published+timedelta(hours=c['news_delay_hours'])).isoformat(),
            source='Federal Reserve FOMC statement',url=meta['url'],headline=title,text=text,entities=['Federal Reserve','United States'],
            categories=['monetary_policy'],precision=precision,features=features,classifier='deterministic_v1',confidence=None,sha256=meta['sha256']))
    countries=set(c['gdelt_countries']); seen=set()
    for path in sorted((out/'data/news/gdelt').glob('*.zip')):
        try:
            with zipfile.ZipFile(path) as z:
                raw=pd.read_csv(BytesIO(z.read(z.namelist()[0])),sep='\t',header=None,dtype=str,keep_default_na=False)
        except (zipfile.BadZipFile,pd.errors.EmptyDataError) as e:
            coverage.append(dict(snapshot=path.stem,rows=0,retained=0,status=str(e))); continue
        retained=0
        if raw.shape[1]!=61: raise ValueError(f'Unexpected GDELT schema {path}: {raw.shape}')
        sampled=pd.to_datetime(path.stem,format='%Y%m%d%H%M%S',utc=True)
        records.append(dict(identifier='coverage_'+path.stem,source='Archive coverage marker',
            availability_timestamp=(sampled+pd.Timedelta(c['news_delay_hours'],unit='h')).isoformat(),
            snapshot=path.stem,record_type='coverage_not_news',features={n:0 for n in NEWS}))
        for _,r in raw.iterrows():
            if not ({r[7],r[17]} & countries): continue
            root=r[28].zfill(2); code=r[26]
            conflict=root in ('18','19','20') or code.startswith('163')
            if not conflict: continue
            url=r[60]; identity=url or r[0]
            if identity in seen: continue
            seen.add(identity)
            ingested=pd.to_datetime(r[59],format='%Y%m%d%H%M%S',utc=True)
            sampled=pd.to_datetime(path.stem,format='%Y%m%d%H%M%S',utc=True)
            if ingested>sampled+pd.Timedelta(15,unit='min'): raise ValueError('GDELT record later than claimed snapshot')
            records.append(dict(identifier='gdelt_'+r[0],publication_timestamp=None,publication_date=None,
                availability_timestamp=(max(ingested,sampled)+pd.Timedelta(c['news_delay_hours'],unit='h')).isoformat(),
                source='GDELT 2.0 Events fixed archive sample',url=url,
                headline=None,text=None,event_description=f'CAMEO {code}: {r[6]} / {r[16]}',
                entities=[r[6],r[16]],countries=[r[7],r[17]],categories=['conflict_or_sanction'],
                precision='ingestion timestamp; original publication unknown',event_date=r[1],ingestion_timestamp=str(ingested),snapshot=path.stem,
                features={n:float(n=='conflict_attention') for n in NEWS},classifier='deterministic_cameo_v1',confidence=None))
            retained+=1
        coverage.append(dict(snapshot=path.stem,rows=len(raw),retained=retained,status='ok'))
    manifest=json.loads((out/'logs/download_manifest.json').read_text())
    for failure in manifest:
        if 'error' in failure and '/gdelt/' in failure['path']:
            coverage.append(dict(snapshot=Path(failure['path']).stem,rows=None,retained=None,status='MISSING ARCHIVE: '+failure['error']))
    records.sort(key=lambda r:r['availability_timestamp'])
    for r in records:
        if r['source']=='Archive coverage marker': continue
        features=r['features']; factors=[]; channels=[]; directions={}
        if features['policy_direction']:
            factors+=['interest_rates']; channels+=['financing_costs','discount_rates']
            directions['policy_rate']='positive' if features['policy_direction']>0 else 'negative'
        if features['inflation_attention']:
            factors+=['inflation']; channels+=['input_costs','purchasing_power']
        if features['growth_weakness']:
            factors+=['growth']; channels+=['aggregate_demand']; directions['growth']='negative'
        if features['trade_attention']:
            factors+=['trade']; channels+=['cross_border_costs','supply_chains']
        if features['conflict_attention']:
            factors+=['geopolitical_risk','energy']; channels+=['supply_disruption','risk_aversion']
        r.update(affected_factors=sorted(set(factors)),economic_channels=sorted(set(channels)),
            expected_directions=directions,time_horizon='weeks_to_months: heuristic, not calibrated',
            affected_companies=[],affected_industries=[],model_identifier=None,
            interpretation='deterministic mechanism proxy; not LLM output, probability or market surprise')
    with (out/'data/news/records.jsonl').open('w') as f:
        for record in records: f.write(json.dumps(record)+'\n')
    pd.DataFrame(coverage).to_csv(out/'data/news/coverage.csv',index=False)
    (out/'data/news/llm_status.json').write_text(json.dumps({'status':'NOT_RUN','reason':'No configured LLM provider credential or repository interface','classifications':0,'model_identifier':None,'deterministic_records':len(records)},indent=2))
    return records


def news_vector(records,origin,lookback_days=35):
    cutoff=pd.Timestamp(origin).tz_localize('America/New_York')+pd.Timedelta(16,unit='h')
    selected=[r for r in records if cutoff-pd.Timedelta(lookback_days,unit='D')<=pd.Timestamp(r['availability_timestamp'])<=cutoff]
    fed=[r for r in selected if r['source'].startswith('Federal')]
    result=np.zeros(len(NEWS))
    for r in fed: result+=np.array([r['features'][n] for n in NEWS])
    if fed: result/=len(fed)
    gdelt=[r for r in selected if r['source'].startswith('GDELT')]
    # Fixed nonlinear compression of sampled-event volume, NOT event probability.
    observed=sum(r['source']=='Archive coverage marker' for r in selected)
    if observed==0:
        raise ValueError(f'No archived news coverage before {origin}; do not substitute zero events')
    result[-1]+=np.log1p(len(gdelt)/observed)/5
    return result,[r['identifier'] for r in selected]


def ridge_posterior(x,y,penalty=6.,prior=None):
    """Gaussian ridge posterior, with optional tempered cross-regime distribution.

    Columns must have fixed prespecified units/scales. No unregularized intercept:
    alpha is a separately regularized constant column supplied by the caller.
    """
    x=np.asarray(x,float); y=np.asarray(y,float)
    if y.ndim==1: y=y[:,None]
    p=x.shape[1]; n_assets=y.shape[1]
    means=np.zeros((p,n_assets)); covs=np.zeros((n_assets,p,p)); sigmas=np.maximum(np.var(y,axis=0,ddof=1),1e-4)
    for i in range(n_assets):
        precision=np.eye(p)*penalty; rhs=x.T@y[:,i]
        if prior is not None and prior.get('available',np.ones(n_assets,dtype=bool))[i]:
            # Window A posterior uncertainty is weakened and max precision capped.
            vals,vec=np.linalg.eigh(prior['covariances'][i]*4)
            extra=vec@np.diag(np.minimum(sigmas[i]/np.maximum(vals,1e-8),6.))@vec.T
            precision+=extra; rhs+=extra@prior['mean'][:,i]
        inv=np.linalg.inv(x.T@x+precision)
        means[:,i]=inv@rhs; covs[i]=sigmas[i]*inv
    return dict(mean=means,covariances=covs,residual_variance=sigmas,n=len(y))


def ar_forecast(changes,horizon=21,penalty=12.):
    """Zero-intercept ridge AR(1) on completed monthly changes, not market surprise."""
    changes=np.asarray(changes,float)
    if len(changes)<3: return np.zeros(changes.shape[1])
    lag,now=changes[:-1],changes[1:]
    scale=np.maximum(np.sqrt(np.mean(lag**2,axis=0)),1e-8)
    a=((lag/scale)*(now/scale)).sum(axis=0)/(((lag/scale)**2).sum(axis=0)+penalty)
    a=np.clip(a,-.8,.8)
    return a*changes[-1]*(horizon/21)


def block_difference_ci(values,seed=20261019,replicates=2000,block=3):
    """Circular moving blocks of whole market-month paired differences."""
    values=np.asarray(values,float); n=len(values); rng=np.random.default_rng(seed)
    if n<2: return (float(values.mean()),np.nan,np.nan)
    starts=rng.integers(0,n,(replicates,int(np.ceil(n/block))))
    indices=(starts[:,:,None]+np.arange(block))%n
    means=values[indices.reshape(replicates,-1)[:,:n]].mean(axis=1)
    return float(values.mean()),*map(float,np.quantile(means,[.025,.975]))
