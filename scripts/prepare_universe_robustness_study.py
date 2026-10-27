#!/usr/bin/env python3
"""Freeze design and publish an honest data-readiness report; never run survivor draws."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.metadata,json,sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import pandas as pd
from stock_api.universe_robustness import SECTOR_QUOTAS,benchmark_weights
from stock_api.portfolio import joint_gp,optimize_weights,trade
from portfolio_gp_report import table

OUT=ROOT/'experiments/portfolio_gp_universe_robustness_2026-10-19'
PRIOR=ROOT/'experiments/portfolio_gp_macro_news_growth20_2026-10-19'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze_config():
    c=dict(study=OUT.name,seed=20261019,model='unchanged price-only estimated-drift heteroskedastic GP',
        windows={'A':dict(selection='2019-07-16T16:00:00-04:00',gp_history_reset='2019-01-01',evaluation_start='2019-07-17',end='2023-01-31'),
                 'B':dict(selection='2025-07-16T16:00:00-04:00',gp_history_reset='2025-01-01',evaluation_start='2025-07-17',end='2026-10-16')},
        groups={'A':dict(name='Broad US market',portfolios=10,stocks=20,sector_quotas=SECTOR_QUOTAS),
                'B':dict(name='Technology',portfolios=10,stocks=20),
                'C':dict(name='Historical momentum top quintile',portfolios=10,stocks=20)},
        sampling='SeedSequence([20261019, window_index, group_index, portfolio_index]); sorted permanent security IDs; independent draws without replacement within portfolio; overlap allowed between portfolios',
        parent_universe='Historical major-US-exchange USD operating-company common equities/ADRs, including subsequently delisted securities; historical sector classification required. Public S&P lists are audit proxies, not silently substituted study universes.',
        eligibility=dict(minimum_price_exclusive=10.,price_units='actual historical unadjusted share price, not current split-adjusted close',minimum_median_dollar_volume=10_000_000.,liquidity_sessions=63,minimum_prior_price_observations=253,share_classes='one per issuer, highest trailing 63-session median dollar volume, stable security ID tie break',exclude='ETFs, preferreds, warrants, SPAC/blank-check companies, OTC-only securities'),
        momentum=dict(long_lag=252,skip_lag=21,definition='log(P[t-21]/P[t-252]) in total-return price units',selection_pool='top ceil(0.2*N) eligible stocks; stable ID tie break'),
        gp_lookback=30,horizon='exact sessions from origin through next rebalance preceding close; final period through cutoff close',
        horizon_clarification='Task mentions 30-day forecast horizon, but preserved prior implementation uses a 30-session lookback and calendar monthly horizons. No numerical-model change.',
        paths=10000,lambda_=10.,cap=.2,initial_dollars=2000.,cost_bps=10.,correlation_shrinkage=.5,
        gp_seed='20261019 + YYYYMMDD(origin); cache key additionally includes ordered permanent security IDs, prices, core hashes and horizon',
        execution='previous close information; adjusted open first trading day on/after 17th; next scheduled adjusted open exit, final cutoff adjusted close',
        benchmarks=dict(equal='1/N',inverse_volatility='inverse sample SD of 63 trailing daily log returns; capped proportional water filling',momentum='cross-sectional average rank of 12-2 momentum, ascending positive ranks; capped proportional water filling',spy='buy and hold, same dates and entry fee, exemption from stock cap'),
        corporate_actions='No replacements. Preserve verified losses/proceeds. Corporate cash held until next scheduled rebalance, then reallocated among tradable original constituents. Share consideration needs actual documented conversion and liquidation accounting. No assumed last-close liquidation, perpetual stale marks or pre-listing fills. Stop on unknown terminal outcomes; stop if remaining tradable original names cannot satisfy 20% cap.',
        statistics=dict(bootstrap_replicates=2000,block_holding_months=3,resampling='one synchronized block-index stream per window across all portfolio IDs/groups/benchmarks',interpretation='conditional on observed allocations and fixed universes; not a refit bootstrap; portfolios/stock-months are not independent evidence',primary='GP minus equal-weight cumulative net return and win fraction',interval=.95),
        failure_policy='Missing candidate eligibility history, sector provenance, identifiers or terminal outcomes blocks dependent sampling/backtests; never exclude/redraw based on future performance or data availability',
        core_hashes={p:sha(ROOT/p) for p in ['stock_api/heteroskedastic.py','stock_api/portfolio.py','scripts/run_portfolio_gp_22.py']},
        estimated_compute=dict(joint_forecasts=1740,univariate_GP_fits_without_deduplication=34800,simulated_joint_paths=17_400_000),
        forbidden='macro/news variants, GP tuning, reducing paths, current survivor selection, outcome-based redraws')
    c['lambda']=c.pop('lambda_')
    path=OUT/'config.yaml'
    if path.exists() and json.loads(path.read_text())!=c: raise ValueError('Frozen design changed; use an explicitly versioned new study')
    path.write_text(json.dumps(c,indent=2)+'\n')
    return c

def synthetic_smoke(c):
    """Numerical integration only: generated inputs are never experimental observations."""
    path=OUT/'logs/synthetic_smoke.json'
    fingerprint=hashlib.sha256(json.dumps(c['core_hashes'],sort_keys=True).encode()).hexdigest()
    if path.exists() and json.loads(path.read_text())['core_fingerprint']==fingerprint: return json.loads(path.read_text())
    rng=np.random.default_rng(c['seed']); market=rng.normal(0,.007,(30,1)); returns=market+rng.normal(0,.012,(30,20))
    model=joint_gp(returns,21,mean_gp=True,n_paths=10000,seed=c['seed'],shrinkage=.5)
    w,info=optimize_weights(model['mu'],model['covariance'],np.zeros(20),risk_aversion=10,cost_rate=.001,cap=.2)
    dollars,fee,buy,sell=trade(2000,np.zeros(20),w,.001)
    assert model['samples'].shape==(10000,20) and all(p['converged'] for p in model['parameters'])
    assert np.linalg.eigvalsh(model['covariance']).min()>0 and abs(w.sum()-1)<1e-8 and w.min()>=0 and w.max()<=.2+1e-8
    assert abs(dollars.sum()+fee-2000)<1e-8 and abs(fee-.001*(buy+sell))<1e-8
    synthetic_prices=100*np.exp(np.cumsum(rng.normal(0,.01,(253,20)),axis=0))
    for name in ['equal','inverse_volatility','momentum']:
        b=benchmark_weights(synthetic_prices,name); assert abs(b.sum()-1)<1e-8 and b.min()>=0 and b.max()<=.2+1e-8
    result=dict(status='PASS',classification='SYNTHETIC NUMERICAL INTEGRATION TEST ONLY; NOT A PORTFOLIO-WINDOW EVALUATION',assets=20,paths=10000,horizon=21,unchanged_GP=True,all_GP_fits_converged=True,optimizer_success=info['optimizer_success'],core_fingerprint=fingerprint,checks=['PSD','sum weights','nonnegative','cap','self-financing fees','benchmark allocations'])
    path.write_text(json.dumps(result,indent=2)); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--skip-numerical-smoke',action='store_true'); args=ap.parse_args()
    for folder in ['logs','figures','data','source']: (OUT/folder).mkdir(parents=True,exist_ok=True)
    old_hash_path=OUT/'logs/original_growth_hashes.json'
    old_hashes=json.loads(old_hash_path.read_text()) if old_hash_path.exists() else {str(p.relative_to(PRIOR)):sha(p) for p in PRIOR.rglob('*') if p.is_file()}
    if not old_hash_path.exists(): old_hash_path.write_text(json.dumps(old_hashes,indent=2))
    c=freeze_config(); source=OUT/'data/source_audit'
    history=pd.read_csv(source/'historical_membership_csv.body')
    assert history.date.is_monotonic_increasing
    raw_caches=[ROOT/'reports/monthly-stock-screen-2026-10-18/data',PRIOR/'data/prices/raw']
    rows=[]; audit_rows=[]
    for window,w in c['windows'].items():
        day=w['selection'][:10]; snapshot=history[history.date<=day].iloc[-1]
        tickers=snapshot.tickers.split(','); assert len(tickers)==len(set(tickers))
        for ticker in tickers:
            symbol=ticker.replace('.','-'); paths=[folder/(symbol+'.csv') for folder in raw_caches]; cached=next((p for p in paths if p.exists()),None)
            available=0; checksum=None
            if cached is not None:
                meta=json.loads(cached.with_suffix('.json').read_text()); checksum=sha(cached); assert checksum==meta['sha256']
                f=pd.read_csv(cached,index_col='Date',parse_dates=True); available=int((f.index<=day).sum())
            rows.append(dict(window=window,selection_date=day,reconstructed_membership_date=snapshot.date,ticker=ticker,
                cached_direct_symbol_history=cached is not None,prior_rows=available,at_least_253_prior_rows=available>=253,
                security_identity_verified=False,historical_raw_price_convention_verified=False,
                source_file=str(cached.relative_to(ROOT)) if cached else None,source_sha256=checksum))
        g=pd.DataFrame(rows); g=g[g.window==window]
        audit_rows.append(dict(window=window,selection_date=day,reconstructed_membership_date=snapshot.date,members=len(tickers),direct_symbol_cached=int(g.cached_direct_symbol_history.sum()),not_cached=int((~g.cached_direct_symbol_history).sum()),cached_253_prior_rows=int(g.at_least_253_prior_rows.sum()),verified_study_universe=False))
    coverage=pd.DataFrame(rows); coverage.to_csv(OUT/'data/historical_membership_cache_audit.csv',index=False)
    audit_frame=pd.DataFrame(audit_rows); audit_frame.to_csv(OUT/'data/availability_summary.csv',index=False)
    # All planned rows remain explicitly BLOCKED. NaN means not evaluated, never 0% return.
    planned=[dict(window=window,group=group,portfolio_id=f'{group}{i:02d}',status='BLOCKED_DATA',gp_net_return=np.nan,equal_net_return=np.nan,inverse_volatility_net_return=np.nan,momentum_net_return=np.nan,spy_net_return=np.nan,gp_minus_equal_pp=np.nan) for window in ['A','B'] for group in ['A','B','C'] for i in range(1,11)]
    pd.DataFrame(planned).to_csv(OUT/'portfolio_results.csv',index=False)
    summary=pd.DataFrame([dict(window=window,group=group,group_name=c['groups'][group]['name'],planned=10,completed=0,median_gp_excess_pp=np.nan,mean_gp_excess_pp=np.nan,gp_win_fraction=np.nan,median_sharpe_difference=np.nan,median_drawdown_difference=np.nan,excess_standard_deviation_pp=np.nan,paired_lower95=np.nan,paired_upper95=np.nan,status='BLOCKED_DATA') for window in ['A','B'] for group in ['A','B','C']])
    summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(columns=['portfolio_id','window','ticker','security_id','group','selection_date','sector','historical_price','median_dollar_volume_63','momentum_12_2','membership_available_at']).to_csv(OUT/'universes.csv',index=False)
    pd.DataFrame([dict(window=r['window'],group=r['group'],portfolio_id=r['portfolio_id'],coverage90=np.nan,standardized_residual_rms=np.nan,crps=np.nan,status='BLOCKED_DATA') for r in planned]).to_csv(OUT/'forecast_calibration.csv',index=False)
    smoke=None if args.skip_numerical_smoke else synthetic_smoke(c)
    # Audit figure only; no invented return/calibration distributions.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(2)
    ax.barh(x,audit_frame.direct_symbol_cached,label='Ticker-matching cached file (identity unverified)')
    ax.barh(x,audit_frame.not_cached,left=audit_frame.direct_symbol_cached,label='No ticker-matching cached file',color='tab:red')
    ax.set_yticks(x,['Before July 2019','Before July 2025']); ax.set_xlabel('Securities in public historical S&P 500 reconstruction'); ax.set_title('Data-readiness audit — not performance results'); ax.legend(fontsize=8,loc='lower right')
    for i,r in enumerate(audit_frame.itertuples()): ax.text(r.members+3,i,f'{r.not_cached} missing / {r.members}',va='center',fontsize=9)
    ax.set_xlim(0,650); fig.tight_layout(); fig.savefig(OUT/'figures/data_availability_audit.png',dpi=200); fig.savefig(OUT/'figures/data_availability_audit.pdf'); plt.close(fig)
    probes=json.loads((OUT/'logs/delisted_price_probes.json').read_text())
    snapshot2019=json.loads((source/'github_membership_2019.body').read_text())[0]['commit']['committer']['date']
    snapshot2025=json.loads((source/'github_membership_2025.body').read_text())[0]['commit']['committer']['date']
    actual_iwb=(source/'iwb_dated_latest_probe.body').read_text(encoding='utf-8-sig').splitlines()[1]
    reasons=['No verified historical US-listed security master with sector history and permanent identifiers is configured.',
        'Historical candidate prices and terminal outcomes are missing for disappeared securities; Yahoo probes CELG/ABMD/TWTR/SIVB each returned zero rows.',
        'Current cache was selected from October 2026 holdings and price thresholds; filtering to it would reintroduce survivorship and future-price selection.',
        'No authenticated historical/delisted market-data access found. Nasdaq anonymous probe returned HTTP 429; paid-data availability and entitlement not established.']
    status=dict(status='BLOCKED_DATA',historical_portfolio_windows_completed=0,planned=60,universes_drawn=0,gp_backtest_fits=0,synthetic_integration=smoke,blockers=reasons,
        remaining_implementation='Provider-specific normalization, verified delisting/merger cash/share accounting and the full multi-portfolio execution/figure driver remain necessary once inputs can be validated. The preparation command is an audit resume, not a completed backtest runner.')
    (OUT/'logs/status.json').write_text(json.dumps(status,indent=2))
    provider_contract='''# Required historical data contract

Supply a licensed export or configured provider covering June 2018–January 2023 and June 2024–October 16, 2026. A provider name or current constituent list alone is insufficient.

1. Historical parent-universe membership and sector histories with permanent security/issuer IDs, historical ticker mappings, exchange/security type, first-trade date, effective dates and known-at timestamps. Membership must include later-disappearing companies. GICS-like sector quotas need classifications actually valid at selection; a provider's current sector metadata is not enough.
2. Daily quotes and total-return data for every historically eligible candidate, not just survivors or sampled winners. Eligibility requires actual contemporaneous raw close and volume. Current split-adjusted Yahoo Close cannot automatically establish a historical $10 price threshold. Momentum uses total-return price indices at t−252 and t−21; selection cannot use any later prices. Known IPO dates can justify insufficient-history exclusions; missing files cannot.
3. Full daily evaluation paths for sampled securities, including halts, ticker changes, distributions, splits, merger conversion terms and proceeds, delisting returns and bankruptcy recoveries. Permanent IDs must distinguish ticker reuse. Missing terminal values cannot be replaced by the last quote or a zero return. Share consideration needs a documented liquidation convention; proceeds remain cash until the next rebalance. Do not double-count delisting returns already embedded in a vendor total-return field.
4. SPY on the identical execution/calendar convention, source metadata, timestamps, hashes, and provider adjustment definitions. Independent source validation is required before describing the sample as survivorship-bias-free.

The pure `eligible_universe` adapter expects membership columns `security_id, issuer_id, ticker, sector, available_at, first_trade`, prices keyed by permanent security ID with `raw_close, raw_volume, total_return_close`, and an exchange-session calendar. This is the pre-selection interface only; it is not a completed provider adapter or corporate-action accounting engine. After the data gate passes, freeze universe draws before reading evaluation outcomes. Preserve every draw even if its later outcome is catastrophic.

Potential routes: CRSP/WRDS equity data plus historical GICS/sector history; or an appropriate Norgate licensed export plus verified historical sector classifications. Sharadar SEP documents active/delisted prices, but access, adjustment/terminal-return conventions and point-in-time sector coverage still need validation. No subscription was purchased and no account was created.
'''
    (OUT/'DATA_REQUIREMENTS.md').write_text(provider_contract)
    readme='''# Monte Carlo portfolio-universe robustness study

Primary objective: test whether the unchanged price-only heteroskedastic GP improves allocations over equal weight across historically selected universes. Intended design: 10 broad sector-quota samples, 10 technology samples, 10 historical momentum-quintile samples; 20 distinct issuers each; repeat independently in both historical windows. Seed 20261019. Overlapping samples are not independent market replications.

**Current status: BLOCKED_DATA, 0/60 historical portfolio-window evaluations.** See RESULTS.md. No survivor-only fallback or performance-based redraws were run. Output rows marked BLOCKED_DATA are planned slots with missing numerical results, not zero returns. `universes.csv` has only headers because no valid draws were made. `data/historical_membership_cache_audit.csv` is a source audit, not the selected study universe.

Implemented: frozen configuration, source downloads and provenance audit, strict point-in-time eligibility, reproducible stratified/technology/momentum sampling, capped benchmark weights, synchronized paired block resampling, unit tests and an unchanged-GP synthetic numerical integration test. Missing: validated provider data/adapter, terminal corporate-action accounting, full multi-portfolio backtest/analysis driver, and historical results. The five requested performance figures cannot be generated from unobserved outcomes; only a clearly labeled data-availability figure is supplied.

The old implementation uses 30 trading days of lookback, exact monthly-session forecast horizons, and 10,000 paths. `config.yaml` resolves the task's horizon wording in favor of preserving the actual model. Kernel/priors/optimizer/lambda/costs/cap remain unchanged. Benchmark history is longer only for ex-ante momentum and realized-volatility inputs; GP histories reset per window.

Execution and resumption of the data audit:

```bash
.conda-env/bin/python scripts/audit_universe_robustness_data.py
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/prepare_universe_robustness_study.py
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python -m pytest tests/test_universe_robustness.py tests/test_portfolio.py -q
```

The first command needs network access; successful source responses and delisted probes are cached. Remove/rearchive selected probe metadata deliberately to refresh it after changing provider access. These are audit/test commands, not a claim that adding a key launches 60 backtests. Resolve DATA_REQUIREMENTS.md, implement/validate the provider and lifecycle adapter, then run a genuine representative historical portfolio before the full executor. All former experiment outputs remain untouched.
'''
    (OUT/'README.md').write_text(readme)
    display=summary[['window','group_name','planned','completed']].copy(); display['GP win fraction']='Not estimable'; display['Median excess']='Not estimable'
    report=table(display)+f'''

# Universe robustness study: execution blocked by historical-data gaps

**No scientific return results are available: 0/60 portfolio-window evaluations, 0 valid universe draws.** This is a data-readiness result, not a failed GP-performance result. Missing return/win-fraction fields in the CSVs are blank, not zero. Execution date (UTC): {datetime.now(timezone.utc).date()}; directory retains the requested study identifier.

## What was executed and observed

Inspected the growth20 experiment, its configuration, `stock_api/heteroskedastic.py`, `stock_api/portfolio.py`, shared monthly accounting, price ingestion and cache provenance. The original GP is reusable. The existing stock cache was sampled from **October 2026 IWB holdings**, with current-price restrictions, so it is not an admissible historical sampling universe. No macro/news models were run.

Downloaded a public historical S&P 500 membership reconstruction as a diagnostic and checked direct-symbol cache coverage before either selection date:

{table(audit_frame[['window','selection_date','reconstructed_membership_date','members','direct_symbol_cached','not_cached','cached_253_prior_rows']])}

These counts do **not** establish a verified broad-US universe. They measure exact-symbol cache presence only. Missing symbols include renamed firms and absent downloads as well as delisted companies; cached symbols can have issuer-identity ambiguity. The historical S&P reconstruction is a large-cap proxy, lacks historical sector fields and is not a complete US-listed security master. It was not silently substituted for the required universe. No selection or momentum ranking was restricted to the available files.

![Availability audit, not a backtest](figures/data_availability_audit.png)

Provider/source probes:

- Historical iShares IWB AJAX URLs returned HTML rather than historical holdings. The `latest-holdings.csv?asOfDate=20190716` request returned **{actual_iwb}**, not July 2019. Its response was rejected as a historical source, despite HTTP 200.
- The most recent `datasets/s-and-p-500-companies` commit on the queried path before July 2019 was **{snapshot2019}**, too stale to stand in for July 2019 sector/membership data. The analogous July 2025 snapshot was **{snapshot2025}**; this is useful partial historical information, but does not provide delisted prices, terminal payoffs or the all-US universe.
- Yahoo/yfinance probes of **{', '.join(r['ticker'] for r in probes)}** over June 2018–January 2023 each returned **zero rows** with `YFTzMissingError`. Thus even pre-selection historical observations can disappear when a security later delists. SIVB's disappearance after Window A does not justify dropping its earlier history. These probes were data checks, not outcome-selected study portfolios.
- The unauthenticated Sharadar SEP request returned **HTTP 429** (anonymous limit); no configured market-data credentials or local WRDS credential files were found. No successful licensed access was established. This does not prove no paid source exists; it establishes that the required data are unavailable in this environment.
- A **synthetic-only numerical smoke test** exercised the unchanged heteroskedastic GP on 20 generated return series with 10,000 paths, correlated covariance, allocation constraints, fees and all three stock benchmarks. Status: **{smoke['status'] if smoke else 'NOT RUN'}**. Generated inputs are test fixtures, not stock observations, and contribute zero experimental portfolio results. A representative historical portfolio cannot yet pass the data gate.

Source URLs, HTTP statuses, retrieval times, raw responses and SHA256 hashes are preserved under `data/source_audit/` and `logs/`. The current-cache historical-price convention also needs correction/validation before a historical $10 share-price filter: split-adjusted close is not automatically the actual price quoted at that historical date.

## Frozen design and implemented checks

Selection dates are July 16, 2019 and July 16, 2025, before next-day execution. Ten draws per group use independent reproducible seed streams; no within-portfolio duplicate issuer; overlap across portfolios allowed. Broad quotas total 20 exactly. Eligibility: historical raw price >$10, median 63-session dollar volume ≥$10 million, and 253 prior observations so all portfolios support the momentum benchmark. 12–2 momentum is log(P[t−21]/P[t−252]); top ceil(20% × eligible N) forms Group C. Known listing dates can exclude short-history IPOs; unobserved history fails the pool instead of filtering survivors. Share-class choice uses historical liquidity only.

The GP remains estimated drift, unchanged kernels/priors, 30-session **lookback**, 10,000 paths, lambda 10, cap 20%, $2,000 and 10 bp per dollar bought/sold. The task mentions a 30-session forecast horizon, but the prior implementation actually forecasts the exact monthly holding sessions; preserving that implementation avoids an unannounced model change. Equal weight, 63-session inverse volatility and positive cross-sectional momentum-rank weights share capped allocations and the same trading rules. SPY is exempt from the stock cap. Corporate-action losses/proceeds cannot be silently replaced by flat prices or skipped names.

`stock_api/universe_robustness.py` implements strict eligibility, sector/random sampling, capped benchmark mapping and synchronized paired block uncertainty. The sampler was tested for all 60 planned IDs using clearly synthetic candidates. Additional tests cover future-data invariance, unavailable-history rejection, quota/cap compliance, and preservation of uncertainty when an identical portfolio is repeated 30 times. Duplicating a portfolio must not manufacture 30 independent observations.

Planned inference: shared three-month circular blocks across all portfolios/groups/benchmarks in each historical window, 2,000 replicates. Portfolio-overlap and common market shocks stay paired. Cross-universe dispersion is distinct from temporal uncertainty; only two historical regimes exist. Bootstrap intervals condition on observed weights and are not full strategy-refitting confidence guarantees. Neither this design nor partial data availability warrants a claim of complete elimination of survivorship bias.

## Scientific questions: current answers

1. **Majority of universes beating equal weight:** not estimable. GP win fractions against equal weight, inverse volatility, momentum and SPY are all **N/A (0 evaluated)**.
2. **Consistency across broad, technology and momentum groups:** not tested; all six group/window cells are blocked.
3. **Risk-adjusted versus absolute-return benefit:** not tested. No Sharpe, drawdown, volatility or transaction-cost improvement is inferred from missing results.
4. **Persistence across regimes:** not tested. Two windows cannot be replaced with successful contemporary survivors.
5. **Comparison with the hindsight growth universe:** its prior reported gains remain a biased reference, not evidence of universe robustness. No new Monte Carlo point is available to compare with it. The original files remain unchanged.
6. **Predictive calibration:** not evaluated on the requested random portfolios. Synthetic GP smoke-test validity is not empirical forecast calibration.
7. **Exceptional stocks, portfolios or months driving gains:** unassessed because no study portfolios were evaluated. Dropping missing names would itself condition results on survival/data availability.
8. **Economically meaningful net advantage after costs:** unassessed. Fees and cap logic passed software checks, but these do not establish financial efficacy.

**Assessment: this study has not demonstrated robust GP allocation value independent of retrospective selection.** It also has not demonstrated absence of value; the required test remains unperformed because its historical inputs are incomplete. Running 60 survivor-filtered backtests would produce numbers without answering the stated question.

## Sources and precise path to completion

- [Historical membership reconstruction](https://github.com/fja05680/sp500): versioned raw snapshot cached, membership-only diagnostic with documented reconstruction limitations. [Dated constituent snapshots](https://github.com/datasets/s-and-p-500-companies) were fetched at explicit commits; current rows were never treated as historical.
- [CRSP database documentation](https://www.crsp.org/research/crsp-us-stock-databases/) and [permanent identifiers](https://www.crsp.org/research/permno/) describe historical security tracking. [CRSP delisting definitions](https://www.crsp.org/crsp_pdf/crsp-us-stock-indexes-databases-data-descriptions-guide-crspaccess/) distinguish terminal returns from ordinary last-price observations. Access plus historical sector classifications is needed, not merely a ticker list.
- [Norgate data contents](https://norgatedata.com/data-content-tables.php) documents historical constituent and delisted-security coverage; [accessibility](https://norgatedata.com/accessibility.php) documents its integrations. A licensed export is a possible route on this Mac, but historical sector availability and terminal accounting still require verification. No subscription was purchased.
- [Sharadar SEP documentation](https://data.nasdaq.com/databases/SEP/documentation?anchor=sample-data) describes active/delisted price coverage. The observed unauthenticated API request was not usable; sector/identifier/terminal data cannot be inferred from the SEP name alone.

Resolve [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md): provide historical security/sector membership and identity maps, pre-selection raw prices/volumes and total returns for the full candidate pool, and complete evaluation paths/corporate-action settlements for sampled securities. Then implement and validate the provider/lifecycle adapter, freeze all draws, run a representative historical smoke test, and execute the 60 evaluations with per-portfolio checkpoints. The 1,740 joint monthly forecasts represent about 34,800 univariate fits without deduplication; computation is not the current blocker. No model simplification is proposed.

Audit-resume and test commands are in [README.md](README.md). They do not claim to be a finished backtest executor: provider normalization, terminal-event accounting, full portfolio execution and the five requested performance figures remain outstanding. `portfolio_results.csv` and `forecast_calibration.csv` contain explicitly blocked planned slots; `universes.csv` is intentionally empty. No outcomes, plot distributions, delisting values or win fractions have been invented.
'''
    (OUT/'RESULTS.md').write_text(report)
    assert all(sha(PRIOR/p)==h for p,h in old_hashes.items()),'Original growth outputs changed'
    (OUT/'logs/original_outputs_preserved.json').write_text(json.dumps(dict(status='PASS',files=len(old_hashes)),indent=2))
    files=['stock_api/universe_robustness.py','scripts/audit_universe_robustness_data.py','scripts/prepare_universe_robustness_study.py','tests/test_universe_robustness.py']
    import shutil
    for p in files:
        target=OUT/'source'/p; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/p,target)
    (OUT/'logs/manifest.json').write_text(json.dumps(dict(status='BLOCKED_DATA',utc=datetime.now(timezone.utc).isoformat(),python=sys.version,versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','yfinance','matplotlib']},config_sha256=sha(OUT/'config.yaml'),sources={p:sha(ROOT/p) for p in files},core_hashes=c['core_hashes'],command='MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/prepare_universe_robustness_study.py'),indent=2))
    print(json.dumps(status,indent=2)); print('Report:',OUT/'RESULTS.md')
    return 2

if __name__=='__main__': sys.exit(main())
