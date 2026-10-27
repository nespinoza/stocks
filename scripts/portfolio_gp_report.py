"""Numerical report generation for the frozen portfolio experiment."""
import numpy as np
import pandas as pd


def table(frame,formats=None):
    formats=formats or {}
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(formats.get(col,str)(value) for col,value in row.items())+' |')
    return '\n'.join(lines)


def write_report(out,c,primary,comparison,monthly,weights,risk,perturb,diag,ranking,sensitivity,forecast):
    p=comparison.set_index('strategy'); gp=p.loc[primary[0]]; eq=p.loc['Equal weight']; spy=p.loc['SPY']; zero=p.loc['Zero-drift GP']
    d=diag[(diag.lookback==30)&(diag.ticker=='ALL')].set_index('variant'); estimated=d.loc['estimated_drift']; null=d.loc['same_covariance_zero_drift']
    gm=monthly[monthly.strategy==primary[0]]; gw=weights[weights.strategy==primary[0]]; gr=risk[risk.strategy==primary[0]]
    cap_cells=int((gw.weight>=.2-1e-6).sum()); cap_months=int((gm.cap_count>0).sum())
    coverage=((gr.realized_gross_log_return>=gr.q05)&(gr.realized_gross_log_return<=gr.q95)).mean()
    start=gm.start_equity.to_numpy(); changes=gm.end_equity.to_numpy()-start
    top=np.sort(changes)[-2:].sum(); net_gain=gp.net_ending_dollars-c['initial_dollars']
    matched_sensitivity=sensitivity[(sensitivity.risk_aversion==10)&(sensitivity.cost_bps==10)].set_index('lookback')
    opening=pd.read_csv(out/'data'/'adjusted_open.csv',index_col=0); integrity=pd.read_csv(out/'data_integrity.csv')
    par=pd.read_csv(out/'gp_parameter_history.csv'); schedule=pd.read_csv(out/'schedule.csv')
    headline=pd.DataFrame([{'Strategy':name,'Six-month return':f'{p.loc[name,"net_return"]:.2%}','Ending $':f'{p.loc[name,"net_ending_dollars"]:,.2f}',
        'Sharpe':f'{p.loc[name,"net_sharpe"]:.2f}','Max drawdown':f'{p.loc[name,"net_max_drawdown"]:.2%}','Costs':f'${p.loc[name,"costs"]:.2f}'} for name in primary])
    performance=pd.DataFrame([{'Strategy':name,'Gross return':f'{p.loc[name,"gross_return"]:.2%}','Net return':f'{p.loc[name,"net_return"]:.2%}',
        'Ann. return*':f'{p.loc[name,"net_annualized_return"]:.2%}','Ann. vol.':f'{p.loc[name,"net_annualized_volatility"]:.2%}',
        'Best month':f'{p.loc[name,"best_month"]:.2%}','Worst month':f'{p.loc[name,"worst_month"]:.2%}','Positive months':f'{p.loc[name,"positive_month_fraction"]:.0%}'} for name in comparison.strategy[:7]])
    mt=gm[['execution_date','net_return','gross_return','cost','two_sided_turnover','cap_count','effective_holdings']].copy()
    mt.columns=['Rebalance','Net return','Gross return','Cost $','Two-sided turnover','At cap','Effective holdings']
    dt=diag[(diag.ticker=='ALL')][['lookback','variant','mae','rmse','coverage90','coverage68','crps','rms_z','realized_to_predicted_variance']].copy()
    st=sensitivity[['lookback','risk_aversion','cost_bps','net_return','net_sharpe','net_max_drawdown','costs']].copy()
    rankingtable=ranking[['rank','ticker','mae','rmse','coverage90']].copy()
    verdict=lambda a,b:'outperformed' if a>b else 'underperformed'
    text=table(headline)+f'''

# 22-stock heteroskedastic GP portfolio experiment

Completed October 18, 2026. All headline returns and Sharpe ratios are **net**. Initial wealth is $2,000; fractional shares, zero risk-free rate, and 10 bp charged on each dollar bought or sold. No terminal liquidation is charged.

**This is a retrospective conditional backtest.** The 22 stocks were selected on October 18 using information that includes this validation period. Chronological GP estimation does not make the historical evaluation independent of that screening. These results test allocation conditional on a hindsight-selected universe, not a prospective stock-selection-and-allocation strategy.

## Observed portfolio performance

{table(performance)}

*Annualized return is an extrapolation from only six months (252 divided by the number of daily validation returns). Volatility and Sharpe use daily simple marked-to-market returns and 252 sessions/year; Sharpe subtracts a zero daily risk-free return. Maximum drawdown includes the initial cash anchor and uses daily closes. Gross results replay the same target allocations without fees; they do not reoptimize the strategy with a zero cost penalty.*

![Daily net equity curves](figures/equity_curves.png)

![Benchmark comparison](figures/benchmark_comparison.png)

Primary monthly results (open-to-open holdings, final period open-to-close):

{table(mt,{'Net return':lambda x:f'{x:.2%}','Gross return':lambda x:f'{x:.2%}','Cost $':lambda x:f'{x:.2f}','Two-sided turnover':lambda x:f'{x:.2%}','Effective holdings':lambda x:f'{x:.2f}'})}

## Answers to the scientific questions

1. **Incremental performance:** the GP {verdict(gp.net_return,eq.net_return)} equal weight by {(gp.net_return-eq.net_return)*100:+.2f} percentage points net and {(gp.gross_return-eq.gross_return)*100:+.2f} points gross. It {verdict(gp.net_return,spy.net_return)} SPY by {(gp.net_return-spy.net_return)*100:+.2f} points net and {(gp.gross_return-spy.gross_return)*100:+.2f} points gross. Its daily annualized volatility was {gp.net_annualized_volatility:.2%}, versus {eq.net_annualized_volatility:.2%} equal weight and {spy.net_annualized_volatility:.2%} SPY. A positive return alone is not evidence of success.
2. **Value of drift:** the estimated-drift portfolio returned {gp.net_return:.2%}, versus {zero.net_return:.2%} for the identical-covariance zero-mean control, a {(gp.net_return-zero.net_return)*100:+.2f}-point difference. Across 132 stock-month forecasts, drift MAE was {estimated.mae:.6f}, versus {null.mae:.6f} with zero mean; RMSE was {estimated.rmse:.6f}, versus {null.rmse:.6f}; empirical CRPS was {estimated.crps:.6f}, versus {null.crps:.6f}. {'Drift did not improve either MAE or RMSE.' if estimated.mae>=null.mae and estimated.rmse>=null.rmse else 'Some point-error metrics improved, but this small dependent sample cannot establish a reliable forecasting edge.'} No significance claim is justified by six shared market months. The native zero-drift GP also appears above; it refits the existing `mean_gp=False` model and therefore changes volatility as well as removing drift.
3. **Volatility and risk calibration:** the primary 90% and 68% stock-return intervals covered {estimated.coverage90:.1%} and {estimated.coverage68:.1%}; standardized residual RMS was {estimated.rms_z:.3f} (one is the second-moment reference). Realized summed daily squared returns divided by predicted innovation variance was {estimated.realized_to_predicted_variance:.3f}. The GP portfolio's 90% gross holding-log-return interval covered {coverage:.0%} of six months; its standardized residual RMS was {np.sqrt(np.mean(gr.z**2)):.3f}. These are descriptive calibration checks, not proof of calibrated tail risk; cross-asset observations are correlated and portfolio forecast/holding clocks differ slightly.
4. **Allocation stability:** average two-sided monthly traded notional was {gm.two_sided_turnover.mean():.1%} of pretrade wealth, including the initial purchase; subsequent-month average was {gm.two_sided_turnover.iloc[1:].mean():.1%}. Adding independent 10 bp monthly-log-drift perturbations (ten frozen-seed trials per rebalance) changed weights by a median L1 distance {perturb.l1_weight_change.median():.3f}, with maximum {perturb.l1_weight_change.max():.3f}. This measures local allocation sensitivity, not an additional optimized strategy. Mean effective holdings were {gm.effective_holdings.mean():.2f}.
5. **Concentration:** {cap_cells} of 132 primary asset-month weights reached the 20% cap; {cap_months} of six rebalances contained at least one capped weight. Average cap count was {gm.cap_count.mean():.2f}; the largest target weight was {gw.weight.max():.2%}. Full weights, pretrade drifted weights and per-asset gross return contributions are saved.
6. **Distribution of gains:** {int((gm.net_return>0).sum())} of six holding periods were positive. Best/worst monthly returns were {gm.net_return.max():.2%}/{gm.net_return.min():.2%}. The two largest monthly dollar changes summed to ${top:.2f}, versus the total net dollar change ${net_gain:.2f}; this comparison is signed and can exceed total profit when other periods lose money. The monthly table shows whether any apparent gain depended on isolated periods.
7. **Sensitivity:** across all 18 predefined combinations, net returns ranged from {sensitivity.net_return.min():.2%} to {sensitivity.net_return.max():.2%}, and daily Sharpe from {sensitivity.net_sharpe.min():.2f} to {sensitivity.net_sharpe.max():.2f}. These are descriptive alternatives; the primary configuration remains 30 trading days, lambda 10 and 10 bp. No winning validation configuration was promoted to primary.
8. **Further testing:** a prospective, frozen-universe experiment is needed before drawing investment conclusions. Any advantage here must survive an untouched period, the same cost accounting, a wider universe selected without hindsight, and uncertainty estimates that respect common market months. The drift-versus-zero comparison is the direct evidence for expected-return value; comparing zero-drift GP with historical minimum variance examines risk allocation, although both risk estimates use different lookbacks. Six months cannot identify a stable Sharpe advantage or validate rare-event coverage.

## Forecast accuracy and uncertainty

All means, standard deviations, quantiles, MAE, RMSE and CRPS below are in **dimensionless monthly log-return units**. Multiply by 100 for log percentage points. Gaussian log predictive density in the CSV is a moment-matched Gaussian diagnostic, not the exact non-Gaussian mixture density. CRPS uses simulated distributions. Latent drift uncertainty is saved separately from realized innovation variance.

{table(dt,{k:(lambda x:f'{x:.5f}') for k in ['mae','rmse','crps','rms_z','realized_to_predicted_variance']} | {'coverage90':lambda x:f'{x:.1%}','coverage68':lambda x:f'{x:.1%}'})}

Primary pooled predicted/realized-return Pearson correlation: **{estimated.correlation:.3f}**. Mean monthly cross-sectional Spearman correlation: **{estimated.mean_cross_section_spearman:.3f}**. The mean absolute predicted drift was {estimated.mean_absolute_predicted_drift:.6f}; the mean latent-drift fraction of monthly predictive variance was {estimated.mean_drift_variance_share:.2%}. Zero means have undefined rank and Pearson correlations, left blank rather than given artificial scores. Interval levels are 5–95% and 16–84%, not 95% and exactly one normal standard deviation.

![Forecast and variance calibration](figures/forecast_calibration.png)

Ticker ranking, lowest to highest primary monthly log-return MAE (six predictions each, not a reliable investment ranking):

{table(rankingtable,{'mae':lambda x:f'{x:.5f}','rmse':lambda x:f'{x:.5f}','coverage90':lambda x:f'{x:.1%}'})}

## Data, dates and execution

The exact 22 symbols were parsed from the recommendation table in [the source report](../../reports/monthly-stock-screen-2026-10-18/REPORT.md), preserving its order, and verified unique. Neither screen scores nor later ticker ranks enter the GP or allocation rule. `tickers.txt` is the canonical extracted list. SPY is a separate market reference.

Original inputs are the repository's cached Yahoo Finance/yfinance histories, requested with `auto_adjust=False`; each original CSV's SHA-256 matched its retrieval metadata. See [data_integrity.csv](data_integrity.csv) for per-ticker source URLs, retrieval times, original hashes, corporate actions and jump magnitudes. The experiment snapshots the exact date slice under `data/`; no downloads or new external dependencies were needed. There are {len(opening)} price sessions, {len(opening)-1} daily-return observations per instrument, and **zero missing required price observations**, with no forward filling. There are {int(integrity.dividend_events.sum())} dividend-event rows and {int(integrity.split_events.sum())} split-event rows across the 23 instruments in the slice. Latest unadjusted stock closes remain above $10; the historical adjusted series can differ from historical executable nominal prices.

`Adj Close` is Yahoo's split/dividend-adjusted total-return proxy. Adjusted Open is `Open × Adj Close / Close`; this keeps opens and closes in consistent adjusted share units. Dollar accounting treats fractional units as automatically reinvesting distributions without extra reinvestment costs. Vendor revisions and corporate-action adjustment accuracy are not independently audited. These are retrospective vendor files, not point-in-time archived histories; hashes establish reproducibility, not correctness.

Anchor close: October 16, 2025. Training returns: **October 17, 2025–April 16, 2026**. Validation: **April 17–October 16, 2026**. The first half initializes the models and historical covariance; no evaluation data select hyperparameters. Six training months are too few for robust risk-aversion tuning, so lambda is fixed in advance. The GP's rolling 30/60-day inference and its existing regularized hyperparameter rules remain frozen; it is refit as newly observed data arrive. The initial six months are available, but intentionally do not all enter the primary 30-day GP fit.

{table(schedule[['origin','execution','target_end','horizon']])}

At each rebalance the GP uses returns ending at the **previous close**. Trades are priced at the next session's adjusted Open, with weights optimized using the just-observed opening holdings values and frozen forecasts. This is an immediately-after-open daily-bar fill proxy, not a guarantee of executing at an auction price after seeing it; the proportional allowance absorbs modeled trading friction but does not reconstruct bid/ask spreads or market impact. Same-day High/Low/Close do not enter the decision. The adjustment factor is retrospective corporate-action normalization, not a forecasting feature.

**Forecast-clock approximation:** close-return GP forecasts run from prior close through the session before the next rebalance; holdings run from execution open to next execution open (last period to October 16 close). The horizons are similar but overnight endpoints differ. `monthly_forecasts.csv` saves both forecast-aligned realized log returns and actual holding log returns; stock diagnostics use the former, portfolio risk diagnostics the latter. There is no same-close execution using that close's fitted return. This alignment approximation should be removed in a future intraday/execution-aware model.

## GP architecture, dependence and optimization

[ARCHITECTURE.md](ARCHITECTURE.md) documents the existing modules reviewed before coding. The implementation calls the existing `stock_api.heteroskedastic.fit_latent_returns`, `_future` and `_root`; it does not replace the GP with a constant-volatility or independent-day model. The daily likelihood is `r = scale × (f + exp(g/2) ε)`, with Gaussian variational latent drift f and log variance g and Matérn-3/2 temporal kernels. The scale is fitted using only the window's uncentered RMS returns. Kernel amplitudes and length scales are re-estimated at each origin and stay fixed within the forecast; latent f and g evolve and retain their full future temporal covariances. Regularization shrinks drift toward zero. Hyperparameters are plug-in optimized values, not fully Bayesian draws.

Monthly mean log return is `scale × sum(E[f])`. Its variance is `scale² × (sum(Cov[f]) + sum(exp(E[g]+Var[g]/2)))`; this separates latent-drift uncertainty from future realized-return noise. There are 10,000 antithetic joint paths per origin/variant/lookback. Full temporal covariance is propagated rather than multiplying one daily forecast by 21. `gp_parameter_history.csv` saves all fitted kernel parameters, scale, convergence information and objective; `gp_forward_latents.csv` shows daily expected drift, variance and uncertainty evolution.

Contemporaneous correlation uses standardized GP training residuals and fixed **50% shrinkage toward identity**. Residuals are smoothed in-sample latent-fit residuals, not independently held-out innovations. Correlated Gaussian driver families couple asset innovations and latent drift using that matrix; latent log-volatility drivers use its elementwise square (a PSD shared-volatility construction). The three driver families are independent of each other. Each ticker retains its own temporal GP covariance. This is an explicit approximate joint construction, not an inferred multi-output GP or estimated volatility cross-kernel. The existing multitask price GP reports zero cross-covariance and is not a suitable replacement.

The terminal Monte Carlo log-return covariance is PSD and rescaled to exactly preserve analytic GP marginal variances. The simulation rescaling factors are recorded; this finite-sample moment correction slightly adjusts empirical quantiles. The squared-correlation log-volatility construction and 50% shrinkage are assumptions, not empirically selected optimal values. Joint terminal samples, covariance, correlation and marginal moments are saved under `predictive/`.

**Moment limitation:** the existing lognormal variance mixture has finite log-return moments, but exponentiating its Gaussian return mixtures need not produce finite population arithmetic-return moments. Therefore the optimizer explicitly uses **monthly log-return means and covariance**, not unstable Monte Carlo estimates of expected simple returns. This is a documented utility approximation to the requested return objective: weighted asset log returns are not exactly portfolio log returns. Realized wealth uses exact simple price ratios; portfolio predictive log returns use `log(sum(w × exp(R_log)))`. No predictive tail clipping is used. No finite expected dollar return is claimed.

The convex objective is `w·mu_log − lambda/2 w'Σ_log w − cost_rate ||w−w_pre||₁`, with sum one and 0–20% bounds. SciPy SLSQP solves a quadratic objective with linear epigraph constraints and analytic gradients, and failures stop execution. The zero-drift control sets mu to zero with the **identical primary GP covariance**, giving variance minimization plus trading costs. Historical minimum variance uses up to 126 prior daily returns, 50% correlation shrinkage, and horizon scaling, with the same lambda/cost tradeoff and bounds. Equal weight is monthly rebalanced; SPY is buy-and-hold and, as an external single-asset reference, exempt from the 20% cap. The optional maximum-mean portfolio sets lambda to zero but retains the trading penalty.

Pretrade weights reflect actual opening-price drift. The realized fee solves `C = c × sum(abs((V−C)w − old_asset_dollars))`, so holdings plus fee exactly equal available wealth. The optimization's turnover penalty guides allocations; it is not separately subtracted from wealth. Initial purchase costs are included, and no cash is borrowed to pay fees. Two-sided turnover counts all purchases and sales; the CSV also includes half of that as one-way turnover. Monthly holding returns include the fee at their start and exclude the next rebalance fee. Daily equity marks use adjusted closes, so monthly boundaries differ from a naive calendar-close resampling.

![Monthly allocation weights](figures/monthly_weights.png)

## Frozen sensitivity analysis

{table(st,{'net_return':lambda x:f'{x:.2%}','net_sharpe':lambda x:f'{x:.2f}','net_max_drawdown':lambda x:f'{x:.2%}','costs':lambda x:f'{x:.2f}'})}

![Predefined sensitivity grid](figures/sensitivity.png)

These cost variants reoptimize with their stated cost penalty and replay their own net accounting. They differ from gross results, which hold the fee-paying strategy's target allocations fixed. Forecasts are cached across lambda/cost variants, keeping the predictive inputs identical.

## Reproducibility and limits

Run from repository root:

```bash
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python -m unittest tests.test_portfolio
MPLCONFIGDIR=/tmp/stocks-matplotlib-audit OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_portfolio_gp_22.py
```

`config.yaml` (JSON syntax, valid YAML) was written before execution. Seed: {c['seed']}. `manifest.json` records software versions, exact command and source hashes. Checkpoints bind GP inputs, configuration and model-code hashes. There were **{len(par)} successful GP fits**, {int((~par.converged).sum())} recorded nonconverged selected fits, and {int((~monthly.optimizer_success).sum())} portfolio optimization failures. Only successful starts returned by the original two-start GP routine are exposed; a selected fit with one successful start does not imply both starts converged. All six periods and all 22 names were retained. `run.log` records execution and captured warnings.

Primary constraints, covariance PSD and marginal variance preservation, transaction-cost budgets, and chronological separation have lightweight tests. `monthly_weights.csv`, `monthly_performance.csv` and `portfolio_daily.csv` permit independent reconstruction. `forecast_diagnostics.csv` separates mean error, proper scores and variance diagnostics. `ticker_error_ranking.csv` provides the requested low-to-high forecast error ordering for this experiment's different six-month interval.

The principal limitations are hindsight selection and survivorship (current constituents with adequate histories), six shared market months, plug-in hyperparameters and variational approximation, uncertain cross-asset/latent-volatility dependence, a log-utility approximation, vendor-adjusted data revisions, and opening execution/forecast-clock approximations. No hypothesis test treats the 132 asset-months as independent. No definitive claim of predictive drift, stable risk-adjusted superiority, or investable edge follows from this experiment alone.
'''
    text=text.replace('The monthly table shows whether any apparent gain depended on isolated periods.',
        'The other four periods together lost money in dollar terms: the positive total depended on the two strongest periods.')
    text=text.replace('Mean effective holdings were',
        f'The median L1 change corresponds to reallocating roughly {perturb.l1_weight_change.median()/2:.1%} of portfolio weight between names, so the primary allocations were sensitive to small drift perturbations. Mean effective holdings were')
    text=text.replace('These are descriptive alternatives; the primary configuration remains',
        f'At lambda 10 and 10 bp, extending the lookback from 30 to 60 days changed net return from {matched_sensitivity.loc[30,"net_return"]:.2%} to {matched_sensitivity.loc[60,"net_return"]:.2%}. These are descriptive alternatives; the primary configuration remains')
    text=text.replace('Six months cannot identify a stable Sharpe advantage or validate rare-event coverage.',
        f'The zero-drift GP also underperformed historical minimum variance ({zero.net_return:.2%} versus {p.loc["Minimum variance","net_return"]:.2%} net), so this sample does not demonstrate incremental portfolio value from its volatility model. Six months cannot identify a stable Sharpe advantage or validate rare-event coverage.')
    text=text.replace('Primary constraints, covariance PSD',
        'The independent reconstruction audit in `audit.json` checks all 24 strategy paths, their costs and daily holdings against saved prices. Source files and environment specifications are snapshotted under `source/`. Run `.conda-env/bin/python scripts/audit_portfolio_gp_22.py` to repeat it.\n\nPrimary constraints, covariance PSD')
    (out/'RESULTS.md').write_text(text)
