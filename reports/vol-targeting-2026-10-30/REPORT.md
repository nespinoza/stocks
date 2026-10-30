# Volatility targeting of SPY and 60/40, no leverage

Date: 2026-10-30. Reproduce with
`OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_vol_target_study.py`
(it needs SPY, IEF and BIL in `data_store/etf/`; BIL comes from
`configs/cash_etf.json`). Outputs are `strategies.csv`, `run_info.json` and the
forecast CSVs.

**Rule.**
- At each decision close, risky weight = min(target / forecast volatility, 1).
  The rest goes into BIL (T-bills).
- Forecasts:
  - SPY: 21-day log-HAR with the Garman–Klass proxy.
  - 60/40: the same model on the portfolio's squared returns.
  - EWMA (λ 0.94) is run alongside for comparison.
- Targets were fixed in advance (SPY 10% and 15%; 60/40 6% and 8%).
- Rebalancing is monthly or weekly, with or without a 5% no-trade band.
- Trades execute at the next close at 5 bp. February 2010 to October 2026.

**Fair comparison.** A managed strategy holds less equity on average, so it is
compared with a *static* mix of the same base and BIL at the same average
risky weight. The alpha is the monthly intercept against that mix (Newey–West t).

## Results (no band)

| Strategy | Vol | Max DD | Static mix: vol | Static mix: max DD | Sharpe (excess) | Static mix: Sharpe | α t |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPY unmanaged | 17.0% | −33.7% | | | 0.79 | | |
| SPY, HAR 10%, monthly | 10.1% | **−15.8%** | 11.8% | −24.0% | 0.77 | 0.80 | 0.14 |
| SPY, HAR 15%, monthly | 13.5% | **−23.0%** | 15.3% | −30.6% | 0.81 | 0.80 | 0.72 |
| SPY, EWMA 15%, weekly | 13.1% | −20.4% | 15.4% | −30.8% | 0.81 | 0.80 | 0.18 |
| 60/40 unmanaged | 9.8% | −21.2% | | | 0.87 | | |
| 60/40, HAR 6%, monthly | 6.0% | **−10.9%** | 6.8% | −15.1% | 0.87 | 0.87 | 1.11 |
| 60/40, HAR 8%, monthly | 7.6% | **−14.4%** | 8.4% | −18.4% | 0.87 | 0.87 | 1.08 |

The full grid of 32 variants is in `strategies.csv`. The deflated Sharpe ratio of the best
hedged active return across the 32 variants is 0.43.

## Answers

1. **No return gain.** At the same average equity weight, volatility targeting
   has the same Sharpe ratio as a static mix: 0.77–0.89 against 0.80–0.87. No
   alpha is significant (the largest t is 1.17), and weekly rebalancing is no
   better than monthly.
2. **It cuts drawdowns and volatility for free.** For the same average
   exposure, the worst drawdown is about a third smaller (SPY at 10%:
   −16% vs −24%; 60/40 at 6%: −11% vs −15%), and volatility is 1–2 points
   lower. Volatility clusters, so cutting exposure once forecast volatility
   rises avoids part of the losses that follow; that it did not also raise
   the Sharpe ratio means the calmer periods it kept were not more profitable.
3. **Use it as the risk dial for the base portfolio.** Monthly HAR is enough. It
   costs nothing beyond the T-bill ETF and a few trades a year.
4. **It is not a way to beat SPY.** Matching SPY's return would need leverage
   (margin costs), and the evidence that this kind of timing earns a higher
   Sharpe ratio with leverage is weak in this sample.
