# Lazy Prices: 10-K/10-Q text changes and small/mid-cap returns, 2019–2026

Date: 2026-10-30. All data are free. Reproduce with:

```
SEC_USER_AGENT="<name> <email>" .conda-env/bin/python scripts/collect_filing_similarity.py --since 2018-01-01 --threads 8
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_lazy_prices_study.py
```

(It also needs the broad universe and prices; see
`reports/insider-purchases-2026-10-30/REPORT.md`.) Outputs: `ic.csv`,
`portfolios.csv`, `run_info.json`.

## Data

**Filings:** every 10-K and 10-Q of the 1,596 broad-universe companies with a
report date from 2018 on, taken from EDGAR. That is 47,385 filings from 1,588
companies; the other 8 file foreign-issuer forms.
- The collection streamed the primary documents (about 2 hours, 8 threads).
  It stored only the scores (9 MB); no text was kept.
- Each filing is compared with the same form for the same fiscal period a year
  earlier (report date 335–395 days before). 41,065 filings have such a match,
  and 35,729 also have a Risk Factors section in both years.

**Measures** (fixed in advance, after Cohen, Malloy & Nguyen 2020): cosine
similarity of word counts and Jaccard overlap of word sets, for the whole
document and for Item 1A Risk Factors. Numbers are excluded.

**Timing:** a filing counts as public at the close of the session after its
filing date. Each month end uses the latest filing made public in the previous
63 sessions, and the portfolio trades at the next close.

**Sample:** 90 month-ends, April 2019 to September 2026. Eligible stocks have a
price of at least $3, trade at least $1M a day and have a score: about 1,130 a
month, or 1,000 for the Risk Factors measures.

**Survivorship:** the universe is today's listed companies, as in the other
broad-universe studies.

## Results

**IC** (monthly Spearman, 21 and 63 sessions from the next close). "Neutral"
removes log dollar volume, the trailing 21- and 252-day returns, log document
length and |log length change|:

| Measure | 21d raw | 21d neutral | 63d raw | 63d neutral |
|---|---:|---:|---:|---:|
| Cosine | −0.002 (t −0.3) | −0.000 (−0.0) | +0.002 (−0.4) | +0.005 (0.5) |
| Jaccard | +0.004 (0.4) | +0.003 (0.4) | +0.007 (0.1) | +0.007 (0.5) |
| Risk Factors cosine | +0.002 (0.3) | −0.002 (−0.2) | +0.006 (−0.2) | +0.001 (−0.2) |
| Risk Factors Jaccard | +0.007 (0.9) | +0.004 (0.7) | +0.009 (0.1) | +0.004 (−0.6) |

The 63-day t-statistics use non-overlapping quarters (29 dates), so their sign
can differ from the mean over all months.

**Portfolios** (equal weight, monthly, about 230 stocks per quintile, about 54%
turnover per rebalance). Active return is against all eligible stocks; FF α
regresses it on FF5 + momentum + reversal. Long–short pays costs on both legs.

| Measure | Non-changers, active %/yr (t) | Changers, active %/yr (t) | Long–short %/yr (t) | Long–short FF α (t) | L–S 2019–21 | L–S 2022–26 | L–S at 20 bp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosine | −0.7 (−0.4) | +1.1 (0.5) | −1.7 (−0.5) | −0.9 (−0.3) | −9.4 | +2.5 | −4.3 |
| Jaccard | −0.9 (−0.5) | +2.0 (0.8) | −2.8 (−0.8) | −2.9 (−0.9) | −8.5 | +0.4 | −5.3 |
| Risk Factors cosine | −0.3 (−0.2) | +1.0 (0.6) | −1.2 (−0.4) | −0.7 (−0.2) | −7.4 | +2.1 | −3.9 |
| Risk Factors Jaccard | +0.8 (0.5) | +2.2 (1.1) | −1.4 (−0.4) | −1.2 (−0.4) | −6.0 | +1.1 | −4.0 |

The deflated Sharpe ratio of the best gross long–short (Risk Factors cosine) is
0.27; its Sharpe is negative.

## Answers

1. **No Lazy Prices effect in this universe and period.**
   - Every IC is within ±0.01 with |t| < 1, before and after controls.
   - The paper's trade (buy the stocks whose filings changed least, short the
     ones that changed most) lost 1–3%/yr before costs. No t-statistic reaches 1.
   - At 20 bp per trade it loses 4–5%/yr.
2. **The two sub-periods disagree.**
   - 2019–21: the sign is opposite to the paper. Changers beat non-changers by
     6–9%/yr.
   - 2022–26: the sign is the paper's, but small: +0.4 to +2.5%/yr, which is
     below costs.
   - Neither period is significant on its own. The original result comes from
     1995–2014; the paper was published in 2020, and this sample starts right
     before that. Its absence here is consistent with the decay commonly seen
     in published anomalies, but this test cannot say why.
3. **Recommendation:** drop filing text changes as a return signal. The
   collector costs nothing to re-run each quarter, and the scores are kept in
   `data_store/filings/` in case a later risk model wants a "this company
   rewrote its risk factors" flag. That use has not been tested.
