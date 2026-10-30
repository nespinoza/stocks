# Wikipedia attention and small/mid-cap returns, 2015–2026

Date: 2026-10-30. A free stand-in for social-media attention. Reproduce with:

```
.conda-env/bin/python scripts/collect_pageviews.py --end 2026-10-29
cd scripts && OPENBLAS_NUM_THREADS=1 ../.conda-env/bin/python run_attention_study.py
```

(It also needs the broad universe and prices; see
`reports/insider-purchases-2026-10-30/REPORT.md`.) Outputs: `ic.csv`,
`volatility_channel.csv`, `portfolios.csv`, `run_info.json`.

## Data

**Coverage:** 696 of the 1,596 broad-universe stocks have an English Wikipedia
article that Wikidata links to them.
- 482 are matched by SEC CIK.
- 214 are matched by a current NYSE/Nasdaq ticker whose article title shares a
  word with the SEC name.
- The least-known stocks often have no article, so they are not covered.

**Views:** daily user (non-bot) pageviews, July 2015 to October 2026. The median
stock gets 44 views a day.

**Sample:** about 530 eligible stocks per day over 2,761 sessions. Eligible
means price ≥ $3, at least $1M a day traded, and at least 5 views a day.

**Signals** (fixed in advance, using views through the previous UTC day):
- **Shock:** log(mean of the last 5 days) − log(median of days −61 to −6).
- **Level:** log(median views over 60 days), a measure of how well-known the stock is.

## Results

**IC** (Spearman; t-statistics on non-overlapping dates). "Neutral" removes the
trailing 5-day return and its absolute value, the trailing 21-day return, log
dollar volume and abnormal volume:

| Signal | Outcome | Mean IC | t |
|---|---|---:|---:|
| Shock | next day, from today's close | +0.001 | 0.6 |
| Shock, neutral | next day | +0.002 | 1.5 |
| Shock, neutral | 5 days, lag 1 | −0.001 | −0.0 |
| Shock, neutral | 21 days, lag 1 | −0.003 | −1.4 |
| Level | 21 days, lag 1 | −0.021 | −2.7 |
| Level, neutral (vs log dollar volume) | 21 days, lag 1 | −0.004 | −0.6 |

**Volatility.** The outcome is the log of next-5-day realized variance over the
trailing 63-day variance; standard errors are clustered by date, n = 1.46M.
- Shock alone: b = +0.044 (t 3.2).
- With all controls: b = +0.049 (t 3.4).

**Portfolios** (equal weight, lag 1; active return against all eligible stocks;
FF α on FF5 + momentum + reversal):

| Portfolio | Stocks | Active %/yr, gross (t) | FF α %/yr (t) | 2015–18 | 2019–22 | 2023–26 | Active at 20 bp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top shock quintile, weekly | 107 | +2.5 (2.1) | +2.7 (2.1) | +3.8 | +0.7 | +3.3 | −10.0 |
| Bottom shock quintile, weekly | 107 | +0.6 (0.5) | +0.1 (0.1) | +1.7 | +1.1 | −0.8 | −12.2 |
| **Least-viewed quintile, monthly** | 109 | **+3.1 (1.95)** | **+4.5 (2.9)** | +3.3 | +2.0 | +3.9 | +2.7 |
| Most-viewed quintile, monthly | 107 | −4.7 (−2.3) | −5.7 (−3.0) | −1.9 | −3.1 | −8.6 | −4.8 |

Deflated Sharpe ratio of the best gross variant (least-viewed) across the four
designs: 0.41.

## Answers

1. **Attention shocks predict volatility, not returns.** A jump in pageviews is
   followed by higher variance, even after controlling for recent returns and
   trading volume (t 3.4). Its return IC is about zero at every horizon. The
   weekly top-quintile portfolio's +2.5%/yr gross becomes −10%/yr at 20 bp. Like
   the 8-K volatility flag, the shock is a free input for the risk model.
2. **"Less-known stocks do better" shows up, but this test cannot confirm it.**
   The least-viewed fifth beat the universe by +3.1%/yr (FF α +4.5%, t 2.9), and
   the most-viewed fifth lagged by −4.7%/yr, in every sub-period. Three things
   argue against taking this at face value:
   - **Most of it is liquidity.** After removing log dollar volume, the IC of
     the attention level falls from −0.021 (t −2.7) to −0.004 (t −0.6). Seen
     this way it looks like the known illiquidity premium, not neglect.
   - **The universe-selection bias points the same way.** The universe is today's
     $300M–$10B companies.
     - A famous company in that range today most likely fell into it.
     - Well-known companies that grew past $10B are missing.
     - Obscure firms that failed are missing too.

     All three flatter the least-viewed group and hurt the most-viewed one. The
     bias is strongest near the build date, where the gap is largest (−8.6%/yr
     for the most-viewed fifth in 2023–26).
   - **It fails the overfitting check.** The deflated Sharpe ratio is 0.41, below
     the 0.95 required.
3. **The clean test is forward-looking.** A universe fixed today and then
   followed forward has no selection bias. The "least-viewed fifth" rule is cheap
   to run (24% turnover a month, about the same at 20 bp costs), so it is a good
   candidate for the paper-trading sleeve, measured against the equal-weighted
   universe.

**Partial historical fix (free, optional):** rebuild the universe without the
$10B upper bound, and apply the size band at each date using historical SEC
share counts. That brings back the large-cap winners. Firms that died would
still be missing, and that remaining bias favours the least-viewed group.
