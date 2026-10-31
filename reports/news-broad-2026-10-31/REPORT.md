# LLM-scored 8-K filings in small/mid caps: the larger post-cutoff test

Date: 2026-10-31. This repeats the step 5 study
(`reports/news-pipeline-2026-10-27/`) on the 1,596-stock broad universe instead
of 112 large caps, to confirm or reject its next-day IC of +0.23. Reproduce with:

```
SEC_USER_AGENT="<name> <email>" .conda-env/bin/python scripts/collect_news.py --start 2026-07-01 --end 2026-10-31 \
    --sources edgar --config configs/broad_universe.json --store data_store/news_broad/documents.jsonl
NEWS_DIR=data_store/news_broad .conda-env/bin/python scripts/score_news.py --config configs/broad_universe.json --limit 100000
NEWS_DIR=data_store/news_broad OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_news_study.py \
    --config configs/broad_universe.json --prices data_store/broad --output reports/news-broad-2026-10-31
NEWS_DIR=data_store/news_broad .conda-env/bin/python scripts/news_open_check.py
```

Outputs: `ic.csv`, `volatility_channel.csv`, `open_check.csv`, `open_check.json`,
`run_info.json`. If the plan's usage limit stops the scoring step, re-running
it continues from the first unscored batch.

## Data

**Filings:** every 8-K from the 1,596 stocks between 2026-07-01 and 2026-10-30,
which is 5,408 filings. All are after Haiku 5.5's June 2026 training cutoff.
- Event types: 1,897 earnings, 928 capital, 895 management, 465 M&A, 262
  guidance, and other.
- 95% were filed outside market hours.

**Scoring:** same prompt (`news-extract-v1`) and model as step 5. Every filing
was scored, in 217 batches on the Claude Code plan, without reaching the usage
limit. The API-equivalent cost was $1.32.

**Sample:** 85 sessions, each with enough filers for a cross-sectional IC, up
from 29 in step 5.

## Results

**IC** (Spearman among stocks with a filing that session; Newey–West t;
non-overlapping dates for multi-day horizons):

| Signal | Outcome | Mean IC | t | Step 5 (112 large caps) |
|---|---|---:|---:|---:|
| LLM | same day (reaction, close d−1 → d) | +0.143 | 8.2 | +0.107 (1.2) |
| LM lexicon | same day | +0.053 | 3.3 | — |
| LLM | **next day (close d → d+1)** | **−0.025** | **−1.4** | **+0.234 (3.0)** |
| LLM, neutralized vs trailing 5d/21d | next day | −0.024 | −1.2 | +0.175 (2.2) |
| LLM | 1 day after a lag | +0.015 | 0.9 | +0.081 (1.0) |
| LLM | 5 days after a lag | +0.046 | 0.3 | −0.052 (0.1) |
| LLM | 21 days after a lag | +0.002 | −0.8 | — |

The LLM's direction agrees with the Loughran–McDonald dictionary at Spearman
0.39.

**At the next open** (`open_check`; 4,745 stock-sessions with only
out-of-hours filings):

| Outcome | Mean IC | t |
|---|---:|---:|
| Overnight gap (close d−1 → open d) | +0.224 | 13.2 |
| Open d → close d (tradable at the open) | −0.007 | −0.4 |
| Open → close, controlling for the gap | +0.005 | 0.3 |

Buying stocks with a positive score and shorting those with a negative score,
open to close, earned +9 bp a day (t 0.4) before costs.

**Volatility** (log of next-5-day variance over trailing 63-day variance,
clustered by date, n = 973k):
- Days with a filing are followed by *lower* relative variance (b −0.15, t −4.9).
  Step 5 had the same sign.
- The LLM's volatility-intensity score adds nothing once that is controlled for
  (t −1.2). In step 5 it had t 2.7.

## Answers

1. **The step 5 result was noise.** On a sample about 15 times larger, the
   next-day IC is −0.025 (t −1.4), against +0.23 before. The earlier 29-date
   estimate, the best of about 33 tests, did not replicate. That's consistent
   with the Bonferroni-adjusted p of about 0.2 reported then.
2. **The LLM reads filings well; the market is simply faster.** Its score
   explains the reaction strongly (overnight-gap IC +0.22, t 13), almost three
   times as well as the dictionary. But the whole reaction happens in the gap
   before the first trade. From the open onward there is nothing left (IC −0.007).
3. **Capturing the gap would mean trading after hours or pre-market, within
   minutes of the filing.** That is the same race against automated readers as
   with insider purchases. It also needs intraday data and execution this
   project doesn't have.
4. **The 8-K volatility flag did not replicate either**, so it should not go
   into the risk model on this evidence.
5. **What still holds:** the extraction itself is accurate, cheap (about $0.25
   per 1,000 filings at API prices) and restartable. It's a sound component for
   any LLM idea where the information is *not* priced within minutes, such as
   text the market reads slowly, or aggregation across many documents.
