# News pipeline: LLM-scored 8-K filings, first post-cutoff study

Date: 2026-10-27. 112 large-cap US stocks (`configs/news_universe.json`), mapped
to 10 SPDR sector ETFs. Reproduce with:

```
SEC_USER_AGENT="<name> <email>" .conda-env/bin/python scripts/collect_news.py --start 2026-07-01 --end 2026-10-27 --sources edgar
.conda-env/bin/python scripts/score_news.py --limit 500          # LM lexicon for all, Haiku via `claude -p` for new items
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_news_study.py
```

Outputs are `ic.csv`, `volatility_channel.csv`, `etf_ic.csv` and
`run_info.json`. The raw text, extractions and lexicon scores stay in git-ignored
`data_store/news/`. The Loughran–McDonald dictionary is downloaded at run time
and never committed, because its licence allows academic use only.

## What was built

| Stage | Implementation |
|---|---|
| Sources | SEC EDGAR 8-Ks: acceptance timestamp, item codes, and text from the Exhibit 99.x press release, or from the item body when there is no exhibit. GDELT DOC 2.0 headlines: a rolling ~3-month search window, throttled. |
| Store | append-only JSONL de-duplicated by `doc_id`. Syndicated headlines are collapsed by normalized title; filings stay distinct. |
| Extraction | one JSON-schema call per batch of 25 items. Fields: relevance, event type, direction, surprise, magnitude, affects-volatility, confidence. Cached by (prompt version, model, item). Backends: headless Claude Code on the plan (`claude -p --model haiku`, used here), the Anthropic API (`claude-haiku-5-5`, for paper trading) and a mock. |
| Baseline | Loughran–McDonald tone of the same text, with negation handling |
| Timing | an item belongs to the first session whose 16:00 New York close follows it. The engine trades at the next close (lag 1). |
| Signals | relevance × magnitude × direction, summed per stock and session. Exponential decay, cross-sectional z-scores, an equal-weighted roll-up to sector ETFs, and a variance-bump hook for the risk model. |

Evaluation window: 2026-07-01 to 2026-10-26, after Haiku 5.5's June 2026
training cutoff, so the model cannot have seen the outcomes. 82 sessions.

## Data actually used

- **371 8-K filings** from all 112 stocks, every one scored by Haiku (about $0.09
  at API prices). Event types: 141 earnings, 68 management, 65 capital, 26
  guidance, 22 M&A, 54 other.
  - 94% were filed outside market hours: 54% after the close and 40% before the
    open. So the "same-day" outcome contains the full price reaction.
- **Headlines are missing.** GDELT rate-limited this machine for most of the day,
  even at one request per 20 seconds. Only 2 of 8 monthly back-fill requests
  succeeded, giving 575 headlines for 3 stocks. These were not scored or used.
  - The collector now resumes from its log and has a `--days-back` mode. A
    small daily pull, part of the step 6 cron job, will build the headline sample
    from here on.
- Only **29 sessions had at least 5 stocks filing**, the minimum for a
  cross-sectional IC. Every result below is preliminary.

## Results

IC is the cross-sectional Spearman correlation among stocks that filed that
session. t-statistics use Newey–West standard errors; for multi-day horizons
they use non-overlapping dates only.

| Signal | Outcome | Mean IC | t | Dates |
|---|---|---|---|---|
| LLM | same day (reaction) | +0.107 | 1.21 | 29 |
| LLM | **next day** (close d → d+1) | **+0.234** | **2.99** | 29 |
| LLM | next 1 day after a lag (d+1 → d+2) | +0.081 | 1.04 | 28 |
| LLM, neutralized vs trailing 5d/21d returns | next day | +0.175 | 2.19 | 29 |
| LLM, neutralized | next 1 day after a lag | +0.160 | 2.68 | 28 |
| LLM | 5 days after a lag | −0.052 | 0.07 | 5 independent |
| LM lexicon | same day / next day | −0.092 / +0.049 | −1.14 / 0.57 | 30 |
| LM lexicon, neutralized | next day | −0.047 | −0.61 | 30 |
| Sector ETFs (LLM, 5-day decay) | 5 days after a lag | +0.093 | 1.32 | 16 independent |

Volatility channel (`volatility_channel.csv`, 71k stock-days, standard errors
clustered by date). The outcome is the log of realized variance over the next 5
days divided by the trailing 63-day variance:

| Specification | b (news day) | t | b (LLM vol intensity) | t |
|---|---|---|---|---|
| news only | −0.194 | −3.42 | | |
| news + intensity | −0.287 | −4.14 | +0.199 | 2.66 |

## Answers

1. **The LLM score carries information the dictionary does not.** It has a
   next-day IC of +0.23 (t 3.0), which falls to +0.18 (t 2.2) after
   neutralizing recent returns. The dictionary is about zero everywhere. The
   two agree only moderately: their Spearman correlation is 0.39, and the
   dictionary gives exactly zero tone to 26% of the filings.
2. **This is not yet a demonstrated signal.** The study ran about 33 IC tests.
   The best p-value is 0.006, which is about 0.2 after a Bonferroni correction.
   There are 29 cross-sections, all in one earnings season. The pattern is also
   odd: the same-day reaction IC (+0.11, not significant) is weaker than the
   next-day IC, although 94% of filings arrive outside market hours and the
   first close should already price them. That fits post-announcement drift in
   large caps, but it also fits chance.
3. **The tradable horizon is short.** Positive IC appears only on the next day
   or two, and nothing survives to 5 or 21 days. A monthly rebalance would miss
   it. The sector-ETF roll-up is positive (5-day IC +0.09) but not significant,
   with 16 independent dates.
4. **The volatility flag is the most usable output.** Filing days are followed by
   *lower* variance than the trailing average. Filings cluster in earnings
   season, after which volatility mean-reverts. Among filings, though, the items
   Haiku flags as volatility-relevant are followed by higher variance (t 2.7).
   That is the input the `variance_bump` hook needs.
5. **Earnings press releases score near-neutral to mildly positive.** The mean
   direction is +0.25, because companies frame their own releases favourably and
   carry no analyst consensus to judge surprise against. Cross-sectional z-scoring
   removes the level, but surprise needs either consensus estimates or the news
   coverage of the release, which the headline feed will supply.

**Recommendation:** keep the pipeline as built and let it accumulate data
prospectively. Do not put a news view into the Black–Litterman layer yet.
Re-run this study once the daily collector has 2–3 months of headlines plus the
next earnings season of filings, and require the next-day IC to survive with
more than 60 cross-sections. Meanwhile, the volatility flag can be tested as an
event-variance bump in the risk engine, which needs no return predictability.

## Caveats

- One model (Haiku 5.5) and one prompt version (`news-extract-v1`). Scoring ran
  on the Claude Code plan, so `total_cost_usd` is an API-equivalent figure. The
  run cost about $0.006 per batch of 25 at API prices.
- No market-move items occurred in the filings, so `llm_ex_market_move` equals
  `llm`. That control matters only once headlines are included.
- Sessions use the NYSE calendar from the price files. Exchange holidays fall
  through to the next session.
