# Insider purchases (SEC Form 4) in small/mid caps, 2010–2026

Date: 2026-10-30. All data are free. Reproduce with:

```
SEC_USER_AGENT="<name> <email>" .conda-env/bin/python scripts/build_broad_universe.py
.conda-env/bin/python scripts/download_etf_universe.py --config configs/broad_universe.json --out data_store/broad --start 2010-01-01 --end 2026-10-30 --skip-missing
SEC_USER_AGENT="<name> <email>" .conda-env/bin/python scripts/collect_insider_trades.py --first 2009q4
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/run_insider_study.py
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python scripts/insider_timing_check.py
```

Outputs: `portfolios.csv`, `event_study.csv`, `ic.csv`, `timing_check.txt`,
`run_info.json`.

## Data

**Universe** (`configs/broad_universe.json`): 1,596 NYSE and Nasdaq operating
companies.
- Market cap $300M–$10B (median $2.0B), price ≥ $3, median daily dollar volume
  ≥ $2M, measured on 2026-10-30.
- Built from SEC's ticker list, SEC XBRL share counts and Yahoo prices.

**Trades:** SEC's quarterly Form 345 data sets, 2009 Q4 to 2026 Q3.
- 2.1 million open-market trades (code P or S).
- In the universe, from officers and directors: 54,908 purchases (in 1,483
  stocks) and 171,548 sales.

**Timing:**
- The data give the filing *date* only, so a filing is treated as public at the
  close of the next session. Portfolios trade one session later still.
- The timing check (below) looks at each day around the filing date directly.

**Survivorship bias:** the universe is the companies listed today, so firms
that failed or were acquired since 2010 are missing.
- The equal-weighted benchmark has the same bias, which offsets part of it.
- 2023–26 is reported separately as the least-affected window.

## Results

**Portfolios** (equal weight, monthly; active return is against the
equal-weighted eligible universe; "FF α" regresses the active return on FF5 +
momentum + short-term reversal, through Aug 2026):

| Portfolio | Stocks | Turnover/mo | Active %/yr, no costs (t) | 2023–26 | FF α %/yr (t) | Active %/yr at 20 bp | at 40 bp |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≥1 buyer, last month | 81 | 163% | −0.6 (−0.5) | +2.0 | −0.5 (−0.4) | −4.2 | −7.9 |
| ≥1 buyer, last 3 months | 196 | 54% | +1.0 (1.3) | +1.2 | +1.4 (1.8) | −0.1 | −1.2 |
| ≥2 buyers, last month | 32 | 169% | −2.1 (−1.2) | +4.4 | −1.7 (−0.8) | −5.9 | −9.7 |
| ≥1 opportunistic buyer, last month | 72 | 173% | −0.9 (−0.4) | −1.3 | −1.0 (−0.5) | −4.8 | −8.7 |
| ≥2 sellers, no buyer (contrast) | 150 | 133% | −0.7 (−0.4) | +0.5 | −1.3 (−1.2) | −3.6 | −6.6 |

The deflated Sharpe ratio of the best gross variant (3-month window) is 0.56,
well below 0.95.

**Event study** (cumulative simple abnormal return starting from the close
after the public session; t from monthly averages; random eligible stock-days as
the placebo):

| Event | Events | 5 days | 21 days | 63 days (t) |
|---|---:|---:|---:|---:|
| Purchase | 24,500 | −0.02% | +0.11% | +0.59% (1.2) |
| Opportunistic purchase | 9,917 | −0.10% | +0.12% | +0.84% (0.2) |
| Sale | 115,910 | +0.00% | −0.06% | −0.25% (−0.7) |
| Placebo | 25,000 | −0.00% | +0.11% | +0.06% (0.2) |

**Cross-sectional IC** of the 63-day net purchase ratio across about 620
stocks with insider trades:
- 21-day IC +0.009 (t 1.8).
- −0.005 (t −1.1) after controlling for trailing returns and liquidity.

**When the information is priced** (`timing_check.txt`, all purchase filings,
mean abnormal return by session relative to the filing date):

| Day | −2 | −1 | **0** | **+1** | +2 | +3 | +4 | +5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Purchase | −0.19% | +0.01% | **+0.37%** | **+0.85%** | +0.08% | +0.05% | +0.05% | +0.02% |

Days +2 to +22, more liquid half (median $14M/day): +0.09%. Less liquid half
(median $0.8M/day): +1.09%.

## Answers

1. **Insider purchases are informative, but the market prices them within a day
   of the filing.** About +1.2% of abnormal return arrives on the filing day and
   the next day (t > 13). After that, the drift in liquid stocks is about zero.
2. **No tradable edge with end-of-day execution and free data.** Gross of costs,
   no portfolio beats the equal-weighted universe significantly. The best is
   the 3-month window at +1.0%/yr (t 1.3; factor α t 1.8; deflated Sharpe 0.56).
   At realistic small-cap costs, every variant loses to the benchmark. The
   Cohen–Malloy–Pomorski "opportunistic" filter does not help in this sample.
3. **What's left is not reachable cheaply.**
   - The day-0/+1 reaction could only be captured by trading within minutes of
     the EDGAR filing. That is a race against automated readers, and testing it
     needs intraday data, which isn't free.
   - The slower drift in the least liquid stocks is about the size of their
     round-trip trading costs.
4. **Recommendation:** drop insider trades as a return signal. Keep the
   collector, because purchases are a cheap event flag for the risk side, in the
   same way as the 8-K volatility flag.
