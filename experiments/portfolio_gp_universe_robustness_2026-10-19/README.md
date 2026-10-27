# Monte Carlo portfolio-universe robustness study

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
