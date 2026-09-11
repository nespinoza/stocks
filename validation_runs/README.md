# Saved example experiments

The [SDE benchmark report](sde-2026-09-01-benchmarks/REPORT.md) contains the
seven-model comparison over 30/60/90/180-day lookbacks, full tables, plots,
terminal-return diagnostics, calibration checks and noted failure cases.

`shared-timescale-comparison-79f7def3-d710-4038-a38e-71009e839357.*` compares
the original Yahoo-data runs with `last_price` and the revised
`shared_timescale_gp_v2`, on the same saved data and folds. The new GP has one
shared Matérn 3/2 timescale and independent per-ticker amplitudes/noises.
Its mean normalized MAE is 3.772%, versus 2.485% for `last_price`.
The old `multitask_gp` row in that comparison denotes the earlier GP v1.

Both runs predict AMZN using GOOGL and AAPL where supported, with a 30-calendar-day
rolling training window, a seven-calendar-day forecast horizon, 12 monthly folds,
and seed 42. Prices are evaluated on XNYS sessions only.

- `yahoo-2026-09-01-75d14252-6844-4462-bfb2-7bc39983e085.*`: observed Yahoo data;
  forecast origins September 2025 through August 2026. Mean fold MAE: random walk
  6.006252, multitask GP 9.248408, VAR 8.944674 (adjusted price units).
- `synthetic-2027-09-01-41b0a756-74a4-41c1-bcf9-2c883f528d58.*`: synthetic data only;
  demonstrates the requested future cutoff and is not evidence about stock-market
  predictive performance.

Each run has a JSON snapshot/record, summary CSV, and error-by-fold PNG.
Additional `*-normalized.csv` and `*-normalized.png` files report error as a
percentage of mean observed price. Loading the original JSON derives these
metrics automatically. The Yahoo run's mean normalized MAE is about 2.485% for
the random walk, 3.690% for the GP, and 3.651% for VAR (see CSV for exact values).
Load a JSON with `ValidationResult.load(...)`, then use `.summary()`, `.plot()`,
or `.rerun(models)` to evaluate new models against its exact data. These two
experiments have different datasets and periods and cannot be merged as a fair
model comparison. See [the validation guide](../VALIDATION.md).
