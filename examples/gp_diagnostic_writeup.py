"""Render the interpretation and complete tables from verified saved GP diagnostics."""
from pathlib import Path


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def number(value):
    return f'{value:.4g}'


def write_report(root, summary, findings, predictions, sampling):
    root = Path(root)
    parts = ["""# GP inference report: AMZN, November 1, 2025

The old two-trading-day volatility floor restricts the 60/90/180-day fits. Extending
that floor while retaining the old regularization reveals better sub-two-day
solutions. It does **not** establish a precisely measured sub-day correlation time.
The mean-return GP remains compatible with negligible amplitude at every lookback;
its old approximately 20-day length is primarily a regularization artifact.

The ordinary log-price GPs are better constrained, especially the shared-timescale
multitask model. Their interior optimizer solutions agree closely with the best
profile solutions, although posterior uncertainty can be much wider than a point
estimate suggests. These are inference findings for one selected origin, not
forecasting-performance or trading-alpha evidence.

[Browse every plot](index.html) · [All parameter intervals](hyperparameter-summary.csv) ·
[Identifiability diagnostics](identifiability.csv) · [Predictive comparison](predictive-comparison.csv) ·
[Sampling checks](sampling-checks.csv) · [Artifact verification](verification.json)

## Data, inference and interpretation

The experiment reuses the September 1, 2026 benchmark snapshots: AMZN is the target,
with GOOGL/AAPL in the multitask model. Training ends October 31, 2025; the held-out
prices are November 3–7 (five trading sessions within the saved seven-day horizon).
Lookbacks of 30/60/90/180 calendar days contain 22/44/64/126 prices, respectively,
and one fewer return. The origin was chosen because existing volatility fits
approached their bound, so it is deliberately informative about that issue rather
than a random representative sample of market conditions.

All diagnostic kernel lengths use **trading sessions**. Ordinary GPs fit log prices,
not dollar prices, with per-ticker training-only scaling. The old point comparison
is also evaluated on trading-session coordinates; legacy calendar-coordinate fits
are retained in each `old-optimizer.json`. Do not directly equate their lengths or
this experiment's predictions with the saved historical benchmark's price-GP fits.

Positive covariance parameters were already optimized in log space. Explicit
sampling priors use log-uniform lengths [0.5, 300], price amplitudes/training scatter
[0.001, 100], price noise/scatter [0.0001, 10], return-mean amplitude/return scatter
[0.001, 10], and dimensionless log-variance amplitude [0.01, 3]. The signed
log-variance level has a training-centered Normal prior, SD 2, truncated at ±5 SD.
These priors are configurable; no alternate-prior experiment was run here.

Ordinary GP samples use dynesty and the exact Gaussian marginal likelihood.
Return-GP samples use joint latent-variable MCMC, analytically integrating the mean
function conditional on latent volatility. Return-GP **profiles use an unregularized
variational ELBO**, not the MCMC likelihood. Two-dimensional surfaces are conditional
slices with the other coordinates fixed; their overlaid posterior contours are
marginal distributions. Differences between the surfaces and contours do not by
themselves indicate faulty sampling. Multistart runs use 30 starts for each of the
old and broad-domain objectives; profiles use 41 log-spaced points and two starts.

## Answers to the inference questions

1. **Are the old boundaries real or optimizer artifacts?** For return volatility,
   the old floor excludes better solutions at 60/90/180 days even with its penalties
   retained. Many starts reproduce the boundary, so initialization alone does not
   explain it. For ordinary GP lengths the best solutions are interior; poor local
   endpoints exist but the original point solution agrees with the best profile.
2. **Is there an interior volatility mode below two days?** The unregularized profile
   grid prefers 1.305/0.948/0.688/0.500 sessions as lookback grows. At 30/60/90 days,
   the lower edge is only 0.346/0.173/0.048 ELBO units below the maximum. These shallow
   optima and broad posteriors do not measure a precise short timescale. At 180 days,
   the profile maximum is at the new lower edge and approximately 98% of posterior
   mass is below two days: temporal correlation is unresolved at daily cadence.
3. **Is the mean length constrained when its amplitude is non-negligible?** No
   useful constraint emerges here. Even conditional on amplitude ≥0.1 times training
   return scatter, the 95% intervals extend from approximately 0.51–0.57 to
   250–257 sessions. The mean process fails the activity criterion at every lookback.
4. **Does posterior exploration change predictions?** Yes. Terminal 95% widths
   change by factors 0.816–1.572 versus the old point fits. This comparison combines
   hyperparameter marginalization, different priors, and (for return GPs) full
   latent sampling versus variational inference; it is not an isolated measurement
   of marginalization alone. Details for every fit appear below.
5. **Do priors dominate some fits?** Mean-length marginals are nearly log-uniform:
   their binned KL divergences from that prior are only 0.001–0.021 nats. The 30-day
   return volatility and single-stock GP length posteriors are broad, and small
   amplitudes make some lengths weakly identifiable. These are signs of weak data
   information under the chosen priors, not a completed prior-sensitivity study.

## Bounds, inactive components and multimodality

The bound-only profile retains the old penalties and other old parameter bounds,
changing only the profiled length domain. Its volatility maxima are approximately
2.109/1.797/1.305/0.808 sessions for both return models. By contrast, removing
regularization leaves the mean-length profile essentially flat (total range at
most 0.000108 ELBO units). Retaining regularization puts that mean-length profile's
maximum back at approximately 19.788 sessions in all four windows. This directly
supports the regularization explanation for the old approximately 20-day estimate.

“Inactive” means amplitude below 0.1 times training scatter for mean/log-price GPs,
or log-variance amplitude below 0.1 for volatility. A component is declared active
only when posterior inactive probability is below 5%. These are operational
thresholds, not posterior probability of exactly zero amplitude (the priors exclude
zero). In particular, absence of activity does not prove that a component is zero.

Both return models allow nearly constant volatility at 30 days (inactive mass
8–13%); at longer lookbacks that mass falls below 5%. The 180-day volatility lengths
put about 44% of mass in the lowest 5% of the log-prior range, versus 5% for an
unchanged log-uniform prior. Mean lengths retain substantial mass near both edges.
Full edge masses, including amplitude/noise parameters, are tabulated per fit.

Multistart endpoints include substantially worse solutions even when the optimizer
reports success. Return fits show near-zero-amplitude/flat-length alternatives,
and ordinary GPs show amplitude/noise tradeoffs. The corner plots and traces show
broad ridges and long volatility-length excursions; these data do not establish a
count of distinct posterior modes. A different optimizer endpoint is not by itself
a statistically distinct mode. The successful-start and near-best counts below
quantify this limitation without assuming uniqueness.

## Sampling checks and numerical limits

All 16 selected runs pass the checks implemented in this repository. For return
models, maximum split R-hat ranges from 1.011 to 1.036 and minimum autocorrelation
ESS from 103.7 to 269.5. These checks cover hyperparameter coordinates, not every
latent value or predictive tail, and are not rank-normalized/tail-ESS diagnostics.
Some traces show long excursions; the 90-day joint return fit is only just above
the ESS >100 threshold. Passing these checks is not proof of exhaustive exploration.

The 30-day return fits use 2,000 retained draws per chain. Longer fits use refined
interwoven updates and 5,000 draws per chain, except the 90/180-day volatility-only
fits, which use 10,000 including continuation from their refined chains. Four
chains are retained throughout. The gallery selects `refined_long`, then `refined`,
then original output, in that order; the original samples remain available.

For dynesty the saved stopping flag means the likelihood-call budget was not hit;
the configured stopping target was dlogz=0.1, with 150 live points. Weighted ESS is
699–1,086 and reported log-evidence error is 0.205–0.432. This is one nested run per
fit, not a repeated-run stability study. Do not compare single-stock and multitask
logZ as a Bayes factor: they condition on different observed ticker sets. Return
models do not have a reported logZ.

Each predictive comparison uses 2,000 saved price paths. Quantiles and probabilities
therefore have Monte Carlo error as well as posterior sampling uncertainty. Small
differences around P(up)=0.5 should not be read as directional evidence. The
volatility-only model is symmetric in log return; its population median stays at
the last price and population P(up)=0.5. Sample deviations are simulation noise.
Return-model price tails can be extremely heavy, so finite simulated price means
or standard deviations should not be treated as stable population moments.

## Visual findings

The mean/volatility latent plots show mean-return credible bands spanning zero,
while volatility rises around large observed returns and then reverts quickly.
At 180 days this looks like a response to individual outliers, not identification
of a smooth, persistent volatility regime. Held-out values are displayed only
after fitting. The November origin follows a large training-day jump, which also
helps explain the ordinary price GPs' visible pull toward their historical level.

Representative figures (the gallery contains all model/lookback combinations):

- [30-day joint return posterior](lookback_30/heteroskedastic_gp_returns/corner.png): near-prior-shaped mean length and broad volatility uncertainty.
- [60-day joint length surface](lookback_60/heteroskedastic_gp_returns/refined/surface-ell_mu-ell_sigma.png): little mean-length objective information.
- [90-day volatility traces](lookback_90/volatility_gp_returns/refined_long/chains.png): long excursions despite passing scalar checks.
- [180-day bound-only comparison](lookback_180/heteroskedastic_gp_returns/bound-sensitivity.png): separates the floor effect from regularization.
- [180-day latent functions](lookback_180/heteroskedastic_gp_returns/refined/latents.png): negligible mean and rapid volatility changes.
- [30-day ordinary GP forecast](lookback_30/gp/forecast.png): posterior versus point-fit predictive widths.

## Complete per-fit results

Lengths are trading sessions. A_AMZN/A_GOOGL/A_AAPL and white-noise amplitudes are
log-price units. A_mu is daily log-return units; A_sigma is dimensionless latent
log-variance amplitude; m_g is the signed log-variance level in original return
units. “Best profile” is the full optimized solution, while the length grid optimum
reported separately is restricted to the 41 grid values. For inactive components,
these numerical optima are not measured timescales. Edge masses refer to the lowest
and highest 5% of each internal prior-coordinate range; the reference 5% mass
applies to log-uniform parameters, not the Normal level prior.
"""]
    for days in [30, 60, 90, 180]:
        parts.append(f'## {days}-calendar-day lookback\n')
        for model in ['gp', 'multitask_gp', 'volatility_gp_returns', 'heteroskedastic_gp_returns']:
            key = (summary.lookback_days == days) & (summary.model == model)
            block = summary[key]
            f = findings[(findings.lookback_days == days) & (findings.model == model)].iloc[0]
            p = predictions[(predictions.lookback_days == days) & (predictions.model == model)].iloc[0]
            s = sampling[(sampling.lookback_days == days) & (sampling.model == model)].iloc[0]
            parts.append(f'### {model}\n\n[Plots](index.html#{days}-{model}) · [Selected samples]({s.selected_sampling}/posterior.csv)\n')
            parts.append(table(['Parameter', 'Old', 'Best profile', 'Posterior median', '68% interval', '95% interval', 'Lower/upper edge mass'],
                [[r.parameter, number(r.old), number(r.best_profile), number(r.median),
                  f'[{number(r.lower68)}, {number(r.upper68)}]', f'[{number(r.lower95)}, {number(r.upper95)}]',
                  f'{r.mass_lower_5pct_prior_coordinate:.1%} / {r.mass_upper_5pct_prior_coordinate:.1%}'] for r in block.itertuples()]))
            parts.append(f'\nAll 30 old-objective and 30 broad-objective starts report optimizer success; '
                         f'{int(f["old_starts_within_0.1_best"])}/30 old and '
                         f'{int(f["broad_starts_within_0.1_best"])}/30 broad starts finish within 0.1 objective units of their respective best. '
                         f'Objective ranges: {f.old_objective_range:.3g} old, {f.broad_objective_range:.3g} broad. '
                         f'Failed profile points: {int(f.failed_profile_points)}.\n')
            if model in ['gp', 'multitask_gp']:
                tickers = ['AMZN'] if model == 'gp' else ['AMZN', 'GOOGL', 'AAPL']
                parts.append('Amplitude inactive probabilities: '+', '.join(f'{t} {f["inactive_probability_"+t]:.1%}' for t in tickers)+'. '
                             f'Length-grid optimum {f.ell_profile_grid_best:.3g}; losses at lower/upper edges '
                             f'{f.ell_profile_loss_lower_edge:.3g}/{f.ell_profile_loss_upper_edge:.3g} log-likelihood units. '
                             f'Nested weighted ESS {s.weighted_ess:.1f}; logZ {s.logZ:.3f} ± {s.logZ_error:.3f}.\n')
            else:
                parts.append(f'Volatility-grid optimum {f.ell_sigma_profile_grid_best:.3g}; retaining old penalties '
                             f'while extending that length gives {f.ell_sigma_old_penalties_extended_best:.3g}. '
                             f'Lower/upper-edge profile losses {f.ell_sigma_profile_loss_lower_edge:.3g}/'
                             f'{f.ell_sigma_profile_loss_upper_edge:.3g} ELBO units. '
                             f'P(ell_sigma <1 session)={f.ell_sigma_prob_below_1:.1%}; '
                             f'P(<2)={f.ell_sigma_prob_below_2:.1%}. '
                             f'Volatility inactive probability {f.volatility_inactive_probability:.1%}; '
                             f'activity criterion {bool(f.volatility_process_active)}. '
                             f'Max split R-hat {s.max_split_rhat:.3f}, min ESS {s.min_autocorrelation_ess:.1f}.\n')
                if model == 'heteroskedastic_gp_returns':
                    parts.append(f'Mean inactive probability {f.mean_inactive_probability:.1%}; mean_process_active=False. '
                                 'Mean amplitude consistent with zero; mean length scale unconstrained. '
                                 f'Conditional on active amplitude, ell_mu 95% interval '
                                 f'[{f.ell_mu_conditional_active_lower95:.3g}, {f.ell_mu_conditional_active_upper95:.3g}]. '
                                 f'Full mean-length profile range {f.ell_mu_profile_range:.3g} ELBO units; '
                                 f'marginal KL from log-uniform prior {f.ell_mu_marginal_KL_vs_log_uniform_nats:.4f} nats.\n')
            parts.append(f'Terminal posterior median **${p.terminal_price_median:.2f}**, 95% interval '
                         f'**[${p.terminal_price_lower95:.2f}, ${p.terminal_price_upper95:.2f}]**; '
                         f'old point fit ${p.old_terminal_price_median:.2f} '
                         f'[${p.old_terminal_price_lower95:.2f}, ${p.old_terminal_price_upper95:.2f}]. '
                         f'Width ratio **{p.interval_width_ratio:.3f}**. '
                         f'Sampled P(up)={p.probability_up:.3f}; terminal log-return sigma={p.terminal_log_return_sigma:.4f}.\n')
    parts.append("""## Verification and reproducibility

Regenerate this report and the CSV summaries using only saved artifacts:

```bash
OPENBLAS_NUM_THREADS=1 .conda-env/bin/python examples/gp_diagnostic_report.py
```

The generator checks all 16 fits against the original benchmark snapshot identities
and values, training checksums and date separation, posterior CSV/NPZ coordinates
and weights, all weighted 68/95% intervals, stored chain diagnostics or nested
weighted ESS/call-budget flags, and predictive quantiles/probabilities. It fails
on missing selected artifacts rather than silently omitting a fit. It does not
instantiate models, run optimization, draw samples, download prices or rerun a
backtest. The source artifact hashes in [verification.json](verification.json)
are checked before/after generation. The recorded environment in
[runtime.json](runtime.json) describes report generation, not original inference.

Original per-fit inference metadata predates the last activity-diagnostic code
change; refreshed activity probabilities live in this report and identifiability.csv.
The saved draws and original per-fit files are preserved. The preceding repository
audit passed all 133 tests; after report completion the 13 diagnostic tests passed
again. Regenerating all eight report/summary files produced identical hashes,
479 source artifacts were preserved, and 297 local links resolved. A deliberately
corrupted posterior-weight copy was correctly rejected. Dependency deprecation
warnings remain. Artifact verification
establishes consistency and reproducibility of the reported numbers, not an
independent proof of the statistical model or MCMC implementation.

## Decision before any larger backtest

The saved results support completing this diagnostic experiment. They do not support
interpreting the old two-day volatility or 20-day mean point estimates literally.
Treat mean-return length as unidentified, and short volatility length as unresolved
at daily cadence. Ordinary GP lengths, especially shared multitask lengths, carry
more data information but still need posterior uncertainty.

A future inference study should assess prior sensitivity and stronger/repeated
sampling diagnostics, particularly for the short-lookback and lower-edge cases,
before claiming robust hyperparameter measurements. No large historical backtest
or production-model change is part of this report completion.
""")
    (root/'REPORT.md').write_text('\n\n'.join(parts)+'\n')
