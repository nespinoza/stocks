# Missing time-series models worth testing

October 17, 2026. Targeted literature review for the repository's weekly AMZN forecasts and planned external regressors. This is a research recommendation, not a new model benchmark. No forecasting models were added or fitted.

## Recommendation

**The most important omission is GARCH, particularly an asymmetric specification such as GJR-GARCH with both Gaussian and Student-t innovations.** Before choosing the heteroskedastic return GP as the long-term architecture, test it against this family and against a regularized predictive regression with GARCH residual variance.

The GP remains a useful candidate. Its ability to accept predictors after extension is not unique: regression-GARCH and state-space models also support them. Existing GP results cannot establish superiority over those untested alternatives.

The repository currently has persistence with random-walk intervals, constant-volatility GBM variants, OU dynamics on returns, a constant-innovation-covariance VAR, price GPs, and latent return GPs. It lacks the ARCH/GARCH family, EWMA variance, classical autoregressive stochastic volatility, predictive regression with time-varying coefficients, and fitted regime-switching models. Its existing retrospective regime labels are analysis categories, not a model of state transitions. The return GPs already belong broadly to stochastic-volatility modeling and generate heavy-tailed predictive mixtures; the missing alternatives are specific volatility dynamics and innovation distributions, not all stochastic volatility or all heavy tails.

## First priority: GARCH and asymmetric GARCH

GARCH(1,1) makes today's conditional variance depend on yesterday's squared surprise and variance. GJR-GARCH adds a separate response to a negative surprise; EGARCH provides another asymmetric formulation using log variance. This is a different inductive assumption from smoothing latent log variance over time with a GP.

Hansen and Lunde (2005) compared 330 ARCH-type specifications. In their IBM experiment, models allowing an asymmetric volatility response outperformed ordinary GARCH(1,1). Their exchange-rate result differed. This supports testing asymmetric equity-volatility models, not assuming a universal winner. Crucially, their Student-t variants were not uniformly superior: Gaussian variants performed better on average in the IBM variance-forecast comparison. [Paper](https://faculty.washington.edu/ezivot/econ589/DoesAnythingBeatGarch11.pdf).

Proposed first batch, all initially using the same zero-log-return mean to isolate variance/distribution effects:

| Proposed model | What it tests |
|---|---|
| EWMA variance with Gaussian innovations | Whether simple recency weighting is sufficient |
| GARCH(1,1), Gaussian | Persistent conditional variance with mean reversion |
| GARCH(1,1), Student-t | Incremental benefit of a heavy-tailed innovation distribution |
| GJR-GARCH(1,1), Gaussian | Incremental benefit of a sign-dependent volatility response |
| GJR-GARCH(1,1), Student-t | Combination of asymmetric dynamics and heavy-tailed innovations |

EWMA is the inexpensive RiskMetrics-style control, not a proposed source of directional information. [RiskMetrics technical document](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a). GJR and standardized Student-t components have direct implementations in the `arch` package; that package is not currently a repository dependency. [Official modeling documentation](https://bashtage.github.io/arch/univariate/univariate_volatility_modeling.html).

**If only one substantive model can be added first, choose GJR-GARCH with Student-t errors**, while retaining a Gaussian version as the immediate distributional control. That priority is my judgment about the gaps in this repository; it is not an AMZN result reported by the cited papers. EGARCH is a sensible subsequent sensitivity comparison, rather than a reason to begin with a large parameter search.

## Second priority: regression or ARX with GARCH errors

For future explanatory variables, test a conditional mean such as

`return[t] = intercept + phi * return[t-1] + beta' * features[t-1] + error[t]`,

with GJR-GARCH conditional variance for the error. Start with a small feature set; shrink coefficients when the number of predictors grows. Compare a fixed regularized regression against the feature-aware GP on identical predictors and forecast origins. This separates the information in the predictors from the benefit of nonlinear GP structure.

An ARX mean with GARCH errors is directly supported in `arch`. Its documented multi-step forecasts require future predictor values or forecasts; simply supplying regressors does not solve their availability problem. The package's usual exogenous-input interface targets the **mean**, not automatically the variance. A GARCH-X variance equation would be a separate extension. [Official exogenous-regressor forecasting documentation](https://bashtage.github.io/arch/univariate/univariate_forecasting_with_exogenous_variables.html).

For a seven-calendar-day target, either simulate the required future covariate paths and propagate their uncertainty, or fit a direct horizon-return regression using only origin-known features. In the latter case, validate it as a distinct horizon model; it is not automatically a daily GARCH path model. Scheduled earnings indicators can be available ahead, whereas realized future market returns and volume cannot be used as inputs.

Low-order ARMA on returns, with GARCH errors, is also a useful small baseline. It is not identical to the current OU implementation, which restricts its autoregressive coefficient to the stable positive OU range and often falls back to IID returns. I would not prioritize an unrestricted automated ARIMA search over the missing volatility benchmarks.

## Third priority: state-space alternatives

Two related but distinct questions merit testing:

**Autoregressive stochastic volatility:** model latent log variance as a persistent AR(1), rather than with a dense GP covariance. Kim, Shephard, and Chib establish likelihood-based Bayesian inference and comparisons with ARCH models. This is the natural parametric comparator for the existing latent-volatility GP. [Paper](https://www.nuff.ox.ac.uk/economics/papers/1996/w26/chib8.pdf).

**Stochastic volatility with leverage:** allow return and volatility shocks to be correlated. Omori, Chib, Shephard, and Nakajima develop efficient inference for this extension and describe heavy-tailed variants. That dependence is another substantive difference from the repository's current independent latent mean/variance GP priors. [Author-hosted publication page](https://shephard.scholars.harvard.edu/publications/stochastic-volatility-leverage-fast-and-efficient-likelihood-inference), [working-paper text](https://www.economics.uci.edu/files/docs/colloqpapers/w06/chib.pdf).

**Time-varying predictive coefficients:** let a predictor's coefficient change gradually, with shrinkage toward a fixed coefficient. Dangl and Halling (2012) study Bayesian predictive regressions with changing coefficients for monthly S&P 500 returns. This motivates testing changing predictor effects; it does not establish superiority for weekly single-stock returns. [Paper and author manuscript](https://orbilu.uni.lu/handle/10993/57249).

I would implement the fixed-coefficient regression-GARCH baseline before combining changing coefficients and stochastic volatility. A simple linear-Gaussian state-space model can use a Kalman filter, but latent stochastic volatility or Student-t observations require additional approximation or simulation; efficient inference should not be assumed automatically.

## Conditional priority: HAR realized volatility

Corsi's HAR-RV combines daily, weekly, and monthly realized-volatility components in a parsimonious autoregression. Its multi-horizon structure is particularly relevant to our weekly forecasts. [Corsi (2009)](https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf).

**The main prerequisite is a suitable realized-volatility dataset**, usually derived from intraday returns. Our frozen file contains only daily adjusted closes. A model of daily squared returns or rolling daily variance could be a useful proxy-based experiment, but should not be presented as reproducing the original HAR-RV evidence. Moreover, a variance forecast needs an additional observation/distribution model to produce the return probabilities and price intervals required by this repository. HAR-RV by itself is not a directional-return model.

## Later priority: regime switching and larger neural models

Hamilton and Susmel explicitly studied regime-dependent ARCH dynamics for weekly US stock returns. A two-state volatility model could test abrupt changes between quieter and more turbulent periods. [Hamilton and Susmel (1994)](https://www.sciencedirect.com/science/article/pii/0304407694900671). I would defer it until the simpler volatility models are evaluated: estimating state transitions adds complexity. Historical forecasts must use filtered state probabilities based only on available observations, not states smoothed using later outcomes.

Neural methods should remain candidates if we later build a much larger stock panel. Gu, Kelly, and Xiu report gains from nonlinear predictor interactions in a broad empirical asset-pricing study, including stock-level and aggregate predictions. Their large monthly stock-panel setting is very different from fitting one stock on 17–126 daily observations per fold. Their findings do not justify prioritizing an LSTM or Transformer on the present setup. [Gu, Kelly, and Xiu (2020)](https://doi.org/10.1093/rfs/hhaa009).

The broader return-prediction evidence is mixed. Goyal, Welch, and Zafirov's 2024 reassessment finds that many published equity-premium predictors lose significance or perform poorly out of sample, while some survive. It concerns aggregate equity-premium prediction, not this Amazon horizon, but reinforces the need for a protected evaluation period. [Published study](https://academic.oup.com/rfs/article/37/11/3490/7749383).

## What the local data suggests

I computed descriptive statistics from the frozen AMZN snapshot for January 4, 2016–August 31, 2026: 2,680 daily log returns, including the return into January 4 from the previous session.

| Statistic | Value |
|---|---:|
| Sample excess kurtosis | 5.400 |
| Lag-one return correlation | −0.0165 |
| Lag-one absolute-return correlation | 0.1987 |
| Lag-one squared-return correlation | 0.1562 |

These full-period, post-hoc diagnostics suggest persistent return magnitudes and heavy unconditional tails, with little simple lag-one mean correlation. They motivate volatility benchmarks; they do not prove leverage asymmetry, Student-t conditional innovations, or a profitable forecast. A Gaussian volatility mixture can itself produce unconditional heavy tails.

Reproduction:

```python
import numpy as np
import pandas as pd

prices = pd.read_csv(
    'historical_validation/AMZN/2016-2026-v1/prices.csv',
    index_col=0, parse_dates=True,
)
r = np.log(prices.AMZN).diff().loc['2016-01-01':].dropna()
print(len(r), r.kurt(), r.autocorr(1), abs(r).autocorr(1), (r*r).autocorr(1))
```

## A fair next experiment

1. **Separate conditional mean and uncertainty experiments.** First compare the five volatility/distribution candidates above with the existing volatility-return GP and random-walk baseline. Then give regression-GARCH and the heteroskedastic GP the same predictors. Add estimated or shrunk means as a declared experiment rather than confounding every comparison.
2. **Use sufficient estimation history.** My proposed development grid is 252, 504, and 756 trading observations for the new parametric models, not a literature-certified minimum. The present 30-calendar-day window has only 17–20 prices. The saved snapshot begins in June 2015, so longer windows cannot cover all original 2016 origins: obtain an explicitly versioned earlier-history snapshot or compare all candidates on common later origins. Retain common-window comparisons where feasible, and disclose model-specific window choices.
3. **Preserve the seven-calendar-day horizon and strict origin cutoff.** Simulate the actual four/five exchange-session returns jointly, updating variance along each path. Do not obtain a weekly predictive distribution by mechanically multiplying a daily interval by the square root of five. Propagate AR dependence, stochastic variance, and uncertain covariates as applicable.
4. **Score what each model predicts.** Use terminal-return WIS or sample CRPS, Brier score, interval coverage and width, and normalized median errors. For variance forecasts, consider QLIKE with an appropriately matched proxy; its robustness requires assumptions about proxy unbiasedness. Daily squared returns are noisy and, with a nonzero conditional mean, do not directly equal a variance proxy without adjustment. Also distinguish the sum of daily variances from terminal-return variance when returns are serially dependent. [Patton (2011)](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf).
5. **Keep numerical failures, runtime, and uncertainty treatment visible.** A maximum-likelihood GARCH fit does not automatically include parameter uncertainty. Label that difference from Bayesian GP forecasts; consider a declared bootstrap or posterior variant if material. With Student-t log-return innovations, constrain degrees of freedom above two for finite return variance, but do not promise finite exponentiated price means: Student-t exponential moments do not exist. Preserve the repository's quantile/median reporting and undefined-price-moment handling.
6. **Use a new frozen experiment and a genuinely uninspected holdout.** The inspected 2016–2026 period is development data. Preserve block-based comparisons and account for the enlarged set of tested models; do not reuse a fixed 96-comparison count after expanding the family. Report results on matching dates and specify any fallback before the final evaluation.

The immediate research choice is **GJR-GARCH plus a regression mean as the leading new challenger to the heteroskedastic GP**. HAR-RV becomes a high-priority volatility comparator once suitable data exists; parametric stochastic volatility and changing-coefficient regression are the next architectural alternatives.
