"""Training-only forecasts using existing models and existing Bayesian GP samplers."""
import warnings
import numpy as np
from scipy.stats import norm
from stock_api.models import MODELS
from stock_api.sde import SDE_MODELS
from stock_api.diagnostics.problems import GaussianProblem,ReturnProblem,component_activity
from stock_api.diagnostics.sampling import sample_gaussian,sample_latent,weighted_quantile
from stock_api.diagnostics.predictive import draw_predictive

GP_MODELS={'gp','multitask_gp','volatility_gp_returns','heteroskedastic_gp_returns'}
LEVELS=(.5,.68,.9,.95)


def forecast(train, steps, model, config, seed):
    """No test observations or full snapshot are accepted by this function."""
    train=train.copy(deep=True)
    if model not in ['multitask_gp','var']:train=train.iloc[:,:1]
    initial=float(train.iloc[-1,0]);rng=np.random.default_rng(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if model in GP_MODELS:
            settings=config.inference.model_copy(update={'seed':seed,'predictive_draws':config.n_paths})
            kind=GaussianProblem if model in ['gp','multitask_gp'] else ReturnProblem
            problem=kind(train,model,settings.priors,fit_old=False)
            samples,weights,latents,chains,info=(sample_gaussian(problem,settings) if kind is GaussianProblem else sample_latent(problem,settings,progress=lambda _:None))
            physical=problem.physical(samples)
            info['hyperparameters']={name:dict(zip(['lower95','lower68','median','upper68','upper95'],map(float,weighted_quantile(physical[:,j],weights)))) for j,name in enumerate(problem.names)}
            info.update(component_activity(problem,physical,weights))
            info['prior_edge_mass']={name:{'lower':float(weights[samples[:,j]<lo+.05*(hi-lo)].sum()),
                                          'upper':float(weights[samples[:,j]>hi-.05*(hi-lo)].sum())}
                                     for j,(name,(lo,hi)) in enumerate(zip(problem.names,problem.bounds))}
            draws=draw_predictive(problem,samples,weights,latents,steps,settings,include_comparison=False)
            paths=draws['paths'][:,1:];terminal=np.log(paths[:,-1]/initial)
            median=np.median(paths,axis=0)
            # A zero-mean return mixture is symmetric. Preserve the exact null
            # location/probability rather than mistaking finite MC noise for alpha.
            zero_mean=model=='volatility_gp_returns'
            if zero_mean: median=np.full(steps,initial)
            heavy=kind is ReturnProblem
            result={'path_mean':None if heavy else paths.mean(axis=0).tolist(),
                    'path_median':median.tolist(),
                    'expected_log_return':0. if zero_mean else float(terminal.mean()),
                    'median_log_return':0. if zero_mean else float(np.median(terminal)),
                    'terminal_sigma':float(terminal.std()),
                    'probability_up':.5 if zero_mean else float((terminal>0).mean()),
                    'distribution':'posterior_mixture','terminal_samples':terminal.tolist(),
                    'normal_location':None,'normal_sigma':None,
                    'intervals':{str(level):{'path_lower':np.quantile(paths,(1-level)/2,axis=0).tolist(),
                                          'path_upper':np.quantile(paths,(1+level)/2,axis=0).tolist(),
                                          'return_lower':float(np.quantile(terminal,(1-level)/2)),
                                          'return_upper':float(np.quantile(terminal,(1+level)/2))} for level in LEVELS},
                    'price_moments_note':'Omitted: latent log-variance mixture may have nonfinite population price moments' if heavy else 'Monte Carlo moments',
                    'parameter_uncertainty':'GP hyperparameters and latent uncertainty marginalized'}
        else:
            kwargs={'seed':seed,'n_paths':config.n_paths} if model in SDE_MODELS else {}
            means,variances,info=MODELS[model](train.to_numpy(),steps,**kwargs)
            ensemble=info.pop('_ensemble',None);info.pop('_statistics',None)
            sd=np.sqrt(variances);location=float(means[-1]-np.log(initial));sigma=float(sd[-1])
            terminal=ensemble.terminal_log_returns if ensemble is not None else rng.normal(location,sigma,config.n_paths)
            median=np.exp(means)
            if model=='last_price':median=np.full(steps,initial);location=0.
            result={'path_mean':np.exp(means+.5*variances).tolist(),'path_median':median.tolist(),
                    'expected_log_return':location,'median_log_return':location,'terminal_sigma':sigma,
                    'probability_up':float(norm.cdf(location/sigma)) if sigma>0 else float(location>0),
                    'distribution':'conditional_gaussian_log_return','terminal_samples':terminal.tolist(),
                    'normal_location':location,'normal_sigma':sigma,
                    'intervals':{str(level):{'path_lower':np.exp(means+norm.ppf((1-level)/2)*sd).tolist(),
                                          'path_upper':np.exp(means+norm.ppf((1+level)/2)*sd).tolist(),
                                          'return_lower':location+float(norm.ppf((1-level)/2))*sigma,
                                          'return_upper':location+float(norm.ppf((1+level)/2))*sigma} for level in LEVELS},
                    'parameter_uncertainty':'Plug-in fitted parameters; innovation uncertainty only',
                    'price_moments_note':'Analytic conditional Gaussian log-price moments'}
            info.setdefault('converged',True)
            info['method']=info.get('transition','existing_gaussian_marginals')
        info['warnings']=list(dict.fromkeys(str(w.message) for w in caught))
    for key in ['path_median','expected_log_return','median_log_return','terminal_sigma','probability_up','terminal_samples']:
        if not np.isfinite(result[key]).all():raise ValueError(f'Nonfinite forecast: {key}')
    if np.any(np.asarray(result['path_median'])<=0):raise ValueError('Nonpositive forecast')
    if result['terminal_sigma']<0 or not 0<=result['probability_up']<=1:raise ValueError('Invalid predictive sigma/probability')
    if result['path_mean'] is not None and not np.isfinite(result['path_mean']).all():raise ValueError('Nonfinite price moments')
    for interval in result['intervals'].values():
        lower=np.asarray(interval['path_lower']);upper=np.asarray(interval['path_upper'])
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower<=0).any() or (lower>upper).any():
            raise ValueError('Invalid predictive price intervals')
        if not np.isfinite([interval['return_lower'],interval['return_upper']]).all() or interval['return_lower']>interval['return_upper']:
            raise ValueError('Invalid predictive return intervals')
    result['terminal_price_median']=float(result['path_median'][-1])
    result['terminal_price_lower95']=result['intervals']['0.95']['path_lower'][-1]
    result['terminal_price_upper95']=result['intervals']['0.95']['path_upper'][-1]
    result['diagnostics']=info
    return result
