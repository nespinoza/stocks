"""Coupled variational latent Matérn GPs for daily log returns (SciPy only).

q(f) q(g) is a Gaussian mean-field approximation, with q(f) analytically
collapsed in the ELBO. q(g)=N(m_g+L_g u, (K_g^-1+diag(lambda))^-1).
Whitened means and the stationary covariance parameterization follow heteroskedastic GP
regression; both latent processes are inferred through the same likelihood.
Hyperparameters maximize a regularized ELBO, not a full Bayesian posterior.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm

from stock_api.sde import PredictiveEnsemble, _rng, _simulation_inputs

JITTER = 1e-6


def _kernel(x, z, amplitude, length):
    r = np.sqrt(3)*np.abs(x[:, None]-z[None, :])/length
    k = amplitude**2*(1+r)*np.exp(-r)
    return k, amplitude**2*r*r*np.exp(-r)


def _objective(params, y, mean_gp, state=False):
    """Negative collapsed ELBO and analytic gradients in log-positive parameters."""
    n = len(y)
    u = params[:n]
    lam = np.exp(params[n:2*n])
    h = 2*n
    ag, ellg = np.exp(params[h:h+2])
    mg = params[h+2]
    af, ellf = np.exp(params[h+3:h+5]) if mean_gp else (0., 20.)
    x = np.arange(n, dtype=float)
    kg, dlg = _kernel(x, x, ag, ellg)
    kg = kg+JITTER*np.eye(n)
    lg = np.linalg.cholesky(kg)
    b = np.eye(n)+(lg.T*lam)@lg
    factor_b = cho_factor(b, lower=True)
    sg = lg@cho_solve(factor_b, lg.T)
    sg = (sg+sg.T)/2
    m = mg+lg@u
    s = np.diag(sg)
    log_noise = m-.5*s
    noise = np.exp(log_noise)
    if mean_gp:
        kf, dlf = _kernel(x, x, af, ellf)
        kf = kf+JITTER*np.eye(n)
        cfactor = cho_factor(kf+np.diag(noise), lower=True)
        alpha = cho_solve(cfactor, y)
        qc = cho_solve(cfactor, np.eye(n))-np.outer(alpha, alpha)
        d = .5*np.diag(qc)*noise
        nl = np.log(np.diag(cfactor[0])).sum()+.5*y@alpha
    else:
        kf = None
        d = .5*(1-y*y/noise)
        nl = .5*np.sum(log_noise+y*y/noise)
    kl = .5*(-lam@s+u@u+2*np.log(np.diag(factor_b[0])).sum())
    objective = nl+.25*s.sum()+kl
    c = -.5*d+.25-.5*lam
    kinv_s = np.eye(n)-lam[:, None]*sg
    q = np.diag(lam)-lam[:, None]*sg*lam[None, :]
    gk = .5*q+(kinv_s*c)@kinv_s.T
    gradient = np.empty_like(params)
    gradient[:n] = lg.T@d+u
    gradient[n:2*n] = -lam*((sg*sg).T@c)
    inverse_l = solve_triangular(lg, np.eye(n), lower=True)
    for index, derivative in [(h,2*(kg-JITTER*np.eye(n))), (h+1,dlg)]:
        whitened_derivative = inverse_l@derivative@inverse_l.T
        phi = np.tril(whitened_derivative, -1)+.5*np.diag(np.diag(whitened_derivative))
        gradient[index] = np.sum(gk*derivative)+d@lg@phi@u
    gradient[h+2] = d.sum()
    # Fixed weak log-timescale regularization; f amplitude strongly prefers zero.
    objective += .5*ag**2+.5*((np.log(ellg)-np.log(20))/1.5)**2+.5*(mg/2)**2
    gradient[h] += ag**2
    gradient[h+1] += (np.log(ellg)-np.log(20))/1.5**2
    gradient[h+2] += mg/4
    if mean_gp:
        gradient[h+3] = np.sum(qc*(kf-JITTER*np.eye(n)))+(af/.3)**2
        gradient[h+4] = .5*np.sum(qc*dlf)+(np.log(ellf)-np.log(20))/1.5**2
        objective += .5*(af/.3)**2+.5*((np.log(ellf)-np.log(20))/1.5)**2
    if state:
        result = {'ag':ag,'ellg':ellg,'mg':mg,'af':af,'ellf':ellf,
                  'a':solve_triangular(lg.T,u,lower=False),'q':q,'g_mean':m,'g_cov':sg,'objective':float(objective)}
        if mean_gp:
            result.update(f_mean=kf@alpha, f_cov=kf-kf@cho_solve(cfactor,kf),
                          c_factor=cfactor, alpha=alpha)
        else:
            result.update(f_mean=np.zeros(n), f_cov=np.zeros((n,n)))
        return result
    return float(objective), gradient


def fit_latent_returns(returns, *, mean_gp=False, maxiter=2500):
    """Infer latent functions and separate kernel parameters from training returns only."""
    returns = np.asarray(returns,dtype=float)
    if returns.ndim != 1 or not 8 <= len(returns) <= 512 or not np.isfinite(returns).all():
        raise ValueError('Latent return GPs require 8–512 finite daily training returns')
    scale = max(float(np.sqrt(np.mean(returns**2))), 1e-6)
    y = returns/scale  # Deliberately no centering: zero expected log return means zero.
    n = len(y)
    bounds = [(-8,8)]*n+[(-12,6)]*n+[(np.log(.03),np.log(3)),(np.log(2),np.log(252)),(-6,3)]
    if mean_gp:
        bounds += [(np.log(.01),np.log(2)),(np.log(2),np.log(252))]
    fits = []
    for ellg, ellf in [(20,20),(7,40)]:
        initial = np.r_[np.zeros(n),np.full(n,np.log(.5)), np.log(.4),np.log(ellg),0.]
        if mean_gp:
            initial = np.r_[initial,np.log(.2),np.log(ellf)]
        fit = minimize(_objective,initial,args=(y,mean_gp),jac=True,method='L-BFGS-B',bounds=bounds,
                       options={'maxiter':maxiter,'ftol':1e-9,'gtol':1e-5,'maxls':40,'maxcor':30})
        projected = fit.x-np.clip(fit.x-fit.jac, *np.asarray(bounds).T)
        fit.projected_gradient = float(np.max(np.abs(projected)))
        if np.isfinite(fit.fun) and fit.fun < 1e5 and fit.success and fit.projected_gradient < .02:
            fits.append(fit)
    if not fits:
        raise ValueError('Latent GP variational inference did not converge; no fallback forecast emitted')
    best = min(fits,key=lambda f:f.fun)
    posterior = _objective(best.x,y,mean_gp,state=True)
    posterior.update(scale=scale, training_returns=n, mean_gp=mean_gp,
                     iterations=int(best.nit), converged=True,
                     optimizer_message=str(best.message), starts_converged=len(fits),
                     projected_gradient=best.projected_gradient)
    return posterior


def _future(posterior, steps):
    n = posterior['training_returns']
    x, future = np.arange(n,dtype=float),np.arange(n,n+steps,dtype=float)
    cross,_ = _kernel(x,future,posterior['ag'],posterior['ellg'])
    k,_ = _kernel(future,future,posterior['ag'],posterior['ellg'])
    gm = posterior['mg']+cross.T@posterior['a']
    gc = k+JITTER*np.eye(steps)-cross.T@posterior['q']@cross
    if posterior['mean_gp']:
        cross,_ = _kernel(x,future,posterior['af'],posterior['ellf'])
        k,_ = _kernel(future,future,posterior['af'],posterior['ellf'])
        fm = cross.T@posterior['alpha']
        fc = k+JITTER*np.eye(steps)-cross.T@cho_solve(posterior['c_factor'],cross)
    else:
        fm,fc = np.zeros(steps),np.zeros((steps,steps))
    return fm,fc,gm,gc


def _root(covariance):
    covariance = (covariance+covariance.T)/2
    eig,vec = np.linalg.eigh(covariance)
    if eig.min() < -1e-7*max(1.,np.max(np.abs(eig))):
        raise ValueError('Latent predictive covariance is not positive semidefinite')
    # Remove only floating-point negative eigenvalues, not predictive tails.
    return vec*np.sqrt(np.maximum(eig,0))


def _return_draws(fm,fc,gm,gc,scale,n_paths,rng):
    half = (n_paths+1)//2
    g = gm+rng.standard_normal((half,len(gm)))@_root(gc).T
    f_noise = rng.standard_normal((half,len(fm)))@_root(fc).T
    with np.errstate(over='raise',invalid='raise'):
        noise = f_noise+np.exp(.5*g)*rng.standard_normal(g.shape)
        returns = scale*np.concatenate([fm+noise,fm-noise],axis=0)[:n_paths]
    return returns,g


def paths_from_returns(initial, returns):
    with np.errstate(over='raise',invalid='raise'):
        paths = initial*np.exp(np.column_stack([np.zeros(len(returns)),np.cumsum(returns,axis=1)]))
    if not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError('Posterior predictive price paths overflowed or underflowed')
    return paths


def _sample_statistics(prices, center):
    with np.errstate(over='raise',invalid='raise'):
        statistics = {'mean_price':prices.mean(axis=0),'median_price':np.exp(center),
                      'price_std':prices.std(axis=0,ddof=1),
                      'lower_95':np.quantile(prices,.025,axis=0),
                      'upper_95':np.quantile(prices,.975,axis=0)}
    for n in (1,2,3):
        statistics[f'sigma_{n}_lower'] = np.quantile(prices,norm.cdf(-n),axis=0)
        statistics[f'sigma_{n}_upper'] = np.quantile(prices,norm.cdf(n),axis=0)
    return statistics


def _forecast(prices, steps, *, mean_gp, include_fit=False, n_paths=10_000, seed=None, rng=None):
    prices = np.asarray(prices,dtype=float)
    if prices.ndim != 2 or prices.shape[1] < 1 or (prices[:,0] <= 0).any():
        raise ValueError('Expected positive target prices in column zero')
    _simulation_inputs(float(prices[-1,0]),steps,1,n_paths)
    if n_paths < 100:
        raise ValueError('At least 100 posterior predictive paths are required')
    generator = _rng(seed if rng is not None or seed is not None else 0,rng)
    logs = np.log(prices[:,0])
    posterior = fit_latent_returns(np.diff(logs),mean_gp=mean_gp)
    fm,fc,gm,gc = _future(posterior,steps)
    scale = posterior['scale']
    returns,g_draws = _return_draws(fm,fc,gm,gc,scale,n_paths,generator)
    paths = paths_from_returns(prices[-1,0],returns)
    center = logs[-1]+scale*np.cumsum(fm)
    f_sum_variance = np.array([fc[:i,:i].sum() for i in range(1,steps+1)])
    variance = scale**2*(f_sum_variance+np.cumsum(np.exp(gm+.5*np.diag(gc))))
    statistics = _sample_statistics(paths[:,1:],center)
    conditional_var = scale**2*(f_sum_variance+np.cumsum(np.exp(g_draws),axis=1))
    statistics['probability_up'] = norm.cdf((center-logs[-1])/np.sqrt(conditional_var)).mean(axis=0)
    if not mean_gp:
        statistics['median_price'] = np.full(steps,prices[-1,0])
        statistics['probability_up'] = np.full(steps,.5)
    diagnostics = {
        'inference':'collapsed_gaussian_variational', 'converged':posterior['converged'],
        'projected_gradient':posterior['projected_gradient'],
        'iterations':posterior['iterations'],'optimizer_message':posterior['optimizer_message'],
        'starts_converged':posterior['starts_converged'], 'negative_regularized_elbo':posterior['objective'],
        'mean_gp_amplitude':float(scale*posterior['af']) if mean_gp else None,
        'ell_mu':float(posterior['ellf']) if mean_gp else None,
        'volatility_gp_amplitude':float(posterior['ag']), 'ell_sigma':float(posterior['ellg']),
        'log_variance_mean_level':float(posterior['mg']+2*np.log(scale)),
        'return_scale':scale,'time_unit':'trading_days','n_paths':n_paths,
        'expected_terminal_log_return':float(center[-1]-logs[-1]),
        'terminal_probability_up':float(statistics['probability_up'][-1]),
        'terminal_log_return_sigma':float(np.sqrt(variance[-1])),
        'training_mean_return':(scale*posterior['f_mean']).tolist(),
        'training_expected_variance':(scale**2*np.exp(posterior['g_mean']+.5*np.diag(posterior['g_cov']))).tolist(),
        'price_moment_note':'Finite Monte Carlo summaries; lognormal variance mixing has no finite population price mean.',
        '_statistics':statistics,'_ensemble':PredictiveEnsemble(paths)}
    if include_fit:
        r,_ = _return_draws(posterior['f_mean'],posterior['f_cov'],posterior['g_mean'],
                           posterior['g_cov'],scale,n_paths,generator)
        fit_center = logs[:-1]+scale*posterior['f_mean']
        with np.errstate(over='raise',invalid='raise'):
            fitted_prices = np.exp(logs[:-1]+r)
        fit_var = scale**2*(np.diag(posterior['f_cov'])+
                            np.exp(posterior['g_mean']+.5*np.diag(posterior['g_cov'])))
        diagnostics['_fit'] = (1,fit_center,fit_var)
        diagnostics['_fit_statistics'] = _sample_statistics(fitted_prices,fit_center)
    return center,variance,diagnostics


def volatility_gp_returns(prices,steps,**kwargs):
    return _forecast(prices,steps,mean_gp=False,**kwargs)


def heteroskedastic_gp_returns(prices,steps,**kwargs):
    return _forecast(prices,steps,mean_gp=True,**kwargs)


RETURN_GP_MODELS = {'volatility_gp_returns':volatility_gp_returns,
                    'heteroskedastic_gp_returns':heteroskedastic_gp_returns}
