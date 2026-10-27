"""Portfolio adapter for the existing variational return GP; no GP refitting shortcuts."""
import numpy as np
from scipy.optimize import minimize, brentq
from stock_api.heteroskedastic import fit_latent_returns, _future, _root


def shrunk_covariance(observations, shrinkage=.5):
    observations = np.asarray(observations, float)
    std = observations.std(axis=0, ddof=1)
    if np.any(std <= 0):
        raise ValueError('Constant return/residual series')
    corr = np.corrcoef(observations, rowvar=False)
    corr = (1-shrinkage)*corr + shrinkage*np.eye(len(std))
    return corr * np.outer(std, std), corr


def joint_gp(returns, horizon, *, mean_gp=True, n_paths=10000, seed=0, shrinkage=.5):
    """Temporal GP simulation with correlated f, g and innovation driver families.

    Monthly LOG-return moments exist; arithmetic moments of this lognormal
    variance mixture need not. Do not interpret mu as expected simple return.
    """
    returns = np.asarray(returns, float)
    if n_paths % 2:
        raise ValueError('Antithetic paths require an even count')
    posteriors = [fit_latent_returns(returns[:, i], mean_gp=mean_gp) for i in range(returns.shape[1])]
    residuals = np.column_stack([
        (returns[:, i]-p['scale']*p['f_mean']) /
        (p['scale']*np.exp(.5*(p['g_mean']+.5*np.diag(p['g_cov']))))
        for i, p in enumerate(posteriors)])
    _, corr = shrunk_covariance(residuals, shrinkage)
    rng = np.random.default_rng(seed)
    shape = (n_paths//2, horizon, returns.shape[1])
    fdrivers = rng.standard_normal(shape) @ _root(corr).T
    gdrivers = rng.standard_normal(shape) @ _root(corr*corr).T
    edrivers = rng.standard_normal(shape) @ _root(corr).T
    samples, means, variances, drift_vars, params, latent = [], [], [], [], [], []
    for i, p in enumerate(posteriors):
        fm, fc, gm, gc = _future(p, horizon)
        f = fdrivers[:, :, i] @ _root(fc).T
        g = gm + gdrivers[:, :, i] @ _root(gc).T
        noise = p['scale']*(f + np.exp(.5*g)*edrivers[:, :, i]).sum(axis=1)
        mu = p['scale']*fm.sum()
        dv = p['scale']**2*fc.sum()
        nv = p['scale']**2*np.exp(gm+.5*np.diag(gc)).sum()
        samples.append(np.r_[noise, -noise])
        means.append(mu); variances.append(dv+nv); drift_vars.append(dv)
        params.append({k: p[k] for k in ('scale','ag','ellg','mg','af','ellf','iterations','converged','starts_converged','projected_gradient','optimizer_message','objective')})
        latent.append(np.column_stack([p['scale']*fm, p['scale']**2*np.diag(fc), gm+2*np.log(p['scale']), np.diag(gc), p['scale']**2*np.exp(gm+.5*np.diag(gc))]))
    centered = np.column_stack(samples)
    correction = np.sqrt(np.array(variances)/centered.var(axis=0, ddof=1))
    centered *= correction
    samples = centered + means
    covariance = centered.T @ centered/(n_paths-1)
    if not np.isfinite(samples).all() or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError('Invalid joint predictive distribution')
    return dict(samples=samples, mu=np.array(means), covariance=covariance,
                drift_variance=np.array(drift_vars), correlation=corr,
                mc_variance_correction=correction, parameters=params, latent=np.array(latent))


def optimize_weights(mu, covariance, pre_weights, risk_aversion=10., cost_rate=.001, cap=.2):
    """Convex quadratic program with explicit absolute-turnover epigraph."""
    mu, covariance, pre_weights = map(np.asarray, (mu, covariance, pre_weights))
    n = len(mu)
    if cap*n < 1 or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError('Infeasible cap or indefinite covariance')
    eye = np.eye(n)
    a = np.block([[-eye, eye], [eye, eye]])
    b = np.r_[-pre_weights, pre_weights]
    def objective(x):
        w = x[:n]
        return .5*risk_aversion*w@covariance@w-mu@w+cost_rate*x[n:].sum()
    def gradient(x):
        return np.r_[risk_aversion*covariance@x[:n]-mu, np.full(n, cost_rate)]
    eq_jac = np.r_[np.ones(n), np.zeros(n)]
    initial = np.full(n, 1/n)
    result = minimize(objective, np.r_[initial, np.abs(initial-pre_weights)], jac=gradient,
        method='SLSQP', bounds=[(0,cap)]*n+[(0,None)]*n,
        constraints=[{'type':'eq','fun':lambda x:x[:n].sum()-1,'jac':lambda x:eq_jac},
                     {'type':'ineq','fun':lambda x:a@x-b,'jac':lambda x:a}],
        options={'ftol':1e-12,'maxiter':2000})
    w = result.x[:n]
    if not result.success or abs(w.sum()-1)>1e-8 or w.min() < -1e-8 or w.max()>cap+1e-8 or (a@result.x-b).min() < -1e-8:
        raise ValueError(f'Portfolio optimizer failed: {result.message}')
    return w, {'optimizer_success':bool(result.success), 'optimizer_message':str(result.message),
               'optimizer_iterations':int(result.nit), 'optimizer_objective':float(result.fun)}


def trade(wealth, old_dollars, weights, cost_rate):
    """Self-financing all-invested trade; buys AND sells pay proportional fees."""
    old_dollars, weights = np.asarray(old_dollars), np.asarray(weights)
    if wealth <= 0 or not 0 <= cost_rate < .1 or abs(weights.sum()-1)>1e-8:
        raise ValueError('Invalid trade')
    def equation(fee):
        return fee-cost_rate*np.abs((wealth-fee)*weights-old_dollars).sum()
    fee = brentq(equation, 0, wealth) if cost_rate else 0.
    dollars = (wealth-fee)*weights
    changes = dollars-old_dollars
    return dollars, fee, np.maximum(changes,0).sum(), np.maximum(-changes,0).sum()


def crps(samples, actual):
    x = np.sort(np.asarray(samples), axis=0)
    n = len(x)
    coeff = (2*np.arange(1,n+1)-n-1).reshape((-1,)+(1,)*(x.ndim-1))
    return np.mean(np.abs(x-actual),axis=0)-(coeff*x).sum(axis=0)/n**2


def check_chronology(training_dates, origin, execution, targets):
    if not (max(training_dates) <= origin < execution <= min(targets)):
        raise ValueError('Chronological leakage')
