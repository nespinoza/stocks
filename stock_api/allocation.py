"""Long-only, fully invested allocation rules that need only a risk model."""
import numpy as np
from scipy.optimize import minimize


def inverse_volatility(volatility):
    weights = 1/np.asarray(volatility, float)
    return weights/weights.sum()


def equal_risk_contribution(covariance, tol=1e-12):
    """Risk parity: every asset contributes equally to portfolio variance.

    Solves the convex problem min ½yᵀΣy − (1/n)Σlog y and normalizes (Spinu 2013).
    """
    covariance = np.asarray(covariance, float)
    n = len(covariance)
    start = 1/np.sqrt(np.diag(covariance))
    result = minimize(lambda y: .5*y@covariance@y-np.log(y).sum()/n, start/start.sum(),
                      jac=lambda y: covariance@y-1/(n*y), method='L-BFGS-B',
                      bounds=[(1e-12, None)]*n, options={'ftol': tol, 'gtol': 1e-10, 'maxiter': 5000})
    if not result.success:
        raise ValueError(f'Risk parity failed: {result.message}')
    return result.x/result.x.sum()


def risk_contributions(weights, covariance):
    weights = np.asarray(weights, float)
    marginal = np.asarray(covariance)@weights
    return weights*marginal/(weights@marginal)


def minimum_variance(covariance, cap=1.):
    """Long-only minimum variance with an optional per-asset cap."""
    covariance = np.asarray(covariance, float)
    n = len(covariance)
    if cap*n < 1:
        raise ValueError('Cap too small for a fully invested portfolio')
    result = minimize(lambda w: w@covariance@w, np.full(n, 1/n), jac=lambda w: 2*covariance@w,
                      method='SLSQP', bounds=[(0, cap)]*n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1, 'jac': lambda w: np.ones(n)}],
                      options={'ftol': 1e-14, 'maxiter': 1000})
    if not result.success:
        raise ValueError(f'Minimum variance failed: {result.message}')
    weights = np.clip(result.x, 0, cap)
    return weights/weights.sum()


def mean_variance(mean, covariance, risk_aversion, current=None, cost=0., cap=1.):
    """Long-only, fully invested max of μᵀw − δ/2 wᵀΣw − cost·‖w − current‖₁.

    The L1 term charges proportional trading costs against the current
    (drifted) holdings, which creates a no-trade region around them. It is
    solved exactly by splitting the trade into buys u and sells v.
    """
    mean, covariance = np.asarray(mean, float), np.asarray(covariance, float)
    n = len(mean)
    if cap*n < 1:
        raise ValueError('Cap too small for a fully invested portfolio')
    current = np.full(n, 1/n) if current is None or np.sum(current) <= 0 else np.asarray(current, float)
    current = current/current.sum()
    weights = lambda x: current+x[:n]-x[n:]

    def objective(x):
        w = weights(x)
        return -(mean@w-.5*risk_aversion*w@covariance@w-cost*x.sum())

    def gradient(x):
        g = mean-risk_aversion*covariance@weights(x)
        return -np.r_[g-cost, -g-cost]

    split = np.hstack([np.eye(n), -np.eye(n)])
    constraints = [{'type': 'eq', 'fun': lambda x: x[:n].sum()-x[n:].sum(), 'jac': lambda x: np.r_[np.ones(n), -np.ones(n)]},
                   {'type': 'ineq', 'fun': lambda x: weights(x), 'jac': lambda x: split},
                   {'type': 'ineq', 'fun': lambda x: cap-weights(x), 'jac': lambda x: -split}]
    result = minimize(objective, np.zeros(2*n), jac=gradient, method='SLSQP', bounds=[(0, None)]*(2*n),
                      constraints=constraints, options={'ftol': 1e-15, 'maxiter': 2000})
    if not result.success:
        raise ValueError(f'Mean-variance failed: {result.message}')
    w = np.clip(weights(result.x), 0, cap)
    return w/w.sum()
