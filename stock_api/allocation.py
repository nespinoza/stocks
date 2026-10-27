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
