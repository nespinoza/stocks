"""Black–Litterman posterior and the mapping from signal scores to views.

Returns and covariances are per rebalance period (here one month); callers pass
risk models built only from data available at the decision date.
"""
import numpy as np


def risk_aversion_for_sharpe(covariance, weights, sharpe):
    """Risk aversion δ under which `weights` is mean–variance optimal with Sharpe ratio `sharpe`."""
    weights = np.asarray(weights, float)
    return float(sharpe/np.sqrt(weights@covariance@weights))


def implied_returns(covariance, weights, risk_aversion):
    """Reverse optimization: the expected returns that make `weights` optimal, π = δΣw."""
    return risk_aversion*np.asarray(covariance)@np.asarray(weights, float)


def grinold_alpha(scores, volatility, ic, clip=3.):
    """Expected excess return of a cross-sectional score: α = IC · σ · z (Grinold & Kahn).

    Scores are standardized across assets and winsorized at ±clip; NaN scores get no view (α = 0).
    """
    scores = np.asarray(scores, float)
    valid = np.isfinite(scores)
    z = np.zeros_like(scores)
    if valid.sum() > 1 and scores[valid].std() > 0:
        z[valid] = np.clip((scores[valid]-scores[valid].mean())/scores[valid].std(), -clip, clip)
    return ic*np.asarray(volatility, float)*z


def view_uncertainty(pick, covariance, tau, confidence=.5):
    """He–Litterman diagonal Ω scaled by confidence c ∈ (0, 1): Ω = diag(PτΣPᵀ)(1−c)/c.

    c = 0.5 gives each view the same weight as the prior on that portfolio.
    """
    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0, 1)')
    pick = np.atleast_2d(pick)
    return np.diag(np.diag(tau*pick@covariance@pick.T)*(1-confidence)/confidence)


def black_litterman(prior_mean, covariance, pick, views, omega, tau=.05):
    """Posterior mean and covariance of returns given views P μ = Q + ε, ε ~ N(0, Ω).

    Returns (posterior mean, posterior predictive covariance Σ + M).
    """
    covariance = np.asarray(covariance, float)
    pick = np.atleast_2d(pick)
    prior = tau*covariance
    gain = prior@pick.T@np.linalg.inv(pick@prior@pick.T+np.atleast_2d(omega))
    mean = np.asarray(prior_mean, float)+gain@(np.asarray(views, float)-pick@prior_mean)
    posterior = prior-gain@pick@prior
    return mean, covariance+posterior
