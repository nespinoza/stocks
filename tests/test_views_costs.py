import numpy as np
import pandas as pd
import pytest

from stock_api.allocation import equal_risk_contribution, mean_variance
from stock_api.costs import half_spread_bps, spread_impact_cost
from stock_api.engine import run_backtest
from stock_api.views import (black_litterman, grinold_alpha, implied_returns, risk_aversion_for_sharpe,
                             view_uncertainty)


def covariance(n=6, seed=0, scale=1e-3):
    a = np.random.default_rng(seed).standard_normal((n, n))
    return scale*(a@a.T/n+np.eye(n))


def test_black_litterman_limits_and_one_asset_formula():
    cov = covariance()
    prior = np.linspace(.001, .006, 6)
    pick = np.eye(6)
    views = prior+.01
    vague, _ = black_litterman(prior, cov, pick, views, 1e6*np.eye(6))
    assert np.allclose(vague, prior)
    certain, _ = black_litterman(prior, cov, pick, views, 1e-12*np.eye(6))
    assert np.allclose(certain, views, atol=1e-8)
    # One asset: posterior is the precision-weighted average of prior and view.
    mean, var = black_litterman(np.array([.01]), np.array([[.04]]), np.eye(1), np.array([.03]), np.array([[.002]]), tau=.05)
    assert mean[0] == pytest.approx((.01/.002+.03/.002)/(1/.002+1/.002))
    assert var[0, 0] == pytest.approx(.04+1/(1/.002+1/.002))
    omega = view_uncertainty(pick, cov, .05, confidence=.5)
    assert np.allclose(np.diag(omega), .05*np.diag(cov))


def test_reverse_optimization_recovers_prior_weights():
    cov = covariance()
    prior = equal_risk_contribution(cov)
    delta = risk_aversion_for_sharpe(cov, prior, .15)
    pi = implied_returns(cov, prior, delta)
    assert pi@prior/np.sqrt(prior@cov@prior) == pytest.approx(.15)
    assert np.allclose(mean_variance(pi, cov, delta), prior, atol=1e-5)


def test_mean_variance_tilts_toward_alpha_and_cost_creates_no_trade_region():
    cov = covariance()
    prior = equal_risk_contribution(cov)
    delta = risk_aversion_for_sharpe(cov, prior, .15)
    pi = implied_returns(cov, prior, delta)
    alpha = np.zeros(6)
    alpha[0] = .002
    tilted = mean_variance(pi+alpha, cov, delta)
    assert tilted[0] > prior[0]+.01
    held = mean_variance(pi+alpha, cov, delta, current=prior, cost=.01)
    assert np.allclose(held, prior, atol=1e-6)
    partial = mean_variance(pi+alpha, cov, delta, current=prior, cost=.0002)
    assert prior[0] < partial[0] < tilted[0]
    assert mean_variance(pi+alpha, cov, delta, cap=.2).max() <= .2+1e-9


def test_grinold_alpha_is_centered_and_skips_missing_scores():
    alpha = grinold_alpha([1., 2., 3., np.nan], [.1, .1, .1, .1], ic=.05)
    assert alpha[3] == 0 and alpha[0] < 0 < alpha[2]
    assert alpha[:3].sum() == pytest.approx(0)
    assert alpha[2] == pytest.approx(.05*.1*np.sqrt(1.5))


def test_engine_policy_sees_drifted_holdings_band_and_cost_model():
    dates = pd.bdate_range('2020-01-01', periods=5)
    returns = pd.DataFrame({'A': [0, 0, 1., 0, 0], 'B': 0.}, index=dates)
    seen = []

    def policy(date, held):
        seen.append(held.copy())
        return pd.Series({'A': .5, 'B': .5})
    daily, weights = run_backtest(returns, policy, decision_dates=[dates[0], dates[2]], cost_bps=0)
    assert seen[0].sum() == 0
    # A doubled on dates[2]: held 2/3, 1/3 at that close.
    assert seen[1]['A'] == pytest.approx(2/3)
    assert daily.loc[dates[3], 'turnover'] == pytest.approx(1/3)
    _, banded = run_backtest(returns, policy, decision_dates=[dates[0], dates[2]], band=.2, cost_bps=0)
    assert len(banded) == 2 and banded.iloc[1]['A'] == pytest.approx(2/3)
    daily, _ = run_backtest(returns, policy, decision_dates=[dates[0]], cost_model=lambda date, trade: .01)
    assert daily.cost.iloc[0] == .01


def test_spread_impact_cost():
    assert list(half_spread_bps([2e9, 3e8, 6e7, 1e6])) == [.5, 1.5, 3., 6.]
    # 10% of a $10M book in an asset trading $100M/day at 1% daily vol, 3 bp half spread.
    cost = spread_impact_cost([.1], 1e7, [1e8], [.01], spread_bps=[3.])
    assert cost == pytest.approx(.1*(3e-4+.01*np.sqrt(.01)))
