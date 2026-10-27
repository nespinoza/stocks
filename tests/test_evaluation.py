import numpy as np
import pandas as pd
import pytest

from stock_api.evaluation import (berkowitz, christoffersen, deflated_sharpe, diebold_mariano,
                                  factor_regression, ic_summary, information_coefficient, kupiec,
                                  neutralize, parse_french_csv, pbo_cscv, pit_from_quantiles,
                                  probabilistic_sharpe, qlike, quantile_spread)
from stock_api.evaluation.synthetic import gaussian_forecasts, noise_strategies, signal_panel
from scipy import stats


def test_null_signal_has_no_detected_skill():
    rejections = 0
    for seed in range(40):
        panel = signal_panel(ic=0.0, seed=seed)
        summary = ic_summary(information_coefficient(panel, 'signal', 'outcome'))
        rejections += summary['p'] < .05
    assert rejections <= 6  # about 2 expected at the 5% level


def test_planted_signal_is_recovered():
    panel = signal_panel(dates=120, assets=30, ic=.1, seed=1)
    summary = ic_summary(information_coefficient(panel, 'signal', 'outcome'))
    assert summary['mean_ic'] == pytest.approx(.1, abs=.04)
    assert summary['t'] > 3
    assert quantile_spread(panel, 'signal', 'outcome').mean() > 0


def test_neutralization_removes_a_signal_that_is_only_a_control():
    panel = signal_panel(ic=.15, seed=2)
    panel['copy'] = 2*panel['signal']+.001*np.random.default_rng(0).standard_normal(len(panel))
    panel['residual'] = neutralize(panel, 'copy', ['signal'])
    residual_ic = ic_summary(information_coefficient(panel, 'residual', 'outcome'))
    assert abs(residual_ic['mean_ic']) < .05


def test_diebold_mariano_size_and_power():
    rng = np.random.default_rng(3)
    p_null = [diebold_mariano(rng.standard_normal(200)**2, rng.standard_normal(200)**2)['p']
              for _ in range(200)]
    assert .02 < np.mean(np.array(p_null) < .05) < .1
    better = diebold_mariano(.5*rng.standard_normal(200)**2, rng.standard_normal(200)**2)
    assert better['statistic'] < 0 and better['p'] < .01


def test_calibration_tests_accept_calibrated_and_reject_overconfident():
    mean, std, actual = gaussian_forecasts(n=1000, miscalibration=1.0, seed=4)
    pit = stats.norm.cdf((actual-mean)/std)
    assert berkowitz(pit)['p'] > .01
    hits = np.abs(actual-mean) > 1.645*std
    assert kupiec(hits, .9)['p'] > .01
    assert christoffersen(hits, .9)['p_conditional'] > .01
    mean, std, actual = gaussian_forecasts(n=1000, miscalibration=1.5, seed=4)
    pit = stats.norm.cdf((actual-mean)/std)
    assert berkowitz(pit)['p'] < 1e-4
    assert kupiec(np.abs(actual-mean) > 1.645*std, .9)['p'] < 1e-4


def test_christoffersen_detects_clustered_failures():
    hits = np.zeros(500, bool)
    hits[100:125] = hits[300:325] = True  # 10% failures, all clustered
    result = christoffersen(hits, .9)
    assert kupiec(hits, .9)['p'] > .5
    assert result['p_independence'] < 1e-6


def test_pit_from_quantiles_and_qlike():
    levels = [.05, .16, .5, .84, .95]
    quantiles = [[-2., -1., 0., 1., 2.]]
    assert pit_from_quantiles(levels, quantiles, [0.])[0] == pytest.approx(.5)
    assert pit_from_quantiles(levels, quantiles, [5.])[0] == pytest.approx(.975)
    assert qlike(1., 1.) == pytest.approx(0)
    assert qlike(1., 2.) > 0 and qlike(2., 1.) > 0


def test_deflated_sharpe_penalizes_selecting_the_best_noise_strategy():
    returns = noise_strategies(periods=1000, configurations=100, seed=5)
    sharpes = returns.mean(axis=0)/returns.std(axis=0, ddof=1)
    best = returns[:, np.argmax(sharpes)]
    assert probabilistic_sharpe(best) > .9  # looks significant in isolation
    assert deflated_sharpe(best, 100, sharpes.var(ddof=1))['dsr'] < .9


def test_pbo_is_high_for_noise_and_low_for_a_genuine_edge():
    noise = noise_strategies(periods=600, configurations=20, seed=6)
    assert pbo_cscv(noise, blocks=8)['pbo'] > .3
    edged = noise.copy()
    edged[:, 0] += .003
    assert pbo_cscv(edged, blocks=8)['pbo'] < .1


def test_factor_regression_recovers_planted_alpha_and_beta():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2020-01-01', periods=1500)
    factors = pd.DataFrame(rng.normal(0, .01, (1500, 2)), index=dates, columns=['MKT', 'UMD'])
    factors['RF'] = 0.
    returns = pd.Series(.0004+1.2*factors.MKT+.5*factors.UMD+rng.normal(0, .002, 1500), index=dates)
    result = factor_regression(returns, factors, columns=('MKT', 'UMD'))
    assert result['coefficients']['MKT'] == pytest.approx(1.2, abs=.02)
    assert result['coefficients']['UMD'] == pytest.approx(.5, abs=.02)
    assert result['t']['alpha'] > 4


def test_parse_french_csv_reads_first_block_only():
    text = ('This file was created...\n\n,Mkt-RF,SMB,RF\n20200102,  1.00, -0.50,  0.01\n'
            '20200103,  -2.00, 0.25,  0.01\n\n Annual Factors\n,Mkt-RF,SMB,RF\n2020, 10, 1, 1\n')
    frame = parse_french_csv(text, 'daily')
    assert list(frame.columns) == ['MKT', 'SMB', 'RF']
    assert frame.loc['2020-01-03', 'MKT'] == pytest.approx(-.02)
    assert len(frame) == 2
