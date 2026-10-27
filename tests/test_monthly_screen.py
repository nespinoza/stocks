"""Chronology and scoring checks for the research shortlist script."""
import numpy as np
import pandas as pd
import pytest

from scripts.screen_monthly_stocks import monthly_data, forecasts, summarize, fit_predict, holm


def daily_prices():
    dates = pd.bdate_range('2009-01-01','2026-10-16')
    rng = np.random.default_rng(730)
    return pd.DataFrame({'Adj Close':100*np.exp(rng.normal(.0002,.015,len(dates)).cumsum())}, index=dates)


def test_monthly_inputs_use_only_prior_months_and_exclude_partial_month():
    monthly = monthly_data(daily_prices())
    assert monthly.index[-1] == pd.Timestamp('2026-09-30')
    i = 100
    assert monthly.lag1.iloc[i] == monthly['return'].iloc[i-1]
    assert monthly.lag3.iloc[i] == pytest.approx(monthly['return'].iloc[i-3:i].sum())
    assert monthly.momentum.iloc[i] == pytest.approx(monthly['return'].iloc[i-12:i-1].sum())
    assert monthly.logrv.iloc[i] == pytest.approx(np.log(monthly.rv.iloc[i-1]))


def test_future_prices_cannot_change_earlier_predictions():
    daily = daily_prices()
    first = forecasts(monthly_data(daily))
    daily.loc['2024-07-01':,'Adj Close'] *= 1.5
    second = forecasts(monthly_data(daily))
    pd.testing.assert_frame_equal(first[first.month<'2024-07-01'],second[second.month<'2024-07-01'])
    # July's forecast inputs and training labels precede July; its outcome changes.
    july = first.month=='2024-07-31'
    assert first.loc[july,'probability_up'].iloc[0] == second.loc[july,'probability_up'].iloc[0]
    assert first.loc[july,'realized_return'].iloc[0] != second.loc[july,'realized_return'].iloc[0]
    assert len(first)==117


def test_scoring_baseline_and_training_only_scaling():
    monthly = monthly_data(daily_prices()); f = forecasts(monthly)
    first_date = pd.Timestamp(f.month.iloc[0]); i = monthly.index.get_loc(first_date)
    expected = ((monthly['return'].iloc[i-60:i]>0).sum()+1)/62
    assert f.baseline_probability_up.iloc[0] == pytest.approx(expected)
    assert summarize(f)['brier_skill_pct'] == pytest.approx(100*(1-f.brier.mean()/f.baseline_brier.mean()))
    rng = np.random.default_rng(3); x=rng.normal(size=(60,4)); y=rng.normal(size=60); future=rng.normal(size=4)
    assert fit_predict(x,y,future) == pytest.approx(fit_predict(x*3+7,y,future*3+7))
    np.testing.assert_allclose(holm([.01,.04,.2]),[.03,.08,.2])
