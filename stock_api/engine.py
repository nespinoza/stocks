"""One walk-forward backtest engine for periodic-rebalance, long-only strategies.

Timing: weights are decided from data through the close of a decision date and
executed at the close of the following session (lag=1), paying cost_bps on the
traded fraction of wealth. Between trades, holdings drift with daily returns.
"""
import numpy as np
import pandas as pd


def month_end_dates(index):
    """Last session of each calendar month in a DatetimeIndex."""
    index = pd.DatetimeIndex(index)
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(index.to_period('M')).max().to_numpy())


def run_backtest(returns, decisions, cost_bps=5., lag=1, band=0., cost_model=None, decision_dates=None):
    """Simulate daily net returns.

    returns: DataFrame of daily simple returns (dates × assets).
    decisions: dict {decision date: weight Series/array over returns.columns}, or a
        policy callable policy(date, held) -> weights, called at each of
        `decision_dates` with the drifted holdings at that close (a Series).
    band: per-asset no-trade band; trades smaller than it are skipped and the
        remaining weights renormalized (not applied to the first purchase).
    cost_model: optional callable cost_model(date, trade) -> cost as a fraction
        of wealth, where trade is the signed weight change; replaces cost_bps.
    Returns (daily DataFrame with gross/net/turnover/cost, executed weights DataFrame).
    """
    returns = returns.fillna(0.)
    dates, columns = returns.index, returns.columns
    if callable(decisions):
        if decision_dates is None:
            raise ValueError('A policy needs decision_dates')
        policy, schedule = decisions, list(decision_dates)
    else:
        policy, schedule = (lambda date, held: decisions[date]), list(decisions)
    decide_at = {dates.get_loc(date): date for date in schedule}

    def validate(weights, date):
        weights = (weights.reindex(columns).fillna(0.) if isinstance(weights, pd.Series)
                   else pd.Series(weights, index=columns)).to_numpy(float)
        if abs(weights.sum()-1) > 1e-6 or weights.min() < -1e-9:
            raise ValueError(f'Weights at {date.date()} must be long-only and sum to one')
        return weights

    held = np.zeros(returns.shape[1])
    pending, rows, executed = {}, [], {}
    values = returns.to_numpy()
    for t in range(len(dates)):
        gross = float(held@values[t])
        if held.any():
            held = held*(1+values[t])/(1+gross)
        cost = turnover = 0.
        if t in decide_at and t+lag < len(dates):
            pending[t+lag] = validate(policy(dates[t], pd.Series(held, index=columns)), dates[t])
        if t in pending:
            target = pending.pop(t)
            if band > 0 and held.any():
                target = np.where(np.abs(target-held) > band, target, held)
                target = target/target.sum()
            trade = target-held
            turnover = float(np.abs(trade).sum())
            cost = cost_model(dates[t], trade) if cost_model else cost_bps/1e4*turnover
            held = target.copy()
            executed[dates[t]] = target
        rows.append((gross, (1+gross)*(1-cost)-1, turnover, cost))
    daily = pd.DataFrame(rows, index=dates, columns=['gross', 'net', 'turnover', 'cost'])
    first = dates.get_loc(min(executed)) if executed else len(dates)
    return daily.iloc[first:], pd.DataFrame(executed, index=columns).T


def performance(net, periods_per_year=252):
    """Summary statistics of a daily simple-return series."""
    wealth = (1+net).cumprod()
    years = len(net)/periods_per_year
    drawdown = wealth/wealth.cummax()-1
    return dict(total_return=float(wealth.iloc[-1]-1), cagr=float(wealth.iloc[-1]**(1/years)-1),
                volatility=float(net.std(ddof=1)*np.sqrt(periods_per_year)),
                sharpe=float(net.mean()/net.std(ddof=1)*np.sqrt(periods_per_year)),
                max_drawdown=float(drawdown.min()), start=str(net.index[0].date()), end=str(net.index[-1].date()))
