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


def run_backtest(returns, decisions, cost_bps=5., lag=1):
    """Simulate daily net returns.

    returns: DataFrame of daily simple returns (dates × assets).
    decisions: dict {decision date: weight Series/array over returns.columns}.
    Returns (daily DataFrame with gross/net/turnover/cost, executed weights DataFrame).
    """
    returns = returns.fillna(0.)
    dates = returns.index
    position = {dates.get_loc(date): date for date in decisions}
    executions = {}
    for location, date in position.items():
        weights = decisions[date]
        weights = (weights.reindex(returns.columns).fillna(0.) if isinstance(weights, pd.Series)
                   else pd.Series(weights, index=returns.columns)).to_numpy(float)
        if abs(weights.sum()-1) > 1e-6 or weights.min() < -1e-9:
            raise ValueError(f'Weights at {date.date()} must be long-only and sum to one')
        if location+lag < len(dates):
            executions[location+lag] = weights
    held = np.zeros(returns.shape[1])
    invested = False
    rows, executed = [], {}
    values = returns.to_numpy()
    for t in range(len(dates)):
        gross = float(held@values[t]) if invested else 0.
        if invested:
            held = held*(1+values[t])/(1+gross)
        cost = turnover = 0.
        if t in executions:
            target = executions[t]
            turnover = float(np.abs(target-held).sum()) if invested else 1.
            cost = cost_bps/1e4*turnover
            held, invested = target.copy(), True
            executed[dates[t]] = target
        rows.append((gross, (1+gross)*(1-cost)-1, turnover, cost))
    daily = pd.DataFrame(rows, index=dates, columns=['gross', 'net', 'turnover', 'cost'])
    first = min(executions) if executions else len(dates)
    return daily.iloc[first:], pd.DataFrame(executed, index=returns.columns).T


def performance(net, periods_per_year=252):
    """Summary statistics of a daily simple-return series."""
    wealth = (1+net).cumprod()
    years = len(net)/periods_per_year
    drawdown = wealth/wealth.cummax()-1
    return dict(total_return=float(wealth.iloc[-1]-1), cagr=float(wealth.iloc[-1]**(1/years)-1),
                volatility=float(net.std(ddof=1)*np.sqrt(periods_per_year)),
                sharpe=float(net.mean()/net.std(ddof=1)*np.sqrt(periods_per_year)),
                max_drawdown=float(drawdown.min()), start=str(net.index[0].date()), end=str(net.index[-1].date()))
