"""Daily adjusted closes for US equities, using completed XNYS sessions."""
import pandas as pd
import numpy as np
import exchange_calendars as xcals
import yfinance as yf


class DataError(ValueError):
    pass


class ProviderError(RuntimeError):
    pass


def calendar():
    return xcals.get_calendar("XNYS")


def latest_session(now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    schedule = calendar().schedule
    completed = schedule.loc[schedule["close"] <= now]
    if completed.empty:
        raise DataError("No completed session in the exchange calendar")
    return completed.index[-1].tz_localize(None)


def future_sessions(as_of, horizon, unit):
    end = as_of + pd.Timedelta(days=horizon if unit == "calendar" else horizon * 3 + 10)
    sessions = calendar().sessions_in_range(as_of + pd.Timedelta(days=1), end)
    sessions = sessions.tz_localize(None)
    return sessions if unit == "calendar" else sessions[:horizon]


def load_prices(tickers, lookback, as_of):
    start = as_of - pd.Timedelta(days=lookback * 2 + 30)
    try:
        raw = yf.download(tickers, start=start.date().isoformat(),
                          end=(as_of + pd.Timedelta(days=1)).date().isoformat(),
                          interval="1d", auto_adjust=True, group_by="column",
                          multi_level_index=True, progress=False, threads=False, timeout=15)
    except Exception as exc:
        raise ProviderError("Price provider failed; retry later") from exc
    if raw is None or raw.empty:
        raise ProviderError("No prices returned; check the ticker or retry later")
    try:
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        close = close.reindex(columns=tickers)
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
        close = close.loc[~close.index.duplicated(keep="last")].sort_index()
        sessions = calendar().sessions_in_range(start, as_of).tz_localize(None)[-lookback:]
        close = close.reindex(sessions).astype(float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError("Price provider returned an unexpected format") from exc
    invalid = ~np.isfinite(close.to_numpy()) | (close.to_numpy() <= 0)
    if invalid.any():
        bad = [ticker for i, ticker in enumerate(tickers) if invalid[:, i].any()]
        raise DataError("Missing, stale or invalid completed-session prices for " + ", ".join(bad)
                        + "; choose US equities with sufficient shared history or reduce lookback")
    return close
