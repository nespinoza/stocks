"""Frozen, ticker-independent walk-forward experiment configuration."""
from datetime import date
from typing import Literal
import re
from functools import lru_cache
import pandas as pd
from pydantic import BaseModel, Field, model_validator, ConfigDict
from stock_api.diagnostics import DiagnosticConfig

MODELS = ('last_price','gp','multitask_gp','var','gbm_zero_drift',
          'gbm_estimated_drift','ou_returns','volatility_gp_returns','heteroskedastic_gp_returns')


class HistoricalConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    target_ticker: str
    auxiliary_tickers: list[str] = Field(default_factory=list)
    start_date: date
    end_date: date  # exclusive, including all forecast targets
    forecast_horizon_days: int = Field(default=7, ge=1, le=365)
    origin_frequency: Literal['weekly','monthly','daily'] = 'weekly'
    lookbacks: list[int] = Field(default_factory=lambda:[30,60,90,180])
    models: list[str] = Field(default_factory=lambda:list(MODELS))
    calendar: str = 'XNYS'
    seed: int = Field(default=20260912, ge=0)
    n_paths: int = Field(default=2000, ge=100, le=100000)
    min_train_observations: int = Field(default=10, ge=8)
    inference_mode: Literal['production_validation'] = 'production_validation'
    inference: DiagnosticConfig = Field(default_factory=lambda:DiagnosticConfig(
        profile_surfaces=False,multistart=False,corner_plots=False,latent_plots=False,
        interweave=True,warmup=1500,draws=5000))
    development_end: date | None = None  # exclusive
    untouched_holdout_start: date | None = None
    data_vintage: Literal['retrospective_adjusted','point_in_time_supplied','synthetic'] = 'retrospective_adjusted'
    min_bin_count: int = Field(default=20, ge=2)
    bootstrap_replicates: int = Field(default=2000, ge=100)
    bootstrap_block_weeks: int = Field(default=8, ge=1)

    @model_validator(mode='after')
    def validate_settings(self):
        self.target_ticker = self.target_ticker.strip().upper()
        self.auxiliary_tickers = [t.strip().upper() for t in self.auxiliary_tickers]
        tickers = [self.target_ticker,*self.auxiliary_tickers]
        if any(not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=-]*',t) for t in tickers) or len(tickers)!=len(set(tickers)):
            raise ValueError('Provide distinct nonempty Yahoo-style tickers')
        if not self.lookbacks or len(self.lookbacks)!=len(set(self.lookbacks)) or any(x<14 or x>2000 for x in self.lookbacks):
            raise ValueError('Provide unique lookbacks between 14 and 2000 calendar days')
        if not self.models or len(set(self.models))!=len(self.models) or set(self.models)-set(MODELS):
            raise ValueError('Provide unique existing model names')
        if 'last_price' not in self.models: raise ValueError('Persistence must be included')
        if self.start_date >= self.evaluation_end: raise ValueError('Empty development interval')
        if self.inference.profile_surfaces or self.inference.multistart:
            raise ValueError('Detailed diagnostics must be requested separately for selected folds')
        if not self.inference.posterior_sampling or not self.inference.enabled:
            raise ValueError('GP posterior sampling is required in production validation')
        return self

    @property
    def evaluation_end(self):
        return min(d for d in [self.end_date,self.development_end,self.untouched_holdout_start] if d is not None)


def sessions_for(config):
    start = pd.Timestamp(config.start_date)-pd.Timedelta(days=max(config.lookbacks)+14)
    end = pd.Timestamp(config.evaluation_end)
    return _sessions(config.calendar,start,end)


@lru_cache(maxsize=32)
def _sessions(name,start,end):
    import exchange_calendars as xcals
    cal = xcals.get_calendar(name, start=start-pd.Timedelta(days=7), end=end+pd.Timedelta(days=7))
    index = cal.sessions
    if index.tz is not None: index=index.tz_localize(None)
    return index[(index>=start)&(index<end)]


def make_origins(config):
    sessions = sessions_for(config)
    available = sessions[sessions>=pd.Timestamp(config.start_date)]
    if config.origin_frequency=='weekly':
        # First exchange session in each Monday–Sunday week; observed missing data
        # must not move the origin or choose the session retrospectively.
        available=available[~available.to_period('W-SUN').duplicated()]
    elif config.origin_frequency=='monthly':
        available=available[~available.to_period('M').duplicated()]
    return [str(t.date()) for t in available if t+pd.Timedelta(days=config.forecast_horizon_days)<=pd.Timestamp(config.evaluation_end)]


def fold_dates(config, origin, lookback):
    sessions=sessions_for(config);origin=pd.Timestamp(origin)
    train=sessions[(sessions>=origin-pd.Timedelta(days=lookback)) & (sessions<origin)]
    test=sessions[(sessions>=origin) & (sessions<origin+pd.Timedelta(days=config.forecast_horizon_days))]
    return train,test
