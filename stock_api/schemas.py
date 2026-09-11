"""Validated forecast inputs and results, shared by Python and HTTP clients."""
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, StringConstraints, field_validator, model_validator

Ticker = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9.-]{0,14}$"),
                   BeforeValidator(lambda v: v.strip().upper() if isinstance(v, str) else v)]


class ForecastRequest(BaseModel):
    ticker: Ticker
    related_tickers: list[Ticker] = Field(default_factory=list, max_length=3)
    model: Literal["last_price", "random_walk", "gp", "multitask_gp", "var", "gbm_zero_drift", "gbm_estimated_drift", "ou_returns"] = "multitask_gp"
    horizon: int = Field(default=7, ge=1, le=30)
    horizon_unit: Literal["calendar", "trading"] = "calendar"
    lookback: int = Field(default=120, ge=60, le=252)
    seed: int = Field(default=0, ge=0)
    n_paths: int = Field(default=10_000, ge=100, le=100_000)

    @field_validator("related_tickers")
    @classmethod
    def unique(cls, values):
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def remove_target(self):
        self.related_tickers = [t for t in self.related_tickers if t != self.ticker]
        if self.model in ("last_price", "random_walk", "gp", "gbm_zero_drift", "gbm_estimated_drift", "ou_returns") and self.related_tickers:
            raise ValueError(f"{self.model} uses only the target; omit related_tickers")
        return self


class SigmaInterval(BaseModel):
    lower: float
    upper: float


class FitPoint(BaseModel):
    date: str
    predicted_close: float
    log_std: float
    sigma_1: SigmaInterval
    sigma_2: SigmaInterval
    sigma_3: SigmaInterval


class TrainingPoint(BaseModel):
    date: str
    closes: dict[str, float]


class ForecastPoint(FitPoint):
    lower_95: float
    upper_95: float
    predicted_change_pct: float
    probability_up: float
    mean_price: float | None = None
    median_price: float | None = None
    price_std: float | None = None


class ForecastResponse(BaseModel):
    ticker: str
    related_tickers: list[str]
    model: str
    as_of: str
    last_close: float
    horizon: int
    horizon_unit: str
    training_observations: int
    direction: Literal["up", "down", "flat"]
    predicted_change_pct: float
    forecasts: list[ForecastPoint]
    diagnostics: dict
    warnings: list[str]
    training_data: list[TrainingPoint] = Field(default_factory=list)
    fitted: list[FitPoint] = Field(default_factory=list)
    terminal_log_returns: list[float] = Field(default_factory=list)
    terminal_returns: list[float] = Field(default_factory=list)

    def plot(self, *, history_days=None, forecast_days=None, sigmas=(1, 2, 3),
             show_related=True, show_fit=True, uncertainty="bands", ax=None,
             figsize=(12, 6), show=True):
        """Plot stored data without downloading or refitting.

        Day limits are calendar days relative to as_of; None shows all data.
        sigmas selects 1/2/3 sigma intervals; () hides them. uncertainty is
        'bands' or 'errorbars'. Related prices are rebased to the target at
        as_of. Returns a Matplotlib Axes; show=False suppresses plt.show().
        """
        from stock_api.plotting import plot_result
        return plot_result(self, history_days=history_days, forecast_days=forecast_days,
                           sigmas=sigmas, show_related=show_related, show_fit=show_fit,
                           uncertainty=uncertainty, ax=ax, figsize=figsize, show=show)
