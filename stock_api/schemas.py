"""Validated forecast inputs and results, shared by Python and HTTP clients."""
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, StringConstraints, field_validator, model_validator

from stock_api.inference_options import InferenceName, inference_settings

Ticker = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9.-]{0,14}$"),
                   BeforeValidator(lambda v: v.strip().upper() if isinstance(v, str) else v)]
ModelName = Literal["last_price", "random_walk", "gp", "multitask_gp", "var",
                    "gbm_zero_drift", "gbm_estimated_drift", "ou_returns", "volatility_gp_returns", "heteroskedastic_gp_returns"]


class ForecastRequest(BaseModel):
    ticker: Ticker
    related_tickers: list[Ticker] = Field(default_factory=list, max_length=3)
    model: ModelName | Annotated[list[ModelName], Field(min_length=1, max_length=10)] = "multitask_gp"
    inference: InferenceName | list[InferenceName | None] | None = None
    inference_setup: dict | list[dict | None] | None = None
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
        if isinstance(self.model, list):
            if self.inference is not None and (not isinstance(self.inference, list) or len(self.inference) != len(self.model)):
                raise ValueError('A model list requires an inference list of the same length (or None)')
            if self.inference_setup is not None and (not isinstance(self.inference_setup, list) or len(self.inference_setup) != len(self.model)):
                raise ValueError('A model list requires an inference_setup list of the same length (or None)')
            if self.inference is None and len(set(self.model)) != len(self.model):
                raise ValueError('Repeated model names require an explicit inference list')
            methods = self.inference if self.inference is not None else [None] * len(self.model)
            setups = self.inference_setup if self.inference_setup is not None else [None] * len(self.model)
            for name, method, setup in zip(self.model, methods, setups):
                inference_settings(name, method, setup, seed=self.seed, n_paths=self.n_paths)
            return self
        if isinstance(self.inference, list) or isinstance(self.inference_setup, list):
            raise ValueError('A single model requires a scalar inference and a dictionary inference_setup')
        inference_settings(self.model, self.inference, self.inference_setup, seed=self.seed, n_paths=self.n_paths)
        if self.model in ("last_price", "random_walk", "gp", "gbm_zero_drift", "gbm_estimated_drift", "ou_returns", "volatility_gp_returns", "heteroskedastic_gp_returns") and self.related_tickers:
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


class ForecastComparison(BaseModel):
    """Forecasts from a shared price snapshot, indexed by model name or disambiguated model:inference key."""
    ticker: str
    related_tickers: list[str]
    as_of: str
    last_close: float
    horizon: int
    horizon_unit: str
    training_data: list[TrainingPoint]
    results: dict[str, ForecastResponse]

    def __getitem__(self, model):
        return self.results[model]

    def plot(self, *, history_days=None, forecast_days=None, sigmas=(1, 2, 3),
             show_related=True, show_fit=True, uncertainty="bands", ax=None,
             figsize=(12, 6), show=True):
        """Overlay models: one color per model, dashed fits and solid forecasts.

        Supports the same calendar-day display limits and sigma selection as
        ForecastResponse.plot. Returns a Matplotlib Axes. No downloads or refits.
        """
        from stock_api.plotting import plot_result
        return plot_result(self, history_days=history_days, forecast_days=forecast_days,
                           sigmas=sigmas, show_related=show_related, show_fit=show_fit,
                           uncertainty=uncertainty, ax=ax, figsize=figsize, show=show)
