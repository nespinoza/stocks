"""Opt-in, serializable inference diagnostics; all scales are training-only."""
from pydantic import BaseModel, Field, model_validator


class PriorConfig(BaseModel):
    length: tuple[float, float] = (.5, 300.)
    price_amplitude_ratio: tuple[float, float] = (1e-3, 1e2)
    price_noise_ratio: tuple[float, float] = (1e-4, 10.)
    return_mean_amplitude_ratio: tuple[float, float] = (1e-3, 10.)
    log_variance_amplitude: tuple[float, float] = (.01, 3.)
    log_variance_mean_sd: float = Field(default=2., gt=0)
    log_variance_mean_truncation: float = Field(default=5., ge=2.)
    log_variance_inactive_amplitude: float = Field(default=.1, gt=0)
    inactive_amplitude_ratio: float = Field(default=.1, gt=0)

    @model_validator(mode='after')
    def ranges(self):
        for name in ['length','price_amplitude_ratio','price_noise_ratio',
                     'return_mean_amplitude_ratio','log_variance_amplitude']:
            a,b=getattr(self,name)
            if not 0 < a < b:
                raise ValueError(f'{name} must be an increasing positive interval')
        return self


class DiagnosticConfig(BaseModel):
    enabled: bool = True
    profile_surfaces: bool = True
    posterior_sampling: bool = True
    multistart: bool = True
    latent_plots: bool = True
    corner_plots: bool = True
    priors: PriorConfig = Field(default_factory=PriorConfig)
    profile_points: int = Field(default=41, ge=20)
    surface_points: int = Field(default=25, ge=8)
    starts: int = Field(default=30, ge=1)
    profile_maxiter: int = Field(default=600, ge=1)
    nlive: int = Field(default=150, ge=20)
    nested_dlogz: float = Field(default=.1, gt=0)
    nested_maxcall: int = Field(default=200000, ge=100)
    interweave: bool = False
    chains: int = Field(default=4, ge=2)
    warmup: int = Field(default=1000, ge=10)
    draws: int = Field(default=2000, ge=20)
    thin: int = Field(default=1, ge=1)
    predictive_draws: int = Field(default=2000, ge=100)
    seed: int = Field(default=42, ge=0)
