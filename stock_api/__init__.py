"""Forecast stocks in Python: ``from stock_api import predict``."""
from stock_api.data import DataError, ProviderError
from stock_api.forecast import ForecastError, predict
from stock_api.schemas import ForecastPoint, ForecastResponse
from stock_api.validation import (ModelSpec, TrainingData, ValidationConfig, ValidationResult, ValidationModel,
                                  builtin_models, compare_results, validate)

__all__ = ["predict", "ForecastPoint", "ForecastResponse", "DataError", "ProviderError", "ForecastError"]
__all__ += ["validate", "ValidationConfig", "ValidationResult", "ModelSpec", "TrainingData",
            "builtin_models", "compare_results", "ValidationModel"]
