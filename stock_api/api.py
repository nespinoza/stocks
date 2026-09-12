"""Optional HTTP adapter. Run with uvicorn stock_api.api:app."""
import logging
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException

from stock_api.forecast import predict as forecast, ForecastError
from stock_api.data import DataError, ProviderError
from stock_api.schemas import ForecastRequest, ForecastResponse, ForecastComparison

logger = logging.getLogger(__name__)
app = FastAPI(title="Stock movement API", version="0.1.0",
              description="Experimental US-equity forecasts from adjusted daily closes.")
fit_slot = BoundedSemaphore(1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ForecastResponse | ForecastComparison)
def predict(request: ForecastRequest):
    if not fit_slot.acquire(blocking=False):
        raise HTTPException(503, "A forecast is running; retry shortly", headers={"Retry-After": "5"})
    try:
        return forecast(**request.model_dump())
    except DataError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(502, str(exc)) from exc
    except ForecastError as exc:
        logger.exception("Forecast fit failed")
        raise HTTPException(500, str(exc)) from exc
    finally:
        fit_slot.release()
