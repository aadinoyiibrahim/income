"""FastAPI service: upload a transactions CSV, get next-month forecasts per user.

The model bundle is loaded once at start-up (``INCOME_MODEL_PATH``). Endpoints:

* ``GET  /``         - redirects to the interactive docs (``/docs``)
* ``GET  /health``   - liveness + whether a model is loaded
* ``GET  /model``    - model metadata and hold-out metrics
* ``POST /predict``  - multipart CSV upload -> forecasts
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from . import __version__
from .bundle import ModelBundle, load_bundle
from .config import get_paths
from .predict import predict_next_month
from .preprocessing import read_transactions

log = logging.getLogger(__name__)
MAX_UPLOAD_MB = float(os.environ.get("INCOME_MAX_UPLOAD_MB", "100"))


class UserForecast(BaseModel):
    user_id: str
    cluster: int
    predicted_income: float
    predicted_expenses: float
    predicted_income_breakdown: dict[str, float]
    predicted_expenses_breakdown: dict[str, float]


class PredictionResponse(BaseModel):
    forecast_month: str
    model_version: str
    n_users: int
    predictions: list[UserForecast]


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = get_paths().model_path
    try:
        app.state.bundle = load_bundle(path)
        log.info("Loaded model bundle from %s", path)
    except FileNotFoundError:
        app.state.bundle = None
        log.error("No model bundle at %s; /predict will return 503 until one is provided.", path)
    yield


app = FastAPI(
    title="Income & Expenses Forecast API",
    version=__version__,
    description="Upload ≥1 complete month of transactions per user; get next-month income/expenses forecasts.",
    lifespan=lifespan,
)


def _bundle(request: Request) -> ModelBundle:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train one with `income train`.")
    return bundle


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    return {"status": "ok", "model_loaded": getattr(request.app.state, "bundle", None) is not None}


@app.get("/model")
def model_info(request: Request) -> dict[str, Any]:
    return _bundle(request).summary()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Columns required: ``user_id, transaction_date, transaction_type, amount_n26_currency`` (``mcc_group`` optional).

    The forecast is for the calendar month after the latest month in the file.
    """
    bundle = _bundle(request)
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB:g} MB")
    try:
        # CPU-bound: run off the event loop so /health stays responsive during big uploads
        preds = await run_in_threadpool(lambda: predict_next_month(bundle, read_transactions(io.BytesIO(content))))
    except ValueError as e:  # bad input -> client error with a useful message
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:  # unexpected -> log the trace, don't leak it
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal error during prediction") from e

    return {
        "forecast_month": preds["forecast_month"].iloc[0],
        "model_version": bundle.created_at,
        "n_users": len(preds),
        "predictions": preds.drop(columns="forecast_month").to_dict(orient="records"),
    }
