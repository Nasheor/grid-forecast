"""FastAPI service over the MLflow registry.

``POST /forecast`` with ``{"series": "demand", "origin_utc": "2026-09-30T23:00:00Z"}`` loads the champion of
the series from the registry once, builds the features for that origin from ``data/prepared/``, and returns
24 hourly values with their UTC timestamps, the model name and version, and the seasonal-naive values for
the same origin. The origin must be the one the models were trained for: 00:00 Europe/Dublin for demand and
wind, 10:00 Europe/Dublin on the day before for the price. ``GET /models`` lists the champions with their
fold-mean MAE. ``GET /healthz`` answers when the process is up.

Run with ``python -m grid_forecast serve`` (uvicorn on port 8000).
"""
from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import PREPARED, SERIES, TZ, UNITS, __version__, mlflow_uri
from .baselines import seasonal_naive
from .features import SeriesTable, build_frame, origin_for_day
from .prepare import load_prepared
from .tracking import INPUT_COLUMNS


class ForecastRequest(BaseModel):
    series: str = Field(description="demand, wind or price")
    origin_utc: str = Field(description="forecast origin, ISO 8601, UTC")


class Point(BaseModel):
    ts_utc: str
    value: float


class ForecastResponse(BaseModel):
    series: str
    unit: str
    origin_utc: str
    target_day: str
    model: str
    version: int
    values: list[Point]
    seasonal_naive: list[Point]
    build_seconds: float


def _parse_origin(text: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"origin_utc is not a timestamp: {e}") from e
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def target_day_for(series: str, origin: pd.Timestamp) -> dt.date:
    """The forecast day implied by the origin, or a 422 when the origin is not an allowed one."""
    local = origin.tz_localize("UTC").tz_convert(TZ)
    day = local.date() if series != "price" else local.date() + dt.timedelta(days=1)
    expected = origin_for_day(series, day)
    if expected != origin:
        want = "00:00 Europe/Dublin on the forecast day" if series != "price" else "10:00 Europe/Dublin on the day before the forecast day"
        raise HTTPException(status_code=422, detail=f"origin must be {want}, the nearest allowed origin is {expected.isoformat()}Z")
    return day


def create_app(prepared: Path | None = None, tracking_uri: str | None = None, prefix: str = "grid-forecast") -> FastAPI:
    prepared = Path(prepared or os.environ.get("GRID_FORECAST_PREPARED", PREPARED))
    app = FastAPI(title="grid-forecast", version=__version__)
    state: dict = {"tables": None, "models": {}, "uri": tracking_uri}

    def tables() -> dict[str, SeriesTable]:
        if state["tables"] is None:
            state["tables"] = {s: SeriesTable(load_prepared(s, prepared)) for s in SERIES}
        return state["tables"]

    def champion(series: str) -> dict:
        if series not in state["models"]:
            mlflow.set_tracking_uri(state["uri"] or mlflow_uri())
            name = f"{prefix}-{series}"
            client = mlflow.MlflowClient()
            try:
                mv = client.get_model_version_by_alias(name, "champion")
            except mlflow.exceptions.MlflowException as e:
                raise HTTPException(status_code=503, detail=f"no champion registered for {series}: {e}") from e
            model = mlflow.pyfunc.load_model(f"models:/{name}@{'champion'}")
            state["models"][series] = {"model": model, "version": int(mv.version), "name": mv.tags.get("model", "unknown"),
                                       "mean_fold_mae": float(mv.tags.get("mean_fold_mae", "nan"))}
        return state["models"][series]

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/models")
    def models() -> dict:
        out = {}
        for s in SERIES:
            try:
                c = champion(s)
                out[s] = {"registered_model": f"{prefix}-{s}", "version": c["version"], "model": c["name"], "mean_fold_mae": c["mean_fold_mae"], "unit": UNITS[s]}
            except HTTPException as e:
                out[s] = {"registered_model": f"{prefix}-{s}", "error": e.detail}
        return out

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(req: ForecastRequest) -> ForecastResponse:
        if req.series not in SERIES:
            raise HTTPException(status_code=422, detail=f"series must be one of {SERIES}")
        origin = _parse_origin(req.origin_utc)
        day = target_day_for(req.series, origin)
        t0 = time.perf_counter()
        frame = build_frame(req.series, tables(), day)
        if frame["lag_24h"].isna().all():
            raise HTTPException(status_code=422, detail="no data known at that origin")
        c = champion(req.series)
        X = frame[[col for col in INPUT_COLUMNS if col in frame.columns]].copy()
        X["day"] = X["day"].astype(str)
        yhat = c["model"].predict(X)
        naive = seasonal_naive(frame)
        stamps = pd.DatetimeIndex(frame["ts_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ")
        return ForecastResponse(
            series=req.series, unit=UNITS[req.series], origin_utc=origin.strftime("%Y-%m-%dT%H:%M:%SZ"), target_day=day.isoformat(),
            model=c["name"], version=c["version"],
            values=[Point(ts_utc=t, value=round(float(v), 3)) for t, v in zip(stamps, yhat, strict=True)],
            seasonal_naive=[Point(ts_utc=t, value=round(float(v), 3)) for t, v in zip(stamps, naive, strict=True)],
            build_seconds=round(time.perf_counter() - t0, 4),
        )

    return app


app = create_app()
