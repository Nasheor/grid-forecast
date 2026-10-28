"""grid-forecast: day-ahead demand, wind and price forecasts on one year of public Irish grid and market data.

Three hourly series for the Republic of Ireland (system demand in MW, wind generation in MW, SEM day-ahead
price in EUR/MWh), two baselines, LightGBM and a small GRU, monthly walk-forward folds, MLflow tracking and
a FastAPI endpoint. See README.md for the sources, their terms and the measured results.

Paths and constants shared by every module live here. Timestamps are stored in UTC and calendar features
are computed in Europe/Dublin.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PREPARED = DATA / "prepared"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
LICENCE_CHECK = ROOT / "LICENCE_CHECK.md"

TZ = "Europe/Dublin"
SERIES = ("demand", "wind", "price")
UNITS = {"demand": "MW", "wind": "MW", "price": "EUR/MWh"}

# The twelve evaluated months: the last twelve complete calendar months before the build date (28 October 2026).
PERIOD_START = dt.date(2025, 10, 1)
PERIOD_END = dt.date(2026, 9, 30)
# Training history before the first evaluated month. Demand and wind come from the EirGrid Library workbooks,
# which reach back to January 2025. The hourly price report starts on 1 October 2025, so the price series has
# no warm-up and its first folds wait until three months of prices exist (see README, fold design).
WARMUP_START = dt.date(2025, 7, 1)
SERIES_START = {"demand": WARMUP_START, "wind": WARMUP_START, "price": PERIOD_START}
# Lags reach 168 hours back from the first origin, so the download starts a little earlier still.
DOWNLOAD_START = dt.date(2025, 6, 23)
MIN_TRAIN_MONTHS = 3

ATTRIBUTION = "Supported by EirGrid Group Data"
USER_AGENT = f"grid-forecast/{__version__} (personal research project, Avinash Nagarajan)"

SEED = 42


def mlflow_uri() -> str:
    """Tracking URI: ``MLFLOW_TRACKING_URI`` if set in the process, otherwise the repo's ``mlruns`` folder.

    MLflow 3 refuses its file store unless ``MLFLOW_ALLOW_FILE_STORE`` is set, so the opt-in is set here
    for the local file store that the spec asks for.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        uri = (ROOT / "mlruns").resolve().as_uri()
    return uri
