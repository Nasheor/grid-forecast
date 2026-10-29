"""The API serves 24 timestamped values per series from a registered champion, fast, and refuses bad origins.

The wind champion here is a GRU, so the path that hands the registered model a frame without a target column
is covered as well as the LightGBM path.
"""
import datetime as dt
import os
import time

import pytest
from fastapi.testclient import TestClient

from grid_forecast import SERIES
from grid_forecast.features import build_dataset, days_between, origin_for_day
from grid_forecast.models import make_model
from grid_forecast.tracking import register_champion, setup

EXPECTED = {"demand": "lightgbm", "wind": "gru", "price": "lightgbm"}


@pytest.fixture(scope="module")
def registry(tables, tmp_path_factory):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = tmp_path_factory.mktemp("mlruns").resolve().as_uri()
    setup(uri)
    for s in SERIES:
        dataset = build_dataset(s, tables, days_between(dt.date(2025, 9, 1), dt.date(2025, 11, 30)))
        day = dataset["day"]
        train = dataset[day < dt.date(2025, 11, 17)]
        valid = dataset[day >= dt.date(2025, 11, 17)]
        model = make_model("gru", s, epochs=1) if EXPECTED[s] == "gru" else make_model("lightgbm", s, max_rounds=40, early_stopping_rounds=10)
        model.fit(train, valid)
        register_champion(s, model, valid.tail(24), 12.5, 3)
    return uri


@pytest.fixture(scope="module")
def client(registry, synthetic_dir):
    from grid_forecast.api import create_app

    app = create_app(prepared=synthetic_dir, tracking_uri=registry)
    with TestClient(app) as c:
        yield c


def test_healthz_and_models(client):
    assert client.get("/healthz").json()["status"] == "ok"
    models = client.get("/models").json()
    for s in SERIES:
        assert models[s]["version"] == 1 and models[s]["model"] == EXPECTED[s]


@pytest.mark.parametrize("series", SERIES)
def test_forecast_returns_24_values_fast(client, series):
    origin = origin_for_day(series, dt.date(2025, 12, 15))
    body = {"series": series, "origin_utc": origin.strftime("%Y-%m-%dT%H:%M:%SZ")}
    first = client.post("/forecast", json=body)
    assert first.status_code == 200, first.text
    t0 = time.perf_counter()
    r = client.post("/forecast", json=body)
    warm = time.perf_counter() - t0
    assert r.status_code == 200
    j = r.json()
    assert len(j["values"]) == 24 and len(j["seasonal_naive"]) == 24
    assert j["values"][0]["ts_utc"] == "2025-12-15T00:00:00Z"
    assert j["model"] == EXPECTED[series] and j["version"] == 1
    assert warm < 2.0


def test_bad_origin_and_series(client):
    r = client.post("/forecast", json={"series": "demand", "origin_utc": "2025-12-15T03:00:00Z"})
    assert r.status_code == 422
    r = client.post("/forecast", json={"series": "price", "origin_utc": "2025-12-15T00:00:00Z"})
    assert r.status_code == 422
    r = client.post("/forecast", json={"series": "solar", "origin_utc": "2025-12-15T00:00:00Z"})
    assert r.status_code == 422
