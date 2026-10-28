"""Baselines read the right hours, metrics behave."""
import datetime as dt

import numpy as np
import pandas as pd

from grid_forecast.baselines import last_week, seasonal_naive
from grid_forecast.features import build_frame
from grid_forecast.walkforward import metrics, skill


def test_seasonal_naive_is_same_hour_yesterday(tables):
    for series in ("demand", "wind", "price"):
        day = dt.date(2025, 11, 12)
        frame = build_frame(series, tables, day)
        t = tables[series]
        ts = pd.DatetimeIndex(frame["ts_utc"])
        np.testing.assert_allclose(seasonal_naive(frame), t.at(t.values, ts - pd.Timedelta(hours=24)))
        np.testing.assert_allclose(last_week(frame), t.at(t.values, ts - pd.Timedelta(hours=168)))


def test_metrics_of_a_perfect_forecast_are_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = metrics(y, y.copy(), "demand")
    assert m["mae"] == 0 and m["rmse"] == 0 and m["mape"] == 0 and m["n"] == 4
    m = metrics(y, y + 1.0, "wind")
    assert m["mae"] == 1.0 and "mape" not in m
    assert skill(1.0, 2.0) == 0.5 and skill(3.0, 2.0) == -0.5


def test_metrics_skip_missing_targets():
    y = np.array([1.0, np.nan, 3.0])
    m = metrics(y, np.array([2.0, 5.0, 3.0]), "price")
    assert m["n"] == 2 and m["mae"] == 0.5
