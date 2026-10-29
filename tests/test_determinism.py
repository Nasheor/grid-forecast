"""Two fits with the same seed give the same LightGBM numbers and GRU numbers within 1 %, in process and
across the two full runs recorded in results/determinism.json."""
import datetime as dt
import json

import numpy as np
import pytest

from grid_forecast.features import build_dataset, days_between
from grid_forecast.walkforward import folds_for, run_fold


@pytest.fixture(scope="module")
def small_fold(tables):
    folds = folds_for("demand", n_folds=1, start=dt.date(2025, 10, 1), end=dt.date(2025, 10, 31), series_start=dt.date(2025, 7, 1))
    dataset = build_dataset("demand", tables, days_between(dt.date(2025, 7, 1), dt.date(2025, 10, 31)))
    return dataset, folds[0]


def test_two_runs_match(small_fold):
    dataset, fold = small_fold
    a, pa, _ = run_fold("demand", dataset, fold, {"max_rounds": 60, "early_stopping_rounds": 10}, {"epochs": 2}, seed=42)
    b, pb, _ = run_fold("demand", dataset, fold, {"max_rounds": 60, "early_stopping_rounds": 10}, {"epochs": 2}, seed=42)
    assert a["models"]["lightgbm"]["mae"] == b["models"]["lightgbm"]["mae"]
    assert a["models"]["lightgbm"]["rmse"] == b["models"]["lightgbm"]["rmse"]
    np.testing.assert_array_equal(pa["lightgbm"].to_numpy(), pb["lightgbm"].to_numpy())
    ga, gb = a["models"]["gru"]["mae"], b["models"]["gru"]["mae"]
    assert abs(ga - gb) / max(ga, 1e-9) < 0.01
    assert a["fits"]["lightgbm"]["best_iteration"] == b["fits"]["lightgbm"]["best_iteration"]
    assert a["fits"]["gru"]["best_epoch"] == b["fits"]["gru"]["best_epoch"]


def test_models_beat_nothing_obvious(small_fold):
    dataset, fold = small_fold
    rec, _, _ = run_fold("demand", dataset, fold, {"max_rounds": 200, "early_stopping_rounds": 20}, {"epochs": 3}, seed=42)
    assert rec["models"]["lightgbm"]["mae"] > 0
    assert rec["scored_hours"] == 24 * 31
    assert set(rec["models"]) == {"seasonal_naive", "last_week", "lightgbm", "gru"}


def test_the_two_full_runs_match(repo_root):
    path = repo_root / "results" / "determinism.json"
    assert path.exists(), "run the second full run into results/determinism and then python -m grid_forecast determinism"
    rep = json.loads(path.read_text(encoding="utf-8"))
    summary = json.loads((repo_root / "results" / "summary.json").read_text(encoding="utf-8"))
    assert rep["lightgbm_identical"] is True
    assert rep["gru_within_1_percent"] is True
    assert rep["gru_worst_relative_difference"] < 0.01
    for s, v in rep["series"].items():
        assert len(v["folds"]) == summary["series"][s]["folds"]
        assert all(f["lightgbm_identical"] for f in v["folds"])
