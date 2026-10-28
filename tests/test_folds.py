"""Fold layout: expanding monthly windows, three months of training at least, disjoint splits."""
import datetime as dt

from grid_forecast.features import build_dataset, days_between
from grid_forecast.walkforward import VALID_DAYS, folds_for, split


def test_twelve_months_with_warmup_give_twelve_folds():
    folds = folds_for("demand", start=dt.date(2025, 10, 1), end=dt.date(2026, 9, 30), series_start=dt.date(2025, 7, 1))
    assert len(folds) == 12
    assert folds[0]["train_months"] == 3
    assert folds[0]["test_month"] == "2025-10"
    assert folds[-1]["test_month"] == "2026-09"
    assert [f["train_months"] for f in folds] == list(range(3, 15))
    for f in folds:
        assert dt.date.fromisoformat(f["train_end"]) == dt.date.fromisoformat(f["test_start"]) - dt.timedelta(days=1)
        assert dt.date.fromisoformat(f["train_start"]) == dt.date(2025, 7, 1)
        assert (dt.date.fromisoformat(f["train_end"]) - dt.date.fromisoformat(f["valid_start"])).days == VALID_DAYS - 1


def test_price_without_warmup_starts_after_three_months():
    folds = folds_for("price", start=dt.date(2025, 10, 1), end=dt.date(2026, 9, 30), series_start=dt.date(2025, 10, 1))
    assert [f["test_month"] for f in folds] == [f"2026-{m:02d}" for m in range(1, 10)]
    assert folds[0]["train_months"] == 3


def test_last_n_folds_and_dropped_months():
    folds = folds_for("demand", n_folds=2, start=dt.date(2025, 10, 1), end=dt.date(2026, 9, 30), series_start=dt.date(2025, 7, 1))
    assert [f["fold"] for f in folds] == [1, 2]
    assert [f["test_month"] for f in folds] == ["2026-08", "2026-09"]
    folds = folds_for("demand", start=dt.date(2025, 10, 1), end=dt.date(2026, 9, 30), series_start=dt.date(2025, 7, 1), dropped_months=["2026-02"])
    assert "2026-02" not in [f["test_month"] for f in folds]
    assert len(folds) == 11


def test_split_is_disjoint_and_ordered(tables):
    folds = folds_for("demand", start=dt.date(2025, 10, 1), end=dt.date(2025, 12, 31), series_start=dt.date(2025, 7, 1))
    assert len(folds) == 3
    dataset = build_dataset("demand", tables, days_between(dt.date(2025, 7, 1), dt.date(2025, 12, 31)))
    for f in folds:
        train, valid, test = split(dataset, f)
        assert len(train) and len(valid) and len(test)
        assert train["day"].max() < valid["day"].min()
        assert valid["day"].max() < test["day"].min()
        assert test["day"].min() == dt.date.fromisoformat(f["test_start"])
        assert test["day"].max() == dt.date.fromisoformat(f["test_end"])
        assert len(test) == 24 * ((test["day"].max() - test["day"].min()).days + 1)
