"""Monthly walk-forward folds and the scoring of every model on them.

Folds. The evaluated period is the twelve months from October 2025 to September 2026. A fold tests one
calendar month (Europe/Dublin) and trains on every day from the series' first day up to the day before that
month. The first fold must have at least three months of training, so a series whose data begins later than
the warm-up gets fewer than twelve folds (the price starts on 1 October 2025 and gets nine, January to
September 2026). The last fourteen days of each training window are the early-stopping set.

Metrics per fold and overall: MAE, RMSE, MAPE for demand only (wind can be zero and prices negative),
skill = 1 minus the model MAE divided by the seasonal-naive MAE, and MAE by horizon hour and by local hour
of day. Target hours flagged missing in the data are excluded from every metric.
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd

from . import MIN_TRAIN_MONTHS, PERIOD_END, PERIOD_START, SERIES_START
from .baselines import BASELINES
from .models import make_model

VALID_DAYS = 14
MODEL_NAMES = ("seasonal_naive", "last_week", "lightgbm", "gru")


def month_start(d: dt.date) -> dt.date:
    return d.replace(day=1)


def next_month(d: dt.date) -> dt.date:
    return (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def months_between(a: dt.date, b: dt.date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def folds_for(series: str, n_folds: int | None = None, start: dt.date = PERIOD_START, end: dt.date = PERIOD_END,
              series_start: dt.date | None = None, min_train_months: int = MIN_TRAIN_MONTHS, dropped_months: list[str] | None = None) -> list[dict]:
    """The fold table for a series. ``n_folds`` keeps the last ``n_folds`` folds (the smoke uses two)."""
    first = series_start or SERIES_START[series]
    dropped = set(dropped_months or [])
    out = []
    m = month_start(start)
    k = 0
    while m <= end:
        test_end = min(next_month(m) - dt.timedelta(days=1), end)
        ym = f"{m.year:04d}-{m.month:02d}"
        train_months = months_between(month_start(first), m)
        if train_months >= min_train_months and ym not in dropped:
            k += 1
            train_end = m - dt.timedelta(days=1)
            out.append({
                "fold": k, "series": series, "test_month": ym,
                "train_start": first.isoformat(), "train_end": train_end.isoformat(),
                "valid_start": (train_end - dt.timedelta(days=VALID_DAYS - 1)).isoformat(),
                "test_start": m.isoformat(), "test_end": test_end.isoformat(),
                "train_months": train_months,
            })
        m = next_month(m)
    if n_folds is not None:
        out = out[-int(n_folds):]
        for i, f in enumerate(out, 1):
            f["fold"] = i
    return out


def _dates(f: dict) -> tuple[dt.date, dt.date, dt.date, dt.date, dt.date]:
    return (dt.date.fromisoformat(f["train_start"]), dt.date.fromisoformat(f["valid_start"]), dt.date.fromisoformat(f["train_end"]),
            dt.date.fromisoformat(f["test_start"]), dt.date.fromisoformat(f["test_end"]))


def split(dataset: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Training rows (before the last fourteen days), validation rows (the last fourteen days) and test rows."""
    ts, vs, te, s, e = _dates(fold)
    day = dataset["day"]
    train = dataset[(day >= ts) & (day < vs)]
    valid = dataset[(day >= vs) & (day <= te)]
    test = dataset[(day >= s) & (day <= e)]
    return train, valid, test


def metrics(y: np.ndarray, yhat: np.ndarray, series: str) -> dict:
    ok = np.isfinite(y) & np.isfinite(yhat)
    y = y[ok]
    yhat = yhat[ok]
    err = yhat - y
    out = {"n": int(len(y)), "mae": float(np.mean(np.abs(err))) if len(y) else float("nan"),
           "rmse": float(np.sqrt(np.mean(err ** 2))) if len(y) else float("nan"), "bias": float(np.mean(err)) if len(y) else float("nan")}
    if series == "demand":
        out["mape"] = float(np.mean(np.abs(err) / np.abs(y)) * 100.0) if len(y) else float("nan")
    return out


def mae_by(frame: pd.DataFrame, col: str, pred_col: str) -> dict[str, float]:
    part = frame[~frame["target_missing"]]
    err = (part[pred_col] - part["target"]).abs()
    return {str(k): float(v) for k, v in err.groupby(part[col]).mean().items()}


def skill(model_mae: float, naive_mae: float) -> float:
    return float(1.0 - model_mae / naive_mae) if naive_mae and np.isfinite(naive_mae) else float("nan")


def run_fold(series: str, dataset: pd.DataFrame, fold: dict, lgbm_kwargs: dict | None = None, gru_kwargs: dict | None = None, seed: int = 42) -> tuple[dict, pd.DataFrame, dict]:
    """Fit both models on the fold, score all four, return the metrics record and the test rows with predictions."""
    train, valid, test = split(dataset, fold)
    if len(train) == 0 or len(valid) == 0 or len(test) == 0:
        raise ValueError(f"fold {fold['fold']} for {series} has an empty split")
    preds = test[["ts_utc", "origin_utc", "day", "horizon", "hour", "dow", "month", "holiday", "dst_day", "target", "target_missing", "target_interpolated"]].copy()
    for name, fn in BASELINES.items():
        preds[name] = fn(test)
    fits = {}
    models = {}
    for name, kwargs in (("lightgbm", lgbm_kwargs or {}), ("gru", gru_kwargs or {})):
        model = make_model(name, series, seed=seed, **kwargs)
        t0 = time.perf_counter()
        info = model.fit(train, valid)
        info["fit_seconds"] = round(time.perf_counter() - t0, 2)
        preds[name] = model.predict(test)
        fits[name] = {**info, "settings": model.settings()}
        models[name] = model
    scored = preds[~preds["target_missing"]]
    y = scored["target"].to_numpy(dtype=float)
    rec = {**fold, "models": {}, "fits": fits}
    naive = metrics(y, scored["seasonal_naive"].to_numpy(dtype=float), series)["mae"]
    for name in MODEL_NAMES:
        m = metrics(y, scored[name].to_numpy(dtype=float), series)
        m["skill_vs_seasonal_naive"] = skill(m["mae"], naive)
        m["mae_by_horizon"] = mae_by(preds, "horizon", name)
        m["mae_by_hour"] = mae_by(preds, "hour", name)
        rec["models"][name] = m
    rec["scored_hours"] = int(len(scored))
    rec["excluded_hours"] = int(preds["target_missing"].sum())
    preds.insert(0, "fold", fold["fold"])
    return rec, preds, models


def summarise(series: str, fold_records: list[dict], preds: pd.DataFrame) -> dict:
    """Mean-over-folds and pooled metrics per model, plus the fold count and the champion."""
    scored = preds[~preds["target_missing"]]
    y = scored["target"].to_numpy(dtype=float)
    naive_pooled = metrics(y, scored["seasonal_naive"].to_numpy(dtype=float), series)["mae"]
    out = {"series": series, "folds": len(fold_records), "scored_hours": int(len(scored)), "models": {}}
    for name in MODEL_NAMES:
        fold_mae = [r["models"][name]["mae"] for r in fold_records]
        fold_rmse = [r["models"][name]["rmse"] for r in fold_records]
        fold_skill = [r["models"][name]["skill_vs_seasonal_naive"] for r in fold_records]
        pooled = metrics(y, scored[name].to_numpy(dtype=float), series)
        pooled["skill_vs_seasonal_naive"] = skill(pooled["mae"], naive_pooled)
        out["models"][name] = {
            "mean_fold_mae": float(np.mean(fold_mae)), "mean_fold_rmse": float(np.mean(fold_rmse)),
            "mean_fold_skill": float(np.mean(fold_skill)), "folds_with_positive_skill": int(sum(1 for s in fold_skill if s > 0)),
            "fold_mae": fold_mae, "fold_skill": fold_skill, "pooled": pooled,
            "mae_by_horizon": mae_by(preds, "horizon", name), "mae_by_hour": mae_by(preds, "hour", name),
        }
        if series == "demand":
            out["models"][name]["mean_fold_mape"] = float(np.mean([r["models"][name]["mape"] for r in fold_records]))
    learned = {n: out["models"][n]["mean_fold_mae"] for n in ("lightgbm", "gru")}
    out["champion"] = min(learned, key=learned.get)
    best_all = min(out["models"], key=lambda n: out["models"][n]["mean_fold_mae"])
    out["best_overall"] = best_all
    worst_hour = max(out["models"][out["champion"]]["mae_by_hour"].items(), key=lambda kv: kv[1])
    out["worst_hour_of_day"] = {"hour": int(worst_hour[0]), "mae": float(worst_hour[1])}
    return out
