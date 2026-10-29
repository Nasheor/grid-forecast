"""MLflow bookkeeping: one experiment per series, one run per fold (both learned models and both baselines
logged under their own prefixes), one aggregate run per learned model, and the champion per series registered
as ``grid-forecast-<series>`` with the alias ``champion``, a signature and an input example.

The tracking store is the local file store under ``mlruns/`` (``MLFLOW_TRACKING_URI=file:./mlruns``), which
MLflow 3 allows with ``MLFLOW_ALLOW_FILE_STORE=true``. ``mlflow_uri()`` in the package sets that opt-in.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from . import SEED, mlflow_uri
from .features import FEATURE_COLUMNS, WINDOW_COLUMNS, feature_hash

INPUT_COLUMNS = ["ts_utc", "origin_utc", "day", *FEATURE_COLUMNS["price"], *WINDOW_COLUMNS]


class ForecasterModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper: takes the feature frame (24 rows per origin) and returns one value per row."""

    def __init__(self, forecaster=None):
        self.forecaster = forecaster

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return np.asarray(self.forecaster.predict(model_input), dtype=float)


def setup(uri: str | None = None, prefix: str = "grid-forecast") -> str:
    """Point MLflow at ``uri`` (default: the repo's ``mlruns``), with the file-store opt-in set either way."""
    default = mlflow_uri()
    uri = uri or default
    mlflow.set_tracking_uri(uri)
    return uri


def experiment(series: str, prefix: str = "grid-forecast") -> str:
    name = f"{prefix}-{series}"
    mlflow.set_experiment(name)
    return name


def log_fold(series: str, rec: dict, seed: int = SEED, prefix: str = "grid-forecast") -> str:
    experiment(series, prefix)
    with mlflow.start_run(run_name=f"fold-{rec['fold']:02d}-{rec['test_month']}") as run:
        mlflow.set_tags({"kind": "fold", "series": series})
        mlflow.log_params({
            "fold": rec["fold"], "test_month": rec["test_month"], "train_start": rec["train_start"], "train_end": rec["train_end"],
            "valid_start": rec["valid_start"], "train_months": rec["train_months"], "seed": seed, "feature_hash": feature_hash(series),
            "n_features": len(FEATURE_COLUMNS[series]),
        })
        for model, fit in rec["fits"].items():
            for k, v in fit.get("settings", {}).items():
                mlflow.log_param(f"{model}_{k}", v)
            for k in ("best_iteration", "best_epoch", "fit_seconds"):
                if k in fit:
                    mlflow.log_metric(f"{model}_{k}", float(fit[k]))
        for model, m in rec["models"].items():
            for k in ("mae", "rmse", "mape", "bias", "skill_vs_seasonal_naive"):
                if k in m and np.isfinite(m[k]):
                    mlflow.log_metric(f"{model}_{k}", float(m[k]))
        mlflow.log_dict({k: v for k, v in rec.items() if k != "fits"} | {"fits": rec["fits"]}, "fold_metrics.json")
        return run.info.run_id


def log_aggregate(series: str, model: str, summary: dict, table: pd.DataFrame, prefix: str = "grid-forecast") -> str:
    experiment(series, prefix)
    m = summary["models"][model]
    with mlflow.start_run(run_name=f"aggregate-{model}") as run:
        mlflow.set_tags({"kind": "aggregate", "series": series, "model": model})
        mlflow.log_params({"folds": summary["folds"], "model": model, "feature_hash": feature_hash(series)})
        mlflow.log_metrics({
            "mean_fold_mae": m["mean_fold_mae"], "mean_fold_rmse": m["mean_fold_rmse"], "mean_fold_skill": m["mean_fold_skill"],
            "folds_with_positive_skill": m["folds_with_positive_skill"], "pooled_mae": m["pooled"]["mae"], "pooled_rmse": m["pooled"]["rmse"],
        })
        if "mean_fold_mape" in m:
            mlflow.log_metric("mean_fold_mape", m["mean_fold_mape"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"metrics_table_{series}_{model}.csv"
            table.to_csv(p, index=False)
            mlflow.log_artifact(str(p))
            q = Path(d) / f"summary_{series}_{model}.json"
            q.write_text(json.dumps(m, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(q))
        return run.info.run_id


def register_champion(series: str, forecaster, example: pd.DataFrame, mean_fold_mae: float, folds: int, prefix: str = "grid-forecast") -> dict:
    """Log the final refit as a pyfunc model, register it and point the ``champion`` alias at the new version."""
    experiment(series, prefix)
    name = f"{prefix}-{series}"
    cols = [c for c in INPUT_COLUMNS if c in example.columns]
    X = example[cols].head(24).copy()
    X["day"] = X["day"].astype(str)
    with mlflow.start_run(run_name=f"champion-{forecaster.name}") as run:
        mlflow.set_tags({"kind": "champion", "series": series, "model": forecaster.name})
        mlflow.log_params({"model": forecaster.name, "mean_fold_mae": round(mean_fold_mae, 4), "folds": folds, "feature_hash": feature_hash(series), **forecaster.settings()})
        mlflow.log_metric("mean_fold_mae", mean_fold_mae)
        yhat = np.asarray(forecaster.predict(example.head(24)), dtype=float)
        sig = infer_signature(X, yhat)
        info = mlflow.pyfunc.log_model(name="model", python_model=ForecasterModel(forecaster), signature=sig, input_example=X)
        run_id = run.info.run_id
    mv = mlflow.register_model(info.model_uri, name)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(name, "champion", mv.version)
    client.set_model_version_tag(name, mv.version, "model", forecaster.name)
    client.set_model_version_tag(name, mv.version, "mean_fold_mae", f"{mean_fold_mae:.4f}")
    return {"registered_model": name, "version": int(mv.version), "alias": "champion", "run_id": run_id, "model": forecaster.name}


def run_counts(series_list, prefix: str = "grid-forecast") -> dict:
    out = {}
    for s in series_list:
        runs = mlflow.search_runs(experiment_names=[f"{prefix}-{s}"], output_format="list")
        kinds = {}
        for r in runs:
            k = r.data.tags.get("kind", "other")
            kinds[k] = kinds.get(k, 0) + 1
        out[s] = kinds
    return out
