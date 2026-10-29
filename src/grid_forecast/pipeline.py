"""The full run and the smoke run.

``run`` builds the feature dataset per series from ``data/prepared/``, walks the monthly folds, logs each fold
and each aggregate to MLflow, writes the failure-hour tables and plots, refits the champion of each series on
the whole history and registers it with the ``champion`` alias, checks the API with the TestClient, and
writes every ``results/*.json`` plus the README tables. ``smoke`` is the same chain on the last two folds with
LightGBM capped at 200 rounds and the GRU at 3 epochs, logged to ``mlruns_smoke/`` and written to
``results/smoke/``, so it never overwrites the numbers of the full run. It checks that the README carries
every number a fresh render of the results files produces. ``compare_runs`` compares two full runs for the
determinism criterion.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

import pandas as pd

from . import DOCS, PERIOD_END, PREPARED, RESULTS, ROOT, SEED, SERIES, SERIES_START, mlflow_uri
from .acceptance import write_acceptance
from .failures import write_failures
from .features import SeriesTable, build_dataset, days_between, feature_hash
from .models import make_model
from .prepare import load_prepared
from .report import build_readme, failures_table, render_readme, results_table, write_numbers
from .tracking import log_aggregate, log_fold, register_champion, run_counts, setup
from .walkforward import MODEL_NAMES, VALID_DAYS, folds_for, run_fold, summarise


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _fold_table(series: str, records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"fold": r["fold"], "test_month": r["test_month"], "train_months": r["train_months"], "scored_hours": r["scored_hours"]}
        for m in MODEL_NAMES:
            row[f"{m}_mae"] = round(r["models"][m]["mae"], 3)
            row[f"{m}_rmse"] = round(r["models"][m]["rmse"], 3)
            row[f"{m}_skill"] = round(r["models"][m]["skill_vs_seasonal_naive"], 4)
            if series == "demand":
                row[f"{m}_mape"] = round(r["models"][m]["mape"], 3)
        row["lightgbm_rounds"] = r["fits"]["lightgbm"]["best_iteration"]
        row["gru_epochs"] = r["fits"]["gru"]["best_epoch"]
        row["lightgbm_seconds"] = r["fits"]["lightgbm"]["fit_seconds"]
        row["gru_seconds"] = r["fits"]["gru"]["fit_seconds"]
        rows.append(row)
    return pd.DataFrame(rows)


def api_check(prepared: Path, tracking_uri: str, results: Path) -> dict:
    from fastapi.testclient import TestClient

    from .api import create_app, origin_for_day

    app = create_app(prepared=prepared, tracking_uri=tracking_uri)
    out = {"checked_at": _now(), "series": {}}
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        models = client.get("/models").json()
        out["models"] = models
        for s in SERIES:
            origin = origin_for_day(s, PERIOD_END)
            body = {"series": s, "origin_utc": origin.strftime("%Y-%m-%dT%H:%M:%SZ")}
            t0 = time.perf_counter()
            r = client.post("/forecast", json=body)
            cold = time.perf_counter() - t0
            t0 = time.perf_counter()
            r2 = client.post("/forecast", json=body)
            warm = time.perf_counter() - t0
            rec = {"status": r.status_code, "cold_seconds": round(cold, 3), "warm_seconds": round(warm, 3)}
            if r2.status_code == 200:
                j = r2.json()
                rec.update({"values": len(j["values"]), "first": j["values"][0], "last": j["values"][-1], "model": j["model"], "version": j["version"]})
            else:
                rec["detail"] = r2.text[:500]
            out["series"][s] = rec
    (results / "api_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run(n_folds: int | None = None, lgbm_rounds: int = 2000, gru_epochs: int = 20, prepared: Path = PREPARED, results: Path = RESULTS,
        docs: Path = DOCS, tracking_uri: str | None = None, label: str = "full", seed: int = SEED, write_readme: bool = True) -> dict:
    t_start = time.perf_counter()
    results.mkdir(parents=True, exist_ok=True)
    uri = setup(tracking_uri or mlflow_uri())
    quality = json.loads((RESULTS / "data_quality.json").read_text(encoding="utf-8")) if (RESULTS / "data_quality.json").exists() else {"series": {}}
    tables = {s: SeriesTable(load_prepared(s, prepared)) for s in SERIES}
    summary: dict = {"label": label, "started_at": _now(), "tracking_uri": uri, "seed": seed, "settings": {"lgbm_max_rounds": lgbm_rounds, "gru_max_epochs": gru_epochs, "n_folds": n_folds},
                     "series": {}, "mlflow": {"fold_runs": {}, "aggregate_runs": {}, "champions": {}}}
    all_folds: dict[str, list[dict]] = {}
    failures: dict[str, dict] = {}
    for s in SERIES:
        dropped = quality["series"].get(s, {}).get("dropped_months", [])
        folds = folds_for(s, n_folds, dropped_months=dropped)
        if not folds:
            raise RuntimeError(f"no folds for {s}")
        first_day = dt.date.fromisoformat(folds[0]["train_start"])
        t0 = time.perf_counter()
        dataset = build_dataset(s, tables, days_between(first_day, PERIOD_END))
        build_s = round(time.perf_counter() - t0, 1)
        records = []
        preds = []
        fold_run_ids = []
        for f in folds:
            rec, p, _ = run_fold(s, dataset, f, {"max_rounds": lgbm_rounds}, {"epochs": gru_epochs}, seed=seed)
            fold_run_ids.append(log_fold(s, rec, seed))
            records.append(rec)
            preds.append(p)
            print(f"{s} fold {f['fold']:02d} {f['test_month']}: naive {rec['models']['seasonal_naive']['mae']:.1f} "
                  f"lgbm {rec['models']['lightgbm']['mae']:.1f} gru {rec['models']['gru']['mae']:.1f} "
                  f"({rec['fits']['lightgbm']['fit_seconds']:.0f} s + {rec['fits']['gru']['fit_seconds']:.0f} s)", flush=True)
        preds_df = pd.concat(preds, ignore_index=True)
        summ = summarise(s, records, preds_df)
        table = _fold_table(s, records)
        agg_ids = {m: log_aggregate(s, m, summ, table) for m in ("lightgbm", "gru")}
        (results / f"metrics_{s}.json").write_text(json.dumps({"series": s, "unit": s, "feature_hash": feature_hash(s), "dataset_build_seconds": build_s,
                                                                "dropped_months": dropped, "summary": summ, "folds": records}, indent=2), encoding="utf-8")
        failures[s] = write_failures(s, preds_df, tables, summ["champion"], results, docs)
        # Final refit of the champion type on the whole history, the last fourteen days as the early-stopping set.
        last_day = PERIOD_END
        valid_start = last_day - dt.timedelta(days=VALID_DAYS - 1)
        day = dataset["day"]
        train = dataset[day < valid_start]
        valid = dataset[(day >= valid_start) & (day <= last_day)]
        kwargs = {"max_rounds": lgbm_rounds} if summ["champion"] == "lightgbm" else {"epochs": gru_epochs}
        model = make_model(summ["champion"], s, seed=seed, **kwargs)
        t0 = time.perf_counter()
        fit = model.fit(train, valid)
        refit_s = round(time.perf_counter() - t0, 1)
        champ = register_champion(s, model, valid.tail(24), summ["models"][summ["champion"]]["mean_fold_mae"], summ["folds"])
        champ["refit"] = {k: v for k, v in fit.items() if k != "valid_history"}
        champ["refit_seconds"] = refit_s
        summary["series"][s] = {**summ, "train_start": folds[0]["train_start"], "series_start": SERIES_START[s].isoformat(), "dataset_rows": int(len(dataset)),
                                "dataset_build_seconds": build_s, "fold_seconds": {"lightgbm": [r["fits"]["lightgbm"]["fit_seconds"] for r in records], "gru": [r["fits"]["gru"]["fit_seconds"] for r in records]},
                                "max_fold_seconds": {"lightgbm": max(r["fits"]["lightgbm"]["fit_seconds"] for r in records), "gru": max(r["fits"]["gru"]["fit_seconds"] for r in records)},
                                "lightgbm_rounds": [r["fits"]["lightgbm"]["best_iteration"] for r in records], "gru_epochs": [r["fits"]["gru"]["best_epoch"] for r in records],
                                "failures": {k: v for k, v in failures[s].items() if k != "rows"}}
        summary["mlflow"]["fold_runs"][s] = fold_run_ids
        summary["mlflow"]["aggregate_runs"][s] = agg_ids
        summary["mlflow"]["champions"][s] = champ
        all_folds[s] = folds
    (results / "folds.json").write_text(json.dumps(all_folds, indent=2), encoding="utf-8")
    summary["mlflow"]["run_counts"] = run_counts(SERIES)
    (results / "mlflow_runs.json").write_text(json.dumps(summary["mlflow"], indent=2), encoding="utf-8")
    summary["api_check"] = api_check(prepared, uri, results)
    summary["finished_at"] = _now()
    summary["total_seconds"] = round(time.perf_counter() - t_start, 1)
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results_md, _ = results_table(summary)
    failures_md, _ = failures_table({s: v["failures"] for s, v in summary["series"].items()})
    if write_readme:
        write_acceptance(results)
        build_readme(results)
    print(results_md)
    print(failures_md)
    return summary


def smoke() -> dict:
    """Last two folds, LightGBM at 200 rounds, GRU at 3 epochs, own tracking store, results under results/smoke/."""
    t0 = time.perf_counter()
    smoke_results = RESULTS / "smoke"
    smoke_results.mkdir(parents=True, exist_ok=True)
    uri = (ROOT / "mlruns_smoke").resolve().as_uri()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    summary = run(n_folds=2, lgbm_rounds=200, gru_epochs=3, results=smoke_results, docs=smoke_results, tracking_uri=uri, label="smoke", write_readme=False)
    ok = all(v["status"] == 200 and v.get("values") == 24 for v in summary["api_check"]["series"].values())
    expected = {"metrics_demand.json", "metrics_wind.json", "metrics_price.json", "folds.json", "worst_hours_demand.json", "worst_hours_wind.json",
                "worst_hours_price.json", "api_check.json", "summary.json", "mlflow_runs.json"}
    missing = sorted(expected - {p.name for p in smoke_results.glob("*.json")})
    full = RESULTS / "summary.json"
    readme_ok = False
    if full.exists():
        _, numbers = render_readme(RESULTS)
        write_numbers(numbers, RESULTS / "readme_numbers.json")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_ok = all(v in readme for v in numbers.values())
    out = {
        "smoke_ok": "yes" if ok and not missing and readme_ok else "no",
        "api_ok": ok, "missing_files": missing, "readme_matches_full_summary": readme_ok,
        "seconds": round(time.perf_counter() - t0, 1), "generated_at": _now(),
        "folds": {s: v["folds"] for s, v in summary["series"].items()},
        "lightgbm_mae": {s: round(v["models"]["lightgbm"]["mean_fold_mae"], 3) for s, v in summary["series"].items()},
        "gru_mae": {s: round(v["models"]["gru"]["mean_fold_mae"], 3) for s, v in summary["series"].items()},
    }
    (RESULTS / "smoke.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def compare_runs(a: Path = RESULTS, b: Path = RESULTS / "determinism", out: Path = RESULTS / "determinism.json") -> dict:
    """Compare two full runs with the same seed: LightGBM metrics must match exactly, GRU metrics within 1 %."""
    report: dict = {"run_a": str(a.relative_to(ROOT)), "run_b": str(b.relative_to(ROOT)), "series": {}, "generated_at": _now()}
    lgbm_identical = True
    gru_within_1pct = True
    for s in SERIES:
        ma = json.loads((a / f"metrics_{s}.json").read_text(encoding="utf-8"))
        mb = json.loads((b / f"metrics_{s}.json").read_text(encoding="utf-8"))
        rows = []
        worst_gru = 0.0
        for fa, fb in zip(ma["folds"], mb["folds"], strict=True):
            la, lb = fa["models"]["lightgbm"]["mae"], fb["models"]["lightgbm"]["mae"]
            ga, gb = fa["models"]["gru"]["mae"], fb["models"]["gru"]["mae"]
            rel = abs(ga - gb) / max(abs(ga), 1e-9)
            worst_gru = max(worst_gru, rel)
            same = la == lb and fa["models"]["lightgbm"]["rmse"] == fb["models"]["lightgbm"]["rmse"]
            lgbm_identical &= same
            gru_within_1pct &= rel < 0.01
            rows.append({"fold": fa["fold"], "test_month": fa["test_month"], "lightgbm_mae_a": la, "lightgbm_mae_b": lb, "lightgbm_identical": same,
                         "gru_mae_a": ga, "gru_mae_b": gb, "gru_relative_difference": rel})
        report["series"][s] = {"folds": rows, "gru_worst_relative_difference": worst_gru,
                               "mean_fold_mae_a": {m: ma["summary"]["models"][m]["mean_fold_mae"] for m in ("lightgbm", "gru")},
                               "mean_fold_mae_b": {m: mb["summary"]["models"][m]["mean_fold_mae"] for m in ("lightgbm", "gru")}}
    report["lightgbm_identical"] = bool(lgbm_identical)
    report["gru_within_1_percent"] = bool(gru_within_1pct)
    report["gru_worst_relative_difference"] = max(v["gru_worst_relative_difference"] for v in report["series"].values())
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "series"}, indent=2))
    return report
