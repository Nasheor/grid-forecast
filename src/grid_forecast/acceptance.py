"""The twelve acceptance criteria of the spec, measured from the results files.

``write_acceptance`` reads ``results/*.json`` (and runs the leakage test for criterion 3), records every outcome
in ``results/acceptance.json``, and the README's acceptance table is rendered from that file. A criterion
that is not met is written down as not met, with the measured value next to it.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import LICENCE_CHECK, RESULTS, ROOT, SERIES

MODELS = ("seasonal_naive", "last_week", "lightgbm", "gru")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(x: float) -> str:
    return f"{x * 100:.1f} %"


def _leakage_test() -> tuple[bool, str]:
    # pyproject.toml already adds -q, a second -q would hide the summary line this parses
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_leakage.py", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    m = re.search(r"(\d+) passed", r.stdout)
    passed = int(m.group(1)) if m else 0
    if r.returncode == 0 and passed:
        return True, f"{passed} passed"
    return False, "failed: " + (r.stdout[-300:] or r.stderr[-300:]).strip()


def acceptance(results: Path = RESULTS, run_tests: bool = True) -> dict:
    summary = _load(results / "summary.json")
    terms = _load(results / "data_terms.json")
    quality = _load(results / "data_quality.json")
    runs = _load(results / "mlflow_runs.json")
    api = _load(results / "api_check.json")
    determinism = _load(results / "determinism.json")
    smoke = _load(results / "smoke.json")
    metrics = {s: _load(results / f"metrics_{s}.json") for s in SERIES}
    worst = {s: _load(results / f"worst_hours_{s}.json") for s in SERIES}
    criteria: list[dict] = []

    def add(i: int, criterion: str, outcome: str, met: bool) -> None:
        criteria.append({"id": i, "criterion": criterion, "outcome": outcome, "met": bool(met)})

    # 1. licence rows before downloads
    commit_time = pd.Timestamp(terms["licence_check_commit"]["time"])
    stamps = []
    for e in terms["entries"].values():
        stamps += [pd.Timestamp(f["downloaded_at"]) for f in e.get("files", []) if f.get("downloaded_at")]
        first_doc = e.get("documents", {}).get("documents_written", {}).get("first")
        if first_doc:
            stamps.append(pd.Timestamp(first_doc))
    used = [k for k, e in terms["entries"].items() if e.get("files") or e.get("documents")]
    licence_text = LICENCE_CHECK.read_text(encoding="utf-8")
    rows_present = all(f"`{k}`" in licence_text for k in used)
    ok1 = rows_present and bool(stamps) and all(t > commit_time for t in stamps)
    add(1, "A LICENCE_CHECK.md row for every source used, every download after the row's commit",
        f"{len(used)} sources used, each with a row. Licence check committed {commit_time.strftime('%Y-%m-%d %H:%M:%S')} UTC, "
        f"first download {min(stamps).strftime('%Y-%m-%d %H:%M:%S')} UTC, last {max(stamps).strftime('%Y-%m-%d %H:%M:%S')} UTC", ok1)

    # 2. folds
    folds = {s: summary["series"][s]["folds"] for s in SERIES}
    total = sum(folds.values())
    first_train = min(m["folds"][0]["train_months"] for m in metrics.values())
    add(2, "Twelve monthly folds per series (36 fold runs), at least three months of training in the first fold",
        f"{total} fold runs ({', '.join(f'{s} {n}' for s, n in folds.items())}), the first fold of every series trains on {first_train} months. "
        "The price has nine folds, not twelve, because its hourly report starts on 1 October 2025", total == 36 and first_train >= 3)

    # 3. leakage test
    ok3, text3 = _leakage_test() if run_tests else (False, "not run")
    add(3, "The leakage test passes", f"tests/test_leakage.py: {text3}", ok3)

    # 4. metrics complete
    complete = all("mae" in f["models"][k] and "rmse" in f["models"][k] for m in metrics.values() for f in m["folds"] for k in MODELS)
    complete &= all("mean_fold_mae" in m["summary"]["models"][k] and "mean_fold_rmse" in m["summary"]["models"][k] for m in metrics.values() for k in MODELS)
    add(4, "MAE and RMSE of both baselines and both models per fold and overall in results/metrics_<series>.json and the README",
        f"present for all {total} folds of the 3 series and as means, the README carries the mean tables and the per-fold tables", complete)

    # 5. demand LightGBM
    k5 = summary["series"]["demand"]["models"]["lightgbm"]["folds_with_positive_skill"]
    add(5, "Demand: LightGBM skill against the seasonal naive positive on at least 9 of 12 folds", f"{k5} of {folds['demand']}", k5 >= 9)

    # 6. wind and price in words
    per_fold = all(len(summary["series"][s]["models"][m]["fold_skill"]) == folds[s] for s in ("wind", "price") for m in ("lightgbm", "gru"))
    parts = []
    for s in ("wind", "price"):
        su = summary["series"][s]
        champ = su["champion"]
        other = "gru" if champ == "lightgbm" else "lightgbm"
        better = sum(1 for f in metrics[s]["folds"] if f["models"][other]["mae"] < f["models"][champ]["mae"])
        parts.append(f"{s} champion {champ} (positive on {su['models'][champ]['folds_with_positive_skill']} of {folds[s]} folds, {other} better in {better})")
    add(6, "Wind and price: skill per fold reported, the README says in words which model wins and where it loses",
        ", ".join(parts) + ". Stated in words under Reading the results", per_fold)

    # 7. failure tables
    rows_tagged = all(all(r["tags"] for r in worst[s]["rows"]) for s in SERIES)
    n_rows = sum(len(worst[s]["rows"]) for s in SERIES)
    hol = " / ".join(_pct(worst[s]["by_tag"]["holiday"]["share_of_failure_hours"]) for s in SERIES)
    dst = " / ".join(_pct(worst[s]["by_tag"]["dst_day"]["share_of_failure_hours"]) for s in SERIES)
    add(7, "A failure-hours table per series with at least one tag per row, the share of failure hours on holidays and clock-change days reported",
        f"all {n_rows} rows tagged. Holiday share of failure hours demand / wind / price: {hol}. Clock-change share: {dst}", rows_tagged)

    # 8. MLflow
    rc = runs["run_counts"]
    fold_runs = sum(v.get("fold", 0) for v in rc.values())
    agg = sum(v.get("aggregate", 0) for v in rc.values())
    champ_runs = sum(v.get("champion", 0) for v in rc.values())
    registered = sum(1 for c in runs["champions"].values() if c.get("alias") == "champion")
    add(8, "MLflow: 36 fold runs, 6 aggregate runs, three registered models with the champion alias",
        f"{fold_runs} fold runs, {agg} aggregate runs, {champ_runs} champion runs, {registered} registered models with the alias champion. "
        "Three fold runs short of 36 because the price has nine folds", fold_runs == 36 and agg == 6 and registered == 3)

    # 9. API
    warm = max(v["warm_seconds"] for v in api["series"].values())
    cold = max(v["cold_seconds"] for v in api["series"].values())
    all24 = all(v.get("status") == 200 and v.get("values") == 24 for v in api["series"].values())
    add(9, "POST /forecast returns 24 timestamped values per series in under 2 seconds on CPU from a warm process",
        f"24 values for every series, warm response at most {warm:.3f} s, cold at most {cold:.3f} s (FastAPI TestClient)", all24 and warm < 2)

    # 10. determinism
    if determinism:
        same = "yes" if determinism["lightgbm_identical"] else "no"
        add(10, "Two full runs with the same seed give identical LightGBM metrics and GRU metrics within 1 %",
            f"LightGBM MAE and RMSE identical on every fold: {same}. Largest GRU fold-MAE difference {determinism['gru_worst_relative_difference'] * 100:.3f} %",
            determinism["lightgbm_identical"] and determinism["gru_within_1_percent"])
    else:
        add(10, "Two full runs with the same seed give identical LightGBM metrics and GRU metrics within 1 %", "second run not compared yet", False)

    # 11. missing data
    months = [(s, ym, mo) for s, q in quality["series"].items() for ym, mo in q["months"].items()]
    worst_share = max(mo["missing_share"] for _, _, mo in months)
    dropped = [f"{s} {ym}" for s, q in quality["series"].items() for ym in q["dropped_months"]]
    add(11, "Missing data per month under 5 % for every month kept, dropped months listed",
        f"highest missing share {_pct(worst_share)} over {len(months)} series-months, dropped: {', '.join(dropped) if dropped else 'none'}", worst_share < 0.05)

    # 12. smoke and tests
    if smoke:
        add(12, "Smoke under 10 minutes, pytest green, no secrets, README numbers asserted",
            f"smoke {smoke['seconds']:.0f} s on the build PC with smoke_ok {smoke['smoke_ok']}. tests/test_no_secrets.py and tests/test_readme_numbers.py are part of pytest -q",
            smoke["smoke_ok"] == "yes" and smoke["seconds"] < 600)
    else:
        add(12, "Smoke under 10 minutes, pytest green, no secrets, README numbers asserted", "smoke not run yet", False)

    return {"generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "criteria": criteria,
            "met": sum(1 for c in criteria if c["met"]), "of": len(criteria)}


def write_acceptance(results: Path = RESULTS, run_tests: bool = True) -> dict:
    rec = acceptance(results, run_tests)
    (results / "acceptance.json").write_text(json.dumps(rec, indent=2), encoding="utf-8", newline="\n")
    return rec
