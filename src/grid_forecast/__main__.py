"""Command line entry point.

``download`` fetches the three series for a date range (default: 23 June 2025 to 30 September 2026) after
the licence check, ``prepare`` cleans them to hourly UTC frames, ``run`` walks the monthly folds with both
models, logs to MLflow, registers the champions, checks the API and writes ``results/`` and the README
tables, ``smoke`` is the short version of ``run`` (two folds, capped rounds and epochs, under ten minutes)
that writes ``results/smoke.json`` and ``results/readme_numbers.json``, ``determinism`` compares a second full
run in ``results/determinism/`` with the first, ``readme`` rewrites the README blocks and the acceptance record
from the results files, and ``serve`` starts the API.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from . import DOWNLOAD_START, PERIOD_END


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="grid_forecast")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="fetch demand, wind and price into data/raw/ (licence rows first)")
    d.add_argument("--start", default=DOWNLOAD_START.isoformat())
    d.add_argument("--end", default=PERIOD_END.isoformat())
    d.add_argument("--no-dashboard", action="store_true", help="skip the Smart Grid Dashboard second route")
    sub.add_parser("prepare", help="clean data/raw/ into hourly UTC frames under data/prepared/")
    r = sub.add_parser("run", help="full walk-forward run, MLflow logging, champion registration, API check, README tables")
    r.add_argument("--folds", type=int, default=None, help="keep only the last N folds (default: all)")
    r.add_argument("--lgbm-rounds", type=int, default=2000)
    r.add_argument("--gru-epochs", type=int, default=20)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--results-dir", default=None, help="write results here instead of results/ (the determinism run uses results/determinism)")
    r.add_argument("--docs-dir", default=None, help="write the failure plots here instead of docs/")
    r.add_argument("--tracking-uri", default=None, help="MLflow tracking URI for this run (default: file store under mlruns/)")
    r.add_argument("--label", default="full", help="label written into summary.json")
    r.add_argument("--no-readme", action="store_true", help="do not rewrite the README tables or results/readme_numbers.json")
    sub.add_parser("smoke", help="two folds, capped models, own MLflow store, results/smoke.json")
    sub.add_parser("determinism", help="compare results/ with results/determinism/ (a second full run with the same seed)")
    sub.add_parser("readme", help="rewrite the README blocks and results/acceptance.json from results/*.json")
    s = sub.add_parser("serve", help="start the FastAPI service with uvicorn")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    if args.cmd == "download":
        from .download import download

        terms = download(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end), try_dashboard=not args.no_dashboard)
        for k, v in terms["entries"].items():
            print(k, v.get("status", ""), [(f.get("name"), f.get("rows"), f.get("missing_steps")) for f in v.get("files", [])])
        return 0
    if args.cmd == "prepare":
        from .prepare import prepare

        report = prepare()
        for s_name, q in report["series"].items():
            worst = max(v["missing_share"] for v in q["months"].values())
            print(f"{s_name}: {len(q['months'])} months, worst missing share {worst:.2%}, dropped {q['dropped_months']}")
        return 0
    if args.cmd == "run":
        from pathlib import Path

        from . import DOCS, RESULTS
        from .pipeline import run

        results = Path(args.results_dir) if args.results_dir else RESULTS
        docs = Path(args.docs_dir) if args.docs_dir else (DOCS if args.results_dir is None else results)
        summary = run(n_folds=args.folds, lgbm_rounds=args.lgbm_rounds, gru_epochs=args.gru_epochs, seed=args.seed, results=results, docs=docs,
                      tracking_uri=args.tracking_uri, label=args.label, write_readme=not args.no_readme)
        print(json.dumps({s_name: {"champion": v["champion"], "folds": v["folds"]} for s_name, v in summary["series"].items()}, indent=2))
        return 0
    if args.cmd == "smoke":
        from .pipeline import smoke

        out = smoke()
        return 0 if out["smoke_ok"] == "yes" else 1
    if args.cmd == "determinism":
        from .pipeline import compare_runs

        rep = compare_runs()
        return 0 if rep["lightgbm_identical"] and rep["gru_within_1_percent"] else 1
    if args.cmd == "readme":
        from .acceptance import write_acceptance
        from .report import build_readme

        acc = write_acceptance()
        numbers = build_readme()
        print(f"acceptance: {acc['met']} of {acc['of']} criteria met, {len(numbers)} README numbers written")
        return 0
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("grid_forecast.api:app", host=args.host, port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
