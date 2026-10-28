"""Summary files and the README tables.

``summary.json`` holds the headline numbers per series, ``readme_numbers.json`` the exact strings that must
appear in README.md (asserted by ``tests/test_readme_numbers.py``), and the two README tables are rewritten
between their marker comments by ``update_readme``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import ROOT, UNITS

RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"
FAILURES_START = "<!-- failures:start -->"
FAILURES_END = "<!-- failures:end -->"

PERIOD_LABEL = {"demand": "Oct 2025 to Sep 2026", "wind": "Oct 2025 to Sep 2026", "price": "Jan to Sep 2026 (data from Oct 2025)"}


def f1(x: float) -> str:
    return f"{x:,.1f}"


def pct(x: float) -> str:
    return f"{x * 100:+.1f} %"


def results_table(summary: dict) -> tuple[str, dict[str, str]]:
    """The README results table and the number strings it carries."""
    numbers: dict[str, str] = {}
    lines = [
        "| Series | Period | Folds | Seasonal-naive MAE | Last-week MAE | LightGBM MAE | GRU MAE | Best skill vs seasonal naive | Folds with positive skill | Worst hour of day (MAE) | Champion |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s, r in summary["series"].items():
        m = r["models"]
        unit = UNITS[s]
        best = r["champion"]
        row = {
            "naive": f"{f1(m['seasonal_naive']['mean_fold_mae'])} {unit}",
            "week": f"{f1(m['last_week']['mean_fold_mae'])} {unit}",
            "lgbm": f"{f1(m['lightgbm']['mean_fold_mae'])} {unit}",
            "gru": f"{f1(m['gru']['mean_fold_mae'])} {unit}",
            "skill": pct(m[best]["mean_fold_skill"]),
            "positive": f"{m[best]['folds_with_positive_skill']} of {r['folds']}",
            "worst": f"{r['worst_hour_of_day']['hour']:02d}:00 ({f1(r['worst_hour_of_day']['mae'])} {unit})",
        }
        for k, v in row.items():
            numbers[f"{s}_{k}"] = v
        numbers[f"{s}_folds"] = str(r["folds"])
        numbers[f"{s}_champion"] = best
        lines.append(f"| {s} ({unit}) | {PERIOD_LABEL[s]} | {r['folds']} | {row['naive']} | {row['week']} | {row['lgbm']} | {row['gru']} | {row['skill']} | {row['positive']} | {row['worst']} | {best} |")
    return "\n".join(lines), numbers


def failures_table(failures: dict) -> tuple[str, dict[str, str]]:
    numbers: dict[str, str] = {}
    lines = [
        "| Series | Failure hours (worst 5 %) | Error threshold | Holiday share (failures / all hours) | Clock-change share | Wind-ramp share | Price-spike share | Negative-price share | Interpolated-input share | Evening-peak share | Morning-ramp share |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s, f in failures.items():
        t = f["by_tag"]
        unit = UNITS[s]

        def cell(tag: str, t: dict = t) -> str:
            return f"{t[tag]['share_of_failure_hours'] * 100:.1f} % / {t[tag]['share_of_all_scored_hours'] * 100:.1f} %"

        numbers[f"{s}_failure_hours"] = str(f["failure_hours"])
        numbers[f"{s}_holiday_share"] = cell("holiday")
        numbers[f"{s}_dst_share"] = cell("dst_day")
        lines.append(
            f"| {s} | {f['failure_hours']} of {f['scored_hours']} | {f1(f['error_threshold'])} {unit} | {cell('holiday')} | {cell('dst_day')} | {cell('wind_ramp')} | "
            f"{cell('price_spike')} | {cell('negative_price')} | {cell('interpolated_input')} | {cell('evening_peak')} | {cell('morning_ramp')} |"
        )
    return "\n".join(lines), numbers


def replace_between(text: str, start: str, end: str, body: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    block = f"{start}\n{body}\n{end}"
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    return text.rstrip("\n") + "\n\n" + block + "\n"


def update_readme(results_md: str, failures_md: str, readme: Path = ROOT / "README.md") -> None:
    text = readme.read_text(encoding="utf-8")
    text = replace_between(text, RESULTS_START, RESULTS_END, results_md)
    text = replace_between(text, FAILURES_START, FAILURES_END, failures_md)
    readme.write_text(text, encoding="utf-8")


def write_numbers(numbers: dict[str, str], path: Path) -> None:
    path.write_text(json.dumps(numbers, indent=2), encoding="utf-8")
