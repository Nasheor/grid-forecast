"""Summary files and the README blocks.

``summary.json`` holds the headline numbers per series and ``readme_numbers.json`` the exact strings that must
appear in README.md (asserted by ``tests/test_readme_numbers.py``). Every block between a pair of marker
comments in the README is rewritten from the results files by ``build_readme``: the results table, the
failure-hours table, the mean RMSE table, the per-fold tables, the reading of the results in words, and the
acceptance record. Nothing in those blocks is typed by hand, so the README cannot drift from the run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import RESULTS, ROOT, SERIES, TZ, UNITS

MARKERS = {
    "results": ("<!-- results:start -->", "<!-- results:end -->"),
    "failures": ("<!-- failures:start -->", "<!-- failures:end -->"),
    "rmse": ("<!-- rmse:start -->", "<!-- rmse:end -->"),
    "folds": ("<!-- folds:start -->", "<!-- folds:end -->"),
    "reading": ("<!-- reading:start -->", "<!-- reading:end -->"),
    "acceptance": ("<!-- acceptance:start -->", "<!-- acceptance:end -->"),
}

PERIOD_LABEL = {"demand": "Oct 2025 to Sep 2026", "wind": "Oct 2025 to Sep 2026", "price": "Jan to Sep 2026 (data from Oct 2025)"}
MODEL_LABEL = {"seasonal_naive": "the seasonal naive", "last_week": "last week", "lightgbm": "LightGBM", "gru": "the GRU"}
MODEL_TITLE = {"seasonal_naive": "Seasonal naive", "last_week": "Last week", "lightgbm": "LightGBM", "gru": "The GRU"}
TAG_LABEL = {
    "holiday": "public holidays", "dst_day": "clock-change days", "wind_ramp": "wind ramps", "price_spike": "price spikes",
    "negative_price": "negative prices", "interpolated_input": "interpolated inputs", "first_day_after_gap": "first days after a gap",
    "night": "night hours", "morning_ramp": "morning-ramp hours", "midday": "midday hours", "evening_peak": "evening-peak hours",
    "late_evening": "late-evening hours",
}
MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
LEARNED = ("lightgbm", "gru")
ALL_MODELS = ("seasonal_naive", "last_week", "lightgbm", "gru")


def f1(x: float) -> str:
    return f"{x:,.1f}"


def pct(x: float) -> str:
    return f"{x * 100:+.1f} %"


def share(x: float) -> str:
    return f"{x * 100:.1f} %"


def month_name(ym: str) -> str:
    y, m = ym.split("-")
    return f"{MONTH_NAMES[int(m)]} {y}"


def local_label(ts_utc: str) -> str:
    t = pd.Timestamp(ts_utc).tz_convert(TZ)
    return f"{t.day} {MONTH_NAMES[t.month]} {t.year} at {t:%H}:00 local time"


def months_list(months: list[str]) -> str:
    names = [month_name(m) for m in months]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


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


def rmse_table(summary: dict) -> tuple[str, dict[str, str]]:
    """Mean fold RMSE of the four models per series, and the mean fold MAPE for demand."""
    numbers: dict[str, str] = {}
    lines = [
        "| Series | Seasonal-naive RMSE | Last-week RMSE | LightGBM RMSE | GRU RMSE | MAPE (LightGBM / GRU) |",
        "|---|---|---|---|---|---|",
    ]
    for s, r in summary["series"].items():
        m = r["models"]
        unit = UNITS[s]
        cells = {k: f"{f1(m[k]['mean_fold_rmse'])} {unit}" for k in ALL_MODELS}
        for k, v in cells.items():
            numbers[f"{s}_{k}_rmse"] = v
        if "mean_fold_mape" in m["lightgbm"]:
            mape = f"{m['lightgbm']['mean_fold_mape']:.2f} % / {m['gru']['mean_fold_mape']:.2f} %"
            numbers[f"{s}_mape"] = mape
        else:
            mape = "not defined (values can be zero or negative)"
        lines.append(f"| {s} ({unit}) | {cells['seasonal_naive']} | {cells['last_week']} | {cells['lightgbm']} | {cells['gru']} | {mape} |")
    return "\n".join(lines), numbers


def fold_tables(metrics: dict[str, dict]) -> tuple[str, dict[str, str]]:
    """One collapsed table per series with the MAE and RMSE of every model on every fold."""
    numbers: dict[str, str] = {}
    out = []
    for s in SERIES:
        m = metrics[s]
        unit = UNITS[s]
        out.append(f"<details><summary>Per-fold MAE and RMSE, {s} ({unit})</summary>\n")
        out.append("| Fold | Test month | Training months | Seasonal naive MAE / RMSE | Last week MAE / RMSE | LightGBM MAE / RMSE | GRU MAE / RMSE | LightGBM skill | GRU skill |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for f in m["folds"]:
            cells = {k: f"{f1(f['models'][k]['mae'])} / {f1(f['models'][k]['rmse'])}" for k in ALL_MODELS}
            skills = {k: pct(f["models"][k]["skill_vs_seasonal_naive"]) for k in LEARNED}
            for k, v in cells.items():
                numbers[f"{s}_fold{f['fold']:02d}_{k}"] = v
            for k, v in skills.items():
                numbers[f"{s}_fold{f['fold']:02d}_{k}_skill"] = v
            out.append(f"| {f['fold']} | {month_name(f['test_month'])} | {f['train_months']} | {cells['seasonal_naive']} | {cells['last_week']} | {cells['lightgbm']} | {cells['gru']} | {skills['lightgbm']} | {skills['gru']} |")
        out.append("\n</details>\n")
    return "\n".join(out).rstrip("\n"), numbers


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
            return f"{share(t[tag]['share_of_failure_hours'])} / {share(t[tag]['share_of_all_scored_hours'])}"

        numbers[f"{s}_failure_hours"] = str(f["failure_hours"])
        numbers[f"{s}_holiday_share"] = cell("holiday")
        numbers[f"{s}_dst_share"] = cell("dst_day")
        lines.append(
            f"| {s} | {f['failure_hours']} of {f['scored_hours']} | {f1(f['error_threshold'])} {unit} | {cell('holiday')} | {cell('dst_day')} | {cell('wind_ramp')} | "
            f"{cell('price_spike')} | {cell('negative_price')} | {cell('interpolated_input')} | {cell('evening_peak')} | {cell('morning_ramp')} |"
        )
    return "\n".join(lines), numbers


def _series_paragraph(s: str, su: dict, m: dict, worst: dict, numbers: dict[str, str]) -> str:
    """The reading of one series in words: who wins, where it loses, where the error sits."""
    unit = UNITS[s]
    models = su["models"]
    champ = su["champion"]
    other = "gru" if champ == "lightgbm" else "lightgbm"
    folds = m["folds"]
    n = su["folds"]

    def num(key: str, value: str) -> str:
        numbers[f"{s}_{key}"] = value
        return value

    champ_mae = num("read_champ_mae", f"{f1(models[champ]['mean_fold_mae'])} {unit}")
    naive_mae = num("read_naive_mae", f"{f1(models['seasonal_naive']['mean_fold_mae'])} {unit}")
    week_mae = num("read_week_mae", f"{f1(models['last_week']['mean_fold_mae'])} {unit}")
    skill = num("read_skill", pct(models[champ]["mean_fold_skill"]))
    positive = num("read_positive", f"{models[champ]['folds_with_positive_skill']} of {n}")
    text = [f"**{s.capitalize()}.** {MODEL_TITLE[champ]} is the champion: a mean fold MAE of {champ_mae} against {naive_mae} for the seasonal naive "
            f"and {week_mae} for last week, a skill of {skill}, positive on {positive} folds."]
    if models["last_week"]["mean_fold_mae"] > models["seasonal_naive"]["mean_fold_mae"]:
        text.append("A week-old value is a worse guess than yesterday's for this series.")
    losing = [f for f in folds if f["models"][champ]["skill_vs_seasonal_naive"] <= 0]
    if losing:
        worst_f = min(losing, key=lambda f: f["models"][champ]["skill_vs_seasonal_naive"])
        a = num("read_lose_champ", f"{f1(worst_f['models'][champ]['mae'])} {unit}")
        b = num("read_lose_naive", f"{f1(worst_f['models']['seasonal_naive']['mae'])} {unit}")
        if len(losing) == 1:
            text.append(f"It loses to the naive in {month_name(worst_f['test_month'])} ({a} against {b}).")
        else:
            text.append(f"It loses to the naive in {months_list([f['test_month'] for f in losing])}, worst in {month_name(worst_f['test_month'])} ({a} against {b}).")
    else:
        tight = min(folds, key=lambda f: f["models"][champ]["skill_vs_seasonal_naive"])
        a = num("read_tight_champ", f"{f1(tight['models'][champ]['mae'])} {unit}")
        b = num("read_tight_naive", f"{f1(tight['models']['seasonal_naive']['mae'])} {unit}")
        text.append(f"It beats the naive in every fold, by the least in {month_name(tight['test_month'])} ({a} against {b}).")
    other_mae = num("read_other_mae", f"{f1(models[other]['mean_fold_mae'])} {unit}")
    other_pos = num("read_other_positive", f"{models[other]['folds_with_positive_skill']} of {n}")
    better = [f["test_month"] for f in folds if f["models"][other]["mae"] < f["models"][champ]["mae"]]
    if better:
        text.append(f"{MODEL_TITLE[other]} averages {other_mae}, is positive on {other_pos} folds, and is the better model in {len(better)} of {n} folds: {months_list(better)}.")
    else:
        text.append(f"{MODEL_TITLE[other]} averages {other_mae}, is positive on {other_pos} folds, and never beats {MODEL_LABEL[champ]} on a fold.")
    by_h = {int(k): v for k, v in models[champ]["mae_by_horizon"].items()}
    first_h = min(by_h)
    worst_h = max(by_h, key=by_h.get)
    a = num("read_h_first", f"{f1(by_h[first_h])} {unit}")
    b = num("read_h_worst", f"{f1(by_h[worst_h])} {unit}")
    text.append(f"By horizon, the champion's error is {a} at the first forecast hour and largest at hour {worst_h} ({b}).")
    tags = worst["by_tag"]
    enriched = sorted(
        ((t, v) for t, v in tags.items()
         if v["failure_hours"] >= 5 and v["share_of_all_scored_hours"] > 0 and v["share_of_failure_hours"] / v["share_of_all_scored_hours"] >= 1.4),
        key=lambda kv: kv[1]["share_of_failure_hours"] / kv[1]["share_of_all_scored_hours"], reverse=True)[:3]
    if enriched:
        parts = []
        for t, v in enriched:
            a = share(v["share_of_failure_hours"])
            b = share(v["share_of_all_scored_hours"])
            parts.append(f"{TAG_LABEL[t]} ({num(f'read_tag_{t}', f'{a} of the failure hours against {b} of all hours')})")
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
        text.append("Among the worst 5 % of hours the enriched tags are " + joined + ".")
    by_month = worst["by_month"]
    top_m = max(by_month, key=by_month.get)
    k = num("read_fail_month_hours", f"{by_month[top_m]} of {worst['failure_hours']}")
    text.append(f"{month_name(top_m)} holds the most failure hours ({k}).")
    row = worst["rows"][0]
    actual = num("read_worst_actual", f"{f1(row['target'])} {unit}")
    pred = num("read_worst_pred", f"{f1(row[champ])} {unit}")
    tag_text = row["tags"].replace(",", ", ").replace("_", " ")
    fold_of_row = next((f for f in folds if f["fold"] == row["fold"]), None)
    where = f", in fold {row['fold']} with {fold_of_row['train_months']} months of training" if fold_of_row else ""
    text.append(f"The single worst hour is {local_label(row['ts_utc'])}{where}: {actual} actual against {pred} forecast (tags: {tag_text}).")
    return " ".join(text)


def reading_section(summary: dict, metrics: dict[str, dict], worst: dict[str, dict], quality: dict, terms: dict, determinism: dict | None,
                    gru_check: dict | None) -> tuple[str, dict[str, str]]:
    """The results in words, every number registered so the README test asserts it."""
    numbers: dict[str, str] = {}
    paras = ["### Reading the results", ""]
    for s in SERIES:
        paras.append(_series_paragraph(s, summary["series"][s], metrics[s], worst[s], numbers))
        paras.append("")
    months = [(s, ym, mo) for s, q in quality["series"].items() for ym, mo in q["months"].items()]
    n_months = len(months)
    worst_share = max(mo["missing_share"] for _, _, mo in months)
    interp = sum(mo["interpolated_hours"] for _, _, mo in months)
    long_gap = sum(mo["long_gap_hours"] for _, _, mo in months)
    dropped = [f"{s} {ym}" for s, q in quality["series"].items() for ym in q["dropped_months"]]
    numbers["dq_months"] = str(n_months)
    numbers["dq_worst_share"] = share(worst_share)
    numbers["dq_interpolated"] = str(interp)
    numbers["dq_long_gap"] = str(long_gap)
    per_series = ", ".join(f"{len(q['months'])} for {s}" for s, q in quality["series"].items())
    dropped_text = "no month was dropped" if not dropped else "dropped: " + ", ".join(dropped)
    paras.append(f"**Data quality.** {n_months} series-months were checked ({per_series}). The highest missing share in any of them was "
                 f"{numbers['dq_worst_share']}, {interp} hours were interpolated, {long_gap} hours sat in gaps longer than three hours, and {dropped_text}. "
                 "The 25-hour clock-change day in October and the 23-hour day in March are kept as they are in UTC.")
    paras.append("")
    docs = terms["entries"]["semopx_price"]["documents"]
    numbers["price_days_requested"] = str(docs["days_requested"])
    numbers["price_days_found"] = str(docs["days_found"])
    numbers["price_days_missing"] = str(len(docs["missing_days"]))
    numbers["price_folds_read"] = str(summary["series"]["price"]["folds"])
    where = ""
    if docs["missing_days"]:
        after = pd.Timestamp(max(docs["missing_days"])) + pd.Timedelta(days=1)
        where = f", all of them before {after.day} {MONTH_NAMES[after.month]} {after.year}"
    paras.append(f"**Why the price has {numbers['price_folds_read']} folds.** The SEMOpx hourly report EA-025 starts on 1 October 2025. Of the {docs['days_requested']} delivery days "
                 f"requested from 23 June 2025, {docs['days_found']} documents exist and {len(docs['missing_days'])} are missing{where}. So the price has no warm-up "
                 "before the evaluated year, and the rule of at least three months of training puts its first fold in January 2026.")
    paras.append("")
    if determinism:
        same = "identical" if determinism["lightgbm_identical"] else "not identical"
        numbers["det_gru_worst"] = f"{determinism['gru_worst_relative_difference'] * 100:.3f} %"
        verdict = "within" if determinism["gru_within_1_percent"] else "outside"
        paras.append(f"**Determinism.** A second full run with the same seed (`results/determinism/`, its own MLflow store) gave {same} LightGBM MAE and RMSE on every fold "
                     f"of every series, and the GRU fold MAE differed by at most {numbers['det_gru_worst']}, {verdict} the 1 % criterion. The comparison is in `results/determinism.json`.")
    else:
        paras.append("**Determinism.** The second run has not been compared yet (`python -m grid_forecast determinism`).")
    paras.append("")
    if gru_check:
        chosen = gru_check["settings"][gru_check["chosen"]]
        numbers["gru_batch"] = str(chosen["batch_size"])
        numbers["gru_lr"] = f"{chosen['lr']:g}"
        paras.append(f"**A caveat on the GRU.** Its batch size ({numbers['gru_batch']}) and learning rate ({numbers['gru_lr']}) were chosen on a comparison over "
                     f"{len(gru_check['folds'])} folds that looked at test-month error (`results/gru_settings_check.json`), so the GRU numbers carry some selection "
                     "and are a little optimistic. LightGBM was not tuned. Both models see the same features and the same folds.")
    return "\n".join(paras).rstrip("\n"), numbers


def acceptance_table(acceptance: dict | None) -> tuple[str, dict[str, str]]:
    numbers: dict[str, str] = {}
    if not acceptance:
        return "Acceptance record not written yet (`python -m grid_forecast readme`).", numbers
    lines = ["| # | Criterion | Outcome | Met |", "|---|---|---|---|"]
    for c in acceptance["criteria"]:
        numbers[f"acc_{c['id']}"] = c["outcome"]
        lines.append(f"| {c['id']} | {c['criterion']} | {c['outcome']} | {'yes' if c['met'] else 'no'} |")
    lines.append("")
    lines.append(f"{acceptance['met']} of {acceptance['of']} criteria met.")
    numbers["acc_met"] = f"{acceptance['met']} of {acceptance['of']} criteria met"
    return "\n".join(lines), numbers


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def render_readme(results: Path = RESULTS) -> tuple[dict[str, str], dict[str, str]]:
    """Every README block from the results files, and the union of the number strings they carry."""
    summary = _load(results / "summary.json")
    if summary is None:
        raise FileNotFoundError(f"{results / 'summary.json'} is missing, run the full run first")
    metrics = {s: _load(results / f"metrics_{s}.json") for s in SERIES}
    worst = {s: _load(results / f"worst_hours_{s}.json") for s in SERIES}
    quality = _load(results / "data_quality.json")
    terms = _load(results / "data_terms.json")
    determinism = _load(results / "determinism.json")
    gru_check = _load(results / "gru_settings_check.json")
    acceptance = _load(results / "acceptance.json")
    failures = {s: summary["series"][s]["failures"] for s in SERIES}
    blocks: dict[str, str] = {}
    numbers: dict[str, str] = {}
    rendered = {
        "results": results_table(summary),
        "failures": failures_table(failures),
        "rmse": rmse_table(summary),
        "folds": fold_tables(metrics),
        "reading": reading_section(summary, metrics, worst, quality, terms, determinism, gru_check),
        "acceptance": acceptance_table(acceptance),
    }
    for key, (md, nums) in rendered.items():
        blocks[key] = md
        numbers.update(nums)
    return blocks, numbers


def replace_between(text: str, start: str, end: str, body: str) -> str:
    block = f"{start}\n{body}\n{end}"
    i = text.find(start)
    j = text.find(end, i + len(start)) if i >= 0 else -1
    if i >= 0 and j >= 0:
        return text[:i] + block + text[j + len(end):]
    return text.rstrip("\n") + "\n\n" + block + "\n"


def update_readme(blocks: dict[str, str], readme: Path = ROOT / "README.md") -> None:
    text = readme.read_text(encoding="utf-8")
    for key, body in blocks.items():
        start, end = MARKERS[key]
        text = replace_between(text, start, end, body)
    readme.write_text(text, encoding="utf-8", newline="\n")


def write_numbers(numbers: dict[str, str], path: Path) -> None:
    path.write_text(json.dumps(numbers, indent=2), encoding="utf-8", newline="\n")


def build_readme(results: Path = RESULTS, readme: Path = ROOT / "README.md") -> dict[str, str]:
    """Rewrite every README block from ``results`` and record the numbers the README must carry."""
    blocks, numbers = render_readme(results)
    update_readme(blocks, readme)
    write_numbers(numbers, results / "readme_numbers.json")
    return numbers
