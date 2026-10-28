"""The hours where the models fail.

For the champion of each series, the 5 % of scored test hours with the largest absolute error are listed,
each tagged with every reason that applies: a public holiday, a clock-change day, a wind ramp above the 95th
percentile of hourly wind change, a price above the 95th percentile, a negative price, an interpolated input
(the target hour or the same hour yesterday was interpolated), the first day after a long gap, and the part
of the day (night, morning ramp, midday, evening peak, late evening), so every row carries at least one tag.
The share of failure hours per tag is reported next to the share of all scored hours per tag, so an enriched
tag stands out. ``docs/failures_<series>.png`` plots the champion's error by hour of day and by month next to
the seasonal naive.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import DOCS, RESULTS, UNITS
from .features import SeriesTable

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FAILURE_SHARE = 0.05
PCT = 95


def part_of_day(hour: int) -> str:
    if hour <= 5:
        return "night"
    if hour <= 9:
        return "morning_ramp"
    if hour <= 16:
        return "midday"
    if hour <= 20:
        return "evening_peak"
    return "late_evening"


def _lookup(table: SeriesTable, ts: pd.DatetimeIndex, attr: str = "values") -> np.ndarray:
    v = getattr(table, attr)
    idx = table.index_of(ts)
    out = np.full(len(idx), np.nan if v.dtype.kind == "f" else False)
    ok = (idx >= 0) & (idx < table.n)
    out[ok] = v[idx[ok]]
    return out


def tag_rows(preds: pd.DataFrame, series: str, tables: dict[str, SeriesTable], champion: str) -> pd.DataFrame:
    """Scored prediction rows with the champion's absolute error and one boolean column per tag."""
    rows = preds[~preds["target_missing"]].copy()
    ts = pd.DatetimeIndex(rows["ts_utc"])
    rows["abs_error"] = (rows[champion] - rows["target"]).abs()
    rows["error"] = rows[champion] - rows["target"]
    rows["tag_holiday"] = rows["holiday"].astype(bool)
    rows["tag_dst_day"] = rows["dst_day"].astype(bool)
    wind = tables["wind"]
    dwind = np.abs(np.diff(wind.values, prepend=np.nan))
    ramp_threshold = float(np.nanpercentile(dwind, PCT))
    ramp_at = _lookup(wind, ts) - _lookup(wind, ts - pd.Timedelta(hours=1))
    rows["tag_wind_ramp"] = np.abs(ramp_at) > ramp_threshold
    price = tables["price"]
    spike_threshold = float(np.nanpercentile(price.values, PCT))
    p_at = _lookup(price, ts)
    rows["tag_price_spike"] = p_at > spike_threshold
    rows["tag_negative_price"] = p_at < 0
    target = tables[series]
    interp_now = rows["target_interpolated"].to_numpy(dtype=bool)
    interp_yday = _lookup(target, ts - pd.Timedelta(hours=24), "interpolated").astype(bool)
    rows["tag_interpolated_input"] = interp_now | interp_yday
    gap_days = {d for d, m in zip(target.ts.tz_localize("UTC").tz_convert("Europe/Dublin").date, target.missing, strict=True) if m}
    rows["tag_first_day_after_gap"] = rows["day"].map(lambda d: (d - dt.timedelta(days=1)) in gap_days).astype(bool)
    for name in ("night", "morning_ramp", "midday", "evening_peak", "late_evening"):
        rows[f"tag_{name}"] = rows["hour"].map(part_of_day) == name
    rows.attrs["thresholds"] = {"wind_ramp_mw_per_hour": ramp_threshold, "price_spike_eur_per_mwh": spike_threshold}
    return rows


def failure_table(rows: pd.DataFrame, share: float = FAILURE_SHARE) -> tuple[pd.DataFrame, dict]:
    n_fail = max(1, int(np.ceil(len(rows) * share)))
    worst = rows.sort_values("abs_error", ascending=False).head(n_fail).copy()
    tags = [c for c in rows.columns if c.startswith("tag_")]
    worst["tags"] = [",".join(t[4:] for t in tags if r[t]) for _, r in worst.iterrows()]
    shares = {}
    for t in tags:
        shares[t[4:]] = {"share_of_failure_hours": float(worst[t].mean()), "share_of_all_scored_hours": float(rows[t].mean()),
                         "failure_hours": int(worst[t].sum())}
    summary = {
        "failure_hours": int(n_fail), "scored_hours": int(len(rows)), "share": share,
        "error_threshold": float(worst["abs_error"].min()), "mean_abs_error_of_failures": float(worst["abs_error"].mean()),
        "by_tag": shares, "thresholds": rows.attrs.get("thresholds", {}),
        "by_month": {str(k): int(v) for k, v in worst.groupby(worst["day"].map(lambda d: f"{d.year:04d}-{d.month:02d}")).size().items()},
        "by_hour": {str(int(k)): int(v) for k, v in worst.groupby("hour").size().items()},
    }
    return worst, summary


def write_failures(series: str, preds: pd.DataFrame, tables: dict[str, SeriesTable], champion: str, results: Path = RESULTS, docs: Path = DOCS) -> dict:
    rows = tag_rows(preds, series, tables, champion)
    worst, summary = failure_table(rows)
    keep = ["fold", "ts_utc", "day", "hour", "horizon", "target", champion, "seasonal_naive", "error", "abs_error", "tags"]
    out = worst[keep].copy()
    out["ts_utc"] = pd.DatetimeIndex(out["ts_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["day"] = out["day"].astype(str)
    rec = {"series": series, "unit": UNITS[series], "champion": champion, **summary, "rows": json.loads(out.round(3).to_json(orient="records"))}
    results.mkdir(exist_ok=True)
    (results / f"worst_hours_{series}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    plot_failures(series, rows, champion, docs)
    return rec


def plot_failures(series: str, rows: pd.DataFrame, champion: str, docs: Path = DOCS) -> Path:
    docs.mkdir(exist_ok=True)
    by_hour = rows.groupby("hour").apply(lambda g: pd.Series({"champion": g["abs_error"].mean(), "naive": (g["seasonal_naive"] - g["target"]).abs().mean()}), include_groups=False)
    month = rows["day"].map(lambda d: f"{d.year:04d}-{d.month:02d}")
    by_month = rows.groupby(month).apply(lambda g: pd.Series({"champion": g["abs_error"].mean(), "naive": (g["seasonal_naive"] - g["target"]).abs().mean()}), include_groups=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(by_hour.index, by_hour["naive"], color="#9aa0a6", marker="o", label="seasonal naive")
    axes[0].plot(by_hour.index, by_hour["champion"], color="#1f5fbf", marker="o", label=champion)
    axes[0].set_xlabel("hour of day (Europe/Dublin)")
    axes[0].set_ylabel(f"MAE ({UNITS[series]})")
    axes[0].set_title(f"{series}: error by hour of day")
    axes[0].legend(frameon=False)
    x = np.arange(len(by_month))
    axes[1].bar(x - 0.2, by_month["naive"], width=0.4, color="#9aa0a6", label="seasonal naive")
    axes[1].bar(x + 0.2, by_month["champion"], width=0.4, color="#1f5fbf", label=champion)
    axes[1].set_xticks(x, by_month.index, rotation=45, ha="right")
    axes[1].set_ylabel(f"MAE ({UNITS[series]})")
    axes[1].set_title(f"{series}: error by test month")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = docs / f"failures_{series}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
