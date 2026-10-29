"""Clean each raw series to an hourly UTC frame in ``data/prepared/<series>.parquet`` (with a CSV copy).

Rules: demand and wind are fifteen-minute means and become hourly means (an hour with fewer than four points
is a partial hour, an hour with none is missing). The price is hourly already. Daylight saving is handled by
working in UTC throughout. Gaps of up to three hours are interpolated linearly in time and flagged, longer
gaps stay NaN and are excluded from scoring. Each hour carries its Europe/Dublin calendar day, hour, a public
holiday flag from the ``holidays`` package and a flag for the two clock-change days.

``results/data_quality.json`` records, per series and local month, the expected hours, the hours present,
the missing share, the interpolated hours and the long gaps. A month whose missing share is 5 % or more is
dropped from scoring and listed.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from . import DOWNLOAD_START, PERIOD_END, PREPARED, RAW, RESULTS, SERIES, SERIES_START, TZ

MAX_INTERPOLATED_HOURS = 3
MAX_MISSING_SHARE = 0.05


def _raw_path(series: str, raw: Path = RAW) -> Path:
    """Newest raw CSV for the series: the Library export first, then the dashboard export, then SEMOpx."""
    folders = {"demand": ["eirgrid_library", "smartgriddashboard"], "wind": ["eirgrid_library", "smartgriddashboard"], "price": ["semopx"]}
    for folder in folders[series]:
        hits = sorted((raw / folder).glob(f"{series}_*.csv"), key=lambda p: p.stat().st_mtime)
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"no raw file for {series} under {raw}. Run `python -m grid_forecast download` first.")


def read_raw(series: str, raw: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(_raw_path(series, raw))
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.sort_values("ts_utc").drop_duplicates("ts_utc").reset_index(drop=True)


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly mean of a sub-hourly frame on a complete hourly grid, with the number of points per hour."""
    s = df.set_index("ts_utc")["value"]
    step = pd.Series(s.index).diff().dropna().mode()
    per_hour = int(round(pd.Timedelta(hours=1) / step.iloc[0])) if len(step) else 1
    hourly = s.resample("1h").agg(["mean", "count"])
    hourly.columns = ["value", "n_points"]
    hourly["partial"] = (hourly["n_points"] > 0) & (hourly["n_points"] < per_hour)
    hourly.loc[hourly["n_points"] == 0, "value"] = np.nan
    return hourly.reset_index()


def on_grid(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    grid = pd.DataFrame({"ts_utc": pd.date_range(start, end, freq="1h")})
    out = grid.merge(df, on="ts_utc", how="left")
    if "n_points" not in out:
        out["n_points"] = out["value"].notna().astype(int)
        out["partial"] = False
    out["n_points"] = out["n_points"].fillna(0).astype(int)
    out["partial"] = out["partial"].fillna(False).astype(bool)
    return out


def fill_gaps(df: pd.DataFrame, max_hours: int = MAX_INTERPOLATED_HOURS) -> pd.DataFrame:
    """Interpolate runs of NaN up to ``max_hours`` long and flag them, leave longer runs as NaN and flag those."""
    v = df["value"].to_numpy(dtype=float)
    isna = np.isnan(v)
    interpolated = np.zeros(len(v), dtype=bool)
    missing = np.zeros(len(v), dtype=bool)
    i = 0
    n = len(v)
    while i < n:
        if not isna[i]:
            i += 1
            continue
        j = i
        while j < n and isna[j]:
            j += 1
        run = j - i
        inside = i > 0 and j < n
        if run <= max_hours and inside:
            interpolated[i:j] = True
        else:
            missing[i:j] = True
        i = j
    filled = pd.Series(v).interpolate(method="linear", limit_area="inside").to_numpy()
    v2 = v.copy()
    v2[interpolated] = filled[interpolated]
    out = df.copy()
    out["value"] = v2
    out["interpolated"] = interpolated
    out["missing"] = missing
    return out


def dst_days(years: list[int]) -> set[dt.date]:
    out: set[dt.date] = set()
    for y in years:
        for d in pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D"):
            a = pd.Timestamp(d).tz_localize(TZ).utcoffset()
            b = (pd.Timestamp(d) + pd.Timedelta(hours=23)).tz_localize(TZ).utcoffset()
            if a != b:
                out.add(d.date())
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    local = df["ts_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ)
    out = df.copy()
    out["local_day"] = local.dt.date
    out["local_hour"] = local.dt.hour.astype(int)
    out["dow"] = local.dt.day_of_week.astype(int)
    out["month"] = local.dt.month.astype(int)
    years = sorted({d.year for d in out["local_day"]})
    ie = holidays.country_holidays("IE", years=years)
    out["holiday"] = out["local_day"].map(lambda d: d in ie).astype(bool)
    dst = dst_days(years)
    out["dst_day"] = out["local_day"].map(lambda d: d in dst).astype(bool)
    return out


def quality(df: pd.DataFrame, series: str, start: dt.date | None = None, end: dt.date | None = None) -> dict:
    """Per local month inside ``start`` to ``end``: expected hours, hours present, missing share, interpolated hours, long gaps."""
    inside = df
    if start is not None:
        start = max(start, SERIES_START.get(series, start))
        inside = inside[inside["local_day"] >= start]
    if end is not None:
        inside = inside[inside["local_day"] <= end]
    g = inside.assign(ym=inside["local_day"].map(lambda d: f"{d.year:04d}-{d.month:02d}")).groupby("ym")
    months = {}
    for ym, part in g:
        expected = int(len(part))
        present = int(part["value"].notna().sum())
        missing_share = 1.0 - present / expected if expected else 1.0
        long_gaps = int(part["missing"].sum())
        months[ym] = {
            "expected_hours": expected,
            "hours_present": present,
            "missing_share": round(missing_share, 5),
            "interpolated_hours": int(part["interpolated"].sum()),
            "long_gap_hours": long_gaps,
            "partial_hours": int(part["partial"].sum()),
            "kept": bool(missing_share < MAX_MISSING_SHARE),
        }
    return {"series": series, "months": months, "dropped_months": [m for m, v in months.items() if not v["kept"]],
            "rule": f"a month is dropped when its missing share is {MAX_MISSING_SHARE:.0%} or more"}


def prepare_series(series: str, raw: Path = RAW, out_dir: Path = PREPARED, start: dt.date = DOWNLOAD_START, end: dt.date = PERIOD_END) -> tuple[pd.DataFrame, dict]:
    df = read_raw(series, raw)
    if series in ("demand", "wind"):
        df = to_hourly(df)
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    # The local day starts an hour before the UTC day in summer, so keep one extra UTC hour on each side.
    df = on_grid(df, lo - pd.Timedelta(hours=1), hi + pd.Timedelta(hours=1))
    df = fill_gaps(df)
    df = add_calendar(df)
    q = quality(df, series, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{series}.parquet", index=False)
    csv = df.copy()
    csv["ts_utc"] = csv["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    csv["value"] = csv["value"].round(3)
    csv.to_csv(out_dir / f"{series}.csv", index=False)
    return df, q


def load_prepared(series: str, prepared: Path = PREPARED) -> pd.DataFrame:
    """Prepared hourly frame, from Parquet if present, else from the committed CSV copy."""
    pq = prepared / f"{series}.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(prepared / f"{series}.csv")
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True).dt.tz_localize(None)
        df["local_day"] = pd.to_datetime(df["local_day"]).dt.date
    df["local_day"] = [d if isinstance(d, dt.date) else pd.Timestamp(d).date() for d in df["local_day"]]
    for c in ("partial", "interpolated", "missing", "holiday", "dst_day"):
        df[c] = df[c].astype(bool)
    return df.reset_index(drop=True)


def prepare(raw: Path = RAW, out_dir: Path = PREPARED, results: Path = RESULTS, start: dt.date = DOWNLOAD_START, end: dt.date = PERIOD_END) -> dict:
    results.mkdir(exist_ok=True)
    report = {"generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "series": {}}
    for series in SERIES:
        _, q = prepare_series(series, raw, out_dir, start, end)
        report["series"][series] = q
    (results / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "LICENCE").write_text(
        "Derived from EirGrid Group data (EirGrid Library System Data Qtr Hourly workbooks and SEMOpx report EA-025)\n"
        "under the EirGrid Group Open Data Licence, https://www.smartgriddashboard.com/all/open-data-license/ .\n"
        "Supported by EirGrid Group Data.\n"
        "Hourly means in UTC, cleaned as described in README.md. The code in this repository is MIT licensed.\n",
        encoding="utf-8",
    )
    return report
