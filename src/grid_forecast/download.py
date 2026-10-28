"""Download the three source series into ``data/raw/`` and record what arrived.

``python -m grid_forecast download --start 2025-06-23 --end 2026-09-30`` fetches:

* demand and wind for the Republic of Ireland from the EirGrid Library's "System Data Qtr Hourly" workbooks
  (one workbook per calendar year, fifteen-minute SCADA means in MW, local clock plus a ``GMT Offset`` column),
* the SEM day-ahead hourly price from SEMOpx report EA-025 ("DAM 60Min Harmonised Reference Price", one XML
  file per delivery day, UTC intervals, EUR/MWh), found through the public reports index.

The Smart Grid Dashboard route the spec names first (``/api/chart/`` on smartgriddashboard.com) is kept as the
second route for demand and wind. On the build day it answered ``{"Rows": []}`` for every request and its older
``DashboardService.svc`` answered HTTP 503, so the Library workbooks are the first route.

Every entry refuses to run unless ``LICENCE_CHECK.md`` carries a row with the entry id whose "Training allowed"
cell starts with "Yes" (workspace rule 3). Each file's URL, sha256, size, row count, time span, gaps and download
timestamp go to ``results/data_terms.json`` next to the commit time of the licence file, so the order "licence
row first, bytes second" can be checked. Downloaded files are untrusted data: they are only ever parsed with
pandas or the XML parser from this module, never executed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

from . import ATTRIBUTION, DOWNLOAD_START, LICENCE_CHECK, PERIOD_END, RAW, RESULTS, ROOT, USER_AGENT

LIBRARY_PAGE = "https://www.eirgrid.ie/grid/system-and-renewable-data-reports"
LIBRARY_FILE_BASE = "https://www.eirgrid.ie/sites/default/files/publications/"
# Workbook names found on 2026-10-28. The version suffix changes when EirGrid revises a year, so the names are
# re-discovered from the page (current year) or by probing the suffix (earlier years) when these return 404.
LIBRARY_FILES = {2025: "System-Data-Qtr-Hourly-2025-V12.xlsx", 2026: "System-Data-Qtr-Hourly-2026-V9.xlsx"}
LIBRARY_COLUMNS = {"demand": "IE Demand", "wind": "IE Wind Generation"}

SEMOPX_INDEX = "https://reports.semopx.com/api/v1/documents/static-reports"
SEMOPX_DOCS = "https://reports.semopx.com/documents/"
SEMOPX_REPORT = "EA-025"

DASHBOARD_API = "https://www.smartgriddashboard.com/api/chart/"
DASHBOARD_AREAS = {"demand": ("demand", "demandactual"), "wind": ("wind", "windactual")}

ENTRY_IDS = {"library": "eirgrid_library_qtr_hourly", "demand": "sgd_demand", "wind": "sgd_wind", "price": "semopx_price"}
TERMS_URLS = {
    "eirgrid_library_qtr_hourly": "https://www.smartgriddashboard.com/all/open-data-license/",
    "sgd_demand": "https://www.smartgriddashboard.com/all/open-data-license/",
    "sgd_wind": "https://www.smartgriddashboard.com/all/open-data-license/",
    "semopx_price": "https://www.smartgriddashboard.com/all/open-data-license/",
}

# The SEMOpx document server sometimes hangs for a minute before answering 502, so requests time out early
# and are retried. Four workers fetch the daily price documents side by side.
TIMEOUT = 20.0
PAUSE_S = 0.15
WORKERS = 4


class LicenceRefused(RuntimeError):
    """LICENCE_CHECK.md has no row for the entry, or the row does not allow the use."""


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def licence_row(entry_id: str) -> dict[str, str]:
    """The LICENCE_CHECK.md table row that names ``entry_id``, as a dict keyed by the table header."""
    if not LICENCE_CHECK.exists():
        raise LicenceRefused("LICENCE_CHECK.md is missing")
    header: list[str] | None = None
    for line in LICENCE_CHECK.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "Item":
            header = cells
            continue
        if header and f"`{entry_id}`" in line:
            return dict(zip(header, cells, strict=False))
    raise LicenceRefused(f"no LICENCE_CHECK.md row carries the entry id `{entry_id}`")


def require_allowed(entry_id: str) -> dict[str, str]:
    row = licence_row(entry_id)
    if not row.get("Training allowed", "").lower().startswith("yes"):
        raise LicenceRefused(f"LICENCE_CHECK.md row `{entry_id}` does not allow the use: {row.get('Training allowed')!r}")
    return row


def licence_commit() -> dict[str, str]:
    """Hash and committer time of the last commit that touched LICENCE_CHECK.md (empty strings outside git)."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H %cI", "--", "LICENCE_CHECK.md"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"hash": "", "time": ""}
    if not out:
        return {"hash": "", "time": ""}
    h, _, t = out.partition(" ")
    return {"hash": h, "time": t}


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True)


def _get(client: httpx.Client, url: str, retries: int = 4, **kwargs) -> httpx.Response:
    """GET with a small retry on 429 and 5xx. Honours the licence's throttling clause by pausing between calls."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = client.get(url, **kwargs)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"{r.status_code} for {url}", request=r.request, response=r)
            r.raise_for_status()
            time.sleep(PAUSE_S)
            return r
        except (httpx.HTTPError, httpx.TransportError) as e:
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last}")


def _span_and_gaps(ts: pd.Series, step: pd.Timedelta) -> dict:
    ts = pd.DatetimeIndex(ts).sort_values()
    if len(ts) == 0:
        return {"from": None, "to": None, "rows": 0, "gaps": []}
    full = pd.date_range(ts[0], ts[-1], freq=step)
    missing = full.difference(ts)
    gaps: list[dict] = []
    if len(missing):
        starts = [missing[0]]
        ends = [missing[0]]
        for t in missing[1:]:
            if t - ends[-1] == step:
                ends[-1] = t
            else:
                starts.append(t)
                ends.append(t)
        gaps = [{"from": s.isoformat(), "to": e.isoformat(), "steps": int((e - s) / step) + 1} for s, e in zip(starts, ends, strict=True)]
    return {"from": ts[0].isoformat(), "to": ts[-1].isoformat(), "rows": int(len(ts)), "missing_steps": int(len(missing)), "gaps": gaps[:50]}


# ----------------------------------------------------------------------------------------------------- Library


def discover_library_file(client: httpx.Client, year: int) -> str:
    """Workbook file name for ``year``: the known name if it still exists, else the page link, else a probe."""
    known = LIBRARY_FILES.get(year)
    if known and client.head(LIBRARY_FILE_BASE + known).status_code == 200:
        return known
    page = _get(client, LIBRARY_PAGE).text
    found = sorted(set(re.findall(rf"System-Data-Qtr-Hourly-{year}-V(\d+)\.xlsx", page)), key=int)
    if found:
        return f"System-Data-Qtr-Hourly-{year}-V{found[-1]}.xlsx"
    best = None
    for v in range(1, 60):
        name = f"System-Data-Qtr-Hourly-{year}-V{v}.xlsx"
        if client.head(LIBRARY_FILE_BASE + name).status_code == 200:
            best = name
    if best is None and client.head(LIBRARY_FILE_BASE + f"System-Data-Qtr-Hourly-{year}.xlsx").status_code == 200:
        best = f"System-Data-Qtr-Hourly-{year}.xlsx"
    if best is None:
        raise RuntimeError(f"no System Data Qtr Hourly workbook found for {year}")
    return best


def read_library_workbook(path: Path) -> pd.DataFrame:
    """Fifteen-minute UTC frame with ``ts_utc``, ``demand`` and ``wind`` from one Library workbook."""
    cols = ["DateTime", "GMT Offset", *LIBRARY_COLUMNS.values()]
    df = pd.read_excel(path, sheet_name="System Data", usecols=cols, engine="openpyxl")
    df["ts_utc"] = pd.to_datetime(df["DateTime"]) - pd.to_timedelta(df["GMT Offset"].astype(int), unit="h")
    out = pd.DataFrame({"ts_utc": df["ts_utc"]})
    for series, col in LIBRARY_COLUMNS.items():
        out[series] = pd.to_numeric(df[col], errors="coerce")
    return out.sort_values("ts_utc").drop_duplicates("ts_utc").reset_index(drop=True)


def download_library(client: httpx.Client, start: dt.date, end: dt.date, terms: dict) -> pd.DataFrame:
    entry = ENTRY_IDS["library"]
    require_allowed(entry)
    folder = RAW / "eirgrid_library"
    folder.mkdir(parents=True, exist_ok=True)
    rec = terms["entries"].setdefault(entry, {"terms_url": TERMS_URLS[entry], "files": []})
    frames = []
    for year in range(start.year, end.year + 1):
        name = discover_library_file(client, year)
        path = folder / name
        url = LIBRARY_FILE_BASE + name
        if not path.exists():
            t0 = _now()
            r = _get(client, url)
            path.write_bytes(r.content)
            downloaded_at = t0
        else:
            downloaded_at = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(timespec="seconds")
        frame = read_library_workbook(path)
        frames.append(frame)
        rec["files"].append({
            "name": name, "url": url, "bytes": path.stat().st_size, "sha256": _sha256(path),
            "downloaded_at": downloaded_at, **_span_and_gaps(frame["ts_utc"], pd.Timedelta(minutes=15)),
        })
    df = pd.concat(frames, ignore_index=True).sort_values("ts_utc").drop_duplicates("ts_utc")
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    df = df[(df["ts_utc"] >= lo) & (df["ts_utc"] < hi)].reset_index(drop=True)
    for series in LIBRARY_COLUMNS:
        out = folder / f"{series}_{start.isoformat()}_{end.isoformat()}.csv"
        df[["ts_utc", series]].rename(columns={series: "value"}).to_csv(out, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        rec["files"].append({"name": out.name, "derived_from": "the workbooks above", "sha256": _sha256(out), **_span_and_gaps(df["ts_utc"], pd.Timedelta(minutes=15))})
    return df


# --------------------------------------------------------------------------------------------------- Dashboard


def download_dashboard(client: httpx.Client, series: str, start: dt.date, end: dt.date, terms: dict) -> pd.DataFrame | None:
    """Second route for demand and wind: the dashboard's own chart route, one month per request.

    Returns None when the route answers without rows (as it did on the build day), so the caller keeps the
    Library frame. The row shape expected is the dashboard's ``{"Rows": [{"EffectiveTime", "FieldName", "Value"}]}``.
    """
    entry = ENTRY_IDS[series]
    require_allowed(entry)
    chart, area = DASHBOARD_AREAS[series]
    rec = terms["entries"].setdefault(entry, {"terms_url": TERMS_URLS[entry], "files": [], "status": ""})
    rows: list[dict] = []
    month = dt.date(start.year, start.month, 1)
    while month <= end:
        nxt = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        last = min(nxt - dt.timedelta(days=1), end)
        params = {"region": "ROI", "chartType": chart, "dateRange": "month", "dateFrom": month.strftime("%d-%b-%Y"),
                  "dateTo": last.strftime("%d-%b-%Y"), "areas": area}
        try:
            r = _get(client, DASHBOARD_API, params=params, headers={"Eirgrid-Content-Request": "Nextjs"})
            got = r.json().get("Rows", [])
        except (RuntimeError, ValueError):
            got = []
        if not got:
            rec["status"] = f"route answered without rows on {_now()}, Library workbooks used instead"
            return None
        rows.extend(got)
        month = nxt
    df = pd.DataFrame(rows)
    df["ts_utc"] = pd.to_datetime(df["EffectiveTime"], format="%d-%b-%Y %H:%M:%S", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"Value": "value"})[["ts_utc", "value"]].sort_values("ts_utc")
    folder = RAW / "smartgriddashboard"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{series}_{start.isoformat()}_{end.isoformat()}.csv"
    df.to_csv(out, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    rec["status"] = "rows received"
    rec["files"].append({"name": out.name, "url": DASHBOARD_API, "sha256": _sha256(out), "downloaded_at": _now(), **_span_and_gaps(df["ts_utc"], pd.Timedelta(minutes=15))})
    return df


# ------------------------------------------------------------------------------------------------------ SEMOpx


def semopx_index(client: httpx.Client, start: dt.date, end: dt.date) -> list[dict]:
    """Every EA-025 document between ``start`` and ``end`` inclusive, from the public index (metadata only)."""
    items: list[dict] = []
    page = 1
    while True:
        params = [("DPuG_ID", SEMOPX_REPORT), ("Date", f">={start.isoformat()}"), ("Date", f"<={end.isoformat()}"),
                  ("page_size", "100"), ("page", str(page)), ("sort_by", "Date"), ("order_by", "ASC")]
        body = _get(client, SEMOPX_INDEX, params=params).json()
        items.extend(body.get("items", []))
        pages = int(body.get("pagination", {}).get("totalPages", 1) or 1)
        if page >= pages:
            break
        page += 1
    return items


def parse_ea025(path: Path) -> pd.DataFrame:
    """Hourly UTC rows (``ts_utc``, ``value``) from one DAM 60Min Harmonised Reference Price document."""
    root = ET.parse(path).getroot()
    ns = {"h": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda t: f"h:{t}") if ns else (lambda t: t)
    rows = []
    for period in root.iter(f"{{{ns['h']}}}Period" if ns else "Period"):
        interval = period.find(q("TimeInterval"), ns).get("v")
        res = period.find(q("Resolution"), ns).get("v")
        if res != "PT60M":
            raise ValueError(f"{path.name}: resolution {res}, expected PT60M")
        t0 = pd.Timestamp(interval.split("/")[0]).tz_convert("UTC").tz_localize(None)
        for iv in period.findall(q("Interval"), ns):
            hhmm = iv.find(q("StartTime"), ns).get("v")
            price = iv.find(q("MarketPriceRoundedAmount"), ns).get("v")
            hh, mm = (int(x) for x in hhmm.split(":"))
            ts = t0.normalize() + pd.Timedelta(hours=hh, minutes=mm)
            if ts < t0:
                ts += pd.Timedelta(days=1)
            rows.append({"ts_utc": ts, "value": float(price)})
    df = pd.DataFrame(rows)
    # Interval start times are given as clock hours inside the UTC document interval, so the ones after
    # midnight belong to the next UTC day: order them by their offset from the interval start.
    df = df.sort_values("ts_utc").drop_duplicates("ts_utc").reset_index(drop=True)
    return df


def download_price(client: httpx.Client, start: dt.date, end: dt.date, terms: dict) -> pd.DataFrame:
    entry = ENTRY_IDS["price"]
    require_allowed(entry)
    folder = RAW / "semopx" / SEMOPX_REPORT
    folder.mkdir(parents=True, exist_ok=True)
    rec = terms["entries"].setdefault(entry, {"terms_url": TERMS_URLS[entry], "index_url": SEMOPX_INDEX, "files": []})
    items = semopx_index(client, start, end)
    by_day: dict[str, dict] = {}
    for it in items:
        by_day.setdefault(it["Date"][:10], it)
    expected = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d").tolist()
    missing_days = [d for d in expected if d not in by_day]
    names = [by_day[day]["ResourceName"] for day in expected if day in by_day]
    to_fetch = [n for n in names if not (folder / n).exists()]
    first_download = _now() if to_fetch else None
    n_new = 0

    def fetch(name: str) -> str:
        with _client() as c:
            r = _get(c, SEMOPX_DOCS + name, retries=6)
        (folder / name).write_bytes(r.content)
        return name

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done in pool.map(fetch, to_fetch):
            n_new += 1
            if n_new % 50 == 0:
                print(f"semopx: {n_new} of {len(to_fetch)} documents fetched ({done})", flush=True)
    frames = [parse_ea025(folder / n) for n in names]
    df = pd.concat(frames, ignore_index=True).sort_values("ts_utc").drop_duplicates("ts_utc").reset_index(drop=True)
    out = RAW / "semopx" / f"price_{start.isoformat()}_{end.isoformat()}.csv"
    df.to_csv(out, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    rec["documents"] = {"report": SEMOPX_REPORT, "report_name": "DAM 60Min Harmonised Reference Price", "days_requested": len(expected),
                        "days_found": len(expected) - len(missing_days), "missing_days": missing_days, "downloaded_now": n_new,
                        "first_download_at": first_download, "folder": str(folder.relative_to(ROOT)),
                        "sha256_of_documents": {p.name: _sha256(p) for p in sorted(folder.glob("*.xml"))}}
    rec["files"].append({"name": out.name, "sha256": _sha256(out), "downloaded_at": first_download or _now(), **_span_and_gaps(df["ts_utc"], pd.Timedelta(hours=1))})
    return df


# ---------------------------------------------------------------------------------------------------- Entry


def download(start: dt.date = DOWNLOAD_START, end: dt.date = PERIOD_END, try_dashboard: bool = True) -> dict:
    """Fetch every series for the range and write ``results/data_terms.json``. Returns the terms record."""
    RESULTS.mkdir(exist_ok=True)
    terms: dict = {
        "attribution": ATTRIBUTION,
        "licence_check_commit": licence_commit(),
        "requested": {"start": start.isoformat(), "end": end.isoformat()},
        "evaluated_period": {"start": "2025-10-01", "end": "2026-09-30"},
        "run_started_at": _now(),
        "entries": {},
    }
    with _client() as client:
        lib = download_library(client, start, end, terms)
        if try_dashboard:
            for series in ("demand", "wind"):
                download_dashboard(client, series, start, end, terms)
        download_price(client, start, end, terms)
    terms["run_finished_at"] = _now()
    terms["library_rows"] = int(len(lib))
    (RESULTS / "data_terms.json").write_text(json.dumps(terms, indent=2), encoding="utf-8")
    return terms
