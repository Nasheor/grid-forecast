"""Features for one forecast origin, built only from values that are known at that origin.

Origins. Demand and wind are forecast from 00:00 Europe/Dublin on day D for the 24 hours from that moment.
The price of day D is forecast from 10:00 Europe/Dublin on day D-1, an hour before the SEM day-ahead auction
closes, for the 24 hours from 00:00 on day D.

What is known at an origin. A demand or wind hour is known once it has ended (its start time plus one hour is
at or before the origin). A day-ahead price for a delivery hour is known once SEMOpx has published it, which
is taken as 11:30 Europe/Dublin on the day before delivery, so at 10:00 on D-1 the prices of D-1 are known and
the prices of D are not. Every feature is computed from the masked series, so nothing after the origin can
leak in, and ``tests/test_leakage.py`` checks that.

Columns. ``horizon`` (hours from the origin to the end of the target hour, 1 to 24 for demand and wind,
15 to 38 for the price), target-hour calendar (``hour``, ``dow``,
``month``, ``holiday``, ``dst_day``), lags of the target series aligned to the target hour (24 to 168 hours,
``lag_24h`` is the same hour yesterday and ``lag_168h`` the same hour a week ago), rolling means over the
last 24 and 168 known hours, the last known value, and for the price the previous complete day's demand and
wind aligned to the target hour plus their last known values and 24-hour means. ``win_000`` to ``win_167``
hold the last 168 known values for the GRU. ``target`` holds the value to predict (NaN when the hour is
missing from the data) and ``ts_utc`` the target hour.
"""
from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import TZ

LAGS = (24, 48, 72, 96, 120, 144, 168)
WINDOW = 168
HORIZON = 24
PRICE_PUBLISH_LOCAL = dt.time(11, 30)

CALENDAR_COLUMNS = ["hour", "dow", "month", "holiday", "dst_day"]
BASE_COLUMNS = ["horizon", *CALENDAR_COLUMNS, *[f"lag_{k}h" for k in LAGS], "roll_24h", "roll_168h", "last_known"]
PRICE_EXTRA = ["demand_prev_day", "wind_prev_day", "demand_last_known", "wind_last_known", "demand_roll_24h", "wind_roll_24h"]
WINDOW_COLUMNS = [f"win_{i:03d}" for i in range(WINDOW)]
FEATURE_COLUMNS = {
    "demand": BASE_COLUMNS,
    "wind": BASE_COLUMNS,
    "price": BASE_COLUMNS + PRICE_EXTRA,
}


def feature_hash(series: str) -> str:
    return hashlib.sha256(",".join(FEATURE_COLUMNS[series] + WINDOW_COLUMNS).encode()).hexdigest()[:12]


def local_to_utc(day: dt.date, t: dt.time) -> pd.Timestamp:
    return pd.Timestamp.combine(day, t).tz_localize(TZ).tz_convert("UTC").tz_localize(None)


def origin_for_day(series: str, day: dt.date) -> pd.Timestamp:
    """UTC origin for the forecast of ``day``."""
    if series == "price":
        return local_to_utc(day - dt.timedelta(days=1), dt.time(10, 0))
    return local_to_utc(day, dt.time(0, 0))


def target_times(day: dt.date) -> pd.DatetimeIndex:
    start = local_to_utc(day, dt.time(0, 0))
    return pd.date_range(start, periods=HORIZON, freq="1h")


def known_mask(series: str, ts_utc: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """True where the value at ``ts_utc`` is known at ``origin``."""
    if series == "price":
        local = ts_utc.tz_localize("UTC").tz_convert(TZ)
        delivery_day = local.normalize().tz_localize(None)
        publish_local = (delivery_day - pd.Timedelta(days=1)) + pd.Timedelta(hours=PRICE_PUBLISH_LOCAL.hour, minutes=PRICE_PUBLISH_LOCAL.minute)
        publish_utc = publish_local.tz_localize(TZ, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC").tz_localize(None)
        return np.asarray(publish_utc <= origin)
    return np.asarray(ts_utc + pd.Timedelta(hours=1) <= origin)


class SeriesTable:
    """Hourly values on a complete UTC grid with O(1) lookup by timestamp."""

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("ts_utc").reset_index(drop=True)
        self.ts = pd.DatetimeIndex(df["ts_utc"])
        self.values = df["value"].to_numpy(dtype=float)
        self.start = self.ts[0]
        step = pd.Timedelta(hours=1)
        if len(self.ts) > 1 and not (self.ts[1:] - self.ts[:-1] == step).all():
            raise ValueError("prepared frame must be a complete hourly grid")
        self.n = len(self.ts)
        self.missing = df["missing"].to_numpy(dtype=bool) if "missing" in df else np.isnan(self.values)
        self.interpolated = df["interpolated"].to_numpy(dtype=bool) if "interpolated" in df else np.zeros(self.n, bool)
        self.holiday = df["holiday"].to_numpy(dtype=bool) if "holiday" in df else np.zeros(self.n, bool)
        self.dst_day = df["dst_day"].to_numpy(dtype=bool) if "dst_day" in df else np.zeros(self.n, bool)

    def index_of(self, ts: pd.Timestamp | pd.DatetimeIndex) -> np.ndarray:
        idx = ((pd.DatetimeIndex(np.atleast_1d(ts)) - self.start) / pd.Timedelta(hours=1)).astype(int)
        return np.asarray(idx)

    def masked(self, series: str, origin: pd.Timestamp) -> np.ndarray:
        """Values with everything unknown at ``origin`` set to NaN."""
        v = self.values.copy()
        v[~known_mask(series, self.ts, origin)] = np.nan
        return v

    def at(self, v: np.ndarray, ts: pd.DatetimeIndex) -> np.ndarray:
        idx = self.index_of(ts)
        out = np.full(len(idx), np.nan)
        ok = (idx >= 0) & (idx < self.n)
        out[ok] = v[idx[ok]]
        return out


def _last_known_index(v: np.ndarray) -> int:
    ok = np.flatnonzero(~np.isnan(v))
    return int(ok[-1]) if len(ok) else -1


def _rolling(v: np.ndarray, end: int, hours: int) -> float:
    if end < 0:
        return np.nan
    part = v[max(0, end - hours + 1): end + 1]
    return float(np.nanmean(part)) if np.isfinite(part).any() else np.nan


def _window(v: np.ndarray, end: int, hours: int) -> np.ndarray:
    out = np.full(hours, np.nan)
    if end < 0:
        return out
    part = v[max(0, end - hours + 1): end + 1]
    out[hours - len(part):] = part
    if np.isnan(out).any():
        fill = np.nanmean(out) if np.isfinite(out).any() else 0.0
        out = np.where(np.isnan(out), fill, out)
    return out


def build_frame(series: str, tables: dict[str, SeriesTable], day: dt.date) -> pd.DataFrame:
    """24 feature rows for the forecast of ``day`` from its origin."""
    origin = origin_for_day(series, day)
    tau = target_times(day)
    table = tables[series]
    v = table.masked(series, origin)
    end = _last_known_index(v)
    local = tau.tz_localize("UTC").tz_convert(TZ)
    out = pd.DataFrame({
        "ts_utc": tau,
        "origin_utc": origin,
        "day": [day] * HORIZON,
        "horizon": ((tau - origin) / pd.Timedelta(hours=1)).astype(int) + 1,
        "hour": local.hour.astype(int),
        "dow": local.day_of_week.astype(int),
        "month": local.month.astype(int),
    })
    idx = table.index_of(tau)
    inside = (idx >= 0) & (idx < table.n)
    hol = np.zeros(HORIZON, bool)
    dstd = np.zeros(HORIZON, bool)
    hol[inside] = table.holiday[idx[inside]]
    dstd[inside] = table.dst_day[idx[inside]]
    cols: dict[str, np.ndarray | float] = {"holiday": hol.astype(int), "dst_day": dstd.astype(int)}
    for k in LAGS:
        cols[f"lag_{k}h"] = table.at(v, tau - pd.Timedelta(hours=k))
    cols["roll_24h"] = _rolling(v, end, 24)
    cols["roll_168h"] = _rolling(v, end, 168)
    cols["last_known"] = v[end] if end >= 0 else np.nan
    if series == "price":
        for other in ("demand", "wind"):
            ot = tables[other]
            ov = ot.masked(other, origin)
            oend = _last_known_index(ov)
            cols[f"{other}_prev_day"] = ot.at(ov, tau - pd.Timedelta(hours=48))
            cols[f"{other}_last_known"] = ov[oend] if oend >= 0 else np.nan
            cols[f"{other}_roll_24h"] = _rolling(ov, oend, 24)
    win = _window(v, end, WINDOW)
    win_df = pd.DataFrame(np.repeat(win[None, :], HORIZON, axis=0), columns=WINDOW_COLUMNS)
    target = np.full(HORIZON, np.nan)
    tmiss = np.ones(HORIZON, bool)
    target[inside] = table.values[idx[inside]]
    tmiss[inside] = table.missing[idx[inside]]
    tint = np.zeros(HORIZON, bool)
    tint[inside] = table.interpolated[idx[inside]]
    tail = pd.DataFrame({"target": target, "target_missing": tmiss | np.isnan(target), "target_interpolated": tint})
    return pd.concat([out, pd.DataFrame(cols), win_df, tail], axis=1)


def build_dataset(series: str, tables: dict[str, SeriesTable], days: Iterable[dt.date]) -> pd.DataFrame:
    frames = [build_frame(series, tables, d) for d in days]
    return pd.concat(frames, ignore_index=True)


def days_between(start: dt.date, end: dt.date) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
