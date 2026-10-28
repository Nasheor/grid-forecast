"""Shared fixtures. The synthetic dataset stands in for the real prepared files so the unit tests need no
network and no full run: six months of hourly demand, wind and price with a daily and weekly shape, a
two-hour gap and a six-hour gap, written in the prepared layout."""
import datetime as dt
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def pkg_name() -> str:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["name"].replace("-", "_")


SYN_START = dt.date(2025, 6, 23)
SYN_END = dt.date(2025, 12, 31)


def make_synthetic(out_dir: Path, seed: int = 7) -> dict[str, pd.DataFrame]:
    from grid_forecast.prepare import add_calendar, fill_gaps, on_grid

    rng = np.random.default_rng(seed)
    lo = pd.Timestamp(SYN_START) - pd.Timedelta(hours=1)
    hi = pd.Timestamp(SYN_END) + pd.Timedelta(days=1)
    ts = pd.date_range(lo, hi, freq="1h")
    n = len(ts)
    hours = np.arange(n)
    local = ts.tz_localize("UTC").tz_convert("Europe/Dublin")
    h = local.hour.to_numpy()
    dow = local.day_of_week.to_numpy()
    daily = np.sin(2 * np.pi * (h - 6) / 24)
    weekly = np.where(dow >= 5, -250.0, 0.0)
    demand = 3800 + 600 * daily + weekly + rng.normal(0, 60, n) + 100 * np.sin(2 * np.pi * hours / (24 * 60))
    wind = np.empty(n)
    wind[0] = 1200
    for i in range(1, n):
        wind[i] = 0.97 * wind[i - 1] + rng.normal(40, 120)
    wind = np.clip(wind, 0, 3500)
    price = 80 + 0.03 * (demand - 3800) - 0.02 * wind + rng.normal(0, 8, n)
    spikes = rng.random(n) < 0.01
    price[spikes] += rng.uniform(80, 250, spikes.sum())
    dips = rng.random(n) < 0.005
    price[dips] -= rng.uniform(60, 120, dips.sum())
    frames = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("demand", demand), ("wind", wind), ("price", price)):
        v = values.copy()
        # a two-hour gap on 3 August and a six-hour gap on 24 August 2025, away from the days the tests use
        v[1000:1002] = np.nan
        v[1500:1506] = np.nan
        df = pd.DataFrame({"ts_utc": ts, "value": v})
        df = on_grid(df, lo, hi)
        df = fill_gaps(df)
        df = add_calendar(df)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        frames[name] = df
    return frames


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("prepared")
    make_synthetic(d)
    return d


@pytest.fixture(scope="session")
def tables(synthetic_dir):
    from grid_forecast import SERIES
    from grid_forecast.features import SeriesTable
    from grid_forecast.prepare import load_prepared

    return {s: SeriesTable(load_prepared(s, synthetic_dir)) for s in SERIES}
