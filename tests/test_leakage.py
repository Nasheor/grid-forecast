"""No feature may depend on a value that is unknown at the origin."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from grid_forecast import SERIES
from grid_forecast.features import FEATURE_COLUMNS, WINDOW_COLUMNS, SeriesTable, build_frame, known_mask, origin_for_day

DAYS = [dt.date(2025, 9, 15), dt.date(2025, 10, 26), dt.date(2025, 10, 27), dt.date(2025, 12, 25)]


def _corrupt(table: SeriesTable, series: str, origin, rng) -> SeriesTable:
    df = pd.DataFrame({"ts_utc": table.ts, "value": table.values, "missing": table.missing, "interpolated": table.interpolated,
                       "holiday": table.holiday, "dst_day": table.dst_day})
    unknown = ~known_mask(series, table.ts, origin)
    df.loc[unknown, "value"] = rng.uniform(-1e4, 1e4, unknown.sum())
    return SeriesTable(df)


@pytest.mark.parametrize("series", SERIES)
@pytest.mark.parametrize("day", DAYS)
def test_unknown_values_do_not_change_features(tables, series, day):
    rng = np.random.default_rng(1)
    origin = origin_for_day(series, day)
    before = build_frame(series, tables, day)
    corrupted = {s: _corrupt(t, s, origin, rng) for s, t in tables.items()}
    after = build_frame(series, corrupted, day)
    cols = FEATURE_COLUMNS[series] + WINDOW_COLUMNS
    pd.testing.assert_frame_equal(before[cols].reset_index(drop=True), after[cols].reset_index(drop=True))
    assert before[cols].notna().all().all(), "features must be fully known at the origin on a complete series"
    assert not np.allclose(before["target"], after["target"]), "the targets themselves were changed, so the corruption reached them"


@pytest.mark.parametrize("series", SERIES)
def test_shifting_the_target_by_one_hour_leaves_the_earlier_origin_unchanged(tables, series):
    day = dt.date(2025, 11, 10)
    origin = origin_for_day(series, day)
    before = build_frame(series, tables, day)
    table = tables[series]
    df = pd.DataFrame({"ts_utc": table.ts, "value": table.values, "missing": table.missing, "interpolated": table.interpolated,
                       "holiday": table.holiday, "dst_day": table.dst_day})
    unknown = ~known_mask(series, table.ts, origin)
    shifted = df["value"].shift(-1).to_numpy()
    df.loc[unknown, "value"] = shifted[unknown]
    new = dict(tables)
    new[series] = SeriesTable(df)
    after = build_frame(series, new, day)
    cols = FEATURE_COLUMNS[series] + WINDOW_COLUMNS
    pd.testing.assert_frame_equal(before[cols].reset_index(drop=True), after[cols].reset_index(drop=True))


def test_known_mask_rules(tables):
    t = tables["demand"]
    origin = origin_for_day("demand", dt.date(2025, 11, 10))
    m = known_mask("demand", t.ts, origin)
    last_known = t.ts[m][-1]
    assert last_known == origin - pd.Timedelta(hours=1)
    p = tables["price"]
    origin_p = origin_for_day("price", dt.date(2025, 11, 10))
    mp = known_mask("price", p.ts, origin_p)
    last_known_p = p.ts[mp][-1]
    # at 10:00 on 9 November the prices of 9 November are known (published on the 8th) and those of the 10th are not
    assert last_known_p == pd.Timestamp("2025-11-09 23:00")
    assert origin_p == pd.Timestamp("2025-11-09 10:00")


def test_lags_point_at_known_values(tables):
    day = dt.date(2025, 11, 10)
    frame = build_frame("demand", tables, day)
    t = tables["demand"]
    expect = t.at(t.values, pd.DatetimeIndex(frame["ts_utc"]) - pd.Timedelta(hours=24))
    np.testing.assert_allclose(frame["lag_24h"].to_numpy(), expect)
    assert (frame["horizon"].to_numpy() == np.arange(1, 25)).all()
