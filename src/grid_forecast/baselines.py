"""The two baselines, read straight off the feature frame.

Seasonal naive: the value at the same hour yesterday (``lag_24h``). Last week: the value at the same hour
seven days ago (``lag_168h``). Both are known at every origin used here, because the demand and wind origin
is midnight and yesterday has ended, and the price of yesterday was published the day before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(frame: pd.DataFrame) -> np.ndarray:
    return frame["lag_24h"].to_numpy(dtype=float)


def last_week(frame: pd.DataFrame) -> np.ndarray:
    return frame["lag_168h"].to_numpy(dtype=float)


BASELINES = {"seasonal_naive": seasonal_naive, "last_week": last_week}
