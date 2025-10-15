# track_builder/core/vis_helpers.py
from __future__ import annotations
from typing import Dict, Optional
import warnings
import pandas as pd

from track_builder.core.io_helpers import standardize_columns
from track_builder.config import DATE_CANDS

TRACK_ID_CANDS = ("track_id",)
SEGMENT_ID_CANDS = ("segment_id", "shipid")
MONTH_CANDS = ("month",)


def first_present(df: pd.DataFrame, cands: tuple[str, ...]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def resolve_geo_time_cols(df: pd.DataFrame) -> Dict[str, str]:
    """
    Uses standardization Part 1 to obtain canonical columns.
    - Does not redo I/O validations.
    - Looks for 'date_time_utc' first, then any other date from DATE_CANDS if needed.
    """
    std = standardize_columns(df.copy())

    lat = "latitude" if "latitude" in std.columns else None
    lon = "longitude" if "longitude" in std.columns else None

    # Time column: prefer 'date_time_utc' if present
    if "date_time_utc" in std.columns:
        tcol = "date_time_utc"
    else:
        tcol = next((c for c in DATE_CANDS if c in std.columns), None)

    if not all([lat, lon, tcol]):
        raise KeyError(
            "Impossible to detect latitude/longitude/time after standardize_columns(). "
            f"Present: {list(std.columns)} ; expected: latitude/longitude/({', '.join(('date_time_utc',) + DATE_CANDS)})"
        )

    # Light conversion to datetime if necessary (without re-validating)
    if not pd.api.types.is_datetime64_any_dtype(std[tcol]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            std[tcol] = pd.to_datetime(std[tcol], errors="coerce")

    return {"lat": lat, "lon": lon, "time": tcol, "_df": std}
