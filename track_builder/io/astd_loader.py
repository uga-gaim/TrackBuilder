# io/astd_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, List, Any
import pandas as pd

from track_builder.config import ASTD_USEFUL_COLS, ASTD_DTYPE_MAP
# Internal helper imports
from track_builder.core.io_helpers import (
    DEFAULT_DATA_PATH,
    iter_files,
    read_csv_auto,
    standardize_columns,
    parse_dates,
    validate_coords,
    normalize_strings,
    add_month,
    quality_filter,
    matches_year_month,
    HAS_TQDM,
)

Pathish = Union[str, Path]


def load_astd_data(
    file_paths: Union[Pathish, Iterable[Pathish], None],
    pattern: Optional[str] = None,
    use_optimized_config: bool = True, # parameter to control optimization
    infer_datetime_cols: bool = True,
    standardize_cols: bool = True,
    validate_coordinates: bool = True,
    drop_low_quality: bool = False, 
    low_quality_threshold_minutes: int = 0,
    progress: bool = True,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    High-level wrapper: simplified loading & preprocessing of ASTD datasets.

    Examples (as in project_plan.md):
      df = load_astd_data('ASTD_area_level3_201907.csv')
      df = load_astd_data(['ASTD_area_level3_201907.csv', 'ASTD_area_level3_201908.csv'])
      df = load_astd_data('/path/to/astd/2019/', pattern='ASTD_*.csv')
    """
    files = iter_files(file_paths, pattern)

    # If the option is enabled, we apply our optimized configuration
    if use_optimized_config:
        read_csv_kwargs.setdefault('usecols', ASTD_USEFUL_COLS)
        read_csv_kwargs.setdefault('dtype', ASTD_DTYPE_MAP)

    read_csv_kwargs.setdefault('low_memory', False)
    
    frames: List[pd.DataFrame] = []
    iterator = range(len(files))
    if progress and HAS_TQDM:
        from tqdm import tqdm  # lazy import to avoid dependency if not installed
        iterator = tqdm(iterator, total=len(files), desc="Loading ASTD CSVs")

    for i in iterator:
        f = files[i]
        df = read_csv_auto(f, **read_csv_kwargs)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    if standardize_cols:
        out = standardize_columns(out)
    if infer_datetime_cols:
        out = parse_dates(out)
    if validate_coordinates:
        out = validate_coords(out)

    out = normalize_strings(out)
    out = add_month(out)

    # Ensure presence of key columns (even empty)
    for c in ("shipid", "date_time_utc", "latitude", "longitude",
              "astd_cat", "flagname", "iceclass", "sizegroup_gt", "month"):
        if c not in out.columns:
            out[c] = pd.Series(dtype="object")

    if drop_low_quality:
        out = quality_filter(out, low_quality_threshold_minutes)

    if "date_time_utc" in out.columns:
        out = out.sort_values("date_time_utc").reset_index(drop=True)

    return out


def load_astd_monthly(
    base_path: Optional[Pathish],
    year: int,
    months: Optional[Iterable[int]] = None,
    pattern: Optional[str] = None,
    progress: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    High-level wrapper: convenient loader for monthly ASTD datasets.

    Examples (as in project_plan.md):
      df = load_astd_monthly('/data/astd/', 2019, months=[7, 8, 9])
      df = load_astd_monthly('/data/astd/', 2019)  # full year
    """
    base = Path(base_path).resolve() if base_path is not None else DEFAULT_DATA_PATH
    months = list(months) if months is not None else list(range(1, 13))

    if pattern:
        df = load_astd_data(base, pattern=pattern, progress=progress, **kwargs)
        if "month" in df.columns:
            keep = {f"{year}-{m:02d}" for m in months}
            df = df[df["month"].isin(keep)]
        return df

    candidates = list(base.glob("*.csv"))
    selected = [p for p in candidates if matches_year_month(p.name, year, set(months))]
    if not selected:
        # fallback: broad pattern for the given year
        return load_astd_data(base, pattern=f"*{year}*csv", progress=progress, **kwargs)

    return load_astd_data(selected, progress=progress, **kwargs)
