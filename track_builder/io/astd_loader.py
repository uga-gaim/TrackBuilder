# io/astd_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, List, Any, Sequence
import pandas as pd

from track_builder.config import ASTD_USEFUL_COLS, ASTD_DTYPE_MAP
# Internal helper imports
from track_builder.core.io_helpers import (
    DEFAULT_DATA_PATH,
    iter_files,
    read_csv_auto,
    standardize_columns,
    parse_dates,
    quality_filter,
    matches_year_month,
    HAS_TQDM,
    sample_by_day_of_month,
    make_skipper,
)

Pathish = Union[str, Path]


def load_astd_data(
        file_paths: Union[Pathish, Iterable[Pathish], None],
        pattern: Optional[str] = None,
        usecols: Optional[Union[str, List[str]]] = None,
        remove_nan_rows: Optional[Union[str, List[str]]] = None,
        sampling: Optional[Union[float, Sequence[int]]] = None,
        infer_datetime_cols: bool = True,
        standardize_cols: bool = True,
        quality_threshold_minutes: int = 0,
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

    if usecols in ("default", "essential"):
        read_csv_kwargs.setdefault('usecols', ASTD_USEFUL_COLS)
        read_csv_kwargs.setdefault('dtype', ASTD_DTYPE_MAP)
    elif usecols is not None:
        read_csv_kwargs.setdefault('usecols', usecols)


    read_csv_kwargs.setdefault('low_memory', False)
    rs = read_csv_kwargs.pop("random_state", None)

    frames: List[pd.DataFrame] = []
    iterator = range(len(files))
    if progress and HAS_TQDM:
        from tqdm.auto import tqdm  # lazy import to avoid dependency if not installed
        iterator = tqdm(iterator, total=len(files), desc="Loading ASTD CSVs")

    for i in iterator:
        f = files[i]
        iter_kwargs = read_csv_kwargs.copy()

        if isinstance(sampling, float):
             # For fractional sampling, use skiprows with a callable (fast and memory friendly)
            skipper = make_skipper(sampling, rs)
            iter_kwargs['skiprows'] = skipper
            df = read_csv_auto(f, **iter_kwargs)
        else:
            # For all other cases (None or Sequence), read the entire file
            df = read_csv_auto(f, **iter_kwargs)

        if standardize_cols:
            df = standardize_columns(df)
        if infer_datetime_cols:
            df = parse_dates(df)

            # And if it's a sequence, we filter AFTER reading
        if isinstance(sampling, Sequence) and not isinstance(sampling, str) and len(sampling) > 0:
            df = sample_by_day_of_month(df, sampling)

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    if remove_nan_rows in ("default", "essential"):
        out = out.dropna(subset=ASTD_USEFUL_COLS)
    elif remove_nan_rows is not None:
        if isinstance(remove_nan_rows, str):
            remove_nan_rows = [remove_nan_rows]
        out = out.dropna(subset=remove_nan_rows)

    
    out = out.reset_index(drop=True)


    
    if quality_threshold_minutes and quality_threshold_minutes > 0:
        out = quality_filter(out, quality_threshold_minutes)

    if "date_time_utc" in out.columns:
        out = out.sort_values("date_time_utc").reset_index(drop=True)

    return out



def load_astd_monthly(
        base_path: Optional[Pathish],
        year: int,
        months: Optional[Iterable[int]] = None,
        progress: bool = True,
        **kwargs: Any,
) -> pd.DataFrame:
    """
    High-level wrapper: convenient loader for monthly ASTD datasets.

    This function recursively searches for CSV files in the base_path that match
    the specified year and months in their filenames.

    Examples:
      df = load_astd_monthly('/data/astd/', 2019, months=[7, 8, 9])
      df = load_astd_monthly('/data/astd/', 2019)  # Loads all months for the full year
    """
    base = Path(base_path).resolve() if base_path is not None else DEFAULT_DATA_PATH
    months_to_load = set(months) if months is not None else set(range(1, 13))

    all_csv_files = base.rglob("*.csv")

    selected_files = [
        p for p in all_csv_files if matches_year_month(p.name, year, months_to_load)
    ]

    if not selected_files:
        print(f"No files found for year {year} and months {list(months_to_load)} in {base}.")
        return pd.DataFrame()  # Return empty dataFrame

    return load_astd_data(selected_files, progress=progress, **kwargs)
