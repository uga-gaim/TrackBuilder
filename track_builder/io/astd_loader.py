# io/astd_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, List, Any, Sequence
import pandas as pd
import numpy as np

from track_builder.core.track_helpers import remove_unrealistic_points

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


def load_track_data(
        track_ids: Union[Union[str, int], Sequence[Union[str, int]]],
        track_table: pd.DataFrame,
        *,
        base_path: Optional[Pathish] = None,
        progress: bool = True,
        chunksize: int = 50_000,
        use_preprocessing: bool = True,
) -> pd.DataFrame:
    """
    Load ASTD positions for one or MULTIPLE tracks simultaneously (Batch I/O).

    Optimization:
      Instead of opening the same monthly CSV files multiple times (once per track),
      this function identifies all files needed for the batch of tracks, reads them once,
      and extracts all relevant segments in a single pass.

    Args:
        track_ids: A single ID or a list of IDs to load.
        track_table: The table mapping tracks to segments and months.
        base_path: Path to raw CSVs.
        progress: Show tqdm progress bar.
        chunksize: Pandas read_csv chunksize.
        use_preprocessing: Apply cleaning/filtering steps.

    Returns:
        DataFrame with positions and an added 'track_id' column.
    """

    if isinstance(track_ids, (str, int, np.integer)):
        track_ids = [track_ids]

    track_ids = list(track_ids)
    if not track_ids:
        return pd.DataFrame()

    if "track_id" not in track_table.columns:
        raise KeyError("track_table must contain a 'track_id' column.")
    if "month" not in track_table.columns:
        raise KeyError("track_table must contain a 'month' column (e.g. '2019-07').")

    seg_col = "segment_id" if "segment_id" in track_table.columns else "shipid"
    if seg_col not in track_table.columns:
        raise KeyError("track_table must contain a 'segment_id' or 'shipid' column.")


    requested_ids_str = set(str(x) for x in track_ids)
    tt_subset = track_table[track_table["track_id"].astype(str).isin(requested_ids_str)].copy()

    if tt_subset.empty:
        print(f"Warning: None of the requested track_ids found in track_table.")
        return pd.DataFrame()


    tt_subset["month_dt"] = pd.to_datetime(tt_subset["month"], errors="coerce")


    valid_combinations = set()
    needed_months_by_year: dict[int, set[int]] = {}

    for _, row in tt_subset.iterrows():
        ts = row["month_dt"]
        if pd.isna(ts):
            continue
        sid = str(row[seg_col])
        valid_combinations.add((ts.year, ts.month, sid))
        needed_months_by_year.setdefault(ts.year, set()).add(ts.month)

    if not needed_months_by_year:
        return pd.DataFrame()


    root = Path(base_path).resolve() if base_path is not None else DEFAULT_DATA_PATH
    all_files = iter_files(root, pattern=None)

    selected_files = []
    for p in all_files:
        fname = p.name

        for year, months in needed_months_by_year.items():
            if matches_year_month(fname, year, months):
                selected_files.append(p)
                break

    selected_files = sorted({p.resolve() for p in selected_files})

    if not selected_files:
        return pd.DataFrame()


    frames: list[pd.DataFrame] = []

    iterator = selected_files
    if progress and HAS_TQDM:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, total=len(selected_files),
                        desc=f"Batch loading {len(track_ids)} tracks")

    # Optimization: Pre-compute the set of ALL shipids needed across all months
    # to perform a quick first-pass filter on chunks
    all_needed_shipids = {t[2] for t in valid_combinations}

    for path in iterator:

        reader = pd.read_csv(
            path,
            sep=";",  # or None to use python sniffing, but fixed sep is faster
            usecols=ASTD_USEFUL_COLS,
            dtype=ASTD_DTYPE_MAP,
            low_memory=False,
            chunksize=chunksize,
        )

        for chunk in reader:
            if chunk is None or chunk.empty:
                continue

            chunk = standardize_columns(chunk)

            if "shipid" not in chunk.columns:
                continue

            mask = chunk["shipid"].astype(str).isin(all_needed_shipids)
            sub = chunk.loc[mask].copy()

            if sub.empty:
                continue

            sub = parse_dates(sub)

            frames.append(sub)

    if not frames:
        return pd.DataFrame()

    raw_data = pd.concat(frames, ignore_index=True)



    raw_data["month_str"] = raw_data["date_time_utc"].dt.strftime("%Y-%m")

    tt_subset[seg_col] = tt_subset[seg_col].astype(str)

    tt_subset["month_join"] = tt_subset["month_dt"].dt.strftime("%Y-%m")

    raw_data["shipid_str"] = raw_data["shipid"].astype(str)

    merged = raw_data.merge(
        tt_subset[["track_id", seg_col, "month_join"]],
        left_on=["month_str", "shipid_str"],
        right_on=["month_join", seg_col],
        how="inner"  # Inner join keeps only the positions that are valid for the requested tracks/months
    )

    cols_to_drop = ["month_str", "month_join", "shipid_str"]
    if seg_col != "shipid":
        cols_to_drop.append(seg_col)

    merged = merged.drop(columns=cols_to_drop, errors="ignore")

    if "date_time_utc" in merged.columns:
        merged = merged.sort_values(["track_id", "date_time_utc"]).reset_index(drop=True)

    if use_preprocessing:
        merged = remove_unrealistic_points(merged)

    return merged


def build_light_multi_track_data(
        track_table: pd.DataFrame,
        track_sampling: Optional[Union[int, Sequence[int]]] = None,
        *,
        specific_track_ids: Optional[Sequence[Union[int, str]]] = None,
        positions_df: Optional[pd.DataFrame] = None,
        n_tracks_length: Optional[int] = None,
        base_path: Optional[Pathish] = None,
        chunksize: int = 50_000,
        progress: bool = True,
        point_stride: int = 10,
        random_state: Optional[int] = 42,
        preprocess_positions: bool = True,
) -> pd.DataFrame:
    """
    Build a 'light' DataFrame with positions for multiple tracks,
    ready to be passed to plot_ship_tracks.

    Two modes are possible:
      - positions_df is None  -> read from disk via load_positions_for_track.
      - positions_df provided -> filter within this preloaded DataFrame (e.g., 3 months).

    Parameters
    ----------
    track_table : DataFrame
        Result of build_ship_tracks (must contain 'track_id', 'segment_id', 'month').

    track_sampling :
        - [start, end] (list/tuple of 2 ints)
            * [0, 100] -> from the 1st to the 101st track (indices 0..100)
            * [0, -1]  -> from the 1st to the last
        - int
            * e.g. 20 -> randomly choose 20 tracks among the candidates.

    n_tracks_length : int, optional
        If not None, keep only tracks whose length (number of rows in
        track_table for that track_id) is exactly n_tracks_length.

    positions_df : DataFrame or None
        If provided, already contains the positions (e.g., 3 months) with at least
        'date_time_utc', 'shipid', 'latitude', 'longitude'.
        This DataFrame is filtered instead of calling load_positions_for_track.

    base_path, chunksize, progress :
        Used only if positions_df is None (reading from disk).

    point_stride : int
        Subsampling: keep 1 point every `point_stride` points per track.

    random_state : int or None
        Seed for random sampling of track_id (when track_sampling is an int).
    """

    for col in ("track_id", "segment_id", "month"):
        if col not in track_table.columns:
            raise KeyError(f"track_table must contain column '{col}'.")

    size_by_track = track_table.groupby("track_id").size()
    candidate_ids = size_by_track.index

    if n_tracks_length is not None:
        candidate_ids = candidate_ids[size_by_track.loc[candidate_ids] >= n_tracks_length]

    candidate_ids = list(candidate_ids)
    if not candidate_ids:
        return pd.DataFrame()

    # Define a default sampling if nothing is provided (Safety net)
    if specific_track_ids is None and track_sampling is None:
        track_sampling = 20  # Default behavior: grab 20 random tracks
        print("Info: No selection criteria provided. Defaulting to random sampling of 20 tracks.")

    selected_ids = []

    # Case 1: Explicit selection via specific IDs (Priority)
    if specific_track_ids is not None:
        # Create a set for fast lookup to ensure valid IDs
        candidate_set = set(candidate_ids)

        # Keep only the IDs that actually exist in the track_table
        selected_ids = [tid for tid in specific_track_ids if tid in candidate_set]

        if not selected_ids:
            print("Warning: None of the requested track_ids were found in the track_table.")
            return pd.DataFrame()

    # Case 2: Slicing / Interval (e.g., [0, 100])
    elif isinstance(track_sampling, (list, tuple)) and len(track_sampling) == 2:
        candidate_ids_sorted = sorted(candidate_ids)
        start, end = track_sampling

        if start < 0:
            raise ValueError("start index in track_sampling must be >= 0")

        if end == -1:
            end = len(candidate_ids_sorted) - 1

        if end < start:
            raise ValueError("end index must be >= start index in track_sampling")

        # Clamp the end index to the list boundaries
        end = min(end, len(candidate_ids_sorted) - 1)
        selected_ids = candidate_ids_sorted[start:end + 1]

    # Case 3: Random sampling (Integer)
    elif isinstance(track_sampling, int):
        candidate_ids_sorted = sorted(candidate_ids)
        k = track_sampling

        if k <= 0:
            raise ValueError("track_sampling int must be > 0")

        if k >= len(candidate_ids):
            # If requested samples > available tracks, return all
            selected_ids = candidate_ids_sorted
        else:
            if random_state is not None:
                np.random.seed(random_state)

            # Randomly choose k tracks without replacement
            selected_ids = list(np.random.choice(candidate_ids_sorted, size=k, replace=False))

    else:
        raise TypeError(
            "Invalid arguments: 'track_sampling' must be an int or a list/tuple of length 2, "
            "unless 'specific_track_ids' is provided."
        )

    if not selected_ids:
        return pd.DataFrame()

    # prepare positions_df if provided
    df_pos = None
    if positions_df is not None:
        df_pos = positions_df.copy()
        if "date_time_utc" not in df_pos.columns:
            raise KeyError("positions_df must contain 'date_time_utc'.")
        if "shipid" not in df_pos.columns:
            raise KeyError("positions_df must contain 'shipid'.")
        # Add 'month' if missing, to join as in your notebook
        if "month" not in df_pos.columns:
            df_pos["month"] = pd.to_datetime(df_pos["date_time_utc"]).dt.strftime("%Y-%m")


    if df_pos is None:
        # Mode 1: Read from disk (BATCH MODE)
        # We pass the full list of selected_ids at once.
        # This triggers the optimized single-pass file reading.
        work = load_track_data(
            track_ids=selected_ids,
            track_table=track_table,
            base_path=base_path,
            progress=progress,
            chunksize=chunksize
        )

    else:
        # Mode 2: Filtering pre-loaded dataframe (unchanged logic, just vectorized)
        tt_subset = track_table[track_table["track_id"].isin(selected_ids)][["month", "segment_id", "track_id"]].copy()

        if tt_subset.empty:
            return pd.DataFrame()

        work = df_pos.merge(
            tt_subset,
            left_on=["month", "shipid"],
            right_on=["month", "segment_id"],
            how="inner",
        )
        work = work.drop(columns=["segment_id"], errors="ignore")

        if preprocess_positions:
            work = remove_unrealistic_points(work)

    if work is None or work.empty:
        return pd.DataFrame()

    # Subsample points per track by point_stride
    if "date_time_utc" not in work.columns:
        raise KeyError("Resulting positions must contain 'date_time_utc' to be sampled by time.")

    work = (
        work.sort_values(["track_id", "date_time_utc"])
            .groupby("track_id", group_keys=False)
            .apply(lambda g: g.iloc[::point_stride])
    )

    #  Arctic Zone (optional, here we use your criterion) 
    # if {"latitude", "longitude"}.issubset(work.columns):
    #     work = work.query("latitude >= 60 and longitude >= -80 and longitude <= 40")

    work = work.reset_index(drop=True)

    return work
