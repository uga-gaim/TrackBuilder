# io/astd_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, List, Any, Sequence
import pandas as pd
import numpy as np

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





def load_positions_for_track(
        track_id: Optional[Union[str, int]],
        track_table: pd.DataFrame,
        *,
        base_path: Optional[Pathish] = None,
        progress: bool = True,
        chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Load only the ASTD positions needed for a given track.

    Steps:
      1) Retrieve from track_table the (month, segment_id) entries for this track_id.
      2) Find the corresponding ASTD files on disk (year/month) using
         iter_files + matches_year_month.
      3) Read these files in chunks with pandas.read_csv(..., chunksize=...),
         standardize columns, parse dates, and keep only rows whose shipid
         belongs to the track's segment_id values.
      4) Concatenate all chunks into a single DataFrame.
    """
    if track_id is None:
        raise ValueError("track_id must not be None.")

    if "track_id" not in track_table.columns:
        raise KeyError("track_table must contain a 'track_id' column.")
    if "month" not in track_table.columns:
        raise KeyError("track_table must contain a 'month' column (e.g. '2019-07').")

    seg_col = "segment_id" if "segment_id" in track_table.columns else None
    if seg_col is None:
        raise KeyError("track_table must contain a 'segment_id' column.")

    # subset for this track_id
    tt = track_table[track_table["track_id"].astype(str) == str(track_id)].copy()
    if tt.empty:
        raise ValueError(f"Track '{track_id}' not found in track_table.")

    # Segment IDs for this track
    seg_ids = tt[seg_col].dropna().unique().tolist()
    if not seg_ids:
        return pd.DataFrame()

    seg_ids_str = {str(s) for s in seg_ids}

    # month to load
    months_series = pd.to_datetime(tt["month"], errors="coerce").dropna()
    if months_series.empty:
        return pd.DataFrame()

    months_by_year: dict[int, set[int]] = {}
    for dt64 in months_series.unique():
        ts = pd.Timestamp(dt64)
        months_by_year.setdefault(ts.year, set()).add(ts.month)

    if not months_by_year:
        return pd.DataFrame()

    # find relevant files
    root = Path(base_path).resolve() if base_path is not None else DEFAULT_DATA_PATH
    all_files = iter_files(root, pattern=None)

    selected_files = []
    for p in all_files:
        fname = p.name
        for year, months in months_by_year.items():
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
                        desc=f"Loading positions for track {track_id}")

    for path in iterator:
        reader = pd.read_csv(
            path,
            sep=";",
            usecols=ASTD_USEFUL_COLS,
            dtype=ASTD_DTYPE_MAP,
            low_memory=False,
            chunksize=chunksize,
        )

        # reader is a TextFileReader → iterable of chunks
        for chunk in reader:
            if chunk is None or chunk.empty:
                continue

            # Standardization + dates (as in load_astd_data)
            chunk = standardize_columns(chunk)
            chunk = parse_dates(chunk)

            if "shipid" not in chunk.columns:
                continue

            # Filter only the segments of this track
            mask = chunk["shipid"].astype(str).isin(seg_ids_str)
            sub = chunk.loc[mask]
            if not sub.empty:
                frames.append(sub)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Temporal sorting useful for visualization
    if "date_time_utc" in out.columns:
        out = out.sort_values("date_time_utc").reset_index(drop=True)

    return out


def build_light_multi_track_positions(
        track_table: pd.DataFrame,
        track_sampling: Union[int, Sequence[int]],
        *,
        positions_df: Optional[pd.DataFrame] = None,
        n_tracks_length: Optional[int] = None,
        base_path: Optional[Pathish] = None,
        chunksize: int = 50_000,
        progress: bool = True,
        point_stride: int = 10,
        random_state: Optional[int] = 42,
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

    # sample track_ids
    candidate_ids_sorted = sorted(candidate_ids)

    if isinstance(track_sampling, (list, tuple)) and len(track_sampling) == 2:
        start, end = track_sampling
        if start < 0:
            raise ValueError("start index in track_sampling must be >= 0")

        if end == -1:
            end = len(candidate_ids_sorted) - 1

        if end < start:
            raise ValueError("end index must be >= start index in track_sampling")

        end = min(end, len(candidate_ids_sorted) - 1)
        selected_ids = candidate_ids_sorted[start:end + 1]

    elif isinstance(track_sampling, int):
        k = track_sampling
        if k <= 0:
            raise ValueError("track_sampling int must be > 0")

        if k >= len(candidate_ids):
            selected_ids = candidate_ids_sorted
        else:
            if random_state is not None:
                np.random.seed(random_state)
            selected_ids = list(np.random.choice(candidate_ids_sorted, size=k, replace=False))
    else:
        raise TypeError(
            "track_sampling must be either an int or a list/tuple of length 2 "
            "(e.g., [0, 100], [0, -1], or 20)."
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

    # Load / filter positions for each track and concatenate
    all_frames: list[pd.DataFrame] = []

    for tid in selected_ids:
        if df_pos is None:
            # --- Mode reading from disk ---
            df_t = load_positions_for_track(
                track_id=tid,
                track_table=track_table,
                base_path=base_path,
                progress=progress,
                chunksize=chunksize,
            )
        else:
            # --- Mode filtering in already loaded positions_df ---
            tt_tid = track_table[track_table["track_id"] == tid][["month", "segment_id"]].copy()
            if tt_tid.empty:
                continue

            df_t = df_pos.merge(
                tt_tid,
                left_on=["month", "shipid"],
                right_on=["month", "segment_id"],
                how="inner",
            )
            # You can remove segment_id from the final merge if you want
            df_t = df_t.drop(columns=["segment_id"], errors="ignore")

        if df_t is None or df_t.empty:
            continue

        df_t = df_t.copy()
        df_t["track_id"] = tid
        all_frames.append(df_t)

    if not all_frames:
        return pd.DataFrame()

    work = pd.concat(all_frames, ignore_index=True)

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

    # Remove large jumps per track 
    # def remove_big_jumps(g: pd.DataFrame, max_deg: float = 10.0) -> pd.DataFrame:
    #     g = g.sort_values("date_time_utc")
    #     if "latitude" not in g.columns or "longitude" not in g.columns:
    #         return g
    #     dlat = g["latitude"].diff().abs()
    #     dlon = g["longitude"].diff().abs()
    #     mask = dlat.isna() | ((dlat <= max_deg) & (dlon <= max_deg))
    #     return g[mask]
    #
    # work = work.groupby("track_id", group_keys=False).apply(remove_big_jumps)
    work = work.reset_index(drop=True)

    return work
