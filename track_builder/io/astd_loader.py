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





def load_positions_for_track(
        track_id: Optional[Union[str, int]],
        track_table: pd.DataFrame,
        *,
        base_path: Optional[Pathish] = None,
        progress: bool = True,
        chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Charge uniquement les positions ASTD nécessaires pour un track donné.

    Étapes :
      1) Récupérer, dans track_table, les (month, segment_id) pour ce track_id.
      2) Trouver les fichiers ASTD correspondants sur disque (année/mois) via
         iter_files + matches_year_month.
      3) Lire ces fichiers en chunks avec pandas.read_csv(..., chunksize=...),
         standardiser, parser les dates, et ne garder que les lignes dont shipid
         appartient aux segment_id du track.
      4) Concaténer tous les morceaux en un seul DataFrame.
    """
    if track_id is None:
        raise ValueError("track_id must not be None.")

    # --- Vérifications de base
    if "track_id" not in track_table.columns:
        raise KeyError("track_table must contain a 'track_id' column.")
    if "month" not in track_table.columns:
        raise KeyError("track_table must contain a 'month' column (e.g. '2019-07').")

    seg_col = "segment_id" if "segment_id" in track_table.columns else None
    if seg_col is None:
        raise KeyError("track_table must contain a 'segment_id' column.")

    # --- Sous-table pour ce track
    tt = track_table[track_table["track_id"].astype(str) == str(track_id)].copy()
    if tt.empty:
        raise ValueError(f"Track '{track_id}' not found in track_table.")

    # --- Segment IDs de ce track
    seg_ids = tt[seg_col].dropna().unique().tolist()
    if not seg_ids:
        return pd.DataFrame()

    seg_ids_str = {str(s) for s in seg_ids}

    # --- Extraire (année, mois) à partir de 'month'
    months_series = pd.to_datetime(tt["month"], errors="coerce").dropna()
    if months_series.empty:
        return pd.DataFrame()

    months_by_year: dict[int, set[int]] = {}
    for dt64 in months_series.unique():
        ts = pd.Timestamp(dt64)
        months_by_year.setdefault(ts.year, set()).add(ts.month)

    if not months_by_year:
        return pd.DataFrame()

    # --- Trouver les fichiers ASTD correspondants
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

    # --- Lecture filtrée en chunks (sans read_csv_auto)
    frames: list[pd.DataFrame] = []

    iterator = selected_files
    if progress and HAS_TQDM:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, total=len(selected_files),
                        desc=f"Loading positions for track {track_id}")

    for path in iterator:
        # On force sep=';' car c'est le cas typique pour ASTD.
        reader = pd.read_csv(
            path,
            sep=";",
            usecols=ASTD_USEFUL_COLS,
            dtype=ASTD_DTYPE_MAP,
            low_memory=False,
            chunksize=chunksize,
        )

        # reader est un TextFileReader → itérable de chunks
        for chunk in reader:
            if chunk is None or chunk.empty:
                continue

            # Standardisation + dates (comme dans load_astd_data)
            chunk = standardize_columns(chunk)
            chunk = parse_dates(chunk)

            if "shipid" not in chunk.columns:
                continue

            # Filtrer uniquement les segments de ce track
            mask = chunk["shipid"].astype(str).isin(seg_ids_str)
            sub = chunk.loc[mask]
            if not sub.empty:
                frames.append(sub)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Tri temporel pratique pour la visualisation
    if "date_time_utc" in out.columns:
        out = out.sort_values("date_time_utc").reset_index(drop=True)

    return out