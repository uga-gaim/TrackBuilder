from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, List, Sequence, Any, Callable
import os
import re
import glob

import numpy as np
import pandas as pd

from track_builder.config import COLMAP, DATE_CANDS, STR_COLS

# Optional .env loader for ASTD_DATA_PATH
# (Safe to ignore if python-dotenv is not installed)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# Optional progress bar support using tqdm
try:
    from tqdm import tqdm  # type: ignore

    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

Pathish = Union[str, Path]
DEFAULT_DATA_PATH = Path(os.getenv("ASTD_DATA_PATH", "./data")).resolve()


def iter_files(file_paths: Union[Pathish, Iterable[Pathish], None],
               pattern: Optional[str]) -> List[Path]:
    """
    Helper function to gather all CSV files to load.

    Supports:
      - Single file
      - List of files
      - Directory with optional pattern (defaults to *.csv)
      - None → uses DEFAULT_DATA_PATH
    """

    def dir_glob(d: Path, patt: Optional[str]) -> List[Path]:
        patt = patt or "*.csv"
        return [Path(p) for p in glob.glob(str(d / patt))]

    if file_paths is None:
        files = dir_glob(DEFAULT_DATA_PATH, pattern)
    elif isinstance(file_paths, (str, Path)):
        p = Path(file_paths)
        files = dir_glob(p, pattern) if p.is_dir() else [p]
    else:
        files = []
        for fp in file_paths:
            p = Path(fp)
            files += dir_glob(p, pattern) if p.is_dir() else [p]

    files = sorted({f.resolve() for f in files if f.exists()})
    if not files:
        raise FileNotFoundError("No CSV files found (check path/pattern or ASTD_DATA_PATH).")
    return files


def read_csv_auto(path, **kwargs):
    """
    Reads a CSV file by automatically trying to guess the separator.
    """
    # Semicolon is the most likely separator for ASTD data, so we try it first.
    separators_to_try = [';', ',']
    
    for sep in separators_to_try:
        try:
            # We pass the kwargs (like low_memory=False) to the call to read_csv
            df = pd.read_csv(path, sep=sep, **kwargs)

            if len(df.columns) == 1:
                raise ValueError("Inference failed, produced a single column.")

            return df
        except Exception as e:
            print(f"INFO: Failed to read with separator '{sep}'. Pandas error: {e}")
            continue

    raise ValueError(f"Could not parse CSV {path} with any tried configuration.")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names defined in COLMAP."""
    mapping = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in COLMAP:
            mapping[c] = COLMAP[key]
    return df.rename(columns=mapping) if mapping else df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all potential date columns to datetime (UTC)."""
    for cand in DATE_CANDS:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce", utc=True)
    if "date_time_utc" not in df.columns:
        for cand in DATE_CANDS:
            if cand in df.columns:
                df = df.rename(columns={cand: "date_time_utc"})
                break
    return df


def validate_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert and validate coordinate columns.
    Removes invalid latitude/longitude values.
    """
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df.loc[(df["latitude"] < -90) | (df["latitude"] > 90), "latitude"] = np.nan
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df.loc[(df["longitude"] < -180) | (df["longitude"] > 180), "longitude"] = np.nan
    if {"latitude", "longitude"} <= set(df.columns):
        df = df.dropna(subset=["latitude", "longitude"])
    return df


def quality_filter(df: pd.DataFrame, threshold_minutes: int) -> pd.DataFrame:
    """
    Remove (shipid, month) groups whose average sampling frequency
    is lower (more sparse) than 'threshold_minutes'.
    """
    if threshold_minutes <= 0 or "date_time_utc" not in df.columns or "shipid" not in df.columns:
        return df

    tmp = df.copy()
    tmp["year_month"] = tmp["date_time_utc"].dt.strftime("%Y-%m")
    keep = np.zeros(len(tmp), dtype=bool)

    for (_, _m), g in tmp.groupby(["shipid", "year_month"]):
        g = g.sort_values("date_time_utc")
        if len(g) < 2:
            continue
        deltas = g["date_time_utc"].diff().dt.total_seconds().dropna() / 60.0
        if len(deltas) and deltas.mean() <= threshold_minutes:
            keep[g.index] = True

    return tmp.loc[keep].drop(columns=["year_month"])


def matches_year_month(filename: str, year: int, months: set[int]) -> bool:
    """Return True if filename contains a pattern matching year+month."""
    m = re.findall(r"((?:19|20)\d{2})[-_]?(\d{2})", filename)
    return any(int(y) == year and int(mm) in months for y, mm in m)


def make_skipper(frac: float, seed: Optional[int]) -> Callable[[int], bool]:
    """Return a skiprows callable: True = skip the row (header is never skipped)."""
    if not (0.0 < frac <= 1.0):
        raise ValueError("If 'sampling' is a float, it must be in (0, 1].")
    rng = np.random.default_rng(seed)

    def skipper(i: int) -> bool:
        if i == 0:  # header row
            return False
        return rng.random() > frac

    return skipper


def sample_by_day_of_month(df: pd.DataFrame, indices: Sequence[int]) -> pd.DataFrame:
    """
    Select rows for specific day-of-month indices.
    Assumes a single month per file.
    If 'date_time_utc' is missing or not datetime, parse dates first.
    """
    # Ensure datetime column
    if 'date_time_utc' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date_time_utc']):
        df = parse_dates(df)  # <- ensures/renames to 'date_time_utc' if possible

    if 'date_time_utc' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date_time_utc']):
        print("Warning: 'date_time_utc' column not found or not a datetime type. Cannot sample by day.")
        return df
    if df.empty:
        return df

    # Enforce assignment bounds: -31..30
    normalized_idxs: list[int] = []
    for v in indices:
        if not isinstance(v, int):
            raise ValueError("If 'sampling' is a sequence, it must contain integers.")
        if v < -31 or v > 30:
            raise ValueError("Day indices must be between -31 and 30.")
        normalized_idxs.append(v)

    unique_days = sorted(df['date_time_utc'].dt.day.dropna().unique().tolist())
    if not unique_days:
        return df.iloc[0:0].copy()

    n = len(unique_days)
    chosen: set[int] = set()
    for k in set(normalized_idxs):
        pos = k if k >= 0 else n + k
        if 0 <= pos < n:
            chosen.add(int(unique_days[pos]))

    if not chosen:
        return df.iloc[0:0].copy()

    return df[df['date_time_utc'].dt.day.isin(chosen)].copy()
