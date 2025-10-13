from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, List
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
        except Exception as e:  # On donne un nom à l'exception pour pouvoir l'afficher
            print(f"INFO: Failed to read with separator '{sep}'. Pandas error: {e}")
            continue  # On continue d'essayer les autres séparateurs

    # Si aucun des séparateurs n'a fonctionné, on lève une erreur
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
    tmp["_month"] = tmp["date_time_utc"].dt.to_period("M").astype(str)
    keep = np.zeros(len(tmp), dtype=bool)

    for (_, _m), g in tmp.groupby(["shipid", "_month"]):
        g = g.sort_values("date_time_utc")
        if len(g) < 2:
            continue
        deltas = g["date_time_utc"].diff().dt.total_seconds().dropna() / 60.0
        if len(deltas) and deltas.mean() <= threshold_minutes:
            keep[g.index] = True

    return tmp.loc[keep].drop(columns=["_month"])


def matches_year_month(filename: str, year: int, months: set[int]) -> bool:
    """Return True if filename contains a pattern matching year+month."""
    m = re.findall(r"((?:19|20)\d{2})[-_]?(\d{2})", filename)
    return any(int(y) == year and int(mm) in months for y, mm in m)
