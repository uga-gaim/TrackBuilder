from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union, List, Dict, Any
import os
import re
import glob

import numpy as np
import pandas as pd

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

# --- Minimal column normalization based on use-example.ipynb ---
COLMAP: Dict[str, str] = {
    # Time columns
    "date_time_utc": "date_time_utc",
    "datetime_utc":  "date_time_utc",
    "timestamp":     "date_time_utc",
    # Position columns
    "lat":        "latitude",
    "latitude":   "latitude",
    "lon":        "longitude",
    "long":       "longitude",
    "longitude":  "longitude",
    # Identifier and ship attributes
    "shipid":       "shipid",
    "astd_cat":     "astd_cat",
    "flagname":     "flagname",
    "iceclass":     "iceclass",
    "sizegroup_gt": "sizegroup_gt",
}
STR_COLS   = ("astd_cat", "flagname", "iceclass", "sizegroup_gt")
DATE_CANDS = ("date_time_utc", "datetime_utc", "timestamp")


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


def read_csv_auto(path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """
    Lightweight CSV reader with basic auto-detection logic.

    Order of attempts:
      1) sep=None (let pandas infer) + UTF-8
      2) sep=None + Latin-1
      3) sep=';' + Latin-1

    This is sufficient for most ASTD datasets.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8",
                           low_memory=False, **read_csv_kwargs)
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding="latin-1",
                               low_memory=False, **read_csv_kwargs)
        except Exception:
            return pd.read_csv(path, sep=";", encoding="latin-1",
                               low_memory=False, **read_csv_kwargs)


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


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip common string-based columns (preserving NA)."""
    for c in STR_COLS:
        if c in df.columns:
            # Use pandas StringDtype to keep <NA> instead of 'nan'
            df[c] = df[c].astype("string")
            df[c] = df[c].str.strip().str.lower()
    return df



def add_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'month' column (YYYY-MM) derived from date_time_utc."""
    if "date_time_utc" in df.columns:
        df["month"] = df["date_time_utc"].dt.to_period("M").astype(str)
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
