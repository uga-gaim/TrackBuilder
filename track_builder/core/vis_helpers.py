# track_builder/core/vis_helpers.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union
import warnings
import pandas as pd

from track_builder.core.io_helpers import standardize_columns
from track_builder.config import DATE_CANDS
from track_builder.config import MAPBOX_STYLE_ALIASES
from plotly import graph_objects as go
from plotly import io as pio

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


def resolve_map_style(style: Optional[str]) -> str:
    if not style:
        return "open-street-map"
    key = str(style).strip().lower()
    return MAPBOX_STYLE_ALIASES.get(key, style)


def is_continuous(series: pd.Series, max_unique_for_categorical: int = 20) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) > max_unique_for_categorical
    return False


def discrete_palette(n: int) -> list[str]:
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    if n <= len(base):
        return base[:n]
    out = []
    k = 0
    while len(out) < n:
        out.append(base[k % len(base)])
        k += 1
    return out[:n]


def export_figure(fig: go.Figure, path: Union[str, Path]) -> None:
    """Export a Plotly figure to HTML/PNG/PDF based on file extension.
    Usage:
        export_figure(fig, "map.html")
        export_figure(fig, "map.png")   # requires kaleido
        export_figure(fig, "map.pdf")   # requires kaleido
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".html", ".htm"):
        pio.write_html(fig, file=str(path), auto_open=False, include_plotlyjs="cdn" if ext == ".html" else True)
    elif ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf"):
        fig.write_image(str(path))  # need to install kaleido
    else:
        raise ValueError(f"Unsupported export format: {ext}. Use .html, .png, .pdf, .svg, .jpg")
