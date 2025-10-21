from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union, Literal, Any, Tuple, Sequence
import warnings

import numpy as np
import pandas as pd

from track_builder.core.io_helpers import standardize_columns
from track_builder.config import DATE_CANDS
from track_builder.config import MAPBOX_STYLE_ALIASES
from plotly import graph_objects as go
from plotly import io as pio
import os

TRACK_ID_CANDS = ("track_id",)
SEGMENT_ID_CANDS = ("segment_id", "shipid")
MONTH_CANDS = ("month",)

_LAT_COLS = ("latitude", "lat", "cell_ll_lat")
_LON_COLS = ("longitude", "lon", "lng", "cell_ll_lon")
_TIME_COLS = ("date_time_utc", "timestamp", "date", "time")
_TRACK_COLS = ("track_id", "trackid", "track")


try:
    from . import config
except Exception:
    config = None


def get_mapbox_token() -> str | None:
    tok = os.getenv("MAPBOX_TOKEN")
    if tok:
        return tok
    if config and getattr(config, "MAPBOX_TOKEN", None):
        return config.MAPBOX_TOKEN
    return None


def first_present(df: pd.DataFrame, cands: tuple[str, ...]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None


def get_lat_lon_time_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    lat = next((c for c in _LAT_COLS if c in df.columns), None)
    lon = next((c for c in _LON_COLS if c in df.columns), None)
    tcol = next((c for c in _TIME_COLS if c in df.columns), None)
    if not (lat and lon and tcol):
        raise ValueError("Colonnes lat/lon/temps introuvables.")
    return lat, lon, tcol


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


def get_track_col(df: pd.DataFrame) -> Optional[str]:
    return next((c for c in _TRACK_COLS if c in df.columns), None)


def ensure_datetime(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[tcol]):
        df = df.copy()
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    return df


def build_color_spec(
        df: pd.DataFrame,
        color_by: Optional[str],
        *,
        mode: Literal["auto", "categorical", "continuous"] = "auto",
        max_categories: int = 20,
        colorscale: str = "Viridis",
) -> Dict[str, Any]:
    """
    Returns a dict:
      {
        'enabled': bool,
        'is_cont': bool,
        'series': pd.Series | None,     # possibly cast to str if categorical
        'cats': list[str] | None,       # order of retained categories
        'color_map': dict | None,       # category -> hex color
        'coloraxis_kwargs': dict | None # for fig.update_layout(coloraxis=..)
      }
    """
    if not color_by or color_by not in df.columns:
        return dict(enabled=False, is_cont=False, series=None, cats=None, color_map=None, coloraxis_kwargs=None)

    s = df[color_by]
    if mode == "categorical":
        is_cont = False
    elif mode == "continuous":
        is_cont = True
    else:  # auto
        is_cont = is_continuous(s, max_unique_for_categorical=max_categories)

    if is_cont:
        return dict(
            enabled=True,
            is_cont=True,
            series=s, cats=None, color_map=None,
            coloraxis_kwargs=dict(colorbar_title=color_by, colorscale=colorscale),
        )

    # categorical: normalize to str & truncate to top categories
    s_str = s.astype(str)
    counts = s_str.value_counts(dropna=False)
    keep = counts.index[:max_categories].tolist()
    cats = [str(c) for c in keep]
    # others → "Other" (optional)
    other_label = None
    if len(counts) > max_categories:
        other_label = "Other"
        cats.append(other_label)

    # color map for categories
    palette = discrete_palette(len(cats))
    color_map = dict(zip(cats, palette))

    # grouped series (replace out-of-top with "Other")
    if other_label:
        mask_keep = s_str.isin(keep)
        s_grouped = s_str.where(mask_keep, other_label)
    else:
        s_grouped = s_str

    return dict(
        enabled=True,
        is_cont=False,
        series=s_grouped, cats=cats, color_map=color_map,
        coloraxis_kwargs=None,
    )


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


def build_hover_customdata(
    df_like: pd.DataFrame,
    extra_cols_priority: Optional[Sequence[str]],
) -> tuple[Optional[np.ndarray], str, list[str]]:
    """
    Builds:
      - customdata: np.ndarray | None (for Plotly)
      - suffix: str (to append in the hovertemplate)
      - used_cols: list[str] actually used (present in df_like)

    extra_cols_priority: desired order of columns to display (e.g. ['astd_cat','ship_type', ...]).
    Missing columns are ignored silently. Values are cast to str.
    """
    default_cols = ['astd_cat', 'shipid', 'flagname']
    if not extra_cols_priority:
        extra_cols_priority = default_cols

    used = [c for c in extra_cols_priority if c in df_like.columns]
    if not used:
        return None, "", []

    cdata = df_like[used].astype(str).to_numpy()
    lines = [f"<br><b>{col}:</b> %{{customdata[{i}]}}" for i, col in enumerate(used)]
    return cdata, "".join(lines), used